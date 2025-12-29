"""
Q-Learning Algorithm

Off-policy TD control that learns optimal Q-values directly.
Uses maximum Q-value for next state, regardless of policy.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class QLearningResult:
    """Results from Q-Learning training."""
    q_function: Dict[str, Dict[int, float]]
    policy: Dict[str, int]
    episode_rewards: List[float]
    episode_lengths: List[int]
    td_errors: List[float]
    convergence_history: List[Dict[str, Any]]
    episodes: int


class QLearning:
    """
    Q-Learning algorithm implementation.
    
    Off-policy TD control that updates Q-values:
    Q(s,a) <- Q(s,a) + alpha * [R + gamma * max_a' Q(s',a') - Q(s,a)]
    
    Uses maximum Q-value for next state (greedy), while behavior policy
    can be epsilon-greedy for exploration.
    
    Attributes:
        gamma: Discount factor
        alpha: Learning rate
        epsilon: Exploration rate for behavior policy
        n_episodes: Number of episodes to train
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        n_episodes: int = 1000,
        **kwargs
    ):
        """
        Initialize Q-Learning.
        
        Args:
            gamma: Discount factor (0-1)
            alpha: Learning rate (0-1)
            epsilon: Initial exploration rate (0-1)
            epsilon_decay: Rate of epsilon decay
            epsilon_min: Minimum epsilon value
            n_episodes: Number of episodes
        """
        self.gamma = gamma
        self.alpha = alpha
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        self.n_episodes = n_episodes
        
        # Training state
        self.Q: Dict[str, Dict[int, float]] = defaultdict(lambda: defaultdict(float))
        self.policy: Dict[str, int] = {}
        self.state_action_visits: Dict[Tuple[str, int], int] = defaultdict(int)
        
        # Tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.td_errors: List[float] = []
        self.convergence_history: List[Dict[str, Any]] = []
        self.episode: int = 0
        
        # Callbacks
        self._on_episode: Optional[Callable] = None
        self._stop_requested: bool = False
    
    def train(
        self,
        env,
        on_episode: Optional[Callable] = None
    ) -> QLearningResult:
        """
        Train using Q-Learning.
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_episode: Callback called after each episode
            
        Returns:
            QLearningResult with Q-function, policy, and metrics
        """
        self._on_episode = on_episode
        self._stop_requested = False
        self.epsilon = self.initial_epsilon
        
        n_actions = env.n_actions
        self.Q = defaultdict(lambda: {a: 0.0 for a in range(n_actions)})
        self.policy = {}
        self.state_action_visits = defaultdict(int)
        self.episode_rewards = []
        self.episode_lengths = []
        self.td_errors = []
        self.convergence_history = []
        
        for episode in range(self.n_episodes):
            if self._stop_requested:
                break
                
            self.episode = episode + 1
            
            # Run episode
            total_reward, length, episode_td_errors = self._run_episode(env)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(length)
            self.td_errors.extend(episode_td_errors)
            
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            
            # Update policy
            self._update_policy(env)
            
            # Record metrics periodically
            if episode % max(1, self.n_episodes // 100) == 0:
                metrics = {
                    'episode': self.episode,
                    'total_reward': total_reward,
                    'episode_length': length,
                    'avg_reward_last_100': np.mean(self.episode_rewards[-100:]),
                    'avg_td_error': np.mean(episode_td_errors) if episode_td_errors else 0,
                    'epsilon': self.epsilon,
                    'q_values': {s: dict(q) for s, q in self.Q.items()},
                    'policy': dict(self.policy),
                }
                self.convergence_history.append(metrics)
                
                if self._on_episode:
                    self._on_episode(metrics)
        
        return QLearningResult(
            q_function={s: dict(q) for s, q in self.Q.items()},
            policy=dict(self.policy),
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            td_errors=self.td_errors,
            convergence_history=self.convergence_history,
            episodes=self.episode
        )
    
    def _run_episode(self, env) -> Tuple[float, int, List[float]]:
        """Run single episode with Q-Learning updates."""
        state, _ = env.reset()
        done = False
        total_reward = 0
        length = 0
        td_errors = []
        
        while not done:
            # Select action (epsilon-greedy behavior policy)
            action = self._select_action(env, state)
            
            result = env.step(action)
            
            # Q-Learning update (off-policy: use max Q for next state)
            if result.done:
                max_next_q = 0
            else:
                valid_next_actions = env.get_valid_actions(result.next_state)
                if valid_next_actions:
                    max_next_q = max(self.Q[result.next_state].get(a, 0) for a in valid_next_actions)
                else:
                    max_next_q = 0
            
            td_error = result.reward + self.gamma * max_next_q - self.Q[state][action]
            self.Q[state][action] += self.alpha * td_error
            
            td_errors.append(abs(td_error))
            self.state_action_visits[(state, action)] += 1
            
            total_reward += result.reward
            length += 1
            
            state = result.next_state
            done = result.done
            
            if length > 10000:
                break
        
        return total_reward, length, td_errors
    
    def _select_action(self, env, state: str) -> int:
        """Select action using epsilon-greedy behavior policy."""
        valid_actions = env.get_valid_actions(state)
        
        if not valid_actions:
            return 0
        
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        # Greedy action
        return max(valid_actions, key=lambda a: self.Q[state].get(a, 0))
    
    def _update_policy(self, env) -> None:
        """Update policy to be greedy w.r.t. Q-function."""
        for state in self.Q:
            valid_actions = env.get_valid_actions(state)
            if valid_actions:
                self.policy[state] = max(valid_actions, key=lambda a: self.Q[state].get(a, 0))
    
    def stop(self) -> None:
        """Request training to stop."""
        self._stop_requested = True
    
    def set_alpha(self, alpha: float) -> None:
        """Update learning rate."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon."""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def get_action(self, state: str) -> int:
        """Get action from learned policy (greedy)."""
        return self.policy.get(state, 0)
    
    def get_q_value(self, state: str, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.Q.get(state, {}).get(action, 0.0)
    
    def get_value(self, state: str) -> float:
        """Get value of a state (max Q-value)."""
        if state in self.Q:
            return max(self.Q[state].values())
        return 0.0
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return {
            'episode': self.episode,
            'n_episodes': self.n_episodes,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
            'epsilon': self.epsilon,
            'alpha': self.alpha,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        return {
            'algorithm': 'q_learning',
            'gamma': self.gamma,
            'alpha': self.alpha,
            'epsilon': self.epsilon,
            'Q': {s: dict(q) for s, q in self.Q.items()},
            'policy': dict(self.policy),
            'episode': self.episode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'QLearning':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], alpha=data['alpha'], epsilon=data['epsilon'])
        algo.Q = defaultdict(lambda: defaultdict(float),
                            {s: defaultdict(float, q) for s, q in data['Q'].items()})
        algo.policy = data['policy']
        algo.episode = data['episode']
        return algo

"""
TD(0) Algorithm

Temporal Difference learning with single-step bootstrapping.
Model-free algorithm that updates values after each step.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class TDZeroResult:
    """Results from TD(0) training."""
    value_function: Dict[str, float]
    policy: Dict[str, int]
    episode_rewards: List[float]
    episode_lengths: List[int]
    td_errors: List[float]
    convergence_history: List[Dict[str, Any]]
    episodes: int


class TDZero:
    """
    TD(0) algorithm implementation.
    
    Updates value function using single-step TD error:
    V(s) <- V(s) + alpha * [R + gamma * V(s') - V(s)]
    
    Attributes:
        gamma: Discount factor
        alpha: Learning rate
        n_episodes: Number of episodes to train
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        n_episodes: int = 1000,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        **kwargs
    ):
        """
        Initialize TD(0).
        
        Args:
            gamma: Discount factor (0-1)
            alpha: Learning rate (0-1)
            n_episodes: Number of episodes
            epsilon: Exploration rate for action selection
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.gamma = gamma
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
        
        # Training state
        self.V: Dict[str, float] = defaultdict(float)
        self.policy: Dict[str, int] = {}
        self.state_visits: Dict[str, int] = defaultdict(int)
        
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
    ) -> TDZeroResult:
        """
        Train using TD(0).
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_episode: Callback called after each episode
            
        Returns:
            TDZeroResult with value function, policy, and metrics
        """
        self._on_episode = on_episode
        self._stop_requested = False
        self.epsilon = self.initial_epsilon
        
        self.V = defaultdict(float)
        self.policy = {}
        self.state_visits = defaultdict(int)
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
            
            # Extract policy from V
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
                    'value_function': dict(self.V),
                    'policy': dict(self.policy),
                }
                self.convergence_history.append(metrics)
                
                if self._on_episode:
                    self._on_episode(metrics)
        
        return TDZeroResult(
            value_function=dict(self.V),
            policy=dict(self.policy),
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            td_errors=self.td_errors,
            convergence_history=self.convergence_history,
            episodes=self.episode
        )
    
    def _run_episode(self, env) -> Tuple[float, int, List[float]]:
        """Run single episode with TD(0) updates."""
        state, _ = env.reset()
        done = False
        total_reward = 0
        length = 0
        td_errors = []
        
        while not done:
            action = self._select_action(env, state)
            result = env.step(action)
            
            # TD(0) update
            next_v = 0 if result.done else self.V[result.next_state]
            td_error = result.reward + self.gamma * next_v - self.V[state]
            self.V[state] += self.alpha * td_error
            
            td_errors.append(abs(td_error))
            self.state_visits[state] += 1
            
            total_reward += result.reward
            length += 1
            state = result.next_state
            done = result.done
            
            if length > 10000:
                break
        
        return total_reward, length, td_errors
    
    def _select_action(self, env, state: str) -> int:
        """Select action using epsilon-greedy."""
        valid_actions = env.get_valid_actions(state)
        
        if not valid_actions:
            return 0
        
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        if state in self.policy:
            return self.policy[state]
        
        return np.random.choice(valid_actions)
    
    def _update_policy(self, env) -> None:
        """Update policy based on value function."""
        for state in self.V:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                continue
            
            # Evaluate each action
            best_action = valid_actions[0]
            best_value = float('-inf')
            
            for action in valid_actions:
                # Estimate value of action using expected next states
                try:
                    transitions = env.get_transition_prob(state, action)
                    action_value = sum(
                        prob * (reward + self.gamma * self.V.get(next_s, 0))
                        for next_s, prob, reward in transitions
                    )
                    if action_value > best_value:
                        best_value = action_value
                        best_action = action
                except:
                    pass
            
            self.policy[state] = best_action
    
    def stop(self) -> None:
        """Request training to stop."""
        self._stop_requested = True
    
    def set_alpha(self, alpha: float) -> None:
        """Update learning rate (for real-time parameter updates)."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon (for real-time parameter updates)."""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def get_action(self, state: str) -> int:
        """Get action from learned policy."""
        return self.policy.get(state, 0)
    
    def get_value(self, state: str) -> float:
        """Get value of a state."""
        return self.V.get(state, 0.0)
    
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
            'algorithm': 'td_zero',
            'gamma': self.gamma,
            'alpha': self.alpha,
            'V': dict(self.V),
            'policy': dict(self.policy),
            'episode': self.episode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TDZero':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], alpha=data['alpha'])
        algo.V = defaultdict(float, data['V'])
        algo.policy = data['policy']
        algo.episode = data['episode']
        return algo

"""
N-Step TD Algorithm

Temporal Difference learning with n-step bootstrapping.
Bridges between TD(0) and Monte Carlo methods.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass
from collections import defaultdict, deque


@dataclass
class TDNStepResult:
    """Results from N-Step TD training."""
    value_function: Dict[str, float]
    policy: Dict[str, int]
    episode_rewards: List[float]
    episode_lengths: List[int]
    n_step_returns: List[float]
    convergence_history: List[Dict[str, Any]]
    episodes: int


class TDNStep:
    """
    N-Step TD algorithm implementation.
    
    Uses n-step returns for value updates:
    G_t:t+n = R_{t+1} + gamma*R_{t+2} + ... + gamma^{n-1}*R_{t+n} + gamma^n*V(S_{t+n})
    
    Attributes:
        gamma: Discount factor
        alpha: Learning rate
        n: Number of steps for bootstrapping
        n_episodes: Number of episodes to train
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        alpha: float = 0.1,
        n: int = 4,
        n_episodes: int = 1000,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        **kwargs
    ):
        """
        Initialize N-Step TD.
        
        Args:
            gamma: Discount factor (0-1)
            alpha: Learning rate (0-1)
            n: Number of steps
            n_episodes: Number of episodes
            epsilon: Exploration rate
            epsilon_decay: Epsilon decay rate
            epsilon_min: Minimum epsilon
        """
        self.gamma = gamma
        self.alpha = alpha
        self.n = n
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
        self.n_step_returns: List[float] = []
        self.convergence_history: List[Dict[str, Any]] = []
        self.episode: int = 0
        
        # Callbacks
        self._on_episode: Optional[Callable] = None
        self._stop_requested: bool = False
    
    def train(
        self,
        env,
        on_episode: Optional[Callable] = None
    ) -> TDNStepResult:
        """
        Train using N-Step TD.
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_episode: Callback called after each episode
            
        Returns:
            TDNStepResult with value function, policy, and metrics
        """
        self._on_episode = on_episode
        self._stop_requested = False
        self.epsilon = self.initial_epsilon
        
        self.V = defaultdict(float)
        self.policy = {}
        self.state_visits = defaultdict(int)
        self.episode_rewards = []
        self.episode_lengths = []
        self.n_step_returns = []
        self.convergence_history = []
        
        for episode in range(self.n_episodes):
            if self._stop_requested:
                break
                
            self.episode = episode + 1
            
            # Run episode
            total_reward, length, returns = self._run_episode(env)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(length)
            self.n_step_returns.extend(returns)
            
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
                    'avg_n_step_return': np.mean(returns) if returns else 0,
                    'epsilon': self.epsilon,
                    'n': self.n,
                    'value_function': dict(self.V),
                    'policy': dict(self.policy),
                }
                self.convergence_history.append(metrics)
                
                if self._on_episode:
                    self._on_episode(metrics)
        
        return TDNStepResult(
            value_function=dict(self.V),
            policy=dict(self.policy),
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            n_step_returns=self.n_step_returns,
            convergence_history=self.convergence_history,
            episodes=self.episode
        )
    
    def _run_episode(self, env) -> Tuple[float, int, List[float]]:
        """Run single episode with n-step TD updates."""
        state, _ = env.reset()
        
        # Collect trajectory first
        states = [state]
        rewards = [0.0]  # R_0 is not used
        actions = []
        
        total_reward = 0
        max_steps = 10000
        
        # Collect episode trajectory
        for step in range(max_steps):
            action = self._select_action(env, state)
            result = env.step(action)
            
            actions.append(action)
            states.append(result.next_state)
            rewards.append(result.reward)
            total_reward += result.reward
            
            if result.done:
                break
            state = result.next_state
        
        T = len(states) - 1  # Terminal time step
        returns = []
        
        # Apply n-step TD updates
        for t in range(T):
            tau = t
            
            # Calculate n-step return G
            G = 0
            end_step = min(tau + self.n, T)
            
            for i in range(tau + 1, end_step + 1):
                G += (self.gamma ** (i - tau - 1)) * rewards[i]
            
            if tau + self.n < T:
                G += (self.gamma ** self.n) * self.V[states[tau + self.n]]
            
            returns.append(G)
            
            # Update V(S_tau)
            update_state = states[tau]
            self.V[update_state] += self.alpha * (G - self.V[update_state])
            self.state_visits[update_state] += 1
        
        return total_reward, T, returns
    
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
            
            best_action = valid_actions[0]
            best_value = float('-inf')
            
            for action in valid_actions:
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
        """Update learning rate."""
        self.alpha = max(0.0, min(1.0, alpha))
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon."""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def set_n(self, n: int) -> None:
        """Update n (requires restart to take effect properly)."""
        self.n = max(1, n)
    
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
            'n': self.n,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        return {
            'algorithm': 'td_nstep',
            'gamma': self.gamma,
            'alpha': self.alpha,
            'n': self.n,
            'V': dict(self.V),
            'policy': dict(self.policy),
            'episode': self.episode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'TDNStep':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], alpha=data['alpha'], n=data['n'])
        algo.V = defaultdict(float, data['V'])
        algo.policy = data['policy']
        algo.episode = data['episode']
        return algo

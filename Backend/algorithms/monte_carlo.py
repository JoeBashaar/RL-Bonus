"""
Monte Carlo Algorithm

Model-free RL methods that learn from complete episodes.
Includes Standard MC, Epsilon-Greedy MC, and MC with Exploring Starts.
"""

from typing import Any, Dict, List, Optional, Callable, Tuple
import numpy as np
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class MonteCarloResult:
    """Results from Monte Carlo training."""
    value_function: Dict[str, float]
    q_function: Dict[str, Dict[int, float]]
    policy: Dict[str, int]
    episode_rewards: List[float]
    episode_lengths: List[int]
    convergence_history: List[Dict[str, Any]]
    episodes: int
    state_visit_counts: Dict[str, int]


class MonteCarlo:
    """
    Standard Monte Carlo algorithm (First-Visit or Every-Visit).
    
    Learns value function from complete episode returns.
    Uses greedy policy improvement.
    
    Attributes:
        gamma: Discount factor
        first_visit: If True, use first-visit MC; otherwise every-visit
        n_episodes: Number of episodes to train
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        first_visit: bool = True,
        n_episodes: int = 1000,
        **kwargs
    ):
        """
        Initialize Monte Carlo.
        
        Args:
            gamma: Discount factor (0-1)
            first_visit: Use first-visit MC if True
            n_episodes: Number of episodes to train
        """
        self.gamma = gamma
        self.first_visit = first_visit
        self.n_episodes = n_episodes
        
        # Training state
        self.V: Dict[str, float] = {}
        self.Q: Dict[str, Dict[int, float]] = {}
        self.policy: Dict[str, int] = {}
        self.returns: Dict[Tuple[str, int], List[float]] = defaultdict(list)
        self.state_visits: Dict[str, int] = defaultdict(int)
        
        # Episode tracking
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.convergence_history: List[Dict[str, Any]] = []
        self.episode: int = 0
        
        # Callbacks
        self._on_episode: Optional[Callable] = None
        self._stop_requested: bool = False
    
    def train(
        self,
        env,
        on_episode: Optional[Callable] = None
    ) -> MonteCarloResult:
        """
        Train using Monte Carlo.
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_episode: Callback called after each episode
            
        Returns:
            MonteCarloResult with value function, policy, and metrics
        """
        self._on_episode = on_episode
        self._stop_requested = False
        
        n_actions = env.n_actions
        
        # Initialize Q-function
        self.Q = defaultdict(lambda: {a: 0.0 for a in range(n_actions)})
        self.returns = defaultdict(list)
        self.state_visits = defaultdict(int)
        
        self.episode_rewards = []
        self.episode_lengths = []
        self.convergence_history = []
        
        for episode in range(self.n_episodes):
            if self._stop_requested:
                break
                
            self.episode = episode + 1
            
            # Generate episode
            trajectory, total_reward = self._generate_episode(env)
            
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(len(trajectory))
            
            # Update Q-values
            visited = set()
            G = 0
            
            # Process trajectory in reverse
            for t in range(len(trajectory) - 1, -1, -1):
                state, action, reward = trajectory[t]
                G = self.gamma * G + reward
                
                state_action = (state, action)
                
                # First-visit check
                if self.first_visit and state_action in visited:
                    continue
                    
                visited.add(state_action)
                self.returns[state_action].append(G)
                self.Q[state][action] = np.mean(self.returns[state_action])
                self.state_visits[state] += 1
            
            # Update policy (greedy)
            self._update_policy(env)
            
            # Update V from Q
            self._update_value_function()
            
            # Record metrics periodically
            if episode % max(1, self.n_episodes // 100) == 0:
                metrics = {
                    'episode': self.episode,
                    'total_reward': total_reward,
                    'episode_length': len(trajectory),
                    'avg_reward_last_100': np.mean(self.episode_rewards[-100:]),
                    'value_function': dict(self.V),
                    'policy': dict(self.policy),
                }
                self.convergence_history.append(metrics)
                
                if self._on_episode:
                    self._on_episode(metrics)
        
        return MonteCarloResult(
            value_function=dict(self.V),
            q_function={s: dict(q) for s, q in self.Q.items()},
            policy=dict(self.policy),
            episode_rewards=self.episode_rewards,
            episode_lengths=self.episode_lengths,
            convergence_history=self.convergence_history,
            episodes=self.episode,
            state_visit_counts=dict(self.state_visits)
        )
    
    def _generate_episode(self, env) -> Tuple[List[Tuple[str, int, float]], float]:
        """Generate episode using current policy."""
        trajectory = []
        total_reward = 0
        
        state, _ = env.reset()
        done = False
        
        while not done:
            action = self._select_action(env, state)
            result = env.step(action)
            
            trajectory.append((state, action, result.reward))
            total_reward += result.reward
            
            state = result.next_state
            done = result.done
            
            # Safety limit
            if len(trajectory) > 10000:
                break
        
        return trajectory, total_reward
    
    def _select_action(self, env, state: str) -> int:
        """Select action using current policy."""
        if state in self.policy:
            return self.policy[state]
        
        valid_actions = env.get_valid_actions(state)
        return np.random.choice(valid_actions) if valid_actions else 0
    
    def _update_policy(self, env) -> None:
        """Update policy to be greedy w.r.t. Q-function."""
        for state in self.Q:
            valid_actions = env.get_valid_actions(state)
            if valid_actions:
                best_action = max(valid_actions, key=lambda a: self.Q[state].get(a, 0))
                self.policy[state] = best_action
    
    def _update_value_function(self) -> None:
        """Update V from Q."""
        for state in self.Q:
            if state in self.policy:
                self.V[state] = self.Q[state][self.policy[state]]
            else:
                self.V[state] = max(self.Q[state].values()) if self.Q[state] else 0
    
    def stop(self) -> None:
        """Request training to stop."""
        self._stop_requested = True
    
    def get_action(self, state: str) -> int:
        """Get action from learned policy."""
        return self.policy.get(state, 0)
    
    def get_value(self, state: str) -> float:
        """Get value of a state."""
        return self.V.get(state, 0.0)
    
    def get_q_value(self, state: str, action: int) -> float:
        """Get Q-value for state-action pair."""
        return self.Q.get(state, {}).get(action, 0.0)
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return {
            'episode': self.episode,
            'n_episodes': self.n_episodes,
            'avg_reward': np.mean(self.episode_rewards[-100:]) if self.episode_rewards else 0,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        return {
            'algorithm': 'monte_carlo',
            'gamma': self.gamma,
            'first_visit': self.first_visit,
            'V': dict(self.V),
            'Q': {s: dict(q) for s, q in self.Q.items()},
            'policy': dict(self.policy),
            'episode': self.episode,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MonteCarlo':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], first_visit=data['first_visit'])
        algo.V = data['V']
        algo.Q = defaultdict(dict, {s: dict(q) for s, q in data['Q'].items()})
        algo.policy = data['policy']
        algo.episode = data['episode']
        return algo


class MonteCarloEpsilonGreedy(MonteCarlo):
    """
    Monte Carlo with Epsilon-Greedy exploration.
    
    Uses epsilon-greedy action selection during training.
    Epsilon can decay over time for exploration-exploitation balance.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        first_visit: bool = True,
        n_episodes: int = 1000,
        **kwargs
    ):
        """
        Initialize Epsilon-Greedy Monte Carlo.
        
        Args:
            gamma: Discount factor
            epsilon: Initial exploration rate
            epsilon_decay: Rate of epsilon decay per episode
            epsilon_min: Minimum epsilon value
            first_visit: Use first-visit MC
            n_episodes: Number of episodes
        """
        super().__init__(gamma=gamma, first_visit=first_visit, n_episodes=n_episodes)
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.initial_epsilon = epsilon
    
    def _select_action(self, env, state: str) -> int:
        """Select action using epsilon-greedy policy."""
        valid_actions = env.get_valid_actions(state)
        
        if not valid_actions:
            return 0
        
        if np.random.random() < self.epsilon:
            return np.random.choice(valid_actions)
        
        if state in self.policy:
            return self.policy[state]
        
        # Use Q-values if available
        if state in self.Q:
            return max(valid_actions, key=lambda a: self.Q[state].get(a, 0))
        
        return np.random.choice(valid_actions)
    
    def train(self, env, on_episode: Optional[Callable] = None) -> MonteCarloResult:
        """Train with epsilon decay."""
        self.epsilon = self.initial_epsilon
        
        original_on_episode = on_episode
        
        def on_episode_with_decay(metrics):
            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
            metrics['epsilon'] = self.epsilon
            
            if original_on_episode:
                original_on_episode(metrics)
        
        return super().train(env, on_episode=on_episode_with_decay)
    
    def set_epsilon(self, epsilon: float) -> None:
        """Update epsilon (for real-time parameter updates)."""
        self.epsilon = max(self.epsilon_min, min(1.0, epsilon))
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        data = super().to_dict()
        data['algorithm'] = 'monte_carlo_epsilon_greedy'
        data['epsilon'] = self.epsilon
        data['epsilon_decay'] = self.epsilon_decay
        return data


class MonteCarloExploringStarts(MonteCarlo):
    """
    Monte Carlo with Exploring Starts.
    
    Each episode starts from a random state-action pair.
    Guarantees all state-action pairs are explored.
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        first_visit: bool = True,
        n_episodes: int = 1000,
        **kwargs
    ):
        """
        Initialize MC with Exploring Starts.
        
        Args:
            gamma: Discount factor
            first_visit: Use first-visit MC
            n_episodes: Number of episodes
        """
        super().__init__(gamma=gamma, first_visit=first_visit, n_episodes=n_episodes)
        self.explored_starts: Dict[Tuple[str, int], int] = defaultdict(int)
    
    def _generate_episode(self, env) -> Tuple[List[Tuple[str, int, float]], float]:
        """Generate episode with random initial state-action."""
        trajectory = []
        total_reward = 0
        
        # Reset to random state
        state, _ = env.reset()
        
        # Random first action (exploring start)
        valid_actions = env.get_valid_actions(state)
        if valid_actions:
            first_action = np.random.choice(valid_actions)
            self.explored_starts[(state, first_action)] += 1
            
            result = env.step(first_action)
            trajectory.append((state, first_action, result.reward))
            total_reward += result.reward
            state = result.next_state
            done = result.done
        else:
            done = True
        
        # Continue with greedy policy
        while not done:
            action = self._select_action(env, state)
            result = env.step(action)
            
            trajectory.append((state, action, result.reward))
            total_reward += result.reward
            
            state = result.next_state
            done = result.done
            
            if len(trajectory) > 10000:
                break
        
        return trajectory, total_reward
    
    def get_explored_starts(self) -> Dict[Tuple[str, int], int]:
        """Get count of explored start state-action pairs."""
        return dict(self.explored_starts)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        data = super().to_dict()
        data['algorithm'] = 'monte_carlo_exploring_starts'
        data['explored_starts'] = {f"{s}|{a}": c for (s, a), c in self.explored_starts.items()}
        return data

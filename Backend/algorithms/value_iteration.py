"""
Value Iteration Algorithm

Dynamic programming method that directly computes optimal value function.
Combines policy evaluation and improvement into a single step.
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass


@dataclass
class ValueIterationResult:
    """Results from value iteration training."""
    value_function: Dict[str, float]
    policy: Dict[str, int]
    convergence_history: List[Dict[str, Any]]
    iterations: int
    converged: bool


class ValueIteration:
    """
    Value Iteration algorithm implementation.
    
    This algorithm combines policy evaluation and improvement:
    V(s) = max_a sum_s' P(s'|s,a) [R(s,a,s') + gamma * V(s')]
    
    Attributes:
        gamma: Discount factor
        theta: Convergence threshold
        max_iterations: Maximum number of iterations
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        theta: float = 1e-4,
        max_iterations: int = 100,
        **kwargs
    ):
        """
        Initialize Value Iteration.
        
        Args:
            gamma: Discount factor (0-1)
            theta: Convergence threshold for value changes
            max_iterations: Maximum iterations
        """
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        
        # Training state
        self.V: Dict[str, float] = {}
        self.Q: Dict[str, Dict[int, float]] = {}
        self.policy: Dict[str, int] = {}
        self.convergence_history: List[Dict[str, Any]] = []
        self.iteration: int = 0
        self.converged: bool = False
        
        # Callbacks
        self._on_iteration: Optional[Callable] = None
        self._stop_requested: bool = False
    
    def train(
        self,
        env,
        on_iteration: Optional[Callable] = None
    ) -> ValueIterationResult:
        """
        Train using value iteration.
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_iteration: Callback called after each iteration with current state
            
        Returns:
            ValueIterationResult with final value function, policy, and metrics
        """
        self._on_iteration = on_iteration
        self._stop_requested = False
        
        # Get state space
        states = env.get_state_space()
        n_actions = env.n_actions
        
        # Initialize value function
        self.V = {s: 0.0 for s in states}
        self.Q = {s: {a: 0.0 for a in range(n_actions)} for s in states}
        self.policy = {s: 0 for s in states}
        
        self.convergence_history = []
        self.iteration = 0
        self.converged = False
        
        for iteration in range(self.max_iterations):
            if self._stop_requested:
                break
                
            self.iteration = iteration + 1
            delta = 0
            
            for state in states:
                if env.is_terminal(state):
                    continue
                
                v = self.V[state]
                valid_actions = env.get_valid_actions(state)
                
                if not valid_actions:
                    continue
                
                # Calculate Q-values for all actions
                action_values = []
                for action in valid_actions:
                    transitions = env.get_transition_prob(state, action)
                    
                    q_value = 0
                    for next_state, prob, reward in transitions:
                        next_v = self.V.get(next_state, 0)
                        q_value += prob * (reward + self.gamma * next_v)
                    
                    action_values.append((action, q_value))
                    self.Q[state][action] = q_value
                
                # Take maximum
                best_action, best_value = max(action_values, key=lambda x: x[1])
                self.V[state] = best_value
                self.policy[state] = best_action
                
                delta = max(delta, abs(v - best_value))
            
            # Record convergence metrics
            metrics = {
                'iteration': self.iteration,
                'max_value_delta': delta,
                'converged': delta < self.theta,
                'value_function': dict(self.V),
                'policy': dict(self.policy),
                'q_values': {s: dict(q) for s, q in self.Q.items()},
            }
            self.convergence_history.append(metrics)
            
            # Callback
            if self._on_iteration:
                self._on_iteration(metrics)
            
            # Check for convergence
            if delta < self.theta:
                self.converged = True
                break
        
        return ValueIterationResult(
            value_function=dict(self.V),
            policy=dict(self.policy),
            convergence_history=self.convergence_history,
            iterations=self.iteration,
            converged=self.converged
        )
    
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
            'iteration': self.iteration,
            'max_iterations': self.max_iterations,
            'converged': self.converged,
            'n_states': len(self.V),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        return {
            'algorithm': 'value_iteration',
            'gamma': self.gamma,
            'theta': self.theta,
            'V': dict(self.V),
            'Q': {s: dict(q) for s, q in self.Q.items()},
            'policy': dict(self.policy),
            'iteration': self.iteration,
            'converged': self.converged,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ValueIteration':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], theta=data['theta'])
        algo.V = data['V']
        algo.Q = {s: dict(q) for s, q in data['Q'].items()}
        algo.policy = data['policy']
        algo.iteration = data['iteration']
        algo.converged = data['converged']
        return algo

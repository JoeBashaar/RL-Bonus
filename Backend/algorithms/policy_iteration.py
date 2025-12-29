"""
Policy Iteration Algorithm

Dynamic programming method that iteratively improves policy using policy evaluation.
Requires a complete model of the environment (transition probabilities).
"""

from typing import Any, Dict, List, Optional, Callable
import numpy as np
from dataclasses import dataclass, field


@dataclass
class PolicyIterationResult:
    """Results from policy iteration training."""
    value_function: Dict[str, float]
    policy: Dict[str, int]
    convergence_history: List[Dict[str, Any]]
    iterations: int
    converged: bool
    policy_stable_at: int


class PolicyIteration:
    """
    Policy Iteration algorithm implementation.
    
    This algorithm alternates between:
    1. Policy Evaluation: Compute value function for current policy
    2. Policy Improvement: Update policy to be greedy w.r.t. value function
    
    Attributes:
        gamma: Discount factor
        theta: Convergence threshold for policy evaluation
        max_iterations: Maximum number of policy improvement iterations
    """
    
    def __init__(
        self,
        gamma: float = 0.99,
        theta: float = 1e-4,
        max_iterations: int = 100,
        max_eval_iterations: int = 100,
        **kwargs
    ):
        """
        Initialize Policy Iteration.
        
        Args:
            gamma: Discount factor (0-1)
            theta: Convergence threshold for value changes
            max_iterations: Maximum policy improvement iterations
            max_eval_iterations: Maximum iterations per policy evaluation
        """
        self.gamma = gamma
        self.theta = theta
        self.max_iterations = max_iterations
        self.max_eval_iterations = max_eval_iterations
        
        # Training state
        self.V: Dict[str, float] = {}
        self.policy: Dict[str, int] = {}
        self.convergence_history: List[Dict[str, Any]] = []
        self.iteration: int = 0
        self.converged: bool = False
        self.policy_stable_at: int = -1
        
        # Callbacks
        self._on_iteration: Optional[Callable] = None
        self._stop_requested: bool = False
    
    def train(
        self,
        env,
        on_iteration: Optional[Callable] = None
    ) -> PolicyIterationResult:
        """
        Train using policy iteration.
        
        Args:
            env: Environment implementing BaseEnvironment interface
            on_iteration: Callback called after each iteration with current state
            
        Returns:
            PolicyIterationResult with final value function, policy, and metrics
        """
        self._on_iteration = on_iteration
        self._stop_requested = False
        
        # Get state and action spaces
        states = env.get_state_space()
        n_actions = env.n_actions
        
        # Initialize value function and policy
        self.V = {s: 0.0 for s in states}
        self.policy = {s: 0 for s in states}  # Start with action 0
        
        # Initialize with random policy
        for state in states:
            valid_actions = env.get_valid_actions(state)
            if valid_actions:
                self.policy[state] = np.random.choice(valid_actions)
        
        self.convergence_history = []
        self.iteration = 0
        self.converged = False
        self.policy_stable_at = -1
        
        for iteration in range(self.max_iterations):
            if self._stop_requested:
                break
                
            self.iteration = iteration + 1
            
            # Policy Evaluation
            eval_iterations, max_delta = self._policy_evaluation(env, states)
            
            # Policy Improvement
            policy_stable, policy_changes = self._policy_improvement(env, states, n_actions)
            
            # Record convergence metrics
            metrics = {
                'iteration': self.iteration,
                'eval_iterations': eval_iterations,
                'max_value_delta': max_delta,
                'policy_stable': policy_stable,
                'policy_changes': policy_changes,
                'converged': policy_stable,  # Mark as converged when policy is stable
                'value_function': dict(self.V),
                'policy': dict(self.policy),
            }
            self.convergence_history.append(metrics)
            
            # Callback
            if self._on_iteration:
                self._on_iteration(metrics)
            
            # Check for convergence
            if policy_stable:
                self.converged = True
                self.policy_stable_at = self.iteration
                break
        
        return PolicyIterationResult(
            value_function=dict(self.V),
            policy=dict(self.policy),
            convergence_history=self.convergence_history,
            iterations=self.iteration,
            converged=self.converged,
            policy_stable_at=self.policy_stable_at
        )
    
    def _policy_evaluation(self, env, states: List[str]) -> tuple:
        """
        Evaluate current policy.
        
        Returns:
            Tuple of (iterations, max_delta)
        """
        for eval_iter in range(self.max_eval_iterations):
            delta = 0
            
            for state in states:
                if env.is_terminal(state):
                    continue
                
                v = self.V[state]
                action = self.policy[state]
                
                # Get transition probabilities
                transitions = env.get_transition_prob(state, action)
                
                # Calculate expected value
                new_v = 0
                for next_state, prob, reward in transitions:
                    next_v = self.V.get(next_state, 0)
                    new_v += prob * (reward + self.gamma * next_v)
                
                self.V[state] = new_v
                delta = max(delta, abs(v - new_v))
            
            if delta < self.theta:
                return eval_iter + 1, delta
        
        return self.max_eval_iterations, delta
    
    def _policy_improvement(self, env, states: List[str], n_actions: int) -> tuple:
        """
        Improve policy to be greedy w.r.t. current value function.
        
        Returns:
            Tuple of (policy_stable, policy_changes)
        """
        policy_stable = True
        policy_changes = 0
        
        for state in states:
            if env.is_terminal(state):
                continue
            
            old_action = self.policy[state]
            valid_actions = env.get_valid_actions(state)
            
            if not valid_actions:
                continue
            
            # Find best action
            best_action = valid_actions[0]
            best_value = float('-inf')
            
            for action in valid_actions:
                transitions = env.get_transition_prob(state, action)
                
                action_value = 0
                for next_state, prob, reward in transitions:
                    next_v = self.V.get(next_state, 0)
                    action_value += prob * (reward + self.gamma * next_v)
                
                if action_value > best_value:
                    best_value = action_value
                    best_action = action
            
            self.policy[state] = best_action
            
            if old_action != best_action:
                policy_stable = False
                policy_changes += 1
        
        return policy_stable, policy_changes
    
    def stop(self) -> None:
        """Request training to stop."""
        self._stop_requested = True
    
    def get_action(self, state: str) -> int:
        """Get action from learned policy."""
        return self.policy.get(state, 0)
    
    def get_value(self, state: str) -> float:
        """Get value of a state."""
        return self.V.get(state, 0.0)
    
    def get_training_progress(self) -> Dict[str, Any]:
        """Get current training progress."""
        return {
            'iteration': self.iteration,
            'max_iterations': self.max_iterations,
            'converged': self.converged,
            'policy_stable_at': self.policy_stable_at,
            'n_states': len(self.V),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize algorithm state."""
        return {
            'algorithm': 'policy_iteration',
            'gamma': self.gamma,
            'theta': self.theta,
            'V': dict(self.V),
            'policy': dict(self.policy),
            'iteration': self.iteration,
            'converged': self.converged,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PolicyIteration':
        """Deserialize algorithm state."""
        algo = cls(gamma=data['gamma'], theta=data['theta'])
        algo.V = data['V']
        algo.policy = data['policy']
        algo.iteration = data['iteration']
        algo.converged = data['converged']
        return algo

"""
RL Algorithms Package

This package contains all RL algorithm implementations.
Each algorithm works with the BaseEnvironment interface.
"""

from .policy_iteration import PolicyIteration
from .value_iteration import ValueIteration
from .monte_carlo import MonteCarlo, MonteCarloEpsilonGreedy, MonteCarloExploringStarts
from .td_zero import TDZero
from .td_nstep import TDNStep
from .sarsa import SARSA
from .q_learning import QLearning

__all__ = [
    'PolicyIteration',
    'ValueIteration',
    'MonteCarlo',
    'MonteCarloEpsilonGreedy',
    'MonteCarloExploringStarts',
    'TDZero',
    'TDNStep',
    'SARSA',
    'QLearning',
]

ALGORITHM_REGISTRY = {
    'policy_iteration': PolicyIteration,
    'value_iteration': ValueIteration,
    'mc_first_visit': MonteCarlo,  # first_visit=True by default
    'mc_every_visit': MonteCarlo,  # will set first_visit=False
    'mc_epsilon_greedy': MonteCarloEpsilonGreedy,
    'mc_exploring_starts': MonteCarloExploringStarts,
    'td_zero': TDZero,
    'td_nstep': TDNStep,
    'sarsa': SARSA,
    'q_learning': QLearning,
}

def get_algorithm(name: str) -> type:
    """Get algorithm class by name."""
    if name not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm: {name}. Available: {list(ALGORITHM_REGISTRY.keys())}")
    return ALGORITHM_REGISTRY[name]

def list_algorithms() -> list:
    """List all available algorithms with descriptions."""
    return [
        {
            'name': 'policy_iteration',
            'display_name': 'Policy Iteration',
            'description': 'Dynamic programming method that iteratively improves policy using policy evaluation.',
            'type': 'model_based',
            'parameters': ['gamma', 'theta', 'max_iterations']
        },
        {
            'name': 'value_iteration',
            'display_name': 'Value Iteration',
            'description': 'Dynamic programming method that directly computes optimal value function.',
            'type': 'model_based',
            'parameters': ['gamma', 'theta', 'max_iterations']
        },
        {
            'name': 'mc_first_visit',
            'display_name': 'Monte Carlo (First-Visit)',
            'description': 'First-visit Monte Carlo - counts only first occurrence of each state per episode.',
            'type': 'model_free',
            'parameters': ['gamma', 'n_episodes']
        },
        {
            'name': 'mc_every_visit',
            'display_name': 'Monte Carlo (Every-Visit)',
            'description': 'Every-visit Monte Carlo - counts all occurrences of each state per episode.',
            'type': 'model_free',
            'parameters': ['gamma', 'n_episodes']
        },
        {
            'name': 'mc_epsilon_greedy',
            'display_name': 'Monte Carlo (ε-Greedy)',
            'description': 'Monte Carlo with epsilon-greedy exploration.',
            'type': 'model_free',
            'parameters': ['gamma', 'epsilon', 'n_episodes', 'epsilon_decay']
        },
        {
            'name': 'mc_exploring_starts',
            'display_name': 'Monte Carlo (Exploring Starts)',
            'description': 'Monte Carlo with random initial state-action pairs.',
            'type': 'model_free',
            'parameters': ['gamma', 'n_episodes']
        },
        {
            'name': 'td_zero',
            'display_name': 'TD(0)',
            'description': 'Temporal difference learning with single-step bootstrapping.',
            'type': 'model_free',
            'parameters': ['gamma', 'alpha', 'n_episodes']
        },
        {
            'name': 'td_nstep',
            'display_name': 'N-Step TD',
            'description': 'Temporal difference learning with n-step bootstrapping.',
            'type': 'model_free',
            'parameters': ['gamma', 'alpha', 'n', 'n_episodes']
        },
        {
            'name': 'sarsa',
            'display_name': 'SARSA',
            'description': 'On-policy TD control using state-action-reward-state-action updates.',
            'type': 'model_free',
            'parameters': ['gamma', 'alpha', 'epsilon', 'n_episodes', 'epsilon_decay']
        },
        {
            'name': 'q_learning',
            'display_name': 'Q-Learning',
            'description': 'Off-policy TD control using maximum Q-value updates.',
            'type': 'model_free',
            'parameters': ['gamma', 'alpha', 'epsilon', 'n_episodes', 'epsilon_decay']
        },
    ]


def get_algorithm_parameters(name: str) -> dict:
    """Get parameter definitions for an algorithm."""
    params = {
        'policy_iteration': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'theta': {'type': 'float', 'min': 1e-10, 'max': 1.0, 'default': 1e-4, 'description': 'Convergence threshold'},
            'max_iterations': {'type': 'int', 'min': 1, 'max': 10000, 'default': 100, 'description': 'Maximum iterations'},
        },
        'value_iteration': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'theta': {'type': 'float', 'min': 1e-10, 'max': 1.0, 'default': 1e-4, 'description': 'Convergence threshold'},
            'max_iterations': {'type': 'int', 'min': 1, 'max': 10000, 'default': 100, 'description': 'Maximum iterations'},
        },
        'mc_first_visit': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
        },
        'mc_every_visit': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
        },
        'mc_epsilon_greedy': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'epsilon': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Exploration rate'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
            'epsilon_decay': {'type': 'float', 'min': 0.9, 'max': 1.0, 'default': 0.995, 'description': 'Epsilon decay rate'},
        },
        'mc_exploring_starts': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
        },
        'td_zero': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Learning rate'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
        },
        'td_nstep': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Learning rate'},
            'n': {'type': 'int', 'min': 1, 'max': 100, 'default': 4, 'description': 'Number of steps'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
        },
        'sarsa': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Learning rate'},
            'epsilon': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Exploration rate'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
            'epsilon_decay': {'type': 'float', 'min': 0.9, 'max': 1.0, 'default': 0.995, 'description': 'Epsilon decay rate'},
        },
        'q_learning': {
            'gamma': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.99, 'description': 'Discount factor'},
            'alpha': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Learning rate'},
            'epsilon': {'type': 'float', 'min': 0.0, 'max': 1.0, 'default': 0.1, 'description': 'Exploration rate'},
            'n_episodes': {'type': 'int', 'min': 1, 'max': 100000, 'default': 1000, 'description': 'Number of episodes'},
            'epsilon_decay': {'type': 'float', 'min': 0.9, 'max': 1.0, 'default': 0.995, 'description': 'Epsilon decay rate'},
        },
    }
    
    if name not in params:
        raise ValueError(f"Unknown algorithm: {name}")
    
    return params[name]

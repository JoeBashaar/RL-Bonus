"""
Base Environment Abstract Class

This module defines the abstract base class that all RL environments must implement.
It provides a consistent interface for algorithm interaction.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np
from enum import Enum


@dataclass
class StateInfo:
    """Information about the current state."""
    state_id: str
    is_terminal: bool
    valid_actions: List[int]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StepResult:
    """Result of taking an action in the environment."""
    next_state: str
    reward: float
    done: bool
    info: Dict[str, Any]
    truncated: bool = False


@dataclass
class RenderData:
    """Data for frontend rendering."""
    environment: str
    state_id: str
    entities: List[Dict[str, Any]]
    grid: Optional[Dict[str, Any]]
    metadata: Dict[str, Any]


class BaseEnvironment(ABC):
    """
    Abstract base class for all RL environments.
    
    All environments must implement this interface to work with the RL algorithms.
    The interface is designed to support both tabular and continuous (discretized) environments.
    
    State Space:
        States are represented as strings (state_id) for tabular methods.
        The environment maintains a mapping between state_id and actual state representation.
    
    Action Space:
        Actions are represented as integers from 0 to n_actions-1.
        The environment provides action names for display purposes.
    
    Attributes:
        name (str): Environment name
        n_actions (int): Number of possible actions
        action_names (List[str]): Human-readable action names
        gamma (float): Default discount factor
    """
    
    def __init__(self, name: str, **kwargs):
        """
        Initialize the environment.
        
        Args:
            name: Environment identifier
            **kwargs: Environment-specific configuration
        """
        self.name = name
        self.current_state: Optional[str] = None
        self.step_count: int = 0
        self.episode_reward: float = 0.0
        self.max_steps: int = kwargs.get('max_steps', 1000)
        self.rng = np.random.default_rng(kwargs.get('seed', None))
        
        # To be set by subclasses
        self.n_actions: int = 0
        self.action_names: List[str] = []
        self.gamma: float = 0.99
        
        # State space tracking for tabular methods
        self._state_space: Set[str] = set()
        self._terminal_states: Set[str] = set()
        
    @abstractmethod
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """
        Reset the environment to initial state.
        
        Args:
            **kwargs: Optional reset parameters
            
        Returns:
            Tuple of (initial_state_id, state_info)
        """
        pass
    
    @abstractmethod
    def step(self, action: int) -> StepResult:
        """
        Take an action in the environment.
        
        Args:
            action: Action index (0 to n_actions-1)
            
        Returns:
            StepResult containing next_state, reward, done, info
        """
        pass
    
    @abstractmethod
    def get_valid_actions(self, state_id: Optional[str] = None) -> List[int]:
        """
        Get valid actions for a state.
        
        Args:
            state_id: State to query. If None, use current state.
            
        Returns:
            List of valid action indices
        """
        pass
    
    @abstractmethod
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """
        Get rendering data for frontend visualization.
        
        Args:
            state_id: State to render. If None, use current state.
            
        Returns:
            RenderData for frontend
        """
        pass
    
    @abstractmethod
    def get_state_space(self) -> List[str]:
        """
        Get all possible states (for tabular methods).
        
        Returns:
            List of all state IDs
        """
        pass
    
    @abstractmethod
    def get_transition_prob(self, state: str, action: int) -> List[Tuple[str, float, float]]:
        """
        Get transition probabilities for model-based methods.
        
        Args:
            state: Current state ID
            action: Action to take
            
        Returns:
            List of (next_state, probability, reward) tuples
        """
        pass
    
    def is_terminal(self, state_id: Optional[str] = None) -> bool:
        """
        Check if a state is terminal.
        
        Args:
            state_id: State to check. If None, use current state.
            
        Returns:
            True if state is terminal
        """
        state = state_id if state_id is not None else self.current_state
        return state in self._terminal_states
    
    def get_state_info(self, state_id: Optional[str] = None) -> StateInfo:
        """
        Get information about a state.
        
        Args:
            state_id: State to query. If None, use current state.
            
        Returns:
            StateInfo object
        """
        state = state_id if state_id is not None else self.current_state
        return StateInfo(
            state_id=state,
            is_terminal=self.is_terminal(state),
            valid_actions=self.get_valid_actions(state),
            metadata={'step_count': self.step_count}
        )
    
    def get_action_name(self, action: int) -> str:
        """Get human-readable name for an action."""
        if 0 <= action < len(self.action_names):
            return self.action_names[action]
        return f"action_{action}"
    
    def seed(self, seed: Optional[int] = None) -> None:
        """Set random seed for reproducibility."""
        self.rng = np.random.default_rng(seed)
    
    def close(self) -> None:
        """Clean up environment resources."""
        pass
    
    def get_config(self) -> Dict[str, Any]:
        """Get environment configuration."""
        return {
            'name': self.name,
            'n_actions': self.n_actions,
            'action_names': self.action_names,
            'gamma': self.gamma,
            'max_steps': self.max_steps,
        }
    
    def state_to_index(self, state_id: str) -> int:
        """Convert state ID to numeric index for array storage."""
        states = sorted(list(self._state_space))
        try:
            return states.index(state_id)
        except ValueError:
            # Add new state
            self._state_space.add(state_id)
            states = sorted(list(self._state_space))
            return states.index(state_id)
    
    def index_to_state(self, index: int) -> str:
        """Convert numeric index to state ID."""
        states = sorted(list(self._state_space))
        return states[index]
    
    @abstractmethod
    def encode_state(self, **state_components) -> str:
        """
        Encode state components into a state ID string.
        
        Args:
            **state_components: State components
            
        Returns:
            State ID string
        """
        pass
    
    @abstractmethod
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """
        Decode state ID string into state components.
        
        Args:
            state_id: State ID string
            
        Returns:
            Dictionary of state components
        """
        pass
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert environment state to dictionary for serialization."""
        return {
            'name': self.name,
            'current_state': self.current_state,
            'step_count': self.step_count,
            'episode_reward': self.episode_reward,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BaseEnvironment':
        """Create environment from dictionary."""
        env = cls()
        env.current_state = data.get('current_state')
        env.step_count = data.get('step_count', 0)
        env.episode_reward = data.get('episode_reward', 0.0)
        return env


class DiscreteEnvironment(BaseEnvironment):
    """
    Base class for discrete state/action environments.
    
    Provides utilities for environments with finite, enumerable state spaces.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._transition_matrix: Dict[str, Dict[int, List[Tuple[str, float, float]]]] = {}
        self._reward_matrix: Dict[str, Dict[int, float]] = {}
        
    def build_transition_matrix(self) -> None:
        """Build complete transition matrix for model-based methods."""
        for state in self.get_state_space():
            if state not in self._transition_matrix:
                self._transition_matrix[state] = {}
            for action in self.get_valid_actions(state):
                self._transition_matrix[state][action] = self.get_transition_prob(state, action)
    
    def get_all_transitions(self) -> Dict[str, Dict[int, List[Tuple[str, float, float]]]]:
        """Get complete transition matrix."""
        if not self._transition_matrix:
            self.build_transition_matrix()
        return self._transition_matrix


class ContinuousToDiscreteEnvironment(BaseEnvironment):
    """
    Base class for continuous environments discretized for tabular methods.
    
    Provides utilities for discretizing continuous state spaces.
    """
    
    def __init__(self, name: str, **kwargs):
        super().__init__(name, **kwargs)
        self._bins: Dict[str, np.ndarray] = {}
        
    def discretize(self, value: float, bins: np.ndarray) -> int:
        """Discretize a continuous value into bin index."""
        return int(np.digitize(value, bins))
    
    def set_bins(self, name: str, low: float, high: float, n_bins: int) -> None:
        """Set discretization bins for a continuous dimension."""
        self._bins[name] = np.linspace(low, high, n_bins)
    
    def get_bin_value(self, name: str, bin_index: int) -> float:
        """Get representative value for a bin."""
        bins = self._bins[name]
        if bin_index <= 0:
            return bins[0]
        if bin_index >= len(bins):
            return bins[-1]
        return (bins[bin_index - 1] + bins[bin_index]) / 2

"""
Parameter Validation

Validates and processes algorithm and environment parameters.
"""

from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass


@dataclass
class ValidationResult:
    """Result of parameter validation."""
    valid: bool
    errors: List[str]
    warnings: List[str]
    sanitized_params: Dict[str, Any]


class ParameterValidator:
    """
    Validates parameters for algorithms and environments.
    
    Provides type checking, range validation, and sanitization.
    """
    
    # Algorithm parameter definitions
    ALGORITHM_PARAMS = {
        'policy_iteration': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'theta': {'type': float, 'min': 1e-10, 'max': 1.0, 'default': 1e-4},
            'max_iterations': {'type': int, 'min': 1, 'max': 10000, 'default': 100},
            'max_eval_iterations': {'type': int, 'min': 1, 'max': 1000, 'default': 100},
        },
        'value_iteration': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'theta': {'type': float, 'min': 1e-10, 'max': 1.0, 'default': 1e-4},
            'max_iterations': {'type': int, 'min': 1, 'max': 10000, 'default': 100},
        },
        'mc_first_visit': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
        'mc_every_visit': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
        'mc_epsilon_greedy': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'epsilon': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.995},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
            'first_visit': {'type': bool, 'default': True},
        },
        'mc_exploring_starts': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
            'first_visit': {'type': bool, 'default': True},
        },
        'td_zero': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.995},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
        'td_nstep': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'n': {'type': int, 'min': 1, 'max': 100, 'default': 4},
            'epsilon': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.995},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
        'sarsa': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.995},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
        'q_learning': {
            'gamma': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.99},
            'alpha': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.1},
            'epsilon_decay': {'type': float, 'min': 0.9, 'max': 1.0, 'default': 0.995},
            'epsilon_min': {'type': float, 'min': 0.0, 'max': 1.0, 'default': 0.01},
            'n_episodes': {'type': int, 'min': 1, 'max': 100000, 'default': 1000},
        },
    }
    
    # Environment parameter definitions
    ENVIRONMENT_PARAMS = {
        'football': {
            'grid_width': {'type': int, 'min': 5, 'max': 20, 'default': 10},
            'grid_height': {'type': int, 'min': 4, 'max': 12, 'default': 6},
            'goals_to_win': {'type': int, 'min': 1, 'max': 5, 'default': 2},
            'max_steps': {'type': int, 'min': 50, 'max': 1000, 'default': 200},
        },
        'hill_climbing': {
            'n_position_bins': {'type': int, 'min': 10, 'max': 100, 'default': 50},
            'n_velocity_bins': {'type': int, 'min': 10, 'max': 50, 'default': 30},
            'max_steps': {'type': int, 'min': 100, 'max': 2000, 'default': 500},
        },
        'haunted_house': {
            'grid_width': {'type': int, 'min': 5, 'max': 15, 'default': 8},
            'grid_height': {'type': int, 'min': 5, 'max': 15, 'default': 8},
            'n_levels': {'type': int, 'min': 2, 'max': 5, 'default': 3},
            'time_limit': {'type': int, 'min': 50, 'max': 500, 'default': 100},
        },
        'spider_web': {
            'grid_width': {'type': int, 'min': 5, 'max': 20, 'default': 10},
            'grid_height': {'type': int, 'min': 5, 'max': 20, 'default': 10},
            'time_limit': {'type': int, 'min': 50, 'max': 500, 'default': 200},
            'max_mosquitoes': {'type': int, 'min': 2, 'max': 10, 'default': 5},
        },
        'friend_or_foe': {
            'total_days': {'type': int, 'min': 5, 'max': 50, 'default': 20},
            'death_threshold': {'type': int, 'min': -500, 'max': -50, 'default': -200},
        },
        'train_tracks': {
            'grid_width': {'type': int, 'min': 10, 'max': 30, 'default': 20},
            'grid_height': {'type': int, 'min': 10, 'max': 30, 'default': 20},
            'n_stations': {'type': int, 'min': 3, 'max': 7, 'default': 5},
            'tracks_per_station': {'type': int, 'min': 5, 'max': 20, 'default': 10},
        },
    }
    
    # Parameters that can be updated during training
    LIVE_UPDATE_PARAMS = ['epsilon', 'alpha']
    
    # Parameters that require training restart
    RESTART_PARAMS = ['gamma', 'n', 'n_episodes', 'max_iterations']
    
    @classmethod
    def validate_algorithm_params(
        cls,
        algorithm: str,
        params: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate algorithm parameters.
        
        Args:
            algorithm: Algorithm name
            params: Parameters to validate
            
        Returns:
            ValidationResult with errors, warnings, and sanitized params
        """
        if algorithm not in cls.ALGORITHM_PARAMS:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown algorithm: {algorithm}"],
                warnings=[],
                sanitized_params={}
            )
        
        return cls._validate_params(params, cls.ALGORITHM_PARAMS[algorithm])
    
    @classmethod
    def validate_environment_params(
        cls,
        environment: str,
        params: Dict[str, Any]
    ) -> ValidationResult:
        """
        Validate environment parameters.
        
        Args:
            environment: Environment name
            params: Parameters to validate
            
        Returns:
            ValidationResult with errors, warnings, and sanitized params
        """
        if environment not in cls.ENVIRONMENT_PARAMS:
            return ValidationResult(
                valid=False,
                errors=[f"Unknown environment: {environment}"],
                warnings=[],
                sanitized_params={}
            )
        
        return cls._validate_params(params, cls.ENVIRONMENT_PARAMS[environment])
    
    @classmethod
    def _validate_params(
        cls,
        params: Dict[str, Any],
        param_defs: Dict[str, Dict[str, Any]]
    ) -> ValidationResult:
        """
        Validate parameters against definitions.
        
        Args:
            params: Parameters to validate
            param_defs: Parameter definitions
            
        Returns:
            ValidationResult
        """
        errors = []
        warnings = []
        sanitized = {}
        
        # Check provided params
        for name, value in params.items():
            if name not in param_defs:
                warnings.append(f"Unknown parameter: {name}")
                continue
            
            definition = param_defs[name]
            expected_type = definition['type']
            
            # Type check
            if expected_type == bool:
                if not isinstance(value, bool):
                    try:
                        value = bool(value)
                    except:
                        errors.append(f"{name} must be a boolean")
                        continue
            elif expected_type == int:
                if not isinstance(value, int):
                    try:
                        value = int(value)
                    except:
                        errors.append(f"{name} must be an integer")
                        continue
            elif expected_type == float:
                if not isinstance(value, (int, float)):
                    try:
                        value = float(value)
                    except:
                        errors.append(f"{name} must be a number")
                        continue
                value = float(value)
            
            # Range check
            if 'min' in definition and value < definition['min']:
                errors.append(f"{name} must be >= {definition['min']}, got {value}")
                continue
            if 'max' in definition and value > definition['max']:
                errors.append(f"{name} must be <= {definition['max']}, got {value}")
                continue
            
            sanitized[name] = value
        
        # Add defaults for missing params
        for name, definition in param_defs.items():
            if name not in sanitized and 'default' in definition:
                sanitized[name] = definition['default']
        
        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            sanitized_params=sanitized
        )
    
    @classmethod
    def get_default_params(cls, name: str, param_type: str = 'algorithm') -> Dict[str, Any]:
        """
        Get default parameters for algorithm or environment.
        
        Args:
            name: Algorithm or environment name
            param_type: 'algorithm' or 'environment'
            
        Returns:
            Default parameters
        """
        if param_type == 'algorithm':
            param_defs = cls.ALGORITHM_PARAMS.get(name, {})
        else:
            param_defs = cls.ENVIRONMENT_PARAMS.get(name, {})
        
        return {name: d['default'] for name, d in param_defs.items() if 'default' in d}
    
    @classmethod
    def can_update_live(cls, param_name: str) -> bool:
        """Check if parameter can be updated during training."""
        return param_name in cls.LIVE_UPDATE_PARAMS
    
    @classmethod
    def requires_restart(cls, param_name: str) -> bool:
        """Check if parameter change requires training restart."""
        return param_name in cls.RESTART_PARAMS
    
    @classmethod
    def get_param_info(cls, algorithm: str) -> Dict[str, Dict[str, Any]]:
        """Get parameter information for an algorithm."""
        if algorithm not in cls.ALGORITHM_PARAMS:
            return {}
        
        result = {}
        for name, definition in cls.ALGORITHM_PARAMS[algorithm].items():
            result[name] = {
                **definition,
                'type': definition['type'].__name__,
                'can_update_live': cls.can_update_live(name),
                'requires_restart': cls.requires_restart(name)
            }
        
        return result

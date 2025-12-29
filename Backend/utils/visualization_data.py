"""
Visualization Data Formatter

Formats training data for frontend visualization.
Handles value functions, policies, convergence metrics, and episode data.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict


@dataclass
class ValueFunctionData:
    """Formatted value function data for visualization."""
    values: Dict[str, float]
    min_value: float
    max_value: float
    grid_mapping: Optional[Dict[str, Dict[str, Any]]]


@dataclass
class PolicyData:
    """Formatted policy data for visualization."""
    policy: Dict[str, Dict[str, Any]]
    action_names: List[str]


@dataclass
class ConvergenceData:
    """Convergence metrics data."""
    iterations: List[int]
    value_deltas: List[float]
    policy_changes: List[int]
    converged: bool
    converged_at: Optional[int]


@dataclass
class EpisodeData:
    """Episode-based training data."""
    episodes: List[int]
    rewards: List[float]
    lengths: List[int]
    avg_rewards: List[float]
    td_errors: Optional[List[float]]


class VisualizationFormatter:
    """
    Formats training data for frontend visualization.
    
    Provides methods to convert algorithm results into
    JSON-serializable formats suitable for web rendering.
    """
    
    @staticmethod
    def format_value_function(
        value_function: Dict[str, float],
        env=None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Format value function for visualization.
        
        Args:
            value_function: Dict mapping state IDs to values
            env: Optional environment for grid mapping
            sample_size: Max number of states to include
            
        Returns:
            Formatted value function data
        """
        if not value_function:
            return {
                'values': {},
                'min_value': 0,
                'max_value': 0,
                'grid_mapping': None
            }
        
        # Sample if too large
        if len(value_function) > sample_size:
            # Sort by value and take extremes + random sample
            sorted_states = sorted(value_function.items(), key=lambda x: x[1])
            n_extremes = sample_size // 4
            extreme_states = dict(sorted_states[:n_extremes] + sorted_states[-n_extremes:])
            
            remaining = list(set(value_function.keys()) - set(extreme_states.keys()))
            np.random.shuffle(remaining)
            sample = remaining[:sample_size - len(extreme_states)]
            
            values = {**extreme_states, **{s: value_function[s] for s in sample}}
        else:
            values = dict(value_function)
        
        min_val = min(values.values())
        max_val = max(values.values())
        
        # Create grid mapping if environment provided
        grid_mapping = None
        if env is not None:
            try:
                grid_mapping = {}
                for state in values:
                    decoded = env.decode_state(state)
                    # Extract position info
                    if 'x' in decoded or 'position_bin' in decoded:
                        grid_mapping[state] = {
                            'x': decoded.get('x', decoded.get('ball_x', decoded.get('spider_x', 
                                   decoded.get('seeker_x', decoded.get('position_bin', 0))))),
                            'y': decoded.get('y', decoded.get('ball_y', decoded.get('spider_y',
                                   decoded.get('seeker_y', decoded.get('velocity_bin', 0))))),
                        }
            except Exception:
                grid_mapping = None
        
        return {
            'values': values,
            'min_value': min_val,
            'max_value': max_val,
            'grid_mapping': grid_mapping,
            'n_states': len(value_function),
            'sampled': len(value_function) > sample_size
        }
    
    @staticmethod
    def format_policy(
        policy: Dict[str, int],
        q_values: Optional[Dict[str, Dict[int, float]]] = None,
        action_names: Optional[List[str]] = None,
        sample_size: int = 1000
    ) -> Dict[str, Any]:
        """
        Format policy for visualization.
        
        Args:
            policy: Dict mapping state IDs to actions
            q_values: Optional Q-values for each state-action pair
            action_names: List of action names
            sample_size: Max number of states to include
            
        Returns:
            Formatted policy data
        """
        if not policy:
            return {
                'policy': {},
                'action_names': action_names or [],
                'n_states': 0
            }
        
        # Sample if too large
        states = list(policy.keys())
        if len(states) > sample_size:
            np.random.shuffle(states)
            states = states[:sample_size]
        
        formatted = {}
        for state in states:
            action = policy[state]
            entry = {
                'action': action,
                'action_name': action_names[action] if action_names and action < len(action_names) else f'action_{action}',
                'probability': 1.0  # Deterministic policy
            }
            
            if q_values and state in q_values:
                entry['q_values'] = q_values[state]
            
            formatted[state] = entry
        
        return {
            'policy': formatted,
            'action_names': action_names or [],
            'n_states': len(policy),
            'sampled': len(policy) > sample_size
        }
    
    @staticmethod
    def format_convergence_metrics(
        history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Format convergence metrics from training history.
        
        Args:
            history: List of training history entries
            
        Returns:
            Formatted convergence data
        """
        if not history:
            return {
                'iterations': [],
                'value_deltas': [],
                'policy_changes': [],
                'converged': False,
                'converged_at': None
            }
        
        iterations = []
        value_deltas = []
        policy_changes = []
        converged_at = None
        
        for entry in history:
            iterations.append(entry.get('iteration', entry.get('episode', 0)))
            value_deltas.append(entry.get('max_value_delta', entry.get('avg_td_error', 0)))
            policy_changes.append(entry.get('policy_changes', 0))
            
            if entry.get('converged', False) and converged_at is None:
                converged_at = iterations[-1]
        
        return {
            'iterations': iterations,
            'max_deltas': value_deltas,
            'policy_changes': policy_changes,
            'converged': converged_at is not None,
            'convergence_iteration': converged_at
        }
    
    @staticmethod
    def format_episode_data(
        rewards: List[float],
        lengths: List[int],
        td_errors: Optional[List[float]] = None,
        window_size: int = 100
    ) -> Dict[str, Any]:
        """
        Format episode-based training data.
        
        Args:
            rewards: Episode rewards
            lengths: Episode lengths
            td_errors: Optional TD errors
            window_size: Window for moving average
            
        Returns:
            Formatted episode data
        """
        if not rewards:
            return {
                'episodes': [],
                'rewards': [],
                'lengths': [],
                'avg_rewards': [],
                'td_errors': None
            }
        
        n_episodes = len(rewards)
        episodes = list(range(1, n_episodes + 1))
        
        # Calculate moving average
        avg_rewards = []
        for i in range(n_episodes):
            start = max(0, i - window_size + 1)
            avg_rewards.append(np.mean(rewards[start:i+1]))
        
        # Downsample if too many points
        max_points = 1000
        if n_episodes > max_points:
            indices = np.linspace(0, n_episodes - 1, max_points, dtype=int)
            episodes = [episodes[i] for i in indices]
            rewards = [rewards[i] for i in indices]
            lengths = [lengths[i] for i in indices]
            avg_rewards = [avg_rewards[i] for i in indices]
        
        result = {
            'episodes': episodes,
            'rewards': rewards,
            'lengths': lengths,
            'avg_rewards': avg_rewards,
            'total_episodes': n_episodes
        }
        
        if td_errors:
            # Aggregate TD errors per episode (rough approximation)
            result['avg_td_error'] = np.mean(td_errors) if td_errors else 0
        
        return result
    
    @staticmethod
    def format_state_visitations(
        visit_counts: Dict[str, int],
        sample_size: int = 500
    ) -> Dict[str, Any]:
        """
        Format state visitation counts.
        
        Args:
            visit_counts: Dict mapping states to visit counts
            sample_size: Max states to include
            
        Returns:
            Formatted visitation data
        """
        if not visit_counts:
            return {
                'visitations': {},
                'min_visits': 0,
                'max_visits': 0,
                'total_visits': 0
            }
        
        total = sum(visit_counts.values())
        sorted_states = sorted(visit_counts.items(), key=lambda x: -x[1])
        
        # Take most visited states
        sampled = dict(sorted_states[:sample_size])
        
        return {
            'visitations': sampled,
            'min_visits': min(visit_counts.values()),
            'max_visits': max(visit_counts.values()),
            'total_visits': total,
            'n_visited_states': len(visit_counts),
            'sampled': len(visit_counts) > sample_size
        }
    
    @staticmethod
    def format_training_progress(
        algorithm,
        include_full_data: bool = False
    ) -> Dict[str, Any]:
        """
        Format current training progress.
        
        Args:
            algorithm: Algorithm instance
            include_full_data: Include full value function/policy
            
        Returns:
            Formatted progress data
        """
        progress = algorithm.get_training_progress()
        
        result = {
            'progress': progress,
            'is_complete': progress.get('episode', 0) >= progress.get('n_episodes', float('inf')) or
                          progress.get('iteration', 0) >= progress.get('max_iterations', float('inf')) or
                          progress.get('converged', False)
        }
        
        if include_full_data:
            if hasattr(algorithm, 'V'):
                result['value_function'] = VisualizationFormatter.format_value_function(
                    algorithm.V
                )
            if hasattr(algorithm, 'policy'):
                q_values = algorithm.Q if hasattr(algorithm, 'Q') else None
                result['policy'] = VisualizationFormatter.format_policy(
                    algorithm.policy,
                    q_values=q_values,
                    action_names=getattr(algorithm, 'action_names', None)
                )
        
        return result
    
    @staticmethod
    def format_inference_step(
        state: str,
        action: int,
        reward: float,
        render_data: Any,
        action_names: Optional[List[str]] = None,
        q_values: Optional[Dict[int, float]] = None
    ) -> Dict[str, Any]:
        """
        Format single inference step for visualization.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            render_data: Environment render data
            action_names: Action name mapping
            q_values: Q-values for state
            
        Returns:
            Formatted step data
        """
        return {
            'state': state,
            'action': action,
            'action_name': action_names[action] if action_names and action < len(action_names) else f'action_{action}',
            'reward': reward,
            'render_data': asdict(render_data) if hasattr(render_data, '__dataclass_fields__') else render_data,
            'q_values': q_values
        }

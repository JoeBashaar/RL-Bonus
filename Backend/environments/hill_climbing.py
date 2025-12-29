"""
Car Hill Climbing Environment

Goal: Reach the flag at the top of the hill. Local optimum = negative reward.

MDP Modeling:
- States: (car_position, car_velocity, stuck_counter)
- Actions: {accelerate_left, accelerate_right, no_action}
- Rewards: +1000 for goal, -100 for stuck at local optimum, -1 per step
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import ContinuousToDiscreteEnvironment, StateInfo, StepResult, RenderData


class HillClimbingEnvironment(ContinuousToDiscreteEnvironment):
    """
    Car hill climbing environment where the agent must drive up a hill.
    The car must build momentum to reach the peak.
    """
    
    # Actions
    ACCELERATE_LEFT = 0
    NO_ACTION = 1
    ACCELERATE_RIGHT = 2
    
    def __init__(self, **kwargs):
        super().__init__("hill_climbing", **kwargs)
        
        # Physics parameters
        self.min_position = kwargs.get('min_position', -1.2)
        self.max_position = kwargs.get('max_position', 0.6)
        self.max_velocity = kwargs.get('max_velocity', 0.07)
        self.goal_position = kwargs.get('goal_position', 0.5)
        self.goal_velocity = kwargs.get('goal_velocity', 0.0)
        
        # Discretization
        self.n_position_bins = kwargs.get('n_position_bins', 50)
        self.n_velocity_bins = kwargs.get('n_velocity_bins', 30)
        
        # Set up bins
        self.set_bins('position', self.min_position, self.max_position, self.n_position_bins)
        self.set_bins('velocity', -self.max_velocity, self.max_velocity, self.n_velocity_bins)
        
        # Local optimum detection
        self.stuck_threshold = kwargs.get('stuck_threshold', 20)
        self.local_optimum_position = -0.5  # Valley bottom
        self.local_optimum_range = 0.1
        
        # Action setup
        self.n_actions = 3
        self.action_names = ['accelerate_left', 'no_action', 'accelerate_right']
        
        # Physics constants
        self.force = 0.001
        self.gravity = 0.0025
        
        # State
        self.position: float = 0.0
        self.velocity: float = 0.0
        self.stuck_counter: int = 0
        
        # Default discount factor
        self.gamma = 0.99
        
        # Build state space
        self._build_state_space()
    
    def _build_state_space(self) -> None:
        """Build the discretized state space."""
        for pos_bin in range(self.n_position_bins + 1):
            for vel_bin in range(self.n_velocity_bins + 1):
                for stuck in [0, 1]:  # 0 = not stuck, 1 = stuck
                    state_id = self.encode_state(
                        position_bin=pos_bin,
                        velocity_bin=vel_bin,
                        stuck=stuck
                    )
                    self._state_space.add(state_id)
                    
                    # Goal states are terminal
                    pos = self.get_bin_value('position', pos_bin)
                    if pos >= self.goal_position:
                        self._terminal_states.add(state_id)
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into state ID."""
        return f"{kwargs['position_bin']},{kwargs['velocity_bin']},{kwargs['stuck']}"
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode state ID into components."""
        parts = state_id.split(',')
        pos_bin = int(parts[0])
        vel_bin = int(parts[1])
        stuck = int(parts[2])
        
        return {
            'position_bin': pos_bin,
            'velocity_bin': vel_bin,
            'stuck': stuck,
            'position': self.get_bin_value('position', pos_bin),
            'velocity': self.get_bin_value('velocity', vel_bin)
        }
    
    def _height(self, x: float) -> float:
        """Calculate terrain height at position x."""
        return np.sin(3 * x) * 0.45 + 0.55
    
    def _slope(self, x: float) -> float:
        """Calculate terrain slope at position x."""
        return np.cos(3 * x) * 0.45 * 3
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        self.stuck_counter = 0
        
        # Start at bottom of valley with zero velocity
        self.position = kwargs.get('start_position', -0.5)
        self.velocity = kwargs.get('start_velocity', 0.0)
        
        pos_bin = self.discretize(self.position, self._bins['position'])
        vel_bin = self.discretize(self.velocity, self._bins['velocity'])
        
        self.current_state = self.encode_state(
            position_bin=pos_bin,
            velocity_bin=vel_bin,
            stuck=0
        )
        
        return self.current_state, self.get_state_info()
    
    def step(self, action: int) -> StepResult:
        """Execute action and return result."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        reward = -1  # Time penalty
        done = False
        info = {'action_name': self.action_names[action]}
        
        # Apply action force
        force = (action - 1) * self.force  # -1, 0, or 1 * force
        
        # Physics update
        self.velocity += force - self.gravity * self._slope(self.position)
        self.velocity = np.clip(self.velocity, -self.max_velocity, self.max_velocity)
        self.position += self.velocity
        self.position = np.clip(self.position, self.min_position, self.max_position)
        
        # Bounce off left wall
        if self.position <= self.min_position and self.velocity < 0:
            self.velocity = 0
        
        # Check for local optimum (stuck at valley bottom)
        if abs(self.position - self.local_optimum_position) < self.local_optimum_range and abs(self.velocity) < 0.01:
            self.stuck_counter += 1
            if self.stuck_counter >= self.stuck_threshold:
                reward -= 100
                info['event'] = 'stuck_at_local_optimum'
        else:
            self.stuck_counter = max(0, self.stuck_counter - 1)
        
        # Check for goal
        if self.position >= self.goal_position:
            reward += 1000
            done = True
            info['event'] = 'goal_reached'
        
        # Timeout check
        if self.step_count >= self.max_steps:
            reward -= 10
            done = True
            info['timeout'] = True
        
        self.episode_reward += reward
        
        pos_bin = self.discretize(self.position, self._bins['position'])
        vel_bin = self.discretize(self.velocity, self._bins['velocity'])
        stuck = 1 if self.stuck_counter >= self.stuck_threshold else 0
        
        self.current_state = self.encode_state(
            position_bin=pos_bin,
            velocity_bin=vel_bin,
            stuck=stuck
        )
        
        return StepResult(
            next_state=self.current_state,
            reward=reward,
            done=done,
            info=info
        )
    
    def get_valid_actions(self, state_id: Optional[str] = None) -> List[int]:
        """Get valid actions for state."""
        return list(range(self.n_actions))
    
    def get_state_space(self) -> List[str]:
        """Get all possible states."""
        return list(self._state_space)
    
    def get_transition_prob(self, state: str, action: int) -> List[Tuple[str, float, float]]:
        """Get transition probabilities (deterministic physics)."""
        # Save current state
        old_pos = self.position
        old_vel = self.velocity
        old_stuck = self.stuck_counter
        old_state = self.current_state
        
        # Set to query state
        data = self.decode_state(state)
        self.position = data['position']
        self.velocity = data['velocity']
        self.stuck_counter = data['stuck'] * self.stuck_threshold
        self.current_state = state
        
        # Simulate step
        result = self.step(action)
        
        # Restore state
        self.position = old_pos
        self.velocity = old_vel
        self.stuck_counter = old_stuck
        self.current_state = old_state
        
        return [(result.next_state, 1.0, result.reward)]
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        position = data['position']
        velocity = data['velocity']
        
        # Generate terrain points for rendering
        terrain_points = []
        for x in np.linspace(self.min_position, self.max_position, 100):
            terrain_points.append({
                'x': float(x),
                'y': float(self._height(x))
            })
        
        entities = [
            {
                'type': 'car',
                'position': {'x': position, 'y': self._height(position)},
                'properties': {
                    'velocity': velocity,
                    'direction': 'right' if velocity >= 0 else 'left',
                    'stuck': data['stuck'] == 1
                }
            },
            {
                'type': 'flag',
                'position': {'x': self.goal_position, 'y': self._height(self.goal_position)},
                'properties': {'goal': True}
            },
            {
                'type': 'danger_zone',
                'position': {'x': self.local_optimum_position, 'y': self._height(self.local_optimum_position)},
                'properties': {
                    'width': self.local_optimum_range * 2,
                    'active': data['stuck'] == 1
                }
            }
        ]
        
        return RenderData(
            environment='hill_climbing',
            state_id=state,
            entities=entities,
            grid=None,
            metadata={
                'terrain': terrain_points,
                'position': position,
                'velocity': velocity,
                'stuck_counter': self.stuck_counter,
                'stuck_threshold': self.stuck_threshold,
                'step': self.step_count,
                'goal_position': self.goal_position,
                'min_position': self.min_position,
                'max_position': self.max_position
            }
        )

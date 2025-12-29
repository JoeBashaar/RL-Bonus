"""
Frozen Lake Environment

Goal: Navigate across a frozen lake from start to goal without falling into holes.

MDP Modeling:
- States: (agent_x, agent_y)
- Actions: {left, down, right, up}
- Rewards: +100 reach goal, -100 fall in hole, -1 per step
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import DiscreteEnvironment, StateInfo, StepResult, RenderData


class FrozenLakeEnvironment(DiscreteEnvironment):
    """
    Frozen Lake environment - navigate from Start to Goal avoiding holes.
    
    Grid Legend:
    - S: Start position
    - F: Frozen (safe)
    - H: Hole (game over)
    - G: Goal
    """
    
    # Actions
    LEFT = 0
    DOWN = 1
    RIGHT = 2
    UP = 3
    
    def __init__(self, **kwargs):
        super().__init__("frozen_lake", **kwargs)
        
        # Grid size
        self.size = kwargs.get('size', 8)
        self.grid_width = self.size
        self.grid_height = self.size
        
        # Slippery ice - probability of slipping
        self.is_slippery = kwargs.get('is_slippery', True)
        self.slip_prob = kwargs.get('slip_prob', 0.2)
        
        # Generate or use provided map
        self.map = kwargs.get('map', None)
        if self.map is None:
            self.map = self._generate_map()
        
        # Parse map to find start, goal, holes
        self.start_pos = None
        self.goal_pos = None
        self.holes = set()
        self.frozen = set()
        
        for y, row in enumerate(self.map):
            for x, cell in enumerate(row):
                if cell == 'S':
                    self.start_pos = (x, y)
                    self.frozen.add((x, y))
                elif cell == 'G':
                    self.goal_pos = (x, y)
                    self.frozen.add((x, y))
                elif cell == 'H':
                    self.holes.add((x, y))
                elif cell == 'F':
                    self.frozen.add((x, y))
        
        # Action setup
        self.n_actions = 4
        self.action_names = ['left', 'down', 'right', 'up']
        
        # Movement deltas
        self.action_deltas = {
            self.LEFT: (-1, 0),
            self.DOWN: (0, 1),
            self.RIGHT: (1, 0),
            self.UP: (0, -1)
        }
        
        # Default discount factor
        self.gamma = 0.99
        
        # Agent position
        self.agent_x: int = 0
        self.agent_y: int = 0
        
        # Build state space
        self._build_state_space()
    
    def _generate_map(self) -> List[str]:
        """Generate a random frozen lake map."""
        # Use a classic 8x8 layout
        if self.size == 4:
            return [
                "SFFF",
                "FHFH",
                "FFFH",
                "HFFG"
            ]
        elif self.size == 8:
            return [
                "SFFFFFFF",
                "FFFFFFFF",
                "FFFHFFFF",
                "FFFFFHFF",
                "FFFHFFFF",
                "FHHFFFHF",
                "FHFFHFHF",
                "FFFHFFFG"
            ]
        else:
            # Generate random map
            map_grid = [['F' for _ in range(self.size)] for _ in range(self.size)]
            map_grid[0][0] = 'S'
            map_grid[self.size-1][self.size-1] = 'G'
            
            # Add random holes (about 15% of tiles)
            n_holes = int(self.size * self.size * 0.15)
            holes_placed = 0
            while holes_placed < n_holes:
                x, y = self.rng.integers(0, self.size), self.rng.integers(0, self.size)
                if map_grid[y][x] == 'F':
                    map_grid[y][x] = 'H'
                    holes_placed += 1
            
            return [''.join(row) for row in map_grid]
    
    def _build_state_space(self) -> None:
        """Build the complete state space."""
        for x in range(self.size):
            for y in range(self.size):
                state_id = self.encode_state(agent_x=x, agent_y=y)
                self._state_space.add(state_id)
                
                # Terminal states (holes and goal)
                if (x, y) in self.holes or (x, y) == self.goal_pos:
                    self._terminal_states.add(state_id)
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into state ID."""
        return f"{kwargs['agent_x']},{kwargs['agent_y']}"
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode state ID into components."""
        parts = state_id.split(',')
        return {
            'agent_x': int(parts[0]),
            'agent_y': int(parts[1])
        }
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        
        self.agent_x, self.agent_y = self.start_pos
        
        self.current_state = self.encode_state(
            agent_x=self.agent_x,
            agent_y=self.agent_y
        )
        
        return self.current_state, self.get_state_info()
    
    def step(self, action: int) -> StepResult:
        """Execute action and return result."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        reward = -1  # Step penalty
        done = False
        info = {'action_name': self.action_names[action]}
        
        # Apply slipping if enabled
        actual_action = action
        if self.is_slippery and self.rng.random() < self.slip_prob:
            # Slip to perpendicular direction
            if action in [self.LEFT, self.RIGHT]:
                actual_action = self.rng.choice([self.UP, self.DOWN])
            else:
                actual_action = self.rng.choice([self.LEFT, self.RIGHT])
            info['slipped'] = True
            info['actual_action'] = self.action_names[actual_action]
        
        # Calculate new position
        dx, dy = self.action_deltas[actual_action]
        new_x = max(0, min(self.size - 1, self.agent_x + dx))
        new_y = max(0, min(self.size - 1, self.agent_y + dy))
        
        # Update position
        self.agent_x, self.agent_y = new_x, new_y
        
        # Check for hole
        if (self.agent_x, self.agent_y) in self.holes:
            reward = -100
            done = True
            info['event'] = 'fell_in_hole'
        
        # Check for goal
        elif (self.agent_x, self.agent_y) == self.goal_pos:
            reward = 100
            done = True
            info['event'] = 'reached_goal'
        
        # Timeout check
        if self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
        
        self.episode_reward += reward
        
        self.current_state = self.encode_state(
            agent_x=self.agent_x,
            agent_y=self.agent_y
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
        """Get transition probabilities."""
        data = self.decode_state(state)
        x, y = data['agent_x'], data['agent_y']
        
        # If terminal, stay in place
        if (x, y) in self.holes or (x, y) == self.goal_pos:
            return [(state, 1.0, 0.0)]
        
        transitions = []
        
        if self.is_slippery:
            # Main action with (1 - slip_prob)
            main_prob = 1.0 - self.slip_prob
            
            # Calculate main transition
            dx, dy = self.action_deltas[action]
            new_x = max(0, min(self.size - 1, x + dx))
            new_y = max(0, min(self.size - 1, y + dy))
            
            reward = self._get_reward(new_x, new_y)
            new_state = self.encode_state(agent_x=new_x, agent_y=new_y)
            transitions.append((new_state, main_prob, reward))
            
            # Slip actions
            if action in [self.LEFT, self.RIGHT]:
                slip_actions = [self.UP, self.DOWN]
            else:
                slip_actions = [self.LEFT, self.RIGHT]
            
            slip_prob_each = self.slip_prob / 2.0
            for slip_action in slip_actions:
                dx, dy = self.action_deltas[slip_action]
                new_x = max(0, min(self.size - 1, x + dx))
                new_y = max(0, min(self.size - 1, y + dy))
                
                reward = self._get_reward(new_x, new_y)
                new_state = self.encode_state(agent_x=new_x, agent_y=new_y)
                transitions.append((new_state, slip_prob_each, reward))
        else:
            # Deterministic
            dx, dy = self.action_deltas[action]
            new_x = max(0, min(self.size - 1, x + dx))
            new_y = max(0, min(self.size - 1, y + dy))
            
            reward = self._get_reward(new_x, new_y)
            new_state = self.encode_state(agent_x=new_x, agent_y=new_y)
            transitions.append((new_state, 1.0, reward))
        
        return transitions
    
    def _get_reward(self, x: int, y: int) -> float:
        """Get reward for landing on position."""
        if (x, y) in self.holes:
            return -100
        elif (x, y) == self.goal_pos:
            return 100
        return -1
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        
        # Build cell types grid
        cell_types = []
        for y in range(self.size):
            row = []
            for x in range(self.size):
                if (x, y) in self.holes:
                    row.append('water')  # Holes are water/ice holes
                elif (x, y) == self.goal_pos:
                    row.append('goal')
                elif (x, y) == self.start_pos:
                    row.append('ice')
                else:
                    row.append('ice')
            cell_types.append(row)
        
        entities = [
            {
                'type': 'skater',
                'position': {'x': data['agent_x'], 'y': data['agent_y']},
                'properties': {
                    'on_hole': (data['agent_x'], data['agent_y']) in self.holes,
                    'at_goal': (data['agent_x'], data['agent_y']) == self.goal_pos
                }
            },
            {
                'type': 'flag',
                'position': {'x': self.goal_pos[0], 'y': self.goal_pos[1]},
                'properties': {'goal': True}
            }
        ]
        
        # Add hole markers
        for hx, hy in self.holes:
            entities.append({
                'type': 'hole',
                'position': {'x': hx, 'y': hy},
                'properties': {'danger': True}
            })
        
        return RenderData(
            environment='frozen_lake',
            state_id=state,
            entities=entities,
            grid={
                'width': self.size,
                'height': self.size,
                'cell_types': cell_types
            },
            metadata={
                'agent_x': data['agent_x'],
                'agent_y': data['agent_y'],
                'step': self.step_count,
                'slippery': self.is_slippery,
                'holes_count': len(self.holes)
            }
        )

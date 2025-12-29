"""
Haunted House (Hide and Seek) Environment

Goal: Seeker catches the real hider (not ghost) on multi-level 2D planes.

MDP Modeling:
- States: (seeker_pos, hider_pos, ghost_pos, level, time_remaining)
- Actions: {move_up/down/left/right, climb_ladder_up/down, catch}
- Rewards: +200 catch hider, -150 catch ghost, -1 per step
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import DiscreteEnvironment, StateInfo, StepResult, RenderData


class HauntedHouseEnvironment(DiscreteEnvironment):
    """
    Multi-level hide and seek game where seeker must catch the real hider.
    Ghost mimics hider but catching it results in penalty.
    """
    
    # Actions
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CLIMB_UP = 4
    CLIMB_DOWN = 5
    CATCH = 6
    
    def __init__(self, **kwargs):
        super().__init__("haunted_house", **kwargs)
        
        # Grid dimensions per level
        self.grid_width = kwargs.get('grid_width', 8)
        self.grid_height = kwargs.get('grid_height', 8)
        self.n_levels = kwargs.get('n_levels', 3)
        
        # Time limit
        self.time_limit = kwargs.get('time_limit', 100)
        
        # Ghost catches before game over
        self.max_ghost_catches = kwargs.get('max_ghost_catches', 3)
        
        # Action setup
        self.n_actions = 7
        self.action_names = ['move_up', 'move_down', 'move_left', 'move_right',
                            'climb_ladder_up', 'climb_ladder_down', 'catch']
        
        # Default discount factor
        self.gamma = 0.99
        
        # Ladder positions (shared between levels)
        self.ladder_positions = [
            (1, 1), (6, 6), (3, 4), (5, 2)
        ]
        
        # Obstacles (walls/furniture) per level - list of blocked cells
        self.obstacles = {
            0: [(2, 2), (2, 3), (5, 5), (5, 6)],
            1: [(1, 4), (1, 5), (6, 2), (6, 3)],
            2: [(3, 3), (4, 4), (0, 6), (7, 1)]
        }
        
        # Entity positions
        self.seeker_x: int = 0
        self.seeker_y: int = 0
        self.seeker_level: int = 0
        self.hider_x: int = 0
        self.hider_y: int = 0
        self.hider_level: int = 0
        self.ghost_x: int = 0
        self.ghost_y: int = 0
        self.ghost_level: int = 0
        self.time_remaining: int = 0
        self.ghost_catches: int = 0
        
        # Build simplified state space
        self._build_state_space()
    
    def _build_state_space(self) -> None:
        """Build a representative subset of state space.
        
        Uses simplified encoding: levels + discretized distances + time bin.
        """
        for seeker_level in range(self.n_levels):
            for hider_level in range(self.n_levels):
                for ghost_level in range(self.n_levels):
                    for dist_to_hider in range(5):  # 0-4 representing distance bins
                        for dist_to_ghost in range(5):
                            for time_bin in range(11):  # 0-10 representing time bins
                                state_id = self._encode_simplified(
                                    seeker_level=seeker_level,
                                    hider_level=hider_level,
                                    ghost_level=ghost_level,
                                    dist_to_hider=dist_to_hider,
                                    dist_to_ghost=dist_to_ghost,
                                    time_bin=time_bin
                                )
                                self._state_space.add(state_id)
    
    def _encode_simplified(self, **kwargs) -> str:
        """Encode into simplified state (levels + distances + time)."""
        return (f"{kwargs['seeker_level']},"
                f"{kwargs['hider_level']},"
                f"{kwargs['ghost_level']},"
                f"{kwargs['dist_to_hider']},"
                f"{kwargs['dist_to_ghost']},"
                f"{kwargs['time_bin']}")
    
    def _get_distance_bin(self, x1: int, y1: int, x2: int, y2: int) -> int:
        """Get discretized distance bin (0-4)."""
        dist = abs(x1 - x2) + abs(y1 - y2)  # Manhattan distance
        if dist <= 2:
            return 0
        elif dist <= 4:
            return 1
        elif dist <= 6:
            return 2
        elif dist <= 9:
            return 3
        else:
            return 4
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into simplified state ID.
        
        Uses levels + discretized distances to keep state space manageable.
        """
        dist_to_hider = self._get_distance_bin(
            kwargs['seeker_x'], kwargs['seeker_y'],
            kwargs['hider_x'], kwargs['hider_y']
        )
        dist_to_ghost = self._get_distance_bin(
            kwargs['seeker_x'], kwargs['seeker_y'],
            kwargs['ghost_x'], kwargs['ghost_y']
        )
        time_bin = min(kwargs['time_remaining'] // 10, 10)
        
        return self._encode_simplified(
            seeker_level=kwargs['seeker_level'],
            hider_level=kwargs['hider_level'],
            ghost_level=kwargs['ghost_level'],
            dist_to_hider=dist_to_hider,
            dist_to_ghost=dist_to_ghost,
            time_bin=time_bin
        )
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode simplified state ID into components.
        
        Note: Full position info is not stored, only levels and distances.
        """
        parts = state_id.split(',')
        return {
            'seeker_level': int(parts[0]),
            'hider_level': int(parts[1]),
            'ghost_level': int(parts[2]),
            'dist_to_hider': int(parts[3]),
            'dist_to_ghost': int(parts[4]),
            'time_bin': int(parts[5]),
            # These are approximations since we don't store exact positions
            'seeker_x': 0,
            'seeker_y': 0,
            'hider_x': 7,
            'hider_y': 7,
            'ghost_x': 4,
            'ghost_y': 4,
            'time_remaining': int(parts[5]) * 10,
            'ghost_catches': 0
        }
    
    def _is_blocked(self, x: int, y: int, level: int) -> bool:
        """Check if a position is blocked by obstacle."""
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        return (x, y) in self.obstacles.get(level, [])
    
    def _is_ladder(self, x: int, y: int) -> bool:
        """Check if position has a ladder."""
        return (x, y) in self.ladder_positions
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        self.time_remaining = self.time_limit
        self.ghost_catches = 0
        
        # Random seeker start
        self.seeker_level = 0
        self.seeker_x, self.seeker_y = self._random_valid_position(0)
        
        # Random hider start (different level preferred)
        self.hider_level = self.rng.integers(0, self.n_levels)
        self.hider_x, self.hider_y = self._random_valid_position(self.hider_level)
        
        # Random ghost start
        self.ghost_level = self.rng.integers(0, self.n_levels)
        self.ghost_x, self.ghost_y = self._random_valid_position(self.ghost_level)
        
        self.current_state = self.encode_state(
            seeker_x=self.seeker_x, seeker_y=self.seeker_y, seeker_level=self.seeker_level,
            hider_x=self.hider_x, hider_y=self.hider_y, hider_level=self.hider_level,
            ghost_x=self.ghost_x, ghost_y=self.ghost_y, ghost_level=self.ghost_level,
            time_remaining=self.time_remaining, ghost_catches=self.ghost_catches
        )
        
        return self.current_state, self.get_state_info()
    
    def _random_valid_position(self, level: int) -> Tuple[int, int]:
        """Get random valid position on a level."""
        while True:
            x = self.rng.integers(0, self.grid_width)
            y = self.rng.integers(0, self.grid_height)
            if not self._is_blocked(x, y, level):
                return x, y
    
    def step(self, action: int) -> StepResult:
        """Execute action and return result."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        self.time_remaining -= 1
        reward = -1  # Time penalty
        done = False
        info = {'action_name': self.action_names[action]}
        
        # Execute seeker action
        new_x, new_y, new_level = self.seeker_x, self.seeker_y, self.seeker_level
        
        if action == self.MOVE_UP:
            new_y = max(0, self.seeker_y - 1)
        elif action == self.MOVE_DOWN:
            new_y = min(self.grid_height - 1, self.seeker_y + 1)
        elif action == self.MOVE_LEFT:
            new_x = max(0, self.seeker_x - 1)
        elif action == self.MOVE_RIGHT:
            new_x = min(self.grid_width - 1, self.seeker_x + 1)
        elif action == self.CLIMB_UP:
            if self._is_ladder(self.seeker_x, self.seeker_y) and self.seeker_level < self.n_levels - 1:
                new_level = self.seeker_level + 1
        elif action == self.CLIMB_DOWN:
            if self._is_ladder(self.seeker_x, self.seeker_y) and self.seeker_level > 0:
                new_level = self.seeker_level - 1
        elif action == self.CATCH:
            # Check if hider is adjacent and same level
            hider_dist = abs(self.seeker_x - self.hider_x) + abs(self.seeker_y - self.hider_y)
            ghost_dist = abs(self.seeker_x - self.ghost_x) + abs(self.seeker_y - self.ghost_y)
            
            if self.seeker_level == self.hider_level and hider_dist <= 1:
                reward += 200
                done = True
                info['event'] = 'caught_hider'
            elif self.seeker_level == self.ghost_level and ghost_dist <= 1:
                reward -= 150
                self.ghost_catches += 1
                info['event'] = 'caught_ghost'
                # Reset positions
                self.ghost_x, self.ghost_y = self._random_valid_position(self.ghost_level)
                self.time_remaining -= 10  # Time penalty
                if self.ghost_catches >= self.max_ghost_catches:
                    done = True
                    info['game_over'] = 'too_many_ghost_catches'
            else:
                reward -= 50
                info['event'] = 'catch_missed'
        
        # Update position if not blocked
        if not self._is_blocked(new_x, new_y, new_level):
            self.seeker_x, self.seeker_y, self.seeker_level = new_x, new_y, new_level
        
        # Move ghost (random, every 2 steps)
        if self.step_count % 2 == 0:
            self._move_ghost()
        
        # Move hider (random, every 3 steps)
        if self.step_count % 3 == 0:
            self._move_hider()
        
        # Timeout check
        if self.time_remaining <= 0:
            reward -= 100
            done = True
            info['timeout'] = True
        
        self.episode_reward += reward
        
        self.current_state = self.encode_state(
            seeker_x=self.seeker_x, seeker_y=self.seeker_y, seeker_level=self.seeker_level,
            hider_x=self.hider_x, hider_y=self.hider_y, hider_level=self.hider_level,
            ghost_x=self.ghost_x, ghost_y=self.ghost_y, ghost_level=self.ghost_level,
            time_remaining=self.time_remaining, ghost_catches=self.ghost_catches
        )
        
        return StepResult(
            next_state=self.current_state,
            reward=reward,
            done=done,
            info=info
        )
    
    def _move_ghost(self) -> None:
        """Move ghost randomly."""
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        self.rng.shuffle(moves)
        
        for dx, dy in moves:
            new_x, new_y = self.ghost_x + dx, self.ghost_y + dy
            if not self._is_blocked(new_x, new_y, self.ghost_level):
                self.ghost_x, self.ghost_y = new_x, new_y
                break
        
        # Occasionally change level
        if self.rng.random() < 0.1 and self._is_ladder(self.ghost_x, self.ghost_y):
            if self.rng.random() < 0.5 and self.ghost_level > 0:
                self.ghost_level -= 1
            elif self.ghost_level < self.n_levels - 1:
                self.ghost_level += 1
    
    def _move_hider(self) -> None:
        """Move hider randomly, trying to avoid seeker."""
        moves = [(0, -1), (0, 1), (-1, 0), (1, 0), (0, 0)]  # Include staying
        
        best_move = (0, 0)
        best_dist = 0
        
        for dx, dy in moves:
            new_x, new_y = self.hider_x + dx, self.hider_y + dy
            if not self._is_blocked(new_x, new_y, self.hider_level):
                # Distance from seeker
                if self.hider_level == self.seeker_level:
                    dist = abs(new_x - self.seeker_x) + abs(new_y - self.seeker_y)
                else:
                    dist = 100  # Different level, safe
                
                if dist > best_dist or (dist == best_dist and self.rng.random() < 0.3):
                    best_dist = dist
                    best_move = (dx, dy)
        
        self.hider_x += best_move[0]
        self.hider_y += best_move[1]
        
        # Occasionally change level to escape
        if self.hider_level == self.seeker_level and self._is_ladder(self.hider_x, self.hider_y):
            if self.rng.random() < 0.3:
                if self.hider_level > 0:
                    self.hider_level -= 1
                elif self.hider_level < self.n_levels - 1:
                    self.hider_level += 1
    
    def get_valid_actions(self, state_id: Optional[str] = None) -> List[int]:
        """Get valid actions for state."""
        return list(range(self.n_actions))
    
    def get_state_space(self) -> List[str]:
        """Get all possible states."""
        return list(self._state_space)
    
    def get_transition_prob(self, state: str, action: int) -> List[Tuple[str, float, float]]:
        """Get transition probabilities (stochastic due to ghost/hider movement)."""
        # Simplified: return most likely transition
        old_state = self.current_state
        data = self.decode_state(state)
        
        self.seeker_x = data['seeker_x']
        self.seeker_y = data['seeker_y']
        self.seeker_level = data['seeker_level']
        self.hider_x = data['hider_x']
        self.hider_y = data['hider_y']
        self.hider_level = data['hider_level']
        self.ghost_x = data['ghost_x']
        self.ghost_y = data['ghost_y']
        self.ghost_level = data['ghost_level']
        self.time_remaining = data['time_remaining']
        self.ghost_catches = data['ghost_catches']
        self.current_state = state
        
        result = self.step(action)
        
        self.current_state = old_state
        
        return [(result.next_state, 1.0, result.reward)]
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        
        entities = [
            {
                'type': 'seeker',
                'position': {'x': data['seeker_x'], 'y': data['seeker_y']},
                'properties': {
                    'level': data['seeker_level'],
                    'has_flashlight': True,
                    'vision_cone': {'angle': 90, 'range': 3}
                }
            },
            {
                'type': 'hider',
                'position': {'x': data['hider_x'], 'y': data['hider_y']},
                'properties': {
                    'level': data['hider_level'],
                    'visible': data['seeker_level'] == data['hider_level']
                }
            },
            {
                'type': 'ghost',
                'position': {'x': data['ghost_x'], 'y': data['ghost_y']},
                'properties': {
                    'level': data['ghost_level'],
                    'visible': data['seeker_level'] == data['ghost_level'],
                    'translucent': True,
                    'shimmer': True
                }
            }
        ]
        
        # Add ladders
        for lx, ly in self.ladder_positions:
            entities.append({
                'type': 'ladder',
                'position': {'x': lx, 'y': ly},
                'properties': {'spans_all_levels': True}
            })
        
        # Add obstacles per level
        for level, obs_list in self.obstacles.items():
            for ox, oy in obs_list:
                entities.append({
                    'type': 'obstacle',
                    'position': {'x': ox, 'y': oy},
                    'properties': {'level': level, 'obstacle_type': 'furniture'}
                })
        
        return RenderData(
            environment='haunted_house',
            state_id=state,
            entities=entities,
            grid={
                'width': self.grid_width,
                'height': self.grid_height,
                'levels': self.n_levels
            },
            metadata={
                'time_remaining': data['time_remaining'],
                'time_limit': self.time_limit,
                'ghost_catches': data['ghost_catches'],
                'max_ghost_catches': self.max_ghost_catches,
                'current_level': data['seeker_level'],
                'step': self.step_count
            }
        )

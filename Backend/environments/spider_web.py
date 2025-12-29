"""
Spider Web Mosquito Hunting Environment

Goal: Catch high-value mosquitoes in web before they escape.

MDP Modeling:
- States: (spider_pos, mosquitoes_in_web, time_in_web, mosquito_sizes)
- Actions: {move_up/down/left/right, catch_mosquito}
- Rewards: +size*50 for catching, -30 for escape, -1 per step
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import DiscreteEnvironment, StateInfo, StepResult, RenderData
from dataclasses import dataclass


@dataclass
class Mosquito:
    """Represents a mosquito in the environment."""
    x: int
    y: int
    size: int  # 1=small, 2=medium, 3=large
    time_in_web: int
    in_web: bool
    escape_time: int  # Steps until escape


class SpiderWebEnvironment(DiscreteEnvironment):
    """
    Spider web environment where spider catches mosquitoes.
    Larger mosquitoes are worth more points but escape faster.
    """
    
    # Actions
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    CATCH = 4
    
    # Mosquito sizes
    SMALL = 1
    MEDIUM = 2
    LARGE = 3
    
    def __init__(self, **kwargs):
        super().__init__("spider_web", **kwargs)
        
        # Web dimensions
        self.grid_width = kwargs.get('grid_width', 10)
        self.grid_height = kwargs.get('grid_height', 10)
        
        # Game parameters
        self.time_limit = kwargs.get('time_limit', 200)
        self.max_mosquitoes = kwargs.get('max_mosquitoes', 5)
        self.spawn_rate = kwargs.get('spawn_rate', 0.15)  # Probability per step
        self.web_land_prob = kwargs.get('web_land_prob', 0.2)
        
        # Escape times by size (smaller = slower to escape)
        self.escape_times = {
            self.SMALL: 10,
            self.MEDIUM: 7,
            self.LARGE: 5
        }
        
        # Rewards by size
        self.catch_rewards = {
            self.SMALL: 50,
            self.MEDIUM: 100,
            self.LARGE: 150
        }
        
        # Action setup
        self.n_actions = 5
        self.action_names = ['move_up', 'move_down', 'move_left', 'move_right', 'catch']
        
        # Default discount factor
        self.gamma = 0.99
        
        # State
        self.spider_x: int = 0
        self.spider_y: int = 0
        self.mosquitoes: List[Mosquito] = []
        self.time_remaining: int = 0
        self.total_caught: int = 0
        
        # Build state space
        self._build_state_space()
    
    def _build_state_space(self) -> None:
        """Build representative state space."""
        # Full state space is too large, use simplified representation
        for spider_x in range(self.grid_width):
            for spider_y in range(self.grid_height):
                for n_mosquitoes in range(self.max_mosquitoes + 1):
                    state_id = self.encode_state(
                        spider_x=spider_x,
                        spider_y=spider_y,
                        mosquitoes=[],
                        n_mosquitoes=n_mosquitoes,
                        time_remaining=self.time_limit
                    )
                    self._state_space.add(state_id)
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into state ID.
        
        Uses simplified representation (n_mosquitoes count only) for tabular methods.
        """
        n_mosq = kwargs.get('n_mosquitoes', len(kwargs.get('mosquitoes', [])))
        return (f"{kwargs['spider_x']},{kwargs['spider_y']},"
                f"{n_mosq},{kwargs['time_remaining']}")
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode state ID into components."""
        parts = state_id.split(',')
        
        return {
            'spider_x': int(parts[0]),
            'spider_y': int(parts[1]),
            'n_mosquitoes': int(parts[2]),
            'time_remaining': int(parts[3]),
            'mosquitoes': []  # Simplified - we only track count now
        }
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        self.time_remaining = self.time_limit
        self.total_caught = 0
        
        # Spider starts at center
        self.spider_x = self.grid_width // 2
        self.spider_y = self.grid_height // 2
        
        # Start with a few mosquitoes
        self.mosquitoes = []
        for _ in range(2):
            self._spawn_mosquito()
        
        self.current_state = self.encode_state(
            spider_x=self.spider_x,
            spider_y=self.spider_y,
            mosquitoes=self.mosquitoes,
            time_remaining=self.time_remaining
        )
        
        return self.current_state, self.get_state_info()
    
    def _spawn_mosquito(self) -> None:
        """Spawn a new mosquito near the web."""
        if len(self.mosquitoes) >= self.max_mosquitoes:
            return
        
        # Random position near web edge
        if self.rng.random() < 0.5:
            x = self.rng.choice([0, self.grid_width - 1])
            y = self.rng.integers(0, self.grid_height)
        else:
            x = self.rng.integers(0, self.grid_width)
            y = self.rng.choice([0, self.grid_height - 1])
        
        # Random size (smaller more common)
        size_probs = [0.5, 0.35, 0.15]
        size = self.rng.choice([self.SMALL, self.MEDIUM, self.LARGE], p=size_probs)
        
        self.mosquitoes.append(Mosquito(
            x=x, y=y, size=size,
            time_in_web=0,
            in_web=False,
            escape_time=self.escape_times[size]
        ))
    
    def step(self, action: int) -> StepResult:
        """Execute action and return result."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        self.time_remaining -= 1
        reward = -1  # Time penalty
        done = False
        info = {'action_name': self.action_names[action]}
        
        # Execute spider action
        if action == self.MOVE_UP:
            self.spider_y = max(0, self.spider_y - 1)
        elif action == self.MOVE_DOWN:
            self.spider_y = min(self.grid_height - 1, self.spider_y + 1)
        elif action == self.MOVE_LEFT:
            self.spider_x = max(0, self.spider_x - 1)
        elif action == self.MOVE_RIGHT:
            self.spider_x = min(self.grid_width - 1, self.spider_x + 1)
        elif action == self.CATCH:
            # Try to catch mosquito at spider position
            caught = None
            for m in self.mosquitoes:
                if m.x == self.spider_x and m.y == self.spider_y and m.in_web:
                    caught = m
                    break
            
            if caught:
                reward += self.catch_rewards[caught.size]
                self.mosquitoes.remove(caught)
                self.total_caught += 1
                info['event'] = 'caught_mosquito'
                info['size'] = caught.size
            else:
                reward -= 10
                info['event'] = 'catch_missed'
        
        # Proximity bonus
        for m in self.mosquitoes:
            if m.in_web:
                dist = abs(m.x - self.spider_x) + abs(m.y - self.spider_y)
                if dist <= 2:
                    reward += 5
        
        # Update mosquitoes
        escaped = []
        for m in self.mosquitoes:
            if m.in_web:
                m.time_in_web += 1
                if m.time_in_web >= m.escape_time:
                    escaped.append(m)
            else:
                # Move mosquito randomly
                dx = self.rng.integers(-1, 2)
                dy = self.rng.integers(-1, 2)
                m.x = np.clip(m.x + dx, 0, self.grid_width - 1)
                m.y = np.clip(m.y + dy, 0, self.grid_height - 1)
                
                # Check if lands in web (center area)
                if 2 <= m.x <= self.grid_width - 3 and 2 <= m.y <= self.grid_height - 3:
                    if self.rng.random() < self.web_land_prob:
                        m.in_web = True
                        info['mosquito_landed'] = True
        
        # Handle escapes
        for m in escaped:
            reward -= 30
            self.mosquitoes.remove(m)
            info['mosquito_escaped'] = True
        
        # Spawn new mosquitoes
        if self.rng.random() < self.spawn_rate:
            self._spawn_mosquito()
        
        # Check end conditions
        if self.time_remaining <= 0:
            done = True
            info['timeout'] = True
        elif len(self.mosquitoes) == 0 and self.time_remaining < self.time_limit - 50:
            done = True
            info['no_mosquitoes'] = True
        
        self.episode_reward += reward
        
        self.current_state = self.encode_state(
            spider_x=self.spider_x,
            spider_y=self.spider_y,
            mosquitoes=self.mosquitoes,
            time_remaining=self.time_remaining
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
        """Get transition probabilities (stochastic due to mosquito behavior)."""
        old_state = self.current_state
        old_spider_x = self.spider_x
        old_spider_y = self.spider_y
        old_mosquitoes = [Mosquito(m.x, m.y, m.size, m.time_in_web, m.in_web, m.escape_time) 
                         for m in self.mosquitoes]
        old_time = self.time_remaining
        
        data = self.decode_state(state)
        self.spider_x = data['spider_x']
        self.spider_y = data['spider_y']
        self.mosquitoes = data['mosquitoes']
        self.time_remaining = data['time_remaining']
        self.current_state = state
        
        result = self.step(action)
        
        self.spider_x = old_spider_x
        self.spider_y = old_spider_y
        self.mosquitoes = old_mosquitoes
        self.time_remaining = old_time
        self.current_state = old_state
        
        return [(result.next_state, 1.0, result.reward)]
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        
        size_names = {1: 'small', 2: 'medium', 3: 'large'}
        
        entities = [
            {
                'type': 'spider',
                'position': {'x': data['spider_x'], 'y': data['spider_y']},
                'properties': {
                    'legs_animated': True
                }
            }
        ]
        
        # Add mosquitoes
        for i, m in enumerate(data['mosquitoes']):
            entities.append({
                'type': 'mosquito',
                'position': {'x': m.x, 'y': m.y},
                'properties': {
                    'size': size_names[m.size],
                    'in_web': m.in_web,
                    'time_in_web': m.time_in_web,
                    'escape_time': m.escape_time,
                    'escape_progress': m.time_in_web / m.escape_time if m.in_web else 0,
                    'wings_animated': True,
                    'struggling': m.in_web
                }
            })
        
        # Web center area
        web_cells = []
        for x in range(2, self.grid_width - 2):
            for y in range(2, self.grid_height - 2):
                web_cells.append({'x': x, 'y': y})
        
        return RenderData(
            environment='spider_web',
            state_id=state,
            entities=entities,
            grid={
                'width': self.grid_width,
                'height': self.grid_height,
                'web_cells': web_cells
            },
            metadata={
                'time_remaining': data['time_remaining'],
                'time_limit': self.time_limit,
                'total_caught': self.total_caught,
                'mosquitoes_count': len(data['mosquitoes']),
                'max_mosquitoes': self.max_mosquitoes,
                'step': self.step_count
            }
        )

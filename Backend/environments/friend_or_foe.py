"""
Friend or Foe? (Door Opening Game) Environment

Goal: Survive 20 nights by deciding whether to let in strangers.

MDP Modeling:
- States: (day, phase, total_reward, strangers_inside, stranger_at_door)
- Actions: {let_in, reject, shoot_at_door, shoot_inside}
- Rewards: Various based on stranger type and action
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import DiscreteEnvironment, StateInfo, StepResult, RenderData
from enum import IntEnum


class StrangerType(IntEnum):
    NONE = 0
    HELPFUL = 1
    TRAITOR = 2
    NEUTRAL = 3


class FriendOrFoeEnvironment(DiscreteEnvironment):
    """
    Door opening game where player must decide who to let in.
    Helpful strangers provide rewards, traitors cause harm.
    Survive 20 days/nights to win.
    """
    
    # Actions
    LET_IN = 0
    REJECT = 1
    SHOOT_DOOR = 2
    SHOOT_INSIDE = 3
    
    def __init__(self, **kwargs):
        super().__init__("friend_or_foe", **kwargs)
        
        # Game parameters
        self.total_days = kwargs.get('total_days', 20)
        self.death_threshold = kwargs.get('death_threshold', -200)
        
        # Stranger probabilities
        self.door_prob_day = 0.7
        self.door_prob_night = 0.5
        
        self.type_probs_day = [0.6, 0.2, 0.2]  # helpful, traitor, neutral
        self.type_probs_night = [0.3, 0.5, 0.2]
        
        # Rewards
        self.rewards = {
            'helpful_let_in': 50,
            'traitor_let_in': -80,
            'neutral_let_in': 0,
            'shoot_traitor': 30,
            'shoot_helpful': -100,
            'shoot_neutral': -20,
            'reject_helpful': -15,
            'reject_traitor': 10,
            'reject_neutral': 0,
        }
        
        # Action setup
        self.n_actions = 4
        self.action_names = ['let_in', 'reject', 'shoot_at_door', 'shoot_inside']
        
        # Default discount factor
        self.gamma = 0.99
        
        # State
        self.current_day: int = 1
        self.is_night: bool = False
        self.total_reward: int = 0
        self.strangers_inside: List[StrangerType] = []
        self.stranger_at_door: StrangerType = StrangerType.NONE
        self.pending_effects: List[int] = []  # Rewards to apply next cycle
        
        # Build state space
        self._build_state_space()
    
    def _build_state_space(self) -> None:
        """Build representative state space."""
        for day in range(1, self.total_days + 1):
            for is_night in [False, True]:
                for reward_bin in range(-20, 21):  # Discretized reward
                    for n_inside in range(4):
                        for stranger in range(4):
                            state_id = self.encode_state(
                                day=day,
                                is_night=is_night,
                                total_reward=reward_bin * 10,
                                n_strangers_inside=n_inside,
                                stranger_at_door=stranger
                            )
                            self._state_space.add(state_id)
                            
                            # Terminal states
                            if day > self.total_days or reward_bin * 10 <= self.death_threshold:
                                self._terminal_states.add(state_id)
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into state ID.
        
        Uses simplified representation (count of strangers, not individual types)
        to keep state space manageable for tabular methods.
        """
        n_inside = kwargs.get('n_strangers_inside', len(kwargs.get('strangers_inside', [])))
        return (f"{kwargs['day']},{int(kwargs['is_night'])},"
                f"{kwargs.get('total_reward', 0)},"
                f"{n_inside},"
                f"{kwargs['stranger_at_door']}")
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode state ID into components."""
        parts = state_id.split(',')
        
        return {
            'day': int(parts[0]),
            'is_night': bool(int(parts[1])),
            'total_reward': int(parts[2]),
            'n_strangers_inside': int(parts[3]),
            'stranger_at_door': StrangerType(int(parts[4])),
            'strangers_inside': []  # Simplified - we only track count now
        }
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        
        self.current_day = 1
        self.is_night = False
        self.total_reward = 0
        self.strangers_inside = []
        self.pending_effects = []
        
        # Maybe spawn stranger at door
        self._maybe_spawn_stranger()
        
        self.current_state = self.encode_state(
            day=self.current_day,
            is_night=self.is_night,
            total_reward=self.total_reward,
            strangers_inside=self.strangers_inside,
            stranger_at_door=self.stranger_at_door
        )
        
        return self.current_state, self.get_state_info()
    
    def _maybe_spawn_stranger(self) -> None:
        """Maybe spawn a stranger at the door."""
        door_prob = self.door_prob_night if self.is_night else self.door_prob_day
        
        if self.rng.random() < door_prob:
            type_probs = self.type_probs_night if self.is_night else self.type_probs_day
            stranger_type = self.rng.choice(
                [StrangerType.HELPFUL, StrangerType.TRAITOR, StrangerType.NEUTRAL],
                p=type_probs
            )
            self.stranger_at_door = stranger_type
        else:
            self.stranger_at_door = StrangerType.NONE
    
    def step(self, action: int) -> StepResult:
        """Execute action and return result."""
        if self.current_state is None:
            raise RuntimeError("Environment not initialized. Call reset() first.")
        
        self.step_count += 1
        reward = -1  # Step penalty to encourage efficiency
        done = False
        info = {'action_name': self.action_names[action]}
        
        # Handle action on stranger at door
        if self.stranger_at_door != StrangerType.NONE:
            if action == self.LET_IN:
                # Effects apply next cycle
                self.strangers_inside.append(self.stranger_at_door)
                if self.stranger_at_door == StrangerType.HELPFUL:
                    self.pending_effects.append(self.rewards['helpful_let_in'])
                    info['event'] = 'let_in_helpful'
                elif self.stranger_at_door == StrangerType.TRAITOR:
                    self.pending_effects.append(self.rewards['traitor_let_in'])
                    info['event'] = 'let_in_traitor'
                else:
                    info['event'] = 'let_in_neutral'
                    
            elif action == self.REJECT:
                if self.stranger_at_door == StrangerType.HELPFUL:
                    reward += self.rewards['reject_helpful']
                    info['event'] = 'rejected_helpful'
                elif self.stranger_at_door == StrangerType.TRAITOR:
                    reward += self.rewards['reject_traitor']
                    info['event'] = 'rejected_traitor'
                else:
                    info['event'] = 'rejected_neutral'
                    
            elif action == self.SHOOT_DOOR:
                if self.stranger_at_door == StrangerType.HELPFUL:
                    reward += self.rewards['shoot_helpful']
                    info['event'] = 'shot_helpful'
                elif self.stranger_at_door == StrangerType.TRAITOR:
                    reward += self.rewards['shoot_traitor']
                    info['event'] = 'shot_traitor'
                else:
                    reward += self.rewards['shoot_neutral']
                    info['event'] = 'shot_neutral'
        
        # Handle shooting inside
        if action == self.SHOOT_INSIDE and self.strangers_inside:
            # Shoot first stranger inside (before their effects trigger)
            target = self.strangers_inside.pop(0)
            if target == StrangerType.HELPFUL:
                reward += self.rewards['shoot_helpful']
                info['shot_inside'] = 'helpful'
            elif target == StrangerType.TRAITOR:
                reward += self.rewards['shoot_traitor']
                info['shot_inside'] = 'traitor'
            else:
                reward += self.rewards['shoot_neutral']
                info['shot_inside'] = 'neutral'
            
            # Remove corresponding pending effect
            if self.pending_effects:
                self.pending_effects.pop(0)
        
        # Advance time
        if self.is_night:
            # Night -> Day transition
            self.is_night = False
            self.current_day += 1
            
            # Apply pending effects from strangers let in
            for effect in self.pending_effects:
                reward += effect
            self.pending_effects = []
            self.strangers_inside = []  # They leave after effects
        else:
            # Day -> Night transition
            self.is_night = True
        
        # Update total reward
        self.total_reward += reward
        
        # Check death
        if self.total_reward <= self.death_threshold:
            reward -= 1000
            done = True
            info['death'] = True
        
        # Check win
        if self.current_day > self.total_days:
            done = True
            info['survived'] = True
        
        # Spawn new stranger
        self._maybe_spawn_stranger()
        
        self.episode_reward += reward
        
        self.current_state = self.encode_state(
            day=self.current_day,
            is_night=self.is_night,
            total_reward=self.total_reward,
            strangers_inside=self.strangers_inside,
            stranger_at_door=self.stranger_at_door
        )
        
        return StepResult(
            next_state=self.current_state,
            reward=reward,
            done=done,
            info=info
        )
    
    def get_valid_actions(self, state_id: Optional[str] = None) -> List[int]:
        """Get valid actions for state."""
        state = state_id if state_id else self.current_state
        if state is None:
            return list(range(self.n_actions))
        
        data = self.decode_state(state)
        valid = [self.LET_IN, self.REJECT]
        
        if data['stranger_at_door'] != StrangerType.NONE:
            valid.append(self.SHOOT_DOOR)
        
        if data['strangers_inside']:
            valid.append(self.SHOOT_INSIDE)
        
        return valid
    
    def get_state_space(self) -> List[str]:
        """Get all possible states."""
        return list(self._state_space)
    
    def get_transition_prob(self, state: str, action: int) -> List[Tuple[str, float, float]]:
        """Get transition probabilities (stochastic due to stranger spawning)."""
        old_state = self.current_state
        old_day = self.current_day
        old_night = self.is_night
        old_reward = self.total_reward
        old_inside = self.strangers_inside.copy()
        old_door = self.stranger_at_door
        old_pending = self.pending_effects.copy()
        
        data = self.decode_state(state)
        self.current_day = data['day']
        self.is_night = data['is_night']
        self.total_reward = data['total_reward']
        self.strangers_inside = data['strangers_inside']
        self.stranger_at_door = data['stranger_at_door']
        self.current_state = state
        
        result = self.step(action)
        
        self.current_day = old_day
        self.is_night = old_night
        self.total_reward = old_reward
        self.strangers_inside = old_inside
        self.stranger_at_door = old_door
        self.pending_effects = old_pending
        self.current_state = old_state
        
        return [(result.next_state, 1.0, result.reward)]
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        
        type_names = {
            StrangerType.NONE: 'none',
            StrangerType.HELPFUL: 'helpful',
            StrangerType.TRAITOR: 'traitor',
            StrangerType.NEUTRAL: 'neutral'
        }
        
        entities = []
        
        # House in the center
        entities.append({
            'type': 'house',
            'position': {'x': 2, 'y': 2},
            'properties': {
                'is_night': data['is_night'],
                'day': data['day']
            }
        })
        
        # Door with stranger
        if data['stranger_at_door'] != StrangerType.NONE:
            entities.append({
                'type': 'stranger',
                'position': {'x': 1, 'y': 2},
                'properties': {
                    'silhouette': True,  # Can't see type
                    'actual_type': type_names[data['stranger_at_door']]  # For debugging
                }
            })
        
        # Strangers inside
        for i, s in enumerate(data['strangers_inside']):
            entities.append({
                'type': 'stranger_inside',
                'position': {'x': i + 1, 'y': 0},
                'properties': {
                    'shadowy': True,
                    'actual_type': type_names[s]
                }
            })
        
        return RenderData(
            environment='friend_or_foe',
            state_id=state,
            entities=entities,
            grid=None,
            metadata={
                'day': data['day'],
                'total_days': self.total_days,
                'is_night': data['is_night'],
                'phase': 'night' if data['is_night'] else 'day',
                'total_reward': data['total_reward'],
                'death_threshold': self.death_threshold,
                'strangers_inside_count': len(data['strangers_inside']),
                'stranger_at_door': data['stranger_at_door'] != StrangerType.NONE,
                'step': self.step_count,
                'danger_level': max(0, (self.death_threshold - data['total_reward']) / abs(self.death_threshold))
            }
        )

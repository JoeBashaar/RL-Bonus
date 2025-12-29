"""
Football Game Environment

Goal: First team to score 2 goals wins.

MDP Modeling:
- States: (ball_x, ball_y, player1_pos, player2_pos, score1, score2, possession)
- Actions: {move_up, move_down, move_left, move_right, shoot, pass}
- Rewards: +100 for scoring, -100 for opponent scoring, +500 for winning, -500 for losing
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
from .base_env import DiscreteEnvironment, StateInfo, StepResult, RenderData


class FootballEnvironment(DiscreteEnvironment):
    """
    Football game environment where two players compete to score goals.
    First team to score 2 goals wins.
    """
    
    # Actions
    MOVE_UP = 0
    MOVE_DOWN = 1
    MOVE_LEFT = 2
    MOVE_RIGHT = 3
    SHOOT = 4
    PASS = 5
    
    def __init__(self, **kwargs):
        super().__init__("football", **kwargs)
        
        # Grid dimensions
        self.grid_width = kwargs.get('grid_width', 10)
        self.grid_height = kwargs.get('grid_height', 6)
        
        # Win condition
        self.goals_to_win = kwargs.get('goals_to_win', 2)
        
        # Goal positions (left and right edges)
        self.goal_y_min = self.grid_height // 2 - 1
        self.goal_y_max = self.grid_height // 2 + 1
        
        # Action setup
        self.n_actions = 6
        self.action_names = ['move_up', 'move_down', 'move_left', 'move_right', 'shoot', 'pass']
        
        # Default discount factor
        self.gamma = 0.99
        
        # State components
        self.ball_x: int = 0
        self.ball_y: int = 0
        self.player1_x: int = 0
        self.player1_y: int = 0
        self.player2_x: int = 0
        self.player2_y: int = 0
        self.score1: int = 0
        self.score2: int = 0
        self.possession: int = 0  # 0 = player1, 1 = player2, 2 = neutral
        
        # Shooting success probability (decreases with distance)
        self.base_shot_success = 0.8
        
        # Build state space
        self._build_state_space()
        
    def _build_state_space(self) -> None:
        """Build the complete state space with simplified encoding."""
        # Use simplified state: ball position + scores + possession
        # Player positions are abstracted away
        for ball_x in range(self.grid_width):
            for ball_y in range(self.grid_height):
                for score1 in range(self.goals_to_win + 1):
                    for score2 in range(self.goals_to_win + 1):
                        for poss in range(3):
                            state_id = self._encode_simplified(
                                ball_x=ball_x, ball_y=ball_y,
                                score1=score1, score2=score2,
                                possession=poss
                            )
                            self._state_space.add(state_id)
                            
                            # Terminal states
                            if score1 >= self.goals_to_win or score2 >= self.goals_to_win:
                                self._terminal_states.add(state_id)
    
    def _encode_simplified(self, **kwargs) -> str:
        """Encode into simplified state (ball + scores + possession)."""
        return (f"{kwargs['ball_x']},{kwargs['ball_y']},"
                f"{kwargs['score1']},{kwargs['score2']},"
                f"{kwargs['possession']}")
    
    def encode_state(self, **kwargs) -> str:
        """Encode state components into simplified state ID.
        
        Uses ball position + scores + possession to keep state space manageable.
        Player positions are abstracted away.
        """
        return self._encode_simplified(
            ball_x=kwargs['ball_x'],
            ball_y=kwargs['ball_y'],
            score1=kwargs['score1'],
            score2=kwargs['score2'],
            possession=kwargs['possession']
        )
    
    def decode_state(self, state_id: str) -> Dict[str, Any]:
        """Decode simplified state ID into components."""
        parts = state_id.split(',')
        ball_x = int(parts[0])
        ball_y = int(parts[1])
        return {
            'ball_x': ball_x,
            'ball_y': ball_y,
            'player1_x': ball_x,  # Approximate - player near ball
            'player1_y': ball_y,
            'player2_x': self.grid_width - 1 - ball_x,  # Opposite side
            'player2_y': self.grid_height - 1 - ball_y,
            'score1': int(parts[2]),
            'score2': int(parts[3]),
            'possession': int(parts[4])
        }
    
    def reset(self, **kwargs) -> Tuple[str, StateInfo]:
        """Reset environment to initial state."""
        self.step_count = 0
        self.episode_reward = 0.0
        
        # Center ball
        self.ball_x = self.grid_width // 2
        self.ball_y = self.grid_height // 2
        
        # Player positions
        self.player1_x = self.grid_width // 4
        self.player1_y = self.grid_height // 2
        self.player2_x = 3 * self.grid_width // 4
        self.player2_y = self.grid_height // 2
        
        # Reset scores
        self.score1 = 0
        self.score2 = 0
        
        # Random possession
        self.possession = self.rng.choice([0, 1])
        
        self.current_state = self.encode_state(
            ball_x=self.ball_x, ball_y=self.ball_y,
            player1_x=self.player1_x, player1_y=self.player1_y,
            player2_x=self.player2_x, player2_y=self.player2_y,
            score1=self.score1, score2=self.score2,
            possession=self.possession
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
        
        state = self.decode_state(self.current_state)
        
        # Update state based on action
        if action == self.MOVE_UP:
            if self.possession == 0:
                self.player1_y = max(0, self.player1_y - 1)
                self.ball_y = self.player1_y
            reward -= 0
        elif action == self.MOVE_DOWN:
            if self.possession == 0:
                self.player1_y = min(self.grid_height - 1, self.player1_y + 1)
                self.ball_y = self.player1_y
        elif action == self.MOVE_LEFT:
            if self.possession == 0:
                self.player1_x = max(0, self.player1_x - 1)
                self.ball_x = self.player1_x
        elif action == self.MOVE_RIGHT:
            if self.possession == 0:
                self.player1_x = min(self.grid_width - 1, self.player1_x + 1)
                self.ball_x = self.player1_x
        elif action == self.SHOOT:
            if self.possession == 0:
                # Calculate shot success based on distance to goal
                distance_to_goal = self.grid_width - 1 - self.ball_x
                success_prob = self.base_shot_success * (1 - distance_to_goal / self.grid_width)
                
                if self.goal_y_min <= self.ball_y <= self.goal_y_max:
                    if self.rng.random() < success_prob:
                        self.score1 += 1
                        reward += 100
                        info['event'] = 'goal_scored'
                        # Reset positions after goal
                        self._reset_after_goal()
                    else:
                        reward -= 50
                        info['event'] = 'shot_missed'
                        self.possession = 2  # Neutral
                else:
                    reward -= 50
                    info['event'] = 'shot_missed'
        elif action == self.PASS:
            # Simplified pass - just changes possession
            if self.possession == 0:
                self.possession = 2  # Ball becomes neutral
        
        # Opponent AI (simple)
        self._opponent_move()
        
        # Check win condition
        if self.score1 >= self.goals_to_win:
            reward += 500
            done = True
            info['winner'] = 'player1'
        elif self.score2 >= self.goals_to_win:
            reward -= 500
            done = True
            info['winner'] = 'player2'
        
        # Timeout check
        if self.step_count >= self.max_steps:
            done = True
            info['timeout'] = True
        
        self.episode_reward += reward
        
        self.current_state = self.encode_state(
            ball_x=self.ball_x, ball_y=self.ball_y,
            player1_x=self.player1_x, player1_y=self.player1_y,
            player2_x=self.player2_x, player2_y=self.player2_y,
            score1=self.score1, score2=self.score2,
            possession=self.possession
        )
        
        return StepResult(
            next_state=self.current_state,
            reward=reward,
            done=done,
            info=info
        )
    
    def _opponent_move(self) -> None:
        """Simple opponent AI."""
        if self.possession == 1:
            # Move towards goal
            if self.player2_x > 0:
                self.player2_x -= 1
                self.ball_x = self.player2_x
            # Try to score
            if self.player2_x == 0 and self.goal_y_min <= self.ball_y <= self.goal_y_max:
                if self.rng.random() < 0.3:
                    self.score2 += 1
                    self._reset_after_goal()
        elif self.possession == 2:
            # Try to get ball
            if abs(self.player2_x - self.ball_x) + abs(self.player2_y - self.ball_y) <= 1:
                self.possession = 1
        else:
            # Move towards ball
            if self.player2_x > self.ball_x:
                self.player2_x -= 1
            elif self.player2_x < self.ball_x:
                self.player2_x += 1
            if self.player2_y > self.ball_y:
                self.player2_y -= 1
            elif self.player2_y < self.ball_y:
                self.player2_y += 1
    
    def _reset_after_goal(self) -> None:
        """Reset positions after a goal."""
        self.ball_x = self.grid_width // 2
        self.ball_y = self.grid_height // 2
        self.player1_x = self.grid_width // 4
        self.player1_y = self.grid_height // 2
        self.player2_x = 3 * self.grid_width // 4
        self.player2_y = self.grid_height // 2
        self.possession = self.rng.choice([0, 1])
    
    def get_valid_actions(self, state_id: Optional[str] = None) -> List[int]:
        """Get valid actions for state."""
        return list(range(self.n_actions))
    
    def get_state_space(self) -> List[str]:
        """Get all possible states."""
        return list(self._state_space)
    
    def get_transition_prob(self, state: str, action: int) -> List[Tuple[str, float, float]]:
        """Get transition probabilities."""
        # Simplified - deterministic transitions with stochastic shooting
        if self.current_state != state:
            old_state = self.current_state
            state_data = self.decode_state(state)
            self.ball_x = state_data['ball_x']
            self.ball_y = state_data['ball_y']
            self.player1_x = state_data['player1_x']
            self.player1_y = state_data['player1_y']
            self.player2_x = state_data['player2_x']
            self.player2_y = state_data['player2_y']
            self.score1 = state_data['score1']
            self.score2 = state_data['score2']
            self.possession = state_data['possession']
            self.current_state = state
        
        if action == self.SHOOT and self.possession == 0:
            # Stochastic shooting
            distance_to_goal = self.grid_width - 1 - self.ball_x
            success_prob = self.base_shot_success * (1 - distance_to_goal / self.grid_width)
            
            if self.goal_y_min <= self.ball_y <= self.goal_y_max:
                # Success transition
                new_score1 = min(self.score1 + 1, self.goals_to_win)
                success_state = self.encode_state(
                    ball_x=self.grid_width // 2, ball_y=self.grid_height // 2,
                    player1_x=self.grid_width // 4, player1_y=self.grid_height // 2,
                    player2_x=3 * self.grid_width // 4, player2_y=self.grid_height // 2,
                    score1=new_score1, score2=self.score2,
                    possession=0
                )
                # Fail transition
                fail_state = self.encode_state(
                    ball_x=self.ball_x, ball_y=self.ball_y,
                    player1_x=self.player1_x, player1_y=self.player1_y,
                    player2_x=self.player2_x, player2_y=self.player2_y,
                    score1=self.score1, score2=self.score2,
                    possession=2
                )
                return [
                    (success_state, success_prob, 100 - 1),
                    (fail_state, 1 - success_prob, -50 - 1)
                ]
        
        # Deterministic transition
        result = self.step(action)
        return [(result.next_state, 1.0, result.reward)]
    
    def get_render_data(self, state_id: Optional[str] = None) -> RenderData:
        """Get rendering data for frontend."""
        state = state_id if state_id else self.current_state
        if state is None:
            state, _ = self.reset()
        
        data = self.decode_state(state)
        
        entities = [
            {
                'type': 'ball',
                'position': {'x': data['ball_x'], 'y': data['ball_y']},
                'properties': {'possession': ['player1', 'player2', 'neutral'][data['possession']]}
            },
            {
                'type': 'player',
                'position': {'x': data['player1_x'], 'y': data['player1_y']},
                'properties': {
                    'team': 'team1',
                    'has_ball': data['possession'] == 0,
                    'jersey_number': 10
                }
            },
            {
                'type': 'player',
                'position': {'x': data['player2_x'], 'y': data['player2_y']},
                'properties': {
                    'team': 'team2',
                    'has_ball': data['possession'] == 1,
                    'jersey_number': 7
                }
            },
            {
                'type': 'goal',
                'position': {'x': 0, 'y': self.grid_height // 2},
                'properties': {'team': 'team2', 'width': 1, 'height': 3}
            },
            {
                'type': 'goal',
                'position': {'x': self.grid_width - 1, 'y': self.grid_height // 2},
                'properties': {'team': 'team1', 'width': 1, 'height': 3}
            }
        ]
        
        return RenderData(
            environment='football',
            state_id=state,
            entities=entities,
            grid={
                'width': self.grid_width,
                'height': self.grid_height,
                'cell_types': [['grass'] * self.grid_width for _ in range(self.grid_height)]
            },
            metadata={
                'score': {'team1': data['score1'], 'team2': data['score2']},
                'step': self.step_count,
                'goals_to_win': self.goals_to_win,
                'possession': ['player1', 'player2', 'neutral'][data['possession']]
            }
        )

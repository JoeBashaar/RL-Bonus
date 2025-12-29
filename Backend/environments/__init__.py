"""
RL Learning Tool - Game Environments Package

This package contains all custom game environments for the RL learning tool.
Each environment implements the BaseEnvironment interface.
"""

from .base_env import BaseEnvironment
from .football import FootballEnvironment
from .hill_climbing import HillClimbingEnvironment
from .haunted_house import HauntedHouseEnvironment
from .spider_web import SpiderWebEnvironment
from .friend_or_foe import FriendOrFoeEnvironment
from .frozen_lake import FrozenLakeEnvironment

__all__ = [
    'BaseEnvironment',
    'FootballEnvironment',
    'HillClimbingEnvironment',
    'HauntedHouseEnvironment',
    'SpiderWebEnvironment',
    'FriendOrFoeEnvironment',
    'FrozenLakeEnvironment',
]

ENVIRONMENT_REGISTRY = {
    'football': FootballEnvironment,
    'hill_climbing': HillClimbingEnvironment,
    'haunted_house': HauntedHouseEnvironment,
    'spider_web': SpiderWebEnvironment,
    'friend_or_foe': FriendOrFoeEnvironment,
    'frozen_lake': FrozenLakeEnvironment,
}

def get_environment(name: str) -> type:
    """Get environment class by name."""
    if name not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment: {name}. Available: {list(ENVIRONMENT_REGISTRY.keys())}")
    return ENVIRONMENT_REGISTRY[name]

def list_environments() -> list:
    """List all available environments with descriptions."""
    return [
        {
            'name': 'football',
            'display_name': 'Football Game',
            'description': 'Two-player football game. First team to score 2 goals wins.',
            'state_space_size': 'Medium (~50,000 states)',
            'action_space_size': 6,
        },
        {
            'name': 'hill_climbing',
            'display_name': 'Car Hill Climbing',
            'description': 'Drive a car up a hill to reach the flag. Avoid getting stuck in local optima.',
            'state_space_size': 'Medium (~5,000 states)',
            'action_space_size': 3,
        },
        {
            'name': 'haunted_house',
            'display_name': 'Haunted House (Hide and Seek)',
            'description': 'Multi-level hide and seek. Catch the real hider, not the ghost!',
            'state_space_size': 'Large (~100,000 states)',
            'action_space_size': 7,
        },
        {
            'name': 'spider_web',
            'display_name': 'Spider Web Mosquito Hunting',
            'description': 'Catch high-value mosquitoes in your web before they escape.',
            'state_space_size': 'Large (~50,000 states)',
            'action_space_size': 5,
        },
        {
            'name': 'friend_or_foe',
            'display_name': 'Friend or Foe?',
            'description': 'Survive 20 nights by deciding whether to let strangers in.',
            'state_space_size': 'Medium (~5,000 states)',
            'action_space_size': 4,
        },
        {
            'name': 'train_tracks',
            'display_name': 'Train Track Building',
            'description': 'Build tracks to connect 5 stations without falling into pits.',
            'state_space_size': 'Large (~100,000 states)',
            'action_space_size': 'Variable (based on grid size)',
        },
    ]

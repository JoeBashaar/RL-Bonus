"""
RL Learning Tool - Streamlit Frontend

Interactive web interface for training and visualizing RL algorithms.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time
from typing import Dict, Any, Optional
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'Backend'))

from Backend.environments import get_environment, list_environments, ENVIRONMENT_REGISTRY
from Backend.algorithms import get_algorithm, list_algorithms, get_algorithm_parameters, ALGORITHM_REGISTRY

# Page config
st.set_page_config(
    page_title="RL Learning Tool",
    page_icon="RL",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stApp {
        background-color: #0f172a;
    }
    .css-1d391kg {
        background-color: #1e293b;
    }
    .stSelectbox label, .stSlider label, .stNumberInput label {
        color: #94a3b8 !important;
    }
    .metric-card {
        background-color: #1e293b;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #334155;
    }
    .success-badge {
        background-color: #065f46;
        color: #10b981;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'training_result' not in st.session_state:
    st.session_state.training_result = None
if 'training_history' not in st.session_state:
    st.session_state.training_history = []
if 'env_instance' not in st.session_state:
    st.session_state.env_instance = None
if 'current_state' not in st.session_state:
    st.session_state.current_state = None


def get_env_info(env_name: str) -> Dict[str, Any]:
    """Get environment information."""
    env_class = ENVIRONMENT_REGISTRY.get(env_name)
    if env_class:
        return {
            'name': env_name,
            'display_name': getattr(env_class, 'display_name', env_name.replace('_', ' ').title()),
            'description': getattr(env_class, 'description', 'No description available'),
            'category': getattr(env_class, 'category', 'general'),
        }
    return {'name': env_name, 'display_name': env_name}


def get_algo_info(algo_name: str) -> Dict[str, Any]:
    """Get algorithm information."""
    algo_class = ALGORITHM_REGISTRY.get(algo_name)
    if algo_class:
        return {
            'name': algo_name,
            'display_name': getattr(algo_class, 'display_name', algo_name.replace('_', ' ').title()),
            'description': getattr(algo_class, 'description', 'No description available'),
            'category': getattr(algo_class, 'category', 'general'),
        }
    return {'name': algo_name, 'display_name': algo_name}


# ============== ALGORITHM DEFINITIONS ==============
ALGORITHM_DEFINITIONS = {
    'policy_iteration': {
        'name': 'Policy Iteration',
        'category': 'Dynamic Programming',
        'model_based': True,
        'definition': '''
**Policy Iteration** is a dynamic programming algorithm that finds the optimal policy by alternating between:

1. **Policy Evaluation**: Calculate the value function V(s) for the current policy
2. **Policy Improvement**: Update the policy to be greedy with respect to V(s)

This process repeats until the policy stabilizes (no changes).
''',
        'equation': r'$V^\pi(s) = \sum_a \pi(a|s) \sum_{s\'} P(s\'|s,a)[R + \gamma V^\pi(s\')]$',
        'pros': ['Guaranteed convergence', 'Finds exact optimal policy', 'Works for all states'],
        'cons': ['Requires complete environment model', 'Expensive for large state spaces', 'Memory intensive'],
        'best_for': 'Small to medium discrete state spaces where you have the full transition model'
    },
    'value_iteration': {
        'name': 'Value Iteration',
        'category': 'Dynamic Programming',
        'model_based': True,
        'definition': '''
**Value Iteration** combines policy evaluation and improvement into a single update:

For each state, update: $V(s) \\leftarrow \\max_a \\sum_{s\'} P(s\'|s,a)[R + \\gamma V(s\')]$

The optimal policy is derived from the final value function.
''',
        'equation': r'$V_{k+1}(s) = \max_a \sum_{s\'} P(s\'|s,a)[R + \gamma V_k(s\')]$',
        'pros': ['Faster per iteration than Policy Iteration', 'Simpler implementation', 'Guaranteed convergence'],
        'cons': ['Requires complete environment model', 'May need more iterations than PI'],
        'best_for': 'Same as Policy Iteration - complete model required'
    },
    'monte_carlo': {
        'name': 'Monte Carlo Methods',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**Monte Carlo** methods learn from complete episodes of experience:

1. Generate episodes using current policy
2. For each state visited, calculate the return (sum of discounted rewards)
3. Update value estimates as average of observed returns

**Variants:**
- **First-Visit MC**: Only count first occurrence of state in episode
- **Every-Visit MC**: Count all occurrences
- **ε-Greedy MC**: Use exploration during learning
''',
        'equation': r'$V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$',
        'pros': ['No model needed', 'Works with black-box environments', 'Unbiased estimates'],
        'cons': ['High variance', 'Only works for episodic tasks', 'Slow convergence'],
        'best_for': 'Episodic tasks where you can simulate full episodes'
    },
    'mc_first_visit': {
        'name': 'First-Visit Monte Carlo',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**First-Visit Monte Carlo** only uses the first occurrence of each state in an episode:

For each episode:
1. Record the first time each state is visited
2. Calculate return from that point
3. Update value as running average

This gives unbiased estimates of V(s).
''',
        'equation': r'$V(s) \leftarrow \text{average}(G_t | S_t = s, \text{first visit})$',
        'pros': ['Unbiased estimates', 'Simpler to analyze theoretically', 'Guaranteed convergence'],
        'cons': ['May be slower than every-visit for some problems', 'High variance'],
        'best_for': 'When theoretical guarantees and unbiased estimates matter'
    },
    'mc_every_visit': {
        'name': 'Every-Visit Monte Carlo',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**Every-Visit Monte Carlo** uses all occurrences of each state in an episode:

For each episode:
1. Record every time each state is visited
2. Calculate return from each occurrence
3. Update value using all samples

May converge faster in practice due to more samples per episode.
''',
        'equation': r'$V(s) \leftarrow \text{average}(G_t | S_t = s, \text{all visits})$',
        'pros': ['More samples per episode', 'Often faster convergence in practice', 'Works well with function approximation'],
        'cons': ['Biased estimates (asymptotically unbiased)', 'Correlated samples'],
        'best_for': 'When faster convergence is more important than theoretical purity'
    },
    'mc_epsilon_greedy': {
        'name': 'ε-Greedy Monte Carlo',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**ε-Greedy Monte Carlo** adds exploration to MC methods:

- With probability ε: take random action (explore)
- With probability 1-ε: take best known action (exploit)

ε typically decays over time from high (0.3-1.0) to low (0.01-0.1).
''',
        'equation': r'$\pi(a|s) = \begin{cases} 1-\varepsilon+\frac{\varepsilon}{|A|} & \text{if } a = \arg\max Q(s,a) \\ \frac{\varepsilon}{|A|} & \text{otherwise} \end{cases}$',
        'pros': ['Balances exploration and exploitation', 'Guaranteed to visit all states eventually'],
        'cons': ['Never fully greedy', 'Exploration may hurt final performance'],
        'best_for': 'When exploration is crucial for discovering good policies'
    },
    'td_zero': {
        'name': 'TD(0) - Temporal Difference',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**TD(0)** learns from incomplete episodes using bootstrapping:

After each step: $V(s) \\leftarrow V(s) + \\alpha[R + \\gamma V(s\') - V(s)]$

The term $R + \\gamma V(s\') - V(s)$ is called the **TD error**.

TD combines ideas from MC (learning from experience) and DP (bootstrapping).
''',
        'equation': r'$V(S_t) \leftarrow V(S_t) + \alpha[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD target}} - V(S_t)]$',
        'pros': ['Online learning (no need to wait for episode end)', 'Lower variance than MC', 'Works for continuing tasks'],
        'cons': ['Biased estimates', 'Sensitive to learning rate'],
        'best_for': 'Continuing (non-episodic) tasks, online learning'
    },
    'td_nstep': {
        'name': 'N-Step TD',
        'category': 'Model-Free',
        'model_based': False,
        'definition': '''
**N-Step TD** bridges TD(0) and Monte Carlo:

- **n=1**: Same as TD(0) - one-step lookahead
- **n=∞**: Same as Monte Carlo - full episode return
- **n=4-8**: Often a good middle ground

The n-step return: $G_t^{(n)} = R_{t+1} + \\gamma R_{t+2} + ... + \\gamma^{n-1} R_{t+n} + \\gamma^n V(S_{t+n})$
''',
        'equation': r'$G_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k R_{t+k+1} + \gamma^n V(S_{t+n})$',
        'pros': ['Flexible bias-variance tradeoff', 'Often faster learning than TD(0)'],
        'cons': ['Requires tuning n', 'Must balance n and α'],
        'best_for': 'When TD(0) is too myopic and MC is too slow'
    },
    'sarsa': {
        'name': 'SARSA (On-Policy TD Control)',
        'category': 'Model-Free Control',
        'model_based': False,
        'definition': '''
**SARSA** (State-Action-Reward-State-Action) learns Q-values on-policy:

At each step: $Q(s,a) \\leftarrow Q(s,a) + \\alpha[R + \\gamma Q(s\',a\') - Q(s,a)]$

It updates using the action actually taken (a'), making it on-policy.

The name comes from the tuple (S, A, R, S', A') used in each update.
''',
        'equation': r'$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$',
        'pros': ['On-policy (learns about policy being followed)', 'Safer in dangerous environments'],
        'cons': ['Cannot learn optimal policy while exploring', 'May be suboptimal if exploring'],
        'best_for': 'When safety matters (e.g., cliff walking) - learns a safer path'
    },
    'q_learning': {
        'name': 'Q-Learning (Off-Policy TD Control)',
        'category': 'Model-Free Control',
        'model_based': False,
        'definition': '''
**Q-Learning** learns the optimal Q-values directly (off-policy):

$Q(s,a) \\leftarrow Q(s,a) + \\alpha[R + \\gamma \\max_{a\'} Q(s\',a\') - Q(s,a)]$

It uses the **maximum** Q-value for the next state, regardless of what action was actually taken.

This is **off-policy**: it learns optimal policy while following an exploratory policy.
''',
        'equation': r'$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_a Q(S_{t+1}, a) - Q(S_t, A_t)]$',
        'pros': ['Off-policy (can learn optimal policy while exploring)', 'Converges to optimal Q*'],
        'cons': ['Can overestimate values', 'May learn risky policies'],
        'best_for': 'When you want the optimal policy and can afford some risk during learning'
    }
}

# Parameter interactions and explanations
PARAMETER_INTERACTIONS = {
    'alpha_gamma': {
        'title': 'Learning Rate (α) × Discount Factor (γ)',
        'description': '''
**How they interact:**
- High γ (0.99) means future rewards matter a lot → need more updates to propagate values
- With high γ, you may need lower α for stability

| γ | Recommended α | Why |
|---|---------------|-----|
| 0.99 | 0.05-0.1 | Long-horizon needs careful updates |
| 0.95 | 0.1-0.2 | Balanced setting |
| 0.9 | 0.1-0.3 | Short-horizon, faster learning OK |
'''
    },
    'alpha_n': {
        'title': 'Learning Rate (α) × N-Steps (n)',
        'description': '''
**The Critical Trade-off:**

As **n increases**, the n-step return has:
- ✅ Lower bias (more accurate target)
- ❌ Higher variance (more randomness in return)

To compensate for higher variance, **decrease α**:

| n | Recommended α | Product α×n |
|---|---------------|-------------|
| 1 | 0.1 - 0.3 | 0.1 - 0.3 |
| 2 | 0.08 - 0.2 | 0.16 - 0.4 |
| 4 | 0.05 - 0.15 | 0.2 - 0.6 |
| 8 | 0.02 - 0.08 | 0.16 - 0.64 |
| 16 | 0.01 - 0.05 | 0.16 - 0.8 |

**Rule of thumb:** Keep α × n between **0.2 and 0.8**

**Why?** Large n-step returns can swing wildly. If α is too high, these swings cause instability.
'''
    },
    'epsilon_episodes': {
        'title': 'Exploration Rate (ε) × Training Episodes',
        'description': '''
**Exploration-Exploitation Balance:**

- **Early training**: High ε (0.3-1.0) to discover the state space
- **Late training**: Low ε (0.01-0.1) to exploit learned knowledge

**Decay strategies:**
1. **Linear decay**: ε decreases linearly over episodes
2. **Exponential decay**: ε = ε₀ × decay^episode (common default)
3. **Step decay**: Drop ε at specific episodes

| Episodes | Initial ε | Final ε | Decay Rate |
|----------|-----------|---------|------------|
| 500 | 0.3 | 0.01 | 0.99 |
| 1000 | 0.5 | 0.01 | 0.995 |
| 5000 | 1.0 | 0.01 | 0.999 |
'''
    },
    'gamma_environment': {
        'title': 'Discount Factor (γ) × Environment Type',
        'description': '''
**Choosing γ based on environment:**

| Environment Type | Recommended γ | Reason |
|-----------------|---------------|--------|
| Short episodes (< 20 steps) | 0.9 - 0.95 | Less discounting needed |
| Medium episodes (20-100) | 0.95 - 0.99 | Standard setting |
| Long episodes (100+) | 0.99 - 0.999 | Need long-term planning |
| Continuing (infinite) | 0.9 - 0.99 | Must discount to keep values bounded |

**Signs γ is wrong:**
- Too low: Agent is myopic, ignores distant rewards
- Too high: Slow convergence, values explode in continuing tasks
'''
    }
}


# Parameter hints and convergence tips
PARAMETER_TIPS = {
    'alpha': {
        'description': 'Learning rate - how much new info overrides old',
        'tips': [
            '⬆️ High α (0.3-0.5): Fast learning, may oscillate',
            '⬇️ Low α (0.01-0.1): Stable but slow convergence',
            '💡 If using high n-step, lower α to 0.05-0.1',
            '💡 If not converging, try α=0.1 as baseline',
        ]
    },
    'n': {
        'description': 'N-step lookahead - balances bias vs variance',
        'tips': [
            '⬆️ High n (8-16): More accurate but higher variance',
            '⬇️ Low n (1-4): Faster updates, more biased',
            '💡 Trade-off: ↑n requires ↓α for stability',
            '💡 Try n=4 with α=0.1 as a starting point',
        ]
    },
    'gamma': {
        'description': 'Discount factor - importance of future rewards',
        'tips': [
            '⬆️ High γ (0.99): Long-term planning',
            '⬇️ Low γ (0.8-0.9): Focus on immediate rewards',
            '💡 For episodic tasks, use γ=0.99',
            '💡 For continuous tasks, γ=0.95 works well',
        ]
    },
    'epsilon': {
        'description': 'Exploration rate - random action probability',
        'tips': [
            '⬆️ High ε (0.3+): More exploration, slower learning',
            '⬇️ Low ε (0.05-0.1): Exploitation focused',
            '💡 Start high (0.3) and decay to 0.01',
        ]
    },
    'n_episodes': {
        'description': 'Training episodes - more = better convergence',
        'tips': [
            '💡 Simple envs: 500-1000 episodes',
            '💡 Complex envs: 2000-5000 episodes',
            '💡 If not converging, increase episodes first',
        ]
    },
    'max_iterations': {
        'description': 'Max iterations for DP algorithms',
        'tips': [
            '💡 Usually 100-500 is enough',
            '💡 Converges when delta < threshold',
        ]
    }
}

CONVERGENCE_RECOMMENDATIONS = {
    'not_converging': [
        '1️⃣ Increase n_episodes (try 2x-3x current value)',
        '2️⃣ Adjust α: if oscillating ↓, if too slow ↑',
        '3️⃣ Check γ is not too low for the environment',
        '4️⃣ For TD(n): balance n and α (↑n needs ↓α)',
    ],
    'alpha_n_tradeoff': [
        '📊 **Alpha-N Trade-off Guide:**',
        '• n=1 (TD-0): α can be 0.1-0.3',
        '• n=4: α should be 0.05-0.15',
        '• n=8: α should be 0.02-0.08',
        '• n=16+: α should be 0.01-0.05',
        '',
        '**Rule of thumb:** α × n ≈ 0.4 to 0.8',
    ]
}

# Sprite definitions for different entity types
SPRITES = {
    # Football
    'ball': {'symbol': '⚽', 'color': '#ffffff', 'size': 30},
    'player': {'symbol': '🧍', 'color': '#3b82f6', 'size': 35},
    'goal': {'symbol': '🥅', 'color': '#f59e0b', 'size': 40},
    # Hill Climbing
    'car': {'symbol': '🚗', 'color': '#ef4444', 'size': 35},
    'climber': {'symbol': '🧗', 'color': '#10b981', 'size': 35},
    'peak': {'symbol': '⛰️', 'color': '#6366f1', 'size': 40},
    'flag': {'symbol': '🚩', 'color': '#10b981', 'size': 35},
    'camp': {'symbol': '⛺', 'color': '#f97316', 'size': 30},
    'danger_zone': {'symbol': '⚠️', 'color': '#f59e0b', 'size': 25},
    # Haunted House
    'seeker': {'symbol': '🔦', 'color': '#fbbf24', 'size': 35},
    'hider': {'symbol': '👻', 'color': '#a855f7', 'size': 35},
    'ghost': {'symbol': '👹', 'color': '#ef4444', 'size': 35},
    # Spider Web
    'spider': {'symbol': '🕷️', 'color': '#1f2937', 'size': 35},
    'mosquito': {'symbol': '🦟', 'color': '#84cc16', 'size': 25},
    'web': {'symbol': '🕸️', 'color': '#d1d5db', 'size': 20},
    # Friend or Foe
    'house': {'symbol': '🏠', 'color': '#6366f1', 'size': 45},
    'stranger': {'symbol': '🧑', 'color': '#f97316', 'size': 35},
    'friend': {'symbol': '😊', 'color': '#10b981', 'size': 35},
    'foe': {'symbol': '😈', 'color': '#ef4444', 'size': 35},
    # Frozen Lake
    'skater': {'symbol': '⛸️', 'color': '#3b82f6', 'size': 35},
    'hole': {'symbol': '🕳️', 'color': '#1e293b', 'size': 30},
    'ice': {'symbol': '❄️', 'color': '#67e8f9', 'size': 25},
    # Generic
    'agent': {'symbol': '🤖', 'color': '#3b82f6', 'size': 35},
    'target': {'symbol': '🎯', 'color': '#10b981', 'size': 35},
    'obstacle': {'symbol': '🧱', 'color': '#ef4444', 'size': 30},
    'reward': {'symbol': '💰', 'color': '#fbbf24', 'size': 30},
}


def render_hill_climbing(render_data) -> go.Figure:
    """Render Hill Climbing with terrain visualization."""
    metadata = render_data.metadata or {}
    terrain = metadata.get('terrain', [])
    
    fig = go.Figure()
    
    # Draw the hill terrain
    if terrain:
        x_vals = [p['x'] for p in terrain]
        y_vals = [p['y'] for p in terrain]
        
        # Fill area under curve (the hill)
        fig.add_trace(go.Scatter(
            x=x_vals + [x_vals[-1], x_vals[0]],
            y=y_vals + [0, 0],
            fill='toself',
            fillcolor='rgba(34, 197, 94, 0.4)',  # Green hill
            line=dict(color='#22c55e', width=3),
            name='Hill',
            hoverinfo='skip'
        ))
        
        # Add snow on top (peak area)
        peak_x = [p['x'] for p in terrain if p['y'] > max(y_vals) * 0.8]
        peak_y = [p['y'] for p in terrain if p['y'] > max(y_vals) * 0.8]
        if peak_x:
            fig.add_trace(go.Scatter(
                x=peak_x,
                y=peak_y,
                mode='lines',
                line=dict(color='#ffffff', width=5),
                name='Snow Peak',
                hoverinfo='skip'
            ))
    
    # Add entities (car, flag, danger zone)
    for entity in render_data.entities:
        entity_type = entity.get('type', 'agent')
        pos = entity.get('position', {})
        props = entity.get('properties', {})
        
        sprite = SPRITES.get(entity_type, SPRITES.get('agent'))
        
        if entity_type == 'car':
            # Draw car on the hill
            fig.add_trace(go.Scatter(
                x=[pos.get('x', 0)],
                y=[pos.get('y', 0) + 0.02],
                mode='text+markers',
                text=['🚗'],
                textfont=dict(size=40),
                marker=dict(size=10, color='#ef4444', opacity=0),
                name='Car',
                hovertemplate=f"Car<br>Position: {pos.get('x', 0):.2f}<br>Velocity: {props.get('velocity', 0):.3f}<extra></extra>"
            ))
        elif entity_type == 'flag':
            # Goal flag at the peak
            fig.add_trace(go.Scatter(
                x=[pos.get('x', 0)],
                y=[pos.get('y', 0) + 0.03],
                mode='text+markers',
                text=['🚩'],
                textfont=dict(size=35),
                marker=dict(size=10, color='#10b981', opacity=0),
                name='Goal',
                hovertemplate="Goal 🏁<extra></extra>"
            ))
        elif entity_type == 'danger_zone':
            # Local optimum warning zone
            if props.get('active'):
                fig.add_vrect(
                    x0=pos.get('x', 0) - props.get('width', 0.1) / 2,
                    x1=pos.get('x', 0) + props.get('width', 0.1) / 2,
                    fillcolor="rgba(239, 68, 68, 0.2)",
                    line_width=0,
                    annotation_text="⚠️ Stuck!",
                    annotation_position="top"
                )
    
    # Layout
    min_pos = metadata.get('min_position', -1.2)
    max_pos = metadata.get('max_position', 0.6)
    
    fig.update_layout(
        title="🏔️ Hill Climbing - Mountain Car",
        xaxis=dict(
            range=[min_pos - 0.1, max_pos + 0.1],
            title='Position',
            showgrid=True,
            gridcolor='#374151',
        ),
        yaxis=dict(
            range=[-0.05, 0.15],
            title='Height',
            showgrid=True,
            gridcolor='#374151',
        ),
        plot_bgcolor='#0f172a',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def render_frozen_lake(render_data) -> go.Figure:
    """Render Frozen Lake with ice grid."""
    grid = render_data.grid or {}
    width = grid.get('width', 8)
    height = grid.get('height', 8)
    cell_types = grid.get('cell_types', [])
    
    fig = go.Figure()
    
    # Create color-coded grid
    colors_map = {'ice': 0, 'water': 1, 'goal': 2}
    grid_z = np.zeros((height, width))
    
    for y, row in enumerate(cell_types):
        for x, cell in enumerate(row):
            grid_z[y, x] = colors_map.get(cell, 0)
    
    # Ice grid heatmap
    fig.add_trace(go.Heatmap(
        z=grid_z,
        colorscale=[
            [0, '#a5f3fc'],      # Ice - light cyan
            [0.5, '#1e3a5f'],    # Water/Hole - dark blue
            [1, '#22c55e']       # Goal - green
        ],
        showscale=False,
        hoverongaps=False,
    ))
    
    # Add entities
    for entity in render_data.entities:
        entity_type = entity.get('type', 'agent')
        pos = entity.get('position', {})
        
        sprite = SPRITES.get(entity_type, SPRITES.get('agent'))
        
        if entity_type == 'skater':
            fig.add_trace(go.Scatter(
                x=[pos.get('x', 0)],
                y=[pos.get('y', 0)],
                mode='text',
                text=['⛸️'],
                textfont=dict(size=35),
                name='Skater',
                hovertemplate=f"Skater<br>Pos: ({pos.get('x', 0)}, {pos.get('y', 0)})<extra></extra>"
            ))
        elif entity_type == 'hole':
            fig.add_trace(go.Scatter(
                x=[pos.get('x', 0)],
                y=[pos.get('y', 0)],
                mode='text',
                text=['💧'],
                textfont=dict(size=25),
                name='Hole',
                hoverinfo='skip'
            ))
        elif entity_type == 'flag':
            fig.add_trace(go.Scatter(
                x=[pos.get('x', 0)],
                y=[pos.get('y', 0)],
                mode='text',
                text=['🚩'],
                textfont=dict(size=30),
                name='Goal',
                hoverinfo='skip'
            ))
    
    fig.update_layout(
        title="❄️ Frozen Lake",
        xaxis=dict(
            range=[-0.5, width - 0.5],
            showgrid=True,
            gridcolor='#374151',
            dtick=1,
        ),
        yaxis=dict(
            range=[-0.5, height - 0.5],
            showgrid=True,
            gridcolor='#374151',
            dtick=1,
            scaleanchor='x'
        ),
        plot_bgcolor='#1e3a5f',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig


def render_friend_or_foe(render_data) -> go.Figure:
    """Render Friend or Foe with house scene."""
    metadata = render_data.metadata or {}
    is_night = metadata.get('is_night', False)
    day = metadata.get('day', 1)
    
    fig = go.Figure()
    
    # Background - day or night
    bg_color = '#0f172a' if is_night else '#87CEEB'
    
    # Draw sky
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=10, y1=6,
        fillcolor=bg_color,
        line_width=0
    )
    
    # Draw ground
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=10, y1=1,
        fillcolor='#22c55e' if not is_night else '#1a2e1a',
        line_width=0
    )
    
    # Draw house
    # House body
    fig.add_shape(
        type="rect", x0=3, y0=1, x1=7, y1=4,
        fillcolor='#8b4513',
        line=dict(color='#5c2e0a', width=2)
    )
    # Roof
    fig.add_shape(
        type="path",
        path="M 2.5 4 L 5 6 L 7.5 4 Z",
        fillcolor='#dc2626',
        line=dict(color='#991b1b', width=2)
    )
    # Door
    fig.add_shape(
        type="rect", x0=4.5, y0=1, x1=5.5, y1=2.5,
        fillcolor='#92400e',
        line=dict(color='#78350f', width=1)
    )
    # Windows
    window_color = '#fef08a' if is_night else '#7dd3fc'
    fig.add_shape(
        type="rect", x0=3.5, y0=2.5, x1=4.3, y1=3.3,
        fillcolor=window_color,
        line=dict(color='#0369a1', width=1)
    )
    fig.add_shape(
        type="rect", x0=5.7, y0=2.5, x1=6.5, y1=3.3,
        fillcolor=window_color,
        line=dict(color='#0369a1', width=1)
    )
    
    # Add moon or sun
    if is_night:
        fig.add_trace(go.Scatter(
            x=[8.5], y=[5],
            mode='text',
            text=['🌙'],
            textfont=dict(size=40),
            hoverinfo='skip'
        ))
        # Stars
        for _ in range(5):
            fig.add_trace(go.Scatter(
                x=[np.random.uniform(0.5, 9.5)],
                y=[np.random.uniform(4, 5.8)],
                mode='text',
                text=['⭐'],
                textfont=dict(size=15),
                hoverinfo='skip'
            ))
    else:
        fig.add_trace(go.Scatter(
            x=[8.5], y=[5],
            mode='text',
            text=['☀️'],
            textfont=dict(size=45),
            hoverinfo='skip'
        ))
    
    # Draw entities (stranger at door)
    for entity in render_data.entities:
        entity_type = entity.get('type', 'agent')
        props = entity.get('properties', {})
        
        if entity_type == 'stranger':
            # Stranger at door - show as silhouette
            fig.add_trace(go.Scatter(
                x=[4.2], y=[1.8],
                mode='text',
                text=['🧑‍🦱'],
                textfont=dict(size=35),
                name='Stranger',
                hovertemplate="Stranger at door<br>(Type unknown)<extra></extra>"
            ))
    
    fig.update_layout(
        title=f"🏠 Friend or Foe - {'Night' if is_night else 'Day'} {day}",
        xaxis=dict(range=[0, 10], showgrid=False, showticklabels=False, zeroline=False),
        yaxis=dict(range=[0, 6], showgrid=False, showticklabels=False, zeroline=False),
        plot_bgcolor=bg_color,
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    return fig

def render_environment_visual(env, state_id: str = None) -> go.Figure:
    """Render the environment as a visual Plotly figure with sprites."""
    try:
        render_data = env.get_render_data(state_id)
    except Exception as e:
        return None
    
    # Special rendering for hill_climbing - draw the terrain
    if render_data.environment == 'hill_climbing':
        return render_hill_climbing(render_data)
    
    # Special rendering for frozen_lake - draw ice grid
    if render_data.environment == 'frozen_lake':
        return render_frozen_lake(render_data)
    
    # Special rendering for friend_or_foe - show house scene
    if render_data.environment == 'friend_or_foe':
        return render_friend_or_foe(render_data)
    
    # Get grid dimensions
    grid = render_data.grid or {}
    width = grid.get('width', 10)
    height = grid.get('height', 10)
    
    fig = go.Figure()
    
    # Draw background grid
    cell_types = grid.get('cell_types', [])
    colors = {
        'grass': '#22c55e',
        'floor': '#64748b',
        'wall': '#1e293b',
        'water': '#3b82f6',
        'lava': '#ef4444',
        'ice': '#67e8f9',
        'sand': '#fbbf24',
        'path': '#a1a1aa',
        'web': '#e2e8f0',
    }
    
    # Create grid background
    grid_z = np.zeros((height, width))
    for y, row in enumerate(cell_types):
        for x, cell in enumerate(row):
            grid_z[y, x] = list(colors.keys()).index(cell) if cell in colors else 0
    
    fig.add_trace(go.Heatmap(
        z=grid_z,
        colorscale=[[i/(len(colors)-1), c] for i, c in enumerate(colors.values())],
        showscale=False,
        hoverongaps=False,
        opacity=0.3
    ))
    
    # Draw entities as scatter points with text symbols
    for entity in render_data.entities:
        entity_type = entity.get('type', 'agent')
        pos = entity.get('position', {})
        props = entity.get('properties', {})
        
        # Get sprite info
        sprite = SPRITES.get(entity_type, SPRITES.get('agent'))
        
        # Adjust color based on properties
        color = sprite['color']
        if props.get('team') == 'team2':
            color = '#ef4444'
        elif props.get('team') == 'team1':
            color = '#3b82f6'
        if props.get('is_terminal'):
            color = '#10b981'
        
        fig.add_trace(go.Scatter(
            x=[pos.get('x', 0)],
            y=[pos.get('y', 0)],
            mode='text+markers',
            text=[sprite['symbol']],
            textfont=dict(size=sprite['size']),
            marker=dict(size=sprite['size'] + 10, color=color, opacity=0.3),
            name=entity_type,
            hovertemplate=f"{entity_type}<br>Pos: ({pos.get('x', 0)}, {pos.get('y', 0)})<extra></extra>"
        ))
    
    # Layout
    fig.update_layout(
        title=f"🎮 {render_data.environment.replace('_', ' ').title()}",
        xaxis=dict(
            range=[-0.5, width - 0.5],
            showgrid=True,
            gridcolor='#374151',
            dtick=1,
            zeroline=False
        ),
        yaxis=dict(
            range=[-0.5, height - 0.5],
            showgrid=True,
            gridcolor='#374151',
            dtick=1,
            zeroline=False,
            scaleanchor='x'
        ),
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        showlegend=False,
        height=400,
        margin=dict(l=20, r=20, t=50, b=20)
    )
    
    # Add metadata as annotation
    metadata = render_data.metadata or {}
    info_text = ' | '.join([f"{k}: {v}" for k, v in metadata.items() if k not in ['step']])
    if info_text:
        fig.add_annotation(
            x=0.5, y=1.05,
            xref='paper', yref='paper',
            text=info_text[:80],
            showarrow=False,
            font=dict(size=12, color='#94a3b8')
        )
    
    return fig


def train_algorithm(env_name: str, algo_name: str, params: Dict[str, Any], progress_bar, status_text):
    """Train the algorithm and update progress."""
    # Create environment
    env_class = get_environment(env_name)
    env = env_class()
    
    # Create algorithm
    algo_class = get_algorithm(algo_name)
    
    # Handle special cases for algorithm parameters
    algo_params = params.copy()
    if algo_name == 'mc_every_visit':
        algo_params['first_visit'] = False  # Every-visit MC
    elif algo_name == 'mc_first_visit':
        algo_params['first_visit'] = True   # First-visit MC (explicit)
    
    algo = algo_class(**algo_params)
    
    history = []
    
    def on_update(metrics):
        history.append(metrics.copy())
        iteration = metrics.get('iteration', metrics.get('episode', len(history)))
        max_iter = params.get('max_iterations', params.get('n_episodes', 100))
        progress = min(iteration / max_iter, 1.0)
        progress_bar.progress(progress)
        
        if 'max_value_delta' in metrics:
            status_text.text(f"Iteration {iteration}: Δ = {metrics['max_value_delta']:.6f}")
        elif 'episode_reward' in metrics or 'total_reward' in metrics:
            reward = metrics.get('episode_reward', metrics.get('total_reward', 0))
            status_text.text(f"Episode {iteration}: Reward = {reward:.2f}")
    
    # Train - use appropriate callback name based on algorithm type
    # Policy/Value Iteration use on_iteration, others use on_episode
    if algo_name in ['policy_iteration', 'value_iteration']:
        result = algo.train(env, on_iteration=on_update)
    else:
        result = algo.train(env, on_episode=on_update)
    
    return result, history, env


def plot_value_function_grid(values: Dict[str, float], title: str = "Value Function"):
    """Plot value function as a heatmap grid."""
    if not values:
        return None
    
    # Parse states and aggregate by grid position
    grid_values = {}
    for state_id, value in values.items():
        match = state_id.split(',') if ',' in state_id else [state_id]
        try:
            if len(match) >= 2:
                x, y = int(match[0]), int(match[1])
            else:
                # Single value state
                x, y = int(match[0]) % 10, int(match[0]) // 10
            key = (x, y)
            if key not in grid_values:
                grid_values[key] = []
            grid_values[key].append(value)
        except (ValueError, IndexError):
            continue
    
    if not grid_values:
        return None
    
    # Get grid dimensions
    max_x = max(k[0] for k in grid_values.keys()) + 1
    max_y = max(k[1] for k in grid_values.keys()) + 1
    
    # Create grid
    grid = np.full((max_y, max_x), np.nan)
    for (x, y), vals in grid_values.items():
        grid[y, x] = np.max(vals)  # Take max value for cell
    
    fig = px.imshow(
        grid,
        color_continuous_scale='RdYlGn',
        labels={'color': 'Value'},
        title=title,
        aspect='auto'
    )
    fig.update_layout(
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        xaxis_title='X',
        yaxis_title='Y'
    )
    return fig


def plot_policy_grid(policy: Dict[str, int], title: str = "Policy"):
    """Plot policy as arrows on a grid."""
    if not policy:
        return None
    
    # Parse states and aggregate by grid position
    grid_actions = {}
    for state_id, action in policy.items():
        match = state_id.split(',') if ',' in state_id else [state_id]
        try:
            if len(match) >= 2:
                x, y = int(match[0]), int(match[1])
            else:
                x, y = int(match[0]) % 10, int(match[0]) // 10
            key = (x, y)
            if key not in grid_actions:
                grid_actions[key] = {}
            if action not in grid_actions[key]:
                grid_actions[key][action] = 0
            grid_actions[key][action] += 1
        except (ValueError, IndexError):
            continue
    
    if not grid_actions:
        return None
    
    # Get best action per cell
    best_actions = {}
    for key, action_counts in grid_actions.items():
        best_actions[key] = max(action_counts.items(), key=lambda x: x[1])[0]
    
    max_x = max(k[0] for k in best_actions.keys()) + 1
    max_y = max(k[1] for k in best_actions.keys()) + 1
    
    # Create grid
    grid = np.full((max_y, max_x), np.nan)
    for (x, y), action in best_actions.items():
        grid[y, x] = action
    
    # Arrow directions
    arrows = {0: '←', 1: '→', 2: '↑', 3: '↓', 4: '↖', 5: '↗', 6: '↙', 7: '↘'}
    
    fig = go.Figure()
    
    # Add heatmap
    fig.add_trace(go.Heatmap(
        z=grid,
        colorscale='Viridis',
        showscale=False,
        hovertemplate='Action: %{z}<extra></extra>'
    ))
    
    # Add arrow annotations
    for (x, y), action in best_actions.items():
        fig.add_annotation(
            x=x, y=y,
            text=arrows.get(action, str(action)),
            showarrow=False,
            font=dict(size=16, color='white'),
        )
    
    fig.update_layout(
        title=title,
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        xaxis_title='X',
        yaxis_title='Y'
    )
    return fig


def plot_convergence(history: list):
    """Plot convergence metrics."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    # Check what data we have
    has_delta = 'max_value_delta' in df.columns and df['max_value_delta'].notna().any()
    has_policy = 'policy_changes' in df.columns and df['policy_changes'].notna().any()
    
    # Handle episode-based algorithms (no policy changes)
    if 'episode_reward' in df.columns and not has_delta:
        return None  # Will be handled by episode rewards plot
    
    if has_delta and has_policy:
        fig = make_subplots(rows=1, cols=2, subplot_titles=['Value Delta', 'Policy Changes'])
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['max_value_delta'], 
                mode='lines+markers', 
                name='Max Delta', 
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['policy_changes'], 
                mode='lines+markers', 
                name='Policy Changes', 
                line=dict(color='#a855f7', width=2),
                marker=dict(size=6)
            ),
            row=1, col=2
        )
        
        # Update axes
        fig.update_xaxes(title_text="Iteration", row=1, col=1, gridcolor='#374151')
        fig.update_xaxes(title_text="Iteration", row=1, col=2, gridcolor='#374151')
        fig.update_yaxes(title_text="Max Δ", row=1, col=1, gridcolor='#374151')
        fig.update_yaxes(title_text="# Changes", row=1, col=2, gridcolor='#374151')
        
    elif has_delta:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=list(range(1, len(df) + 1)),
                y=df['max_value_delta'], 
                mode='lines+markers', 
                name='Max Delta', 
                line=dict(color='#3b82f6', width=2),
                marker=dict(size=6)
            )
        )
        fig.update_xaxes(title_text="Iteration", gridcolor='#374151')
        fig.update_yaxes(title_text="Max Value Delta", gridcolor='#374151')
    else:
        return None
    
    fig.update_layout(
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        showlegend=False,
        height=350,
        margin=dict(l=60, r=20, t=40, b=40)
    )
    return fig


def plot_episode_rewards(history: list):
    """Plot episode rewards."""
    if not history:
        return None
    
    df = pd.DataFrame(history)
    
    if 'episode_reward' not in df.columns:
        return None
    
    # Calculate moving average
    window = min(100, len(df) // 10) if len(df) > 10 else 1
    df['avg_reward'] = df['episode_reward'].rolling(window=window, min_periods=1).mean()
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=df['episode_reward'],
        mode='lines',
        name='Episode Reward',
        line=dict(color='#3b82f6', width=1),
        opacity=0.5
    ))
    fig.add_trace(go.Scatter(
        y=df['avg_reward'],
        mode='lines',
        name=f'Moving Avg ({window})',
        line=dict(color='#f59e0b', width=2)
    ))
    
    fig.update_layout(
        title='Episode Rewards',
        plot_bgcolor='#1e293b',
        paper_bgcolor='#1e293b',
        font_color='#e2e8f0',
        xaxis_title='Episode',
        yaxis_title='Reward',
        height=300
    )
    return fig


# ============== MAIN APP ==============

st.title("🎮 RL Learning Tool")
st.markdown("Interactive Reinforcement Learning visualization and training")

# Create tabs for main content and learning resources
main_tab, learn_tab, params_tab = st.tabs([" Training", " Algorithm Guide", " Parameter Guide"])

with learn_tab:
    st.header("📚 Algorithm Definitions")
    st.markdown("Learn about each RL algorithm, when to use it, and how it works.")
    
    # Algorithm selector dropdown
    learn_algo = st.selectbox(
        "Select an algorithm to learn about:",
        list(ALGORITHM_DEFINITIONS.keys()),
        format_func=lambda x: ALGORITHM_DEFINITIONS[x]['name'],
        key="learn_algo_select"
    )
    
    if learn_algo in ALGORITHM_DEFINITIONS:
        algo_def = ALGORITHM_DEFINITIONS[learn_algo]
        
        # Header with category badge
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader(algo_def['name'])
        with col2:
            badge_color = "#3b82f6" if algo_def['model_based'] else "#10b981"
            st.markdown(f"<span style='background-color: {badge_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem;'>{'Model-Based' if algo_def['model_based'] else 'Model-Free'}</span>", unsafe_allow_html=True)
        
        st.caption(f"Category: {algo_def['category']}")
        
        # Definition
        st.markdown(algo_def['definition'])
        
        # Equation
        if 'equation' in algo_def:
            st.markdown("**Key Equation:**")
            st.latex(algo_def['equation'].replace('$', ''))
        
        # Pros and Cons
        col_pro, col_con = st.columns(2)
        with col_pro:
            st.markdown("**✅ Advantages:**")
            for pro in algo_def.get('pros', []):
                st.markdown(f"- {pro}")
        with col_con:
            st.markdown("**❌ Disadvantages:**")
            for con in algo_def.get('cons', []):
                st.markdown(f"- {con}")
        
        # Best for
        st.info(f"**Best for:** {algo_def.get('best_for', 'General use')}")
    
    st.divider()
    
    # Quick comparison table
    st.subheader("Quick Comparison")
    
    comparison_data = []
    for algo_key, algo_def in ALGORITHM_DEFINITIONS.items():
        comparison_data.append({
            'Algorithm': algo_def['name'],
            'Category': algo_def['category'],
            'Model Required': '✅' if algo_def['model_based'] else '❌',
            'Best For': algo_def.get('best_for', '')[:50] + '...'
        })
    
    st.dataframe(comparison_data, width='stretch')

with params_tab:
    st.header("⚙️ Parameter Interactions Guide")
    st.markdown("Understanding how parameters affect each other is crucial for tuning RL algorithms.")
    
    # Parameter interaction selector
    interaction = st.selectbox(
        "Select a parameter interaction to explore:",
        list(PARAMETER_INTERACTIONS.keys()),
        format_func=lambda x: PARAMETER_INTERACTIONS[x]['title'],
        key="param_interaction_select"
    )
    
    if interaction in PARAMETER_INTERACTIONS:
        inter = PARAMETER_INTERACTIONS[interaction]
        st.subheader(inter['title'])
        st.markdown(inter['description'])
    
    st.divider()
    
    # Interactive parameter calculator
    st.subheader("Parameter Calculator")
    
    calc_type = st.radio("Calculate for:", ["α × n Trade-off", "ε Decay Schedule"], horizontal=True)
    
    if calc_type == "α × n Trade-off":
        st.markdown("Find the right learning rate for your n-step setting:")
        
        col1, col2 = st.columns(2)
        with col1:
            n_val = st.slider("N-steps (n)", 1, 20, 4, key="calc_n")
        with col2:
            target_product = st.slider("Target α×n", 0.2, 0.8, 0.5, 0.05, key="calc_product")
        
        recommended_alpha = target_product / n_val
        
        st.success(f"""
        **Recommended Settings:**
        - n = {n_val}
        - α ≈ {recommended_alpha:.3f}
        - α × n = {n_val * recommended_alpha:.2f}
        """)
        
        if recommended_alpha < 0.01:
            st.warning(" α is very low - learning will be slow. Consider using smaller n.")
        elif recommended_alpha > 0.3:
            st.warning("α is high - may cause instability. Monitor convergence carefully.")
    
    else:  # ε Decay Schedule
        st.markdown("Plan your exploration schedule:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            eps_start = st.slider("Initial ε", 0.1, 1.0, 0.5, 0.05, key="eps_start")
        with col2:
            eps_end = st.slider("Final ε", 0.001, 0.2, 0.01, 0.005, key="eps_end")
        with col3:
            n_episodes = st.number_input("Episodes", 100, 10000, 1000, key="eps_episodes")
        
        # Calculate decay rate
        import math
        decay_rate = math.exp(math.log(eps_end / eps_start) / n_episodes)
        
        st.success(f"""
        **Decay Schedule:**
        - Start: ε = {eps_start}
        - End: ε ≈ {eps_end}
        - Decay rate: {decay_rate:.6f}
        - Half-life: ~{int(-math.log(2) / math.log(decay_rate))} episodes
        """)
        
        # Plot decay curve
        episodes = list(range(0, n_episodes, max(1, n_episodes // 100)))
        eps_values = [eps_start * (decay_rate ** e) for e in episodes]
        
        decay_fig = go.Figure()
        decay_fig.add_trace(go.Scatter(x=episodes, y=eps_values, mode='lines', name='ε'))
        decay_fig.update_layout(
            title='Exploration Decay Curve',
            xaxis_title='Episode',
            yaxis_title='ε (Exploration Rate)',
            plot_bgcolor='#1e293b',
            paper_bgcolor='#1e293b',
            font_color='#e2e8f0',
            height=300
        )
        st.plotly_chart(decay_fig, width='stretch')

with main_tab:
    # Sidebar - Configuration
    with st.sidebar:
        st.header(" Configuration")
        
        # Environment Selection
        st.subheader(" Environment")
        env_names = list(ENVIRONMENT_REGISTRY.keys())
        selected_env = st.selectbox(
            "Select Environment",
            env_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        env_info = get_env_info(selected_env)
        st.caption(env_info.get('description', ''))
        
        st.divider()
        
        # Algorithm Selection
        st.subheader("Algorithm")
        algo_names = list(ALGORITHM_REGISTRY.keys())
        selected_algo = st.selectbox(
            "Select Algorithm",
            algo_names,
            format_func=lambda x: x.replace('_', ' ').title()
        )
        
        algo_info = get_algo_info(selected_algo)
        st.caption(algo_info.get('description', ''))
        
        st.divider()
        
        # Algorithm Parameters
        st.subheader("Parameters")
        
        algo_params = get_algorithm_parameters(selected_algo)
        params = {}
        
        for param_name, param_info in algo_params.items():
            default = param_info.get('default', 0)
            min_val = param_info.get('min', 0)
            max_val = param_info.get('max', 1000)
            param_type = param_info.get('type', 'float')
            
            # Get tips for this parameter
            param_tips = PARAMETER_TIPS.get(param_name, {})
            help_text = param_tips.get('description', '')
            
            if param_type == 'int':
                params[param_name] = st.number_input(
                    param_name.replace('_', ' ').title(),
                    min_value=int(min_val),
                    max_value=int(max_val),
                    value=int(default),
                    step=1,
                    help=help_text
                )
            elif param_type == 'bool':
                params[param_name] = st.checkbox(
                    param_name.replace('_', ' ').title(),
                    value=bool(default),
                    help=help_text
                )
            else:
                params[param_name] = st.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=float(min_val),
                    max_value=float(max_val),
                    value=float(default),
                    step=0.01 if max_val <= 1 else 0.1,
                    help=help_text
                )
            
            # Show tips expander for key parameters
            if param_name in ['alpha', 'n'] and param_tips.get('tips'):
                with st.expander(f" {param_name.title()} Tips", expanded=False):
                    for tip in param_tips['tips']:
                        st.caption(tip)
        
        # Show alpha-n trade-off if both parameters exist
        if 'alpha' in params and 'n' in params:
            with st.expander(" Alpha-N Trade-off Guide", expanded=False):
                for line in CONVERGENCE_RECOMMENDATIONS['alpha_n_tradeoff']:
                    st.markdown(line) if line.startswith('•') or line.startswith('**') else st.caption(line)
        
        st.divider()
        
        # Train Button
        train_button = st.button(" Start Training", type="primary", width='stretch')

    # Main content area
    col1, col2 = st.columns([2, 1])

    with col1:
        st.header("Training Results")
        
        if train_button:
            with st.spinner("Training in progress..."):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                try:
                    result, history, env = train_algorithm(
                        selected_env, selected_algo, params, progress_bar, status_text
                    )
                    
                    st.session_state.training_result = result
                    st.session_state.training_history = history
                    st.session_state.env_instance = env
                    
                    progress_bar.progress(1.0)
                    status_text.text(" Training completed!")
                    st.success("Training completed successfully!")
                    
                except Exception as e:
                    st.error(f"Training failed: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())
        
        # Display results if available
        if st.session_state.training_result is not None:
            result = st.session_state.training_result
            history = st.session_state.training_history
            
            # Metrics row
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                n_states = len(result.policy) if hasattr(result, 'policy') else 0
                st.metric("States", n_states)
            
            with metric_cols[1]:
                n_iters = len(history)
                st.metric("Iterations", n_iters)
            
            with metric_cols[2]:
                if history and 'max_value_delta' in history[-1]:
                    final_delta = history[-1]['max_value_delta']
                    st.metric("Final Delta", f"{final_delta:.2e}")
                elif history and 'episode_reward' in history[-1]:
                    final_reward = history[-1]['episode_reward']
                    st.metric("Final Reward", f"{final_reward:.2f}")
            
            with metric_cols[3]:
                converged = any(h.get('converged', False) for h in history)
                st.metric("Converged", "✅ Yes" if converged else "❌ No")
            
            # Show convergence help if not converged
            if not converged:
                with st.expander(" Not Converging? Tips to Help", expanded=True):
                    for rec in CONVERGENCE_RECOMMENDATIONS['not_converging']:
                        st.markdown(rec)
                    st.info(" Non-convergence is common for exploration-based algorithms. Check if the policy reaches the goal in the animation!")
            
            st.divider()
            
            # Charts
            tab1, tab2, tab3, tab4 = st.tabs([" Convergence", " Value Function", " Policy", "Episodes"])
            
            with tab1:
                conv_fig = plot_convergence(history)
                if conv_fig:
                    st.plotly_chart(conv_fig, width='stretch')
                else:
                    st.info("No convergence data available")
            
            with tab2:
                if hasattr(result, 'value_function') and result.value_function:
                    value_fig = plot_value_function_grid(result.value_function)
                    if value_fig:
                        st.plotly_chart(value_fig, width='stretch')
                    else:
                        st.info("Could not visualize value function grid")
                elif hasattr(result, 'q_function') and result.q_function:
                    # Derive V from Q
                    values = {s: max(a.values()) for s, a in result.q_function.items() if a}
                    value_fig = plot_value_function_grid(values)
                    if value_fig:
                        st.plotly_chart(value_fig, width='stretch')
                    else:
                        st.info("Could not visualize value function grid")
                else:
                    st.info("No value function data available")
            
            with tab3:
                if hasattr(result, 'policy') and result.policy:
                    policy_fig = plot_policy_grid(result.policy)
                    if policy_fig:
                        st.plotly_chart(policy_fig, width='stretch')
                    else:
                        st.info("Could not visualize policy grid")
                else:
                    st.info("No policy data available")
            
            with tab4:
                episodes_fig = plot_episode_rewards(history)
                if episodes_fig:
                    st.plotly_chart(episodes_fig, width='stretch')
                else:
                    st.info("No episode data available (this is normal for value/policy iteration)")

    with col2:
        st.header("🎮 Environment Info")
        
        if selected_env:
            env_info = get_env_info(selected_env)
            
            st.markdown(f"**{env_info.get('display_name', selected_env)}**")
            st.caption(env_info.get('description', 'No description'))
            
            # Show environment preview
            try:
                env_class = get_environment(selected_env)
                env = env_class()
                
                st.markdown("**State Space:**")
                if hasattr(env, 'n_states'):
                    st.code(f"States: {env.n_states}")
                
                st.markdown("**Action Space:**")
                if hasattr(env, 'action_names'):
                    for i, name in enumerate(env.action_names):
                        st.text(f"  {i}: {name}")
                elif hasattr(env, 'n_actions'):
                    st.code(f"Actions: {env.n_actions}")
                    
            except Exception as e:
                st.warning(f"Could not load environment preview: {e}")
        
        st.divider()
        
        # Policy playback with visual animation
        if st.session_state.training_result is not None and hasattr(st.session_state.training_result, 'policy'):
            st.subheader("▶️ Visual Policy Simulation")
            
            # Animation controls
            animation_speed = st.slider("⏱️ Speed (ms)", 100, 1000, 400, 50)
            max_steps = st.number_input("Max Steps", 10, 200, 50)
            
            col_a, col_b = st.columns(2)
            with col_a:
                run_animation = st.button("🎬 Run Simulation", width='stretch')
            with col_b:
                step_once = st.button("👆 Step Once", width='stretch')
            
            # Reset button
            if st.button("🔄 Reset Simulation"):
                st.session_state.current_state = None
                st.session_state.env_instance = None
                st.session_state.step_count = 0
                st.session_state.total_reward_sim = 0
                st.rerun()
            
            # Visual display areas
            visual_display = st.empty()
            info_display = st.empty()
            trajectory_display = st.empty()
            
            # Initialize simulation state if needed
            if 'step_count' not in st.session_state:
                st.session_state.step_count = 0
            if 'total_reward_sim' not in st.session_state:
                st.session_state.total_reward_sim = 0
            
            policy = st.session_state.training_result.policy
            action_names_list = None
            
            def find_nearest_state(policy, state):
                """Find the nearest matching state in policy using component-wise matching."""
                if state in policy:
                    return state, 0  # Exact match
                
                # Parse current state components
                try:
                    current_parts = [int(p) if p.lstrip('-').isdigit() else p for p in state.split(',')]
                except:
                    return None, -1
                
                best_match = None
                best_score = -1
                
                for policy_state in policy.keys():
                    try:
                        policy_parts = [int(p) if p.lstrip('-').isdigit() else p for p in policy_state.split(',')]
                    except:
                        continue
                    
                    if len(policy_parts) != len(current_parts):
                        continue
                    
                    # Calculate similarity score (higher = more similar)
                    score = 0
                    for i, (curr, pol) in enumerate(zip(current_parts, policy_parts)):
                        if curr == pol:
                            score += 10  # Exact match bonus
                        elif isinstance(curr, int) and isinstance(pol, int):
                            # Penalize based on distance
                            score -= abs(curr - pol) * 0.1
                    
                    if score > best_score:
                        best_score = score
                        best_match = policy_state
                
                return best_match, best_score
            
            # Get action from policy (handle dict format) with fallback to nearest state
            def get_action(policy, state, use_nearest=True):
                """Get action from policy, with optional nearest-state fallback."""
                if state in policy:
                    action_data = policy[state]
                    if isinstance(action_data, dict):
                        return action_data.get('action', 0), state, True
                    return action_data, state, True
                
                if use_nearest:
                    nearest, score = find_nearest_state(policy, state)
                    if nearest is not None and score > 0:
                        action_data = policy[nearest]
                        if isinstance(action_data, dict):
                            return action_data.get('action', 0), nearest, False
                        return action_data, nearest, False
                
                return None, None, False
            
            # Show current state if we have one (for step-by-step persistence)
            if st.session_state.env_instance is not None and st.session_state.current_state is not None:
                env = st.session_state.env_instance
                state = st.session_state.current_state
                action_names_list = getattr(env, 'action_names', [f'A{i}' for i in range(getattr(env, 'n_actions', 4))])
                
                # Show current state visually
                env_fig = render_environment_visual(env, state)
                if env_fig:
                    env_fig.add_annotation(
                        x=0.98, y=0.02,
                        xref='paper', yref='paper',
                        text=f"Step: {st.session_state.step_count} | Reward: {st.session_state.total_reward_sim:+.1f}",
                        showarrow=False,
                        font=dict(size=12, color='#10b981'),
                        bgcolor='#1e293b',
                        xanchor='right'
                    )
                    visual_display.plotly_chart(env_fig, width='stretch', key="current_state_display")
                
                # Show next action hint
                action, matched_state, is_exact = get_action(policy, state)
                if action is not None:
                    action_name = action_names_list[action] if action < len(action_names_list) else f"A{action}"
                    if is_exact:
                        info_display.info(f"🎯 Next action: **{action_name}** - Click 'Step Once' to execute")
                    else:
                        info_display.warning(f"🎯 Next action: **{action_name}** (using nearest state match)")
                else:
                    # Debug: show state comparison
                    policy_states = list(policy.keys())[:5]
                    info_display.error(f"⚠️ State `{state}` - No similar state found in policy!")
                    with st.expander("🔍 Debug Info"):
                        st.write(f"**Current state:** `{state}`")
                        st.write(f"**Policy has {len(policy)} states**")
                        st.write(f"**Sample policy states:** {policy_states}")
            
            if step_once:
                try:
                    # Initialize environment if first step
                    if st.session_state.env_instance is None or st.session_state.current_state is None:
                        env_class = get_environment(selected_env)
                        env = env_class()
                        state, _ = env.reset()
                        st.session_state.env_instance = env
                        st.session_state.current_state = state
                        st.session_state.step_count = 0
                        st.session_state.total_reward_sim = 0
                        st.rerun()
                    else:
                        env = st.session_state.env_instance
                        state = st.session_state.current_state
                    
                    action_names_list = getattr(env, 'action_names', [f'A{i}' for i in range(getattr(env, 'n_actions', 4))])
                    
                    action, matched_state, is_exact = get_action(policy, state)
                    if action is not None:
                        action_name = action_names_list[action] if action < len(action_names_list) else f"A{action}"
                        
                        # Execute action
                        step_result = env.step(action)
                        next_state = step_result.next_state
                        reward = step_result.reward
                        done = step_result.done
                        
                        # Update session state
                        st.session_state.step_count += 1
                        st.session_state.total_reward_sim += reward
                        
                        if done:
                            info_display.success(f"🎉 Episode Complete! Total reward: {st.session_state.total_reward_sim:+.2f} in {st.session_state.step_count} steps")
                            # Reset for next episode
                            new_state, _ = env.reset()
                            st.session_state.current_state = new_state
                            st.session_state.step_count = 0
                            st.session_state.total_reward_sim = 0
                        else:
                            st.session_state.current_state = next_state
                        
                        st.session_state.env_instance = env
                        st.rerun()
                    else:
                        info_display.error("No matching state found! Environment state space is too large for tabular methods.")
                        st.info("💡 **Tip:** Try a simpler environment like Football or Hill Climbing, or use more training episodes.")
                        
                except Exception as e:
                    info_display.error(f"Step failed: {e}")
            
            if run_animation:
                try:
                    # Create fresh environment for full animation
                    env_class = get_environment(selected_env)
                    env = env_class()
                    state, _ = env.reset()
                    
                    action_names = getattr(env, 'action_names', [f'A{i}' for i in range(getattr(env, 'n_actions', 4))])
                    
                    trajectory = []
                    total_reward = 0
                    done = False
                    steps = 0
                    approx_count = 0  # Track how many times we used approximate matching
                    
                    progress_placeholder = st.empty()
                    
                    while not done and steps < max_steps:
                        action, matched_state, is_exact = get_action(policy, state)
                        if action is None:
                            trajectory.append(f"❓ No matching state at step {steps+1}")
                            info_display.warning("No matching state found - agent got lost! Try a simpler environment.")
                            break
                        
                        if not is_exact:
                            approx_count += 1
                        
                        action_name = action_names[action] if action < len(action_names) else f"A{action}"
                        
                        # Render current state visually
                        env_fig = render_environment_visual(env, state)
                        if env_fig:
                            # Add action annotation
                            match_indicator = "≈" if not is_exact else ""
                            env_fig.add_annotation(
                                x=0.02, y=0.98,
                                xref='paper', yref='paper',
                                text=f"Step {steps+1}: {action_name} {match_indicator}",
                                showarrow=False,
                                font=dict(size=14, color='#fbbf24' if is_exact else '#f97316'),
                                bgcolor='#1e293b',
                                bordercolor='#fbbf24' if is_exact else '#f97316',
                                borderwidth=1
                            )
                            visual_display.plotly_chart(env_fig, width='stretch', key=f"anim_{steps}_{time.time()}")
                        
                        # Execute action
                        step_result = env.step(action)
                        next_state = step_result.next_state
                        reward = step_result.reward
                        done = step_result.done
                        total_reward += reward
                        
                        match_sym = "≈" if not is_exact else "✓"
                        step_info = f"**{steps+1}.** {action_name} {match_sym} → R={reward:+.1f}"
                        trajectory.append(step_info)
                        
                        # Update info
                        info_display.markdown(f"**Step {steps + 1}** | Action: **{action_name}** | Reward: {reward:+.2f} | Total: {total_reward:+.2f}")
                        
                        # Progress bar
                        progress_placeholder.progress(min((steps + 1) / max_steps, 1.0))
                        
                        time.sleep(animation_speed / 1000)
                        state = next_state
                        steps += 1
                    
                    # Final state render
                    env_fig = render_environment_visual(env, state)
                    if env_fig:
                        status_emoji = "🎉" if done else "⏱"
                        env_fig.update_layout(title=f"{status_emoji} Final State")
                        visual_display.plotly_chart(env_fig, width='stretch', key=f"final_{time.time()}")
                    
                    # Final summary
                    status = " Reached goal!" if done else "⏱ Max steps reached"
                    approx_info = f" ({approx_count} approximate matches)" if approx_count > 0 else ""
                    info_display.markdown(f"""
                    ###  Simulation Complete
                    - **Status:** {status}
                    - **Steps:** {steps}{approx_info}
                    - **Total Reward:** {total_reward:+.2f}
                    """)
                    
                    if approx_count > 0:
                        st.warning(f" Used nearest-state matching for {approx_count}/{steps} steps. For better results, use simpler environments or more training episodes.")
                    
                    # Show trajectory log
                    with trajectory_display.expander("📜 Full Trajectory", expanded=False):
                        cols = st.columns(5)
                        for i, t in enumerate(trajectory):
                            cols[i % 5].markdown(t)
                            
                except Exception as e:
                    info_display.error(f"Simulation failed: {e}")
                    import traceback
                    st.code(traceback.format_exc())
# Footer
st.divider()
st.caption("RL Learning Tool • Built with Streamlit • Python Backend")

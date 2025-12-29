"""
RL Learning Tool - FastAPI Backend Server

Main entry point for the backend server providing REST APIs for:
- Environment management
- Algorithm training
- Visualization data
- Inference/demo mode
- Parameter management
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional, Union
import uuid
import asyncio
from dataclasses import asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np

# Local imports
from environments import (
    get_environment, list_environments, ENVIRONMENT_REGISTRY
)
from algorithms import (
    get_algorithm, list_algorithms, get_algorithm_parameters, ALGORITHM_REGISTRY
)
from utils.visualization_data import VisualizationFormatter
from utils.parameter_validation import ParameterValidator


# ============================================================================
# Utility Functions
# ============================================================================

def convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to native Python types for JSON serialization."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy_types(item) for item in obj]
    return obj


# ============================================================================
# FastAPI App Setup
# ============================================================================

app = FastAPI(
    title="RL Learning Tool API",
    description="Backend API for Interactive Reinforcement Learning web tool",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# In-Memory Storage (for training jobs and environment instances)
# ============================================================================

# Training jobs storage
training_jobs: Dict[str, Dict[str, Any]] = {}
training_threads: Dict[str, threading.Thread] = {}

# Environment instances
environment_instances: Dict[str, Any] = {}

# Inference runs storage
inference_runs: Dict[str, Dict[str, Any]] = {}

# Thread pool for background training
executor = ThreadPoolExecutor(max_workers=4)

# ============================================================================
# Pydantic Models
# ============================================================================

class EnvironmentResetRequest(BaseModel):
    env_name: str
    parameters: Optional[Dict[str, Any]] = None
    seed: Optional[int] = None

class EnvironmentStepRequest(BaseModel):
    action: int

class TrainingRequest(BaseModel):
    algorithm: str
    environment: str
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    env_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)
    max_iterations: Optional[int] = None
    n_episodes: Optional[int] = None

class ParameterUpdateRequest(BaseModel):
    parameter: str
    value: Union[float, int, bool]

class InferenceRequest(BaseModel):
    job_id: str
    environment: str
    num_episodes: int = 1
    render_speed: float = 1.0
    env_parameters: Optional[Dict[str, Any]] = None

class ParameterValidationRequest(BaseModel):
    algorithm: Optional[str] = None
    environment: Optional[str] = None
    parameters: Dict[str, Any]

# ============================================================================
# Environment Management Endpoints
# ============================================================================

@app.get("/environments", tags=["Environments"])
async def get_environments():
    """List all available environments with descriptions."""
    return {
        "environments": list_environments(),
        "count": len(ENVIRONMENT_REGISTRY)
    }

@app.post("/environment/reset", tags=["Environments"])
async def reset_environment(request: EnvironmentResetRequest):
    """Reset environment with optional parameters."""
    if request.env_name not in ENVIRONMENT_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Environment '{request.env_name}' not found"
        )
    
    # Validate parameters
    if request.parameters:
        validation = ParameterValidator.validate_environment_params(
            request.env_name, request.parameters
        )
        if not validation.valid:
            raise HTTPException(status_code=400, detail=validation.errors)
        params = validation.sanitized_params
    else:
        params = ParameterValidator.get_default_params(request.env_name, 'environment')
    
    if request.seed is not None:
        params['seed'] = request.seed
    
    # Create environment instance
    env_class = get_environment(request.env_name)
    env = env_class(**params)
    
    # Store instance
    instance_id = str(uuid.uuid4())
    environment_instances[instance_id] = {
        'env': env,
        'name': request.env_name,
        'parameters': params
    }
    
    # Reset and get initial state
    state, state_info = env.reset()
    render_data = env.get_render_data()
    
    return {
        "instance_id": instance_id,
        "state": state,
        "state_info": {
            "state_id": state_info.state_id,
            "is_terminal": state_info.is_terminal,
            "valid_actions": state_info.valid_actions
        },
        "render_data": asdict(render_data),
        "config": env.get_config()
    }

@app.get("/environment/{env_name}/state", tags=["Environments"])
async def get_environment_state(
    env_name: str,
    instance_id: Optional[str] = None
):
    """Get current state and rendering data."""
    if instance_id and instance_id in environment_instances:
        instance = environment_instances[instance_id]
        env = instance['env']
    else:
        # Create temporary instance
        if env_name not in ENVIRONMENT_REGISTRY:
            raise HTTPException(status_code=404, detail=f"Environment '{env_name}' not found")
        env_class = get_environment(env_name)
        env = env_class()
        env.reset()
    
    render_data = env.get_render_data()
    state_info = env.get_state_info()
    
    return {
        "state": env.current_state,
        "state_info": {
            "state_id": state_info.state_id,
            "is_terminal": state_info.is_terminal,
            "valid_actions": state_info.valid_actions
        },
        "render_data": asdict(render_data)
    }

@app.post("/environment/{env_name}/step", tags=["Environments"])
async def step_environment(
    env_name: str,
    request: EnvironmentStepRequest,
    instance_id: Optional[str] = None
):
    """Take action in environment, return next state, reward, done."""
    if not instance_id or instance_id not in environment_instances:
        raise HTTPException(
            status_code=400,
            detail="Invalid instance_id. Please reset the environment first."
        )
    
    instance = environment_instances[instance_id]
    env = instance['env']
    
    if instance['name'] != env_name:
        raise HTTPException(
            status_code=400,
            detail=f"Instance is for '{instance['name']}', not '{env_name}'"
        )
    
    try:
        result = env.step(request.action)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
    render_data = env.get_render_data()
    
    return {
        "next_state": result.next_state,
        "reward": result.reward,
        "done": result.done,
        "truncated": result.truncated,
        "info": result.info,
        "render_data": asdict(render_data)
    }

# ============================================================================
# Algorithm Training Endpoints
# ============================================================================

@app.post("/algorithm/train", tags=["Training"])
async def start_training(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
):
    """Start training with selected algorithm and environment."""
    # Validate algorithm
    if request.algorithm not in ALGORITHM_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Algorithm '{request.algorithm}' not found. Available: {list(ALGORITHM_REGISTRY.keys())}"
        )
    
    # Validate environment
    if request.environment not in ENVIRONMENT_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Environment '{request.environment}' not found"
        )
    
    # Validate algorithm parameters
    algo_validation = ParameterValidator.validate_algorithm_params(
        request.algorithm, request.parameters or {}
    )
    if not algo_validation.valid:
        raise HTTPException(status_code=400, detail=algo_validation.errors)
    
    # Validate environment parameters
    env_validation = ParameterValidator.validate_environment_params(
        request.environment, request.env_parameters or {}
    )
    if not env_validation.valid:
        raise HTTPException(status_code=400, detail=env_validation.errors)
    
    # Apply overrides
    algo_params = algo_validation.sanitized_params
    if request.max_iterations is not None:
        algo_params['max_iterations'] = request.max_iterations
    if request.n_episodes is not None:
        algo_params['n_episodes'] = request.n_episodes
    
    # Create job
    job_id = str(uuid.uuid4())
    
    training_jobs[job_id] = {
        'job_id': job_id,
        'algorithm': request.algorithm,
        'environment': request.environment,
        'algo_params': algo_params,
        'env_params': env_validation.sanitized_params,
        'status': 'starting',
        'progress': {},
        'result': None,
        'error': None,
        'history': []
    }
    
    # Start training in background
    background_tasks.add_task(run_training, job_id)
    
    return {
        "job_id": job_id,
        "status": "starting",
        "algorithm": request.algorithm,
        "environment": request.environment,
        "parameters": algo_params
    }

def run_training(job_id: str):
    """Background task to run training."""
    job = training_jobs[job_id]
    
    try:
        # Create environment
        env_class = get_environment(job['environment'])
        env = env_class(**job['env_params'])
        
        # Create algorithm
        algo_class = get_algorithm(job['algorithm'])
        
        # Handle special cases for algorithm parameters
        algo_params = job['algo_params'].copy()
        if job['algorithm'] == 'mc_every_visit':
            algo_params['first_visit'] = False  # Every-visit MC
        elif job['algorithm'] == 'mc_first_visit':
            algo_params['first_visit'] = True   # First-visit MC (explicit)
        
        algo = algo_class(**algo_params)
        
        job['status'] = 'running'
        job['algorithm_instance'] = algo
        job['environment_instance'] = env
        
        # Training callback
        def on_update(metrics):
            job['progress'] = metrics
            job['history'].append({
                k: v for k, v in metrics.items() 
                if k not in ['value_function', 'policy', 'q_values']
            })
        
        # Run training
        result = algo.train(env, on_iteration=on_update)
        
        # Extract value function - derive from Q-function if needed
        if hasattr(result, 'value_function') and result.value_function:
            value_func = result.value_function
        elif hasattr(result, 'q_function') and result.q_function:
            # Derive V from Q: V(s) = max_a Q(s, a)
            value_func = {
                state: max(actions.values()) if actions else 0.0
                for state, actions in result.q_function.items()
            }
        else:
            value_func = {}
        
        # Store result
        job['status'] = 'completed'
        job['result'] = {
            'value_function': VisualizationFormatter.format_value_function(
                value_func,
                env
            ),
            'policy': VisualizationFormatter.format_policy(
                result.policy,
                q_values=result.q_function if hasattr(result, 'q_function') else None,
                action_names=env.action_names
            ),
            'convergence': VisualizationFormatter.format_convergence_metrics(
                result.convergence_history
            ),
            'episode_data': VisualizationFormatter.format_episode_data(
                result.episode_rewards if hasattr(result, 'episode_rewards') else [],
                result.episode_lengths if hasattr(result, 'episode_lengths') else []
            ) if hasattr(result, 'episode_rewards') else None
        }
        
    except Exception as e:
        job['status'] = 'failed'
        job['error'] = str(e)
        import traceback
        job['traceback'] = traceback.format_exc()

@app.get("/algorithm/training-status/{job_id}", tags=["Training"])
async def get_training_status(job_id: str):
    """Get real-time training progress."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    response = {
        "job_id": job_id,
        "algorithm": job['algorithm'],
        "environment": job['environment'],
        "status": job['status'],
        "progress": convert_numpy_types(job['progress']),
        "error": job['error']
    }
    
    if job['status'] == 'completed' and job['result']:
        response['summary'] = {
            'n_states': job['result']['value_function'].get('n_states', 0),
            'converged': job['result']['convergence'].get('converged', False)
        }
    
    return response

@app.post("/algorithm/stop/{job_id}", tags=["Training"])
async def stop_training(job_id: str):
    """Stop training."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if job['status'] == 'running' and 'algorithm_instance' in job:
        job['algorithm_instance'].stop()
        job['status'] = 'stopped'
    
    return {"job_id": job_id, "status": job['status']}

@app.get("/algorithm/results/{job_id}", tags=["Training"])
async def get_training_results(job_id: str):
    """Get final training results."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(
            status_code=400,
            detail=f"Training not completed. Status: {job['status']}"
        )
    
    return {
        "job_id": job_id,
        "algorithm": job['algorithm'],
        "environment": job['environment'],
        "result": convert_numpy_types(job['result']),
        "history": convert_numpy_types(job['history'][-100:])  # Last 100 entries
    }

# ============================================================================
# Visualization Data Endpoints
# ============================================================================

@app.get("/visualization/value-function/{job_id}", tags=["Visualization"])
async def get_value_function(
    job_id: str,
    sample_size: int = Query(500, ge=10, le=5000)
):
    """Get value function heatmap data."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if job['status'] not in ['running', 'completed']:
        raise HTTPException(status_code=400, detail="Training not started or failed")
    
    if 'algorithm_instance' in job:
        algo = job['algorithm_instance']
        env = job.get('environment_instance')
        v_func = algo.V if hasattr(algo, 'V') else {}
        return convert_numpy_types(VisualizationFormatter.format_value_function(v_func, env, sample_size))
    
    if job['result']:
        return convert_numpy_types(job['result']['value_function'])
    
    return {"values": {}, "min_value": 0, "max_value": 0}

@app.get("/visualization/policy/{job_id}", tags=["Visualization"])
async def get_policy(
    job_id: str,
    sample_size: int = Query(500, ge=10, le=5000)
):
    """Get policy arrows/actions per state."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if job['status'] not in ['running', 'completed']:
        raise HTTPException(status_code=400, detail="Training not started or failed")
    
    if 'algorithm_instance' in job:
        algo = job['algorithm_instance']
        env = job.get('environment_instance')
        policy = algo.policy if hasattr(algo, 'policy') else {}
        q_values = algo.Q if hasattr(algo, 'Q') else None
        action_names = env.action_names if env else None
        return convert_numpy_types(VisualizationFormatter.format_policy(policy, q_values, action_names, sample_size))
    
    if job['result']:
        return convert_numpy_types(job['result']['policy'])
    
    return {"policy": {}, "action_names": []}

@app.get("/visualization/convergence/{job_id}", tags=["Visualization"])
async def get_convergence(job_id: str):
    """Get convergence metrics over time."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    return convert_numpy_types(VisualizationFormatter.format_convergence_metrics(job['history']))

@app.get("/visualization/episode-rewards/{job_id}", tags=["Visualization"])
async def get_episode_rewards(job_id: str):
    """Get rewards per episode."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if 'algorithm_instance' in job:
        algo = job['algorithm_instance']
        rewards = algo.episode_rewards if hasattr(algo, 'episode_rewards') else []
        lengths = algo.episode_lengths if hasattr(algo, 'episode_lengths') else []
        td_errors = algo.td_errors if hasattr(algo, 'td_errors') else None
        return convert_numpy_types(VisualizationFormatter.format_episode_data(rewards, lengths, td_errors))
    
    if job['result'] and job['result'].get('episode_data'):
        return convert_numpy_types(job['result']['episode_data'])
    
    return {"episodes": [], "rewards": [], "lengths": [], "avg_rewards": []}

@app.get("/visualization/state-visitations/{job_id}", tags=["Visualization"])
async def get_state_visitations(
    job_id: str,
    sample_size: int = Query(500, ge=10, le=2000)
):
    """Get state visitation counts."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if 'algorithm_instance' in job:
        algo = job['algorithm_instance']
        visits = algo.state_visits if hasattr(algo, 'state_visits') else {}
        return convert_numpy_types(VisualizationFormatter.format_state_visitations(visits, sample_size))
    
    return {"visitations": {}, "min_visits": 0, "max_visits": 0, "total_visits": 0}

# ============================================================================
# Inference/Demo Mode Endpoints
# ============================================================================

@app.post("/inference/run", tags=["Inference"])
async def start_inference(request: InferenceRequest):
    """Run trained policy in environment."""
    if request.job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{request.job_id}' not found")
    
    job = training_jobs[request.job_id]
    
    if job['status'] != 'completed':
        raise HTTPException(status_code=400, detail="Training not completed")
    
    # Create environment
    env_class = get_environment(request.environment)
    env_params = request.env_parameters or job['env_params']
    env = env_class(**env_params)
    
    # Get trained algorithm
    if 'algorithm_instance' not in job:
        raise HTTPException(status_code=400, detail="No trained algorithm available")
    
    algo = job['algorithm_instance']
    
    # Create inference run
    run_id = str(uuid.uuid4())
    
    inference_runs[run_id] = {
        'run_id': run_id,
        'job_id': request.job_id,
        'environment': env,
        'algorithm': algo,
        'num_episodes': request.num_episodes,
        'render_speed': request.render_speed,
        'current_episode': 0,
        'current_step': 0,
        'episode_trajectory': [],
        'status': 'ready',
        'results': []
    }
    
    # Initialize first episode
    state, _ = env.reset()
    inference_runs[run_id]['current_state'] = state
    
    return {
        "run_id": run_id,
        "status": "ready",
        "initial_state": state,
        "render_data": asdict(env.get_render_data())
    }

@app.get("/inference/{run_id}/step", tags=["Inference"])
async def inference_step(run_id: str):
    """Get next step of inference run with rendering data."""
    if run_id not in inference_runs:
        raise HTTPException(status_code=404, detail=f"Inference run '{run_id}' not found")
    
    run = inference_runs[run_id]
    env = run['environment']
    algo = run['algorithm']
    
    if run['status'] == 'completed':
        return {
            "status": "completed",
            "results": run['results']
        }
    
    # Get action from policy
    state = run['current_state']
    action = algo.get_action(state)
    
    # Get Q-values if available
    q_values = None
    if hasattr(algo, 'Q') and state in algo.Q:
        q_values = dict(algo.Q[state])
    
    # Take step
    result = env.step(action)
    run['current_step'] += 1
    
    # Store step
    step_data = VisualizationFormatter.format_inference_step(
        state=state,
        action=action,
        reward=result.reward,
        render_data=env.get_render_data(),
        action_names=env.action_names,
        q_values=q_values
    )
    run['episode_trajectory'].append(step_data)
    
    # Check if episode done
    if result.done:
        run['results'].append({
            'episode': run['current_episode'],
            'total_reward': sum(s['reward'] for s in run['episode_trajectory']),
            'length': len(run['episode_trajectory']),
            'trajectory': run['episode_trajectory']
        })
        
        run['current_episode'] += 1
        
        if run['current_episode'] >= run['num_episodes']:
            run['status'] = 'completed'
        else:
            # Reset for next episode
            state, _ = env.reset()
            run['current_state'] = state
            run['episode_trajectory'] = []
            run['current_step'] = 0
    else:
        run['current_state'] = result.next_state
    
    return {
        "status": run['status'],
        "step": step_data,
        "episode": run['current_episode'],
        "step_in_episode": run['current_step'],
        "done": result.done
    }

# ============================================================================
# Parameter Management Endpoints
# ============================================================================

@app.get("/parameters/{algorithm}", tags=["Parameters"])
async def get_algorithm_parameters_endpoint(algorithm: str):
    """Get adjustable parameters for algorithm."""
    if algorithm not in ALGORITHM_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm}' not found")
    
    return {
        "algorithm": algorithm,
        "parameters": ParameterValidator.get_param_info(algorithm)
    }

@app.post("/parameters/validate", tags=["Parameters"])
async def validate_parameters(request: ParameterValidationRequest):
    """Validate parameter configuration."""
    if request.algorithm:
        result = ParameterValidator.validate_algorithm_params(
            request.algorithm, request.parameters
        )
    elif request.environment:
        result = ParameterValidator.validate_environment_params(
            request.environment, request.parameters
        )
    else:
        raise HTTPException(
            status_code=400,
            detail="Must specify either 'algorithm' or 'environment'"
        )
    
    return {
        "valid": result.valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "sanitized_params": result.sanitized_params
    }

@app.get("/parameters/defaults/{algorithm}", tags=["Parameters"])
async def get_default_parameters(algorithm: str):
    """Get default parameter values."""
    if algorithm not in ALGORITHM_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Algorithm '{algorithm}' not found")
    
    return {
        "algorithm": algorithm,
        "defaults": ParameterValidator.get_default_params(algorithm, 'algorithm')
    }

@app.post("/parameters/update/{job_id}", tags=["Parameters"])
async def update_training_parameter(job_id: str, request: ParameterUpdateRequest):
    """Update parameter during training (live update)."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    if job['status'] != 'running':
        raise HTTPException(status_code=400, detail="Training not running")
    
    if not ParameterValidator.can_update_live(request.parameter):
        raise HTTPException(
            status_code=400,
            detail=f"Parameter '{request.parameter}' cannot be updated during training"
        )
    
    algo = job.get('algorithm_instance')
    if not algo:
        raise HTTPException(status_code=400, detail="No algorithm instance")
    
    # Update parameter
    setter = getattr(algo, f'set_{request.parameter}', None)
    if setter:
        setter(request.value)
        return {"success": True, "parameter": request.parameter, "value": request.value}
    
    raise HTTPException(
        status_code=400,
        detail=f"Cannot update parameter '{request.parameter}'"
    )

# ============================================================================
# Algorithm Information Endpoints
# ============================================================================

@app.get("/algorithms", tags=["Algorithms"])
async def get_algorithms():
    """List all available algorithms with descriptions."""
    return {
        "algorithms": list_algorithms(),
        "count": len(ALGORITHM_REGISTRY)
    }

# ============================================================================
# Health Check
# ============================================================================

@app.get("/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "environments": len(ENVIRONMENT_REGISTRY),
        "algorithms": len(ALGORITHM_REGISTRY),
        "active_jobs": len([j for j in training_jobs.values() if j['status'] == 'running']),
        "total_jobs": len(training_jobs)
    }

# ============================================================================
# Cleanup Endpoints
# ============================================================================

@app.delete("/job/{job_id}", tags=["System"])
async def delete_job(job_id: str):
    """Delete a training job and free resources."""
    if job_id not in training_jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    
    job = training_jobs[job_id]
    
    # Stop if running
    if job['status'] == 'running' and 'algorithm_instance' in job:
        job['algorithm_instance'].stop()
    
    del training_jobs[job_id]
    
    return {"deleted": job_id}

@app.delete("/environment/{instance_id}", tags=["System"])
async def delete_environment_instance(instance_id: str):
    """Delete an environment instance."""
    if instance_id not in environment_instances:
        raise HTTPException(status_code=404, detail=f"Instance '{instance_id}' not found")
    
    env = environment_instances[instance_id]['env']
    env.close()
    del environment_instances[instance_id]
    
    return {"deleted": instance_id}

# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

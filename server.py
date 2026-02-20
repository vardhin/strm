from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
import os
import asyncio
from datetime import datetime
import uuid

from symbolic import SymbolicRegistry
from agent import SymbolicAgent

app = FastAPI(title="Symbolic Learning API", version="1.0.0")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
class SystemState:
    def __init__(self):
        self.registry: Optional[SymbolicRegistry] = None
        self.agent: Optional[SymbolicAgent] = None
        self.is_training = False
        self.training_jobs: Dict[str, Dict] = {}
        self.learned_functions: Dict[str, int] = {}  # name -> function_id
        
state = SystemState()

# ==========================================
# Request/Response Models
# ==========================================

class TrainingExample(BaseModel):
    inputs: List[float]
    output: float

class TrainRequest(BaseModel):
    function_name: str
    examples: List[TrainingExample]
    num_epochs: Optional[int] = 50
    exploration_bonus: Optional[float] = 0.7

class PredictRequest(BaseModel):
    function_name: str
    inputs: List[float]

class StatusResponse(BaseModel):
    status: str
    message: str
    functions_count: int
    is_training: bool

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

class PredictionResponse(BaseModel):
    function_name: str
    inputs: List[float]
    output: float
    success: bool

class FunctionInfo(BaseModel):
    id: int
    name: str
    arity: int
    layer: Optional[int] = None
    composition: Optional[str] = None

class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "pending", "running", "completed", "failed"
    progress: Optional[float] = None
    message: Optional[str] = None
    result: Optional[Dict] = None

# ==========================================
# Initialization
# ==========================================

@app.on_event("startup")
async def startup_event():
    """Initialize the system on startup"""
    print("Initializing Symbolic Learning System...")
    
    # Create checkpoints directory if it doesn't exist
    os.makedirs("checkpoints", exist_ok=True)
    
    # Initialize registry and agent
    state.registry = SymbolicRegistry()
    
    # Load existing database if available
    if os.path.exists("checkpoints/symbolic.db"):
        state.registry.load("checkpoints/symbolic.db")
        print(f"Loaded {len(state.registry.metadata)} functions from database")
    else:
        print("Starting with fresh registry (primitives only)")
    
    # Initialize agent
    state.agent = SymbolicAgent(
        state.registry,
        d_model=128,
        max_recursion=8,
        input_dim=32,
        max_composition_depth=3
    )
    
    # Load model checkpoint if available
    if os.path.exists("checkpoints/model.pt"):
        state.agent.load_checkpoint("checkpoints/model.pt")
        print("Loaded model checkpoint")
    else:
        print("Starting with fresh model")
    
    # Build function name lookup
    for fid, meta in state.registry.metadata.items():
        state.learned_functions[meta['name']] = fid
    
    print("System ready!")

# ==========================================
# API Endpoints
# ==========================================

@app.get("/", response_model=StatusResponse)
async def root():
    """Get system status"""
    return StatusResponse(
        status="online",
        message="Symbolic Learning API is running",
        functions_count=len(state.registry.metadata) if state.registry else 0,
        is_training=state.is_training
    )

@app.get("/functions", response_model=List[FunctionInfo])
async def list_functions():
    """List all available functions"""
    if not state.registry:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    functions = []
    for fid, meta in state.registry.metadata.items():
        arity = len(meta.get('signature', [])) if 'signature' in meta else -1
        
        # Get composition description
        composition = None
        if 'composition' in meta:
            comp = meta['composition']
            if comp:
                composition = " → ".join([
                    state.registry.metadata[func_id]['name'] 
                    for func_id, _ in comp
                ])
        
        functions.append(FunctionInfo(
            id=fid,
            name=meta['name'],
            arity=arity,
            layer=meta.get('layer'),
            composition=composition
        ))
    
    return functions

@app.post("/train", response_model=TrainingResponse)
async def train_function(request: TrainRequest, background_tasks: BackgroundTasks):
    """Train a new function from examples"""
    if state.is_training:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    if not state.agent or not state.registry:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Validate examples
    if not request.examples:
        raise HTTPException(status_code=400, detail="No examples provided")
    
    if len(request.examples) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 examples")
    
    # Check if function already exists
    if request.function_name in state.learned_functions:
        raise HTTPException(
            status_code=409, 
            detail=f"Function '{request.function_name}' already exists"
        )
    
    # Convert examples to format expected by agent
    # CSV data comes as floats, need to convert to integers
    try:
        examples = []
        for example in request.examples:
            # Convert inputs to integers
            inputs = [int(round(x)) for x in example.inputs]
            # Convert output to integer
            output = int(round(example.output))
            examples.append((inputs, output))
    except (ValueError, TypeError) as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid example data - all values must be numeric: {str(e)}"
        )
    
    # Create job
    job_id = str(uuid.uuid4())
    state.training_jobs[job_id] = {
        'status': 'pending',
        'function_name': request.function_name,
        'start_time': datetime.now().isoformat(),
        'progress': 0.0
    }
    
    # Start training in background
    background_tasks.add_task(
        train_function_task,
        job_id,
        request.function_name,
        examples,
        request.num_epochs,
        request.exploration_bonus
    )
    
    return TrainingResponse(
        job_id=job_id,
        status="pending",
        message=f"Training job created for function '{request.function_name}'"
    )

async def train_function_task(
    job_id: str,
    function_name: str,
    examples: List[Tuple[List, int]],
    num_epochs: int,
    exploration_bonus: float
):
    """Background task to train a function"""
    try:
        state.is_training = True
        state.training_jobs[job_id]['status'] = 'running'
        
        print(f"\n[Job {job_id}] Training {function_name}...")
        print(f"  Examples: {len(examples)}")
        print(f"  Epochs: {num_epochs}")
        
        # Train the function
        success = state.agent.learn_abstraction(
            function_name,
            examples,
            num_epochs=num_epochs,
            exploration_bonus=exploration_bonus
        )
        
        if success:
            # Get the new function ID
            func_id = next(
                fid for fid, m in state.registry.metadata.items() 
                if m['name'] == function_name
            )
            state.learned_functions[function_name] = func_id
            
            # Save checkpoint
            state.agent.save_checkpoint()
            state.registry.save()
            
            state.training_jobs[job_id].update({
                'status': 'completed',
                'progress': 1.0,
                'end_time': datetime.now().isoformat(),
                'result': {
                    'success': True,
                    'function_id': func_id,
                    'message': f"Successfully learned {function_name}"
                }
            })
            print(f"[Job {job_id}] ✓ Success!")
        else:
            state.training_jobs[job_id].update({
                'status': 'failed',
                'progress': 1.0,
                'end_time': datetime.now().isoformat(),
                'result': {
                    'success': False,
                    'message': f"Failed to learn {function_name}"
                }
            })
            print(f"[Job {job_id}] ✗ Failed")
            
    except Exception as e:
        print(f"[Job {job_id}] Error: {e}")
        state.training_jobs[job_id].update({
            'status': 'failed',
            'progress': 1.0,
            'end_time': datetime.now().isoformat(),
            'result': {
                'success': False,
                'message': f"Error: {str(e)}"
            }
        })
    finally:
        state.is_training = False

@app.get("/train/status/{job_id}", response_model=JobStatusResponse)
async def get_training_status(job_id: str):
    """Get status of a training job"""
    if job_id not in state.training_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = state.training_jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job['status'],
        progress=job.get('progress'),
        message=job.get('result', {}).get('message'),
        result=job.get('result')
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictRequest):
    """Make a prediction using a learned function"""
    if not state.registry:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    # Check if function exists
    if request.function_name not in state.learned_functions:
        raise HTTPException(
            status_code=404, 
            detail=f"Function '{request.function_name}' not found"
        )
    
    func_id = state.learned_functions[request.function_name]
    
    try:
        # Convert float inputs to integers (same as training)
        inputs_int = [int(round(x)) for x in request.inputs]
        
        # Execute the function
        result = state.registry.execute_function(func_id, inputs_int)
        
        return PredictionResponse(
            function_name=request.function_name,
            inputs=request.inputs,
            output=float(result),
            success=True
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.delete("/functions/{function_name}")
async def delete_function(function_name: str):
    """Delete a learned function (primitives cannot be deleted)"""
    if function_name not in state.learned_functions:
        raise HTTPException(status_code=404, detail="Function not found")
    
    func_id = state.learned_functions[function_name]
    meta = state.registry.metadata[func_id]
    
    if meta.get('is_primitive', False):
        raise HTTPException(status_code=400, detail="Cannot delete primitive functions")
    
    # Remove from registry (you may need to implement this in SymbolicRegistry)
    del state.registry.metadata[func_id]
    del state.learned_functions[function_name]
    
    # Save changes
    state.registry.save()
    
    return {"message": f"Function '{function_name}' deleted"}

@app.post("/reset")
async def reset_system():
    """Reset the system to initial state"""
    if state.is_training:
        raise HTTPException(status_code=409, detail="Cannot reset while training")
    
    # Reinitialize
    state.registry = SymbolicRegistry()
    state.agent = SymbolicAgent(
        state.registry,
        d_model=128,
        max_recursion=8,
        input_dim=32,
        max_composition_depth=3
    )
    state.learned_functions = {}
    state.training_jobs = {}
    
    return {"message": "System reset successfully"}

# ==========================================
# Run Server
# ==========================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
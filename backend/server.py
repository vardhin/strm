"""
NSSR Experiment Server.

FastAPI server for running reproducible experiments without touching
the core pipeline code. Provides endpoints for:

  - Registry:    list/execute/register functions (calculator mode)
  - Datasets:    define example sets, export/import CSV
  - Experiments: train models with custom curricula, manage checkpoints
  - Testing:     evaluate models against example sets, compare models
"""

import math
import os
import csv
import io
import json
import sqlite3
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

import db
import registry as reg
import executor as exe
import search
import train
import simplify
from model import TRM, create_model, fresh_carry, resize_heads
from main import build_composition, CONFIG, learn, curriculum_tasks


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------

class ExecuteRequest(BaseModel):
    func_id: int
    inputs: list[float]

class ExecuteBatchRequest(BaseModel):
    func_id: int
    input_sets: list[list[float]]

class ExpressionRequest(BaseModel):
    """Calculator mode: evaluate a named function on inputs."""
    func_name: str
    inputs: list[float]

class RegisterRequest(BaseModel):
    name: str
    arity: int
    composition: list[tuple[int, list[int]]]

class DatasetCreate(BaseModel):
    name: str
    description: str = ""
    examples: list[tuple[list[float], float]]

class DatasetFromFunction(BaseModel):
    """Generate a dataset by running a registered function on input sets."""
    name: str
    description: str = ""
    func_id: int
    input_sets: list[list[float]]

class TrainRequest(BaseModel):
    model_name: str = "default"
    dataset_name: str
    target_name: str
    max_search_steps: int = 10
    max_depth: int = 3
    num_epochs: int = 30

class CurriculumItem(BaseModel):
    dataset_name: str
    target_name: str
    max_depth: int = 3
    num_epochs: int = 30

class ExperimentRequest(BaseModel):
    """Run a full experiment: curriculum pre-training + progressive learning."""
    model_name: str = "default"
    curriculum: list[CurriculumItem]
    pre_train: bool = True
    pre_train_epochs: int = 20

class EvalRequest(BaseModel):
    model_name: str = "default"
    dataset_name: str

class CompareRequest(BaseModel):
    model_names: list[str]
    dataset_name: str


# ---------------------------------------------------------------------------
# Global state
# ---------------------------------------------------------------------------

# Multiple isolated model/registry environments keyed by model_name.
# Each has its own conn, state, model, optimizer.
_envs: dict[str, dict] = {}

# Named datasets (in-memory, exportable to CSV)
_datasets: dict[str, dict] = {}

# Experiment logs
_experiments: list[dict] = []


def _get_or_create_env(model_name: str) -> dict:
    """Get or create an isolated environment for a model."""
    if model_name in _envs:
        return _envs[model_name]

    ckpt_dir = os.path.join("checkpoints", model_name)
    os.makedirs(ckpt_dir, exist_ok=True)

    conn = db.init_db(os.path.join(ckpt_dir, "symbolic.db"))
    state = reg.init_registry(conn)

    n_funcs = reg.vocab_size(state)
    model = create_model(
        input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
        d_model=CONFIG["d_model"], n_functions=n_funcs,
        n_layers=CONFIG["n_layers"],
        n_recursions=CONFIG["n_recursions"], T=CONFIG["T"],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG["lr"])

    # Try loading existing checkpoint
    ckpt_path = os.path.join(ckpt_dir, "model.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, weights_only=False)
        old_n = ckpt["n_functions"]
        new_n = n_funcs
        if old_n != new_n:
            tmp = create_model(
                input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                d_model=CONFIG["d_model"], n_functions=old_n,
                n_layers=CONFIG["n_layers"],
                n_recursions=CONFIG["n_recursions"], T=CONFIG["T"],
            )
            tmp.load_state_dict(ckpt["model_state"], strict=False)
            model.load_state_dict(tmp.state_dict(), strict=False)
            resize_heads(model, old_n, new_n)
        else:
            model.load_state_dict(ckpt["model_state"], strict=False)
            try:
                optimizer.load_state_dict(ckpt["optimizer_state"])
            except Exception:
                pass

    env = {
        "conn": conn,
        "state": state,
        "model": model,
        "optimizer": optimizer,
        "ckpt_dir": ckpt_dir,
        "train_history": [],
    }
    _envs[model_name] = env
    return env


def _save_env_checkpoint(env: dict):
    path = os.path.join(env["ckpt_dir"], "model.pt")
    torch.save({
        "model_state": env["model"].state_dict(),
        "optimizer_state": env["optimizer"].state_dict(),
        "n_functions": reg.vocab_size(env["state"]),
    }, path)


def _name_to_id(state: dict, name: str) -> int | None:
    for fid, meta in state["metadata"].items():
        if meta["name"] == name:
            return fid
    return None


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: create default env
    _get_or_create_env("default")
    yield
    # shutdown: close all DB connections
    for env in _envs.values():
        env["conn"].close()


app = FastAPI(
    title="NSSR Experiment Server",
    description="Reproducible experimentation for Neuro-Symbolic Recursive Regression",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================= REGISTRY =====================================

@app.get("/registry/{model_name}")
def list_functions(model_name: str = "default"):
    """List all registered functions (primitives + learned)."""
    env = _get_or_create_env(model_name)
    result = []
    for fid, meta in env["state"]["metadata"].items():
        result.append({
            "id": fid,
            "name": meta["name"],
            "arity": meta["arity"],
            "layer": meta["layer"],
        })
    return {"functions": sorted(result, key=lambda x: (x["layer"], x["id"]))}


@app.post("/registry/{model_name}/execute")
def execute_function(req: ExecuteRequest, model_name: str = "default"):
    """Execute a function by ID on given inputs."""
    env = _get_or_create_env(model_name)
    try:
        result = reg.execute(env["state"], req.func_id, req.inputs)
        return {"func_id": req.func_id, "inputs": req.inputs, "result": result}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/registry/{model_name}/execute_batch")
def execute_batch(req: ExecuteBatchRequest, model_name: str = "default"):
    """Execute a function on multiple input sets."""
    env = _get_or_create_env(model_name)
    results = []
    for inputs in req.input_sets:
        try:
            r = reg.execute(env["state"], req.func_id, inputs)
            results.append({"inputs": inputs, "result": r, "error": None})
        except Exception as e:
            results.append({"inputs": inputs, "result": None, "error": str(e)})
    return {"func_id": req.func_id, "results": results}


@app.post("/registry/{model_name}/eval")
def eval_expression(req: ExpressionRequest, model_name: str = "default"):
    """Calculator mode: evaluate a named function on inputs."""
    env = _get_or_create_env(model_name)
    fid = _name_to_id(env["state"], req.func_name)
    if fid is None:
        raise HTTPException(404, f"Function '{req.func_name}' not found")
    try:
        result = reg.execute(env["state"], fid, req.inputs)
        return {"func_name": req.func_name, "func_id": fid,
                "inputs": req.inputs, "result": result}
    except Exception as e:
        raise HTTPException(400, str(e))


@app.post("/registry/{model_name}/register")
def register_function(req: RegisterRequest, model_name: str = "default"):
    """Register a new composed function from a composition spec."""
    env = _get_or_create_env(model_name)
    composition = [(cid, args) for cid, args in req.composition]

    old_vocab = reg.vocab_size(env["state"])
    fid = reg.register_learned(env["conn"], env["state"], req.name,
                               req.arity, composition)
    new_vocab = reg.vocab_size(env["state"])

    if new_vocab > old_vocab:
        resize_heads(env["model"], old_vocab, new_vocab)
        env["optimizer"] = torch.optim.AdamW(
            env["model"].parameters(), lr=CONFIG["lr"]
        )

    return {"func_id": fid, "name": req.name, "vocab_size": new_vocab}


# ============================= DATASETS =====================================

@app.get("/datasets")
def list_datasets():
    """List all defined datasets."""
    return {
        name: {
            "description": ds["description"],
            "num_examples": len(ds["examples"]),
            "sample": ds["examples"][:3],
        }
        for name, ds in _datasets.items()
    }


@app.post("/datasets")
def create_dataset(req: DatasetCreate):
    """Create a dataset from explicit examples."""
    _datasets[req.name] = {
        "description": req.description,
        "examples": [tuple(e) for e in req.examples],
    }
    return {"name": req.name, "num_examples": len(req.examples)}


@app.post("/datasets/from_function")
def create_dataset_from_function(req: DatasetFromFunction):
    """Generate a dataset by running a registered function on input sets.

    This is the 'calculator -> dataset' workflow: define inputs, let
    the registry compute outputs, store as a reusable dataset.
    """
    env = _get_or_create_env("default")
    examples = []
    errors = []
    for inputs in req.input_sets:
        try:
            result = reg.execute(env["state"], req.func_id, inputs)
            examples.append((inputs, result))
        except Exception as e:
            errors.append({"inputs": inputs, "error": str(e)})

    _datasets[req.name] = {
        "description": req.description,
        "examples": examples,
    }
    return {
        "name": req.name,
        "num_examples": len(examples),
        "errors": errors if errors else None,
    }


@app.get("/datasets/{name}")
def get_dataset(name: str):
    """Get a dataset's full contents."""
    if name not in _datasets:
        raise HTTPException(404, f"Dataset '{name}' not found")
    ds = _datasets[name]
    return {
        "name": name,
        "description": ds["description"],
        "examples": ds["examples"],
    }


@app.delete("/datasets/{name}")
def delete_dataset(name: str):
    if name not in _datasets:
        raise HTTPException(404, f"Dataset '{name}' not found")
    del _datasets[name]
    return {"deleted": name}


@app.get("/datasets/{name}/csv")
def export_dataset_csv(name: str):
    """Export a dataset as CSV (inputs as separate columns + output column)."""
    if name not in _datasets:
        raise HTTPException(404, f"Dataset '{name}' not found")

    ds = _datasets[name]
    if not ds["examples"]:
        raise HTTPException(400, "Dataset is empty")

    max_inputs = max(len(ex[0]) for ex in ds["examples"])
    buf = io.StringIO()
    writer = csv.writer(buf)

    header = [f"input_{i}" for i in range(max_inputs)] + ["output"]
    writer.writerow(header)

    for inputs, output in ds["examples"]:
        row = list(inputs) + [None] * (max_inputs - len(inputs)) + [output]
        writer.writerow(row)

    buf.seek(0)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": f"attachment; filename={name}.csv"},
    )


@app.post("/datasets/{name}/import_csv")
def import_dataset_csv(name: str, description: str = ""):
    """Import a dataset from a CSV file path on disk."""
    # This endpoint expects a query param `path` for simplicity
    raise HTTPException(501, "Use POST /datasets with examples directly, "
                        "or GET /datasets/{name}/csv to export")


# ============================= TRAINING =====================================

@app.post("/train")
def train_single(req: TrainRequest):
    """Train a model to learn a single target function from a dataset.

    The full pipeline: search -> simplify -> register -> resize -> train.
    """
    if req.dataset_name not in _datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found")

    env = _get_or_create_env(req.model_name)
    examples = _datasets[req.dataset_name]["examples"]

    t0 = time.time()
    ok, env["optimizer"], r2 = learn(
        env["conn"], env["state"], env["model"], env["optimizer"],
        req.target_name, examples,
        max_search_steps=req.max_search_steps,
        max_depth=req.max_depth,
        num_epochs=req.num_epochs,
    )
    elapsed = time.time() - t0

    # Auto-save checkpoint
    _save_env_checkpoint(env)

    record = {
        "model_name": req.model_name,
        "target": req.target_name,
        "dataset": req.dataset_name,
        "success": ok,
        "r2_score": round(r2, 6),
        "elapsed_s": round(elapsed, 2),
        "vocab_size": reg.vocab_size(env["state"]),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    env["train_history"].append(record)
    _experiments.append(record)

    return record


@app.post("/train/experiment")
def run_experiment(req: ExperimentRequest):
    """Run a full experiment: optional pre-training + sequential learning."""
    env = _get_or_create_env(req.model_name)
    results = []

    # Phase 1: curriculum pre-training
    if req.pre_train:
        tasks = curriculum_tasks(env["state"])
        for task in tasks:
            train.train_on_examples(
                env["model"], env["optimizer"],
                task["examples"], task["target"],
                input_dim=CONFIG["input_dim"], seq_len=CONFIG["seq_len"],
                num_epochs=req.pre_train_epochs, n_sup=CONFIG["n_sup"],
            )
        results.append({"phase": "pre_training", "tasks": len(tasks), "status": "done"})

    # Phase 2: progressive learning from datasets
    for item in req.curriculum:
        if item.dataset_name not in _datasets:
            results.append({
                "phase": "learn", "target": item.target_name,
                "status": "error", "error": f"Dataset '{item.dataset_name}' not found",
            })
            continue

        examples = _datasets[item.dataset_name]["examples"]
        t0 = time.time()
        ok, env["optimizer"], r2 = learn(
            env["conn"], env["state"], env["model"], env["optimizer"],
            item.target_name, examples,
            max_depth=item.max_depth,
            num_epochs=item.num_epochs,
        )
        elapsed = time.time() - t0

        record = {
            "phase": "learn",
            "target": item.target_name,
            "dataset": item.dataset_name,
            "success": ok,
            "r2_score": round(r2, 6),
            "elapsed_s": round(elapsed, 2),
        }
        results.append(record)
        env["train_history"].append(record)

    _save_env_checkpoint(env)
    return {"model_name": req.model_name, "results": results}


# ============================= MODELS =======================================

@app.get("/models")
def list_models():
    """List all loaded model environments."""
    result = {}
    for name, env in _envs.items():
        result[name] = {
            "vocab_size": reg.vocab_size(env["state"]),
            "num_functions": len(env["state"]["metadata"]),
            "functions": [
                {"id": fid, "name": m["name"], "layer": m["layer"]}
                for fid, m in sorted(env["state"]["metadata"].items())
            ],
            "train_history_count": len(env["train_history"]),
        }
    return result


@app.get("/models/{model_name}")
def get_model_info(model_name: str):
    """Get detailed info about a model."""
    env = _get_or_create_env(model_name)
    state = env["state"]

    total_params = sum(p.numel() for p in env["model"].parameters())
    functions = []
    for fid, m in sorted(state["metadata"].items()):
        entry = {"id": fid, "name": m["name"], "arity": m["arity"], "layer": m["layer"]}
        # For learned functions, show composition
        if m["layer"] > 0:
            comp = db.get_composition(env["conn"], fid)
            entry["composition"] = [
                {"func_id": cid, "func_name": reg.get_name(state, cid), "args": args}
                for cid, args in comp
            ]
            if m.get("constants"):
                entry["constants"] = m["constants"]
                entry["const_mode"] = m.get("const_mode", "multiplicative")
        functions.append(entry)

    return {
        "model_name": model_name,
        "total_params": total_params,
        "d_model": CONFIG["d_model"],
        "n_layers": CONFIG["n_layers"],
        "n_recursions": CONFIG["n_recursions"],
        "T": CONFIG["T"],
        "vocab_size": reg.vocab_size(state),
        "functions": functions,
        "train_history": env["train_history"],
    }


@app.post("/models/{model_name}/save")
def save_model(model_name: str):
    """Save model checkpoint to disk."""
    env = _get_or_create_env(model_name)
    _save_env_checkpoint(env)
    return {"saved": model_name, "path": env["ckpt_dir"]}


@app.delete("/models/{model_name}")
def delete_model(model_name: str):
    """Unload a model from memory (does not delete checkpoint files)."""
    if model_name not in _envs:
        raise HTTPException(404, f"Model '{model_name}' not loaded")
    if model_name == "default":
        raise HTTPException(400, "Cannot delete the default model")
    env = _envs.pop(model_name)
    env["conn"].close()
    return {"unloaded": model_name}


# ============================= TESTING ======================================

@app.post("/test/eval")
def test_eval(req: EvalRequest):
    """Evaluate: for each example in a dataset, run the search/learned function
    and check if the model's registry produces correct outputs.

    Returns per-function R² scores and per-example details.
    """
    if req.dataset_name not in _datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found")

    env = _get_or_create_env(req.model_name)
    examples = _datasets[req.dataset_name]["examples"]
    state = env["state"]

    # Per-example detail
    results = []
    correct = 0
    for inputs, expected in examples:
        matches = []
        for fid, meta in state["metadata"].items():
            if meta["arity"] != len(inputs):
                continue
            try:
                got = reg.execute(state, fid, inputs)
                if math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9):
                    matches.append({"id": fid, "name": meta["name"], "result": got})
            except Exception:
                pass

        is_correct = len(matches) > 0
        if is_correct:
            correct += 1

        results.append({
            "inputs": inputs,
            "expected": expected,
            "correct": is_correct,
            "matching_functions": matches,
        })

    # Per-function R² scores (for all functions with matching arity)
    input_arity = len(examples[0][0]) if examples else 0
    r2_scores = {}
    for fid, meta in state["metadata"].items():
        if meta["arity"] != input_arity:
            continue
        r2 = exe.r_squared(state, fid, examples)
        if r2 > -1e10:  # skip functions that error on all examples
            r2_scores[meta["name"]] = round(r2, 6)

    # Best R²
    best_r2 = max(r2_scores.values()) if r2_scores else 0.0
    best_fn = max(r2_scores, key=r2_scores.get) if r2_scores else None

    return {
        "model_name": req.model_name,
        "dataset": req.dataset_name,
        "total": len(examples),
        "correct": correct,
        "accuracy": round(correct / len(examples), 4) if examples else 0,
        "r2_scores": r2_scores,
        "best_r2": round(best_r2, 6),
        "best_function": best_fn,
        "details": results,
    }


@app.post("/test/compare")
def test_compare(req: CompareRequest):
    """Compare multiple models on the same dataset."""
    if req.dataset_name not in _datasets:
        raise HTTPException(404, f"Dataset '{req.dataset_name}' not found")

    examples = _datasets[req.dataset_name]["examples"]
    comparison = {}

    for model_name in req.model_names:
        env = _get_or_create_env(model_name)
        state = env["state"]

        correct = 0
        for inputs, expected in examples:
            for fid, meta in state["metadata"].items():
                if meta["arity"] != len(inputs):
                    continue
                try:
                    got = reg.execute(state, fid, inputs)
                    if math.isclose(got, expected, rel_tol=1e-6, abs_tol=1e-9):
                        correct += 1
                        break
                except Exception:
                    pass

        # Best R² across all functions with matching arity
        input_arity = len(examples[0][0]) if examples else 0
        best_r2 = -float("inf")
        best_fn = None
        for fid, meta in state["metadata"].items():
            if meta["arity"] != input_arity:
                continue
            r2 = exe.r_squared(state, fid, examples)
            if r2 > best_r2:
                best_r2 = r2
                best_fn = meta["name"]

        comparison[model_name] = {
            "correct": correct,
            "total": len(examples),
            "accuracy": round(correct / len(examples), 4) if examples else 0,
            "best_r2": round(best_r2, 6) if best_r2 > -1e10 else None,
            "best_function": best_fn,
            "vocab_size": reg.vocab_size(state),
            "learned_functions": [
                m["name"] for m in state["metadata"].values() if m["layer"] > 0
            ],
        }

    return {"dataset": req.dataset_name, "comparison": comparison}


@app.post("/test/predict")
def test_predict(model_name: str, inputs: list[float]):
    """Use the TRM model to predict which composition to use for given inputs.

    This shows what the neural network *thinks* the answer is, before
    symbolic verification.
    """
    env = _get_or_create_env(model_name)
    model = env["model"]
    state = env["state"]
    model.eval()

    # Encode inputs as a single example
    input_dim = CONFIG["input_dim"]
    seq_len = CONFIG["seq_len"]
    x = search.format_examples([(inputs, 0)], input_dim=input_dim, seq_len=seq_len)

    carry = fresh_carry(1, seq_len, model.d_model)
    n_funcs = reg.vocab_size(state)

    with torch.no_grad():
        carry, outputs = model(carry, x)

    # Decode predictions
    primary_logits = outputs["primary_logits"][0][:n_funcs]
    secondary_logits = outputs["secondary_logits"][0][:n_funcs]
    comp_logits = outputs["composition_logits"][0]

    comp_types = ["none", "sequential", "nested", "parallel"]
    comp_probs = torch.softmax(comp_logits, dim=0).tolist()

    top_primary = torch.topk(primary_logits, min(5, n_funcs))
    top_secondary = torch.topk(secondary_logits, min(5, n_funcs))

    return {
        "inputs": inputs,
        "predictions": {
            "primary": [
                {"id": int(i), "name": reg.get_name(state, int(i)),
                 "score": round(float(s), 4)}
                for i, s in zip(top_primary.indices, top_primary.values)
            ],
            "secondary": [
                {"id": int(i), "name": reg.get_name(state, int(i)),
                 "score": round(float(s), 4)}
                for i, s in zip(top_secondary.indices, top_secondary.values)
            ],
            "composition": {
                comp_types[i]: round(p, 4)
                for i, p in enumerate(comp_probs[:len(comp_types)])
            },
            "halt_prob": round(float(torch.sigmoid(outputs["halt_logits"][0])), 4),
        },
    }


# ============================= EXPERIMENT LOG ===============================

@app.get("/experiments")
def list_experiments():
    """Get the full experiment log."""
    return {"experiments": _experiments}


# ============================= HEALTH =======================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "loaded_models": list(_envs.keys()),
        "datasets": list(_datasets.keys()),
        "experiments_run": len(_experiments),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)

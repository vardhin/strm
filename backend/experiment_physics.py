"""
Physics Experiment: Can NSSR learn physics equations from examples?

This script talks to the running server (localhost:8000) and runs a
step-by-step experiment:

  Step 1 — Check what the model already knows (primitives + learned fns)
  Step 2 — Teach SQUARE(x) = x * x
  Step 3 — Teach DOUBLE_KE(m, v) = m * v * v   (2 * kinetic energy)
  Step 4 — Teach FORCE(m, a) = m * a            (Newton's 2nd law)
  Step 5 — Teach IMPULSE(F, t) = F * t          (impulse = force * time)
  Step 6 — SYNTHESIS TEST: can it learn POWER(m, a, v) = m * a * v
           from scratch, composing what it already knows?
  Step 7 — Evaluation: test all learned functions on held-out examples

Usage:
    1. Start the server:  cd backend && python server.py
    2. Run this script:   python experiment_physics.py
"""

import sys
import time
import requests

BASE = "http://localhost:8000"
MODEL = "physics"   # isolated model environment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def api(method, path, json=None):
    url = f"{BASE}{path}"
    r = requests.request(method, url, json=json)
    if r.status_code >= 400:
        print(f"  ERROR {r.status_code}: {r.text}")
        return None
    return r.json()


def section(title):
    print(f"\n{'=' * 60}")
    print(f"  {title}")
    print(f"{'=' * 60}\n")


def subsection(title):
    print(f"\n--- {title} ---\n")


def print_functions(funcs):
    for f in funcs:
        layer_tag = f"L{f['layer']}" if 'layer' in f else ""
        print(f"  [{f['id']:>2}] {f['name']:<12} arity={f.get('arity','?')}  {layer_tag}")


def create_dataset(name, description, examples):
    """Create a dataset and print summary."""
    # Format: examples are [(inputs, output), ...]
    formatted = [{"0": inp, "1": out} for inp, out in examples]
    resp = api("POST", "/datasets", {
        "name": name,
        "description": description,
        "examples": [[inp, out] for inp, out in examples],
    })
    if resp:
        print(f"  Created dataset '{name}' with {resp['num_examples']} examples")
        for inp, out in examples[:5]:
            print(f"    {inp} -> {out}")
        if len(examples) > 5:
            print(f"    ... and {len(examples) - 5} more")
    return resp


def train_function(target_name, dataset_name, max_depth=3, num_epochs=30,
                   max_search_steps=10):
    """Train the model to learn a function from a dataset."""
    print(f"  Training '{target_name}' from dataset '{dataset_name}'...")
    print(f"  (max_depth={max_depth}, epochs={num_epochs}, search_steps={max_search_steps})")
    t0 = time.time()
    resp = api("POST", "/train", {
        "model_name": MODEL,
        "dataset_name": dataset_name,
        "target_name": target_name,
        "max_search_steps": max_search_steps,
        "max_depth": max_depth,
        "num_epochs": num_epochs,
    })
    elapsed = time.time() - t0
    if resp:
        status = "PASSED" if resp["success"] else "FAILED"
        print(f"  Result: {status}  ({elapsed:.1f}s)")
        print(f"  Vocab size: {resp['vocab_size']}")
    else:
        print(f"  Training failed!")
    return resp


def evaluate(dataset_name):
    """Evaluate the model on a dataset."""
    resp = api("POST", "/test/eval", {
        "model_name": MODEL,
        "dataset_name": dataset_name,
    })
    if resp:
        print(f"  Accuracy: {resp['correct']}/{resp['total']} = {resp['accuracy']*100:.1f}%")
        for d in resp["details"]:
            mark = "ok" if d["correct"] else "MISS"
            matches = ", ".join(m["name"] for m in d["matching_functions"]) if d["matching_functions"] else "none"
            print(f"    {d['inputs']} -> expected {d['expected']}  [{mark}]  matched: {matches}")
    return resp


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main():
    # --- Preflight: check server ---
    section("PREFLIGHT: Checking server")
    health = api("GET", "/health")
    if not health:
        print("Server not reachable at", BASE)
        print("Start it with:  cd backend && python server.py")
        sys.exit(1)
    print(f"  Server is up. Models loaded: {health['loaded_models']}")

    # --- Step 1: See what's already registered ---
    section("STEP 1: Current model state")
    info = api("GET", f"/models/{MODEL}")
    if info:
        print(f"  Model: {info['model_name']}")
        print(f"  Params: {info['total_params']:,}")
        print(f"  Vocab: {info['vocab_size']}")
        print(f"  Functions:")
        print_functions(info["functions"])

        # Check if MUL already exists
        func_names = [f["name"] for f in info["functions"]]
        has_mul = "MUL" in func_names
        has_add = "ADD" in func_names
        print(f"\n  Has ADD: {has_add}")
        print(f"  Has MUL: {has_mul}")

        if not has_mul:
            print("\n  WARNING: MUL not found. The model needs ADD and MUL as building blocks.")
            print("  Run the base training first (python main.py) or teach ADD+MUL here.\n")

            # Teach ADD first if missing
            if not has_add:
                subsection("Teaching ADD")
                create_dataset("add_train", "Addition training examples", [
                    ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 2),
                    ([2, 3], 5), ([4, 2], 6), ([5, 5], 10), ([7, 3], 10),
                ])
                train_function("ADD", "add_train", max_depth=2, num_epochs=30)

            # Teach MUL
            subsection("Teaching MUL")
            create_dataset("mul_train", "Multiplication training examples", [
                ([0, 0], 0), ([0, 5], 0), ([1, 5], 5), ([2, 3], 6),
                ([3, 4], 12), ([4, 5], 20), ([5, 5], 25),
            ])
            train_function("MUL", "mul_train", max_depth=2, num_epochs=30)

            # Refresh
            info = api("GET", f"/models/{MODEL}")
            print_functions(info["functions"])

    # --- Step 2: Teach SQUARE ---
    section("STEP 2: Teaching SQUARE(x) = x * x")
    print("  This is the simplest 'physics' building block — squaring a value.")
    print("  The system should find: MUL(x, x) or an equivalent.\n")

    create_dataset("square_train", "Square function: x -> x*x", [
        ([1], 1), ([2], 4), ([3], 9), ([4], 16),
        ([5], 25), ([6], 36), ([7], 49),
    ])
    result = train_function("SQUARE", "square_train", max_depth=2, num_epochs=30)

    # Test on held-out values
    subsection("Verifying SQUARE on held-out values")
    create_dataset("square_test", "Square test set", [
        ([8], 64), ([9], 81), ([10], 100),
    ])
    evaluate("square_test")

    # --- Step 3: Teach DOUBLE_KE (2 * kinetic energy) ---
    section("STEP 3: Teaching DOUBLE_KE(m, v) = m * v * v")
    print("  Kinetic energy is (1/2)mv^2. Since we don't have division,")
    print("  we learn 2*KE = m*v*v instead. Should compose as MUL(m, SQUARE(v))")
    print("  or MUL(m, MUL(v, v)).\n")

    create_dataset("double_ke_train", "2*KE = m*v*v", [
        ([1, 1], 1), ([1, 2], 4), ([1, 3], 9),
        ([2, 1], 2), ([2, 2], 8), ([2, 3], 18),
        ([3, 2], 12), ([3, 3], 27), ([4, 2], 16),
        ([5, 2], 20), ([2, 5], 50),
    ])
    result = train_function("DOUBLE_KE", "double_ke_train", max_depth=3, num_epochs=40)

    subsection("Verifying DOUBLE_KE on held-out values")
    create_dataset("double_ke_test", "2*KE test set", [
        ([3, 4], 48), ([4, 3], 36), ([5, 3], 45), ([1, 7], 49),
    ])
    evaluate("double_ke_test")

    # --- Step 4: Teach FORCE ---
    section("STEP 4: Teaching FORCE(m, a) = m * a")
    print("  Newton's second law. This is just MUL with a physics name.")
    print("  Trivial, but it tests that the system recognizes the pattern.\n")

    create_dataset("force_train", "F = m*a", [
        ([1, 1], 1), ([2, 3], 6), ([3, 4], 12),
        ([5, 2], 10), ([4, 5], 20), ([10, 3], 30),
    ])
    result = train_function("FORCE", "force_train", max_depth=2, num_epochs=20)

    subsection("Verifying FORCE on held-out values")
    create_dataset("force_test", "F=ma test set", [
        ([6, 3], 18), ([7, 2], 14), ([3, 8], 24),
    ])
    evaluate("force_test")

    # --- Step 5: Teach IMPULSE ---
    section("STEP 5: Teaching IMPULSE(F, t) = F * t")
    print("  Impulse = Force * time. Again MUL, but named differently.\n")

    create_dataset("impulse_train", "J = F*t", [
        ([5, 2], 10), ([3, 4], 12), ([10, 3], 30),
        ([6, 5], 30), ([2, 7], 14), ([8, 3], 24),
    ])
    result = train_function("IMPULSE", "impulse_train", max_depth=2, num_epochs=20)

    # --- Step 6: SYNTHESIS TEST ---
    section("STEP 6: SYNTHESIS TEST — POWER(m, a, v) = m * a * v")
    print("  Mechanical power = Force * velocity = m * a * v")
    print("  This is a 3-input function the system has NEVER seen.")
    print("  It must COMPOSE existing functions to solve it.")
    print("  Possible decompositions:")
    print("    - MUL(FORCE(m, a), v)")
    print("    - MUL(m, MUL(a, v))")
    print("    - MUL(MUL(m, a), v)")
    print("  Let's see if it can figure it out.\n")

    create_dataset("power_train", "P = m*a*v (mechanical power)", [
        ([1, 1, 1], 1), ([2, 1, 1], 2), ([1, 2, 1], 2), ([1, 1, 2], 2),
        ([2, 3, 1], 6), ([2, 1, 3], 6), ([1, 2, 3], 6),
        ([2, 3, 4], 24), ([3, 2, 4], 24), ([3, 4, 2], 24),
        ([2, 5, 3], 30), ([4, 3, 2], 24), ([5, 2, 3], 30),
    ])
    result = train_function("POWER", "power_train", max_depth=3, num_epochs=50,
                            max_search_steps=15)

    subsection("Verifying POWER on held-out values")
    create_dataset("power_test", "P=m*a*v test set", [
        ([3, 3, 3], 27), ([2, 4, 5], 40), ([5, 5, 2], 50),
        ([1, 6, 4], 24), ([4, 1, 7], 28),
    ])
    evaluate("power_test")

    # --- Step 7: Final overview ---
    section("STEP 7: Final Model State")
    info = api("GET", f"/models/{MODEL}")
    if info:
        print(f"  Model: {info['model_name']}")
        print(f"  Total params: {info['total_params']:,}")
        print(f"  Vocab: {info['vocab_size']}")
        print(f"\n  All registered functions:")
        for f in info["functions"]:
            comp_str = ""
            if f.get("composition"):
                steps = []
                for c in f["composition"]:
                    args = ",".join(str(a) for a in c["args"])
                    steps.append(f"{c['func_name']}({args})")
                comp_str = f"  = {' -> '.join(steps)}"
            print(f"    [{f['id']:>2}] {f['name']:<12} arity={f['arity']}  L{f['layer']}{comp_str}")

        print(f"\n  Training history:")
        for h in info["train_history"]:
            status = "ok" if h["success"] else "FAIL"
            print(f"    {h['target']:<12} [{status}]  {h.get('elapsed_s','')}s")

    # --- Summary ---
    section("SUMMARY")
    experiments = api("GET", "/experiments")
    if experiments:
        total = len(experiments["experiments"])
        passed = sum(1 for e in experiments["experiments"]
                     if e.get("success") and e.get("model_name") == MODEL)
        print(f"  Total training runs: {total}")
        print(f"  Passed (model={MODEL}): {passed}")
        print(f"\n  Key question: Did POWER(m,a,v) = m*a*v succeed?")
        power_results = [e for e in experiments["experiments"]
                         if e.get("target") == "POWER" and e.get("model_name") == MODEL]
        if power_results:
            for r in power_results:
                print(f"    -> {'YES' if r['success'] else 'NO'}")
        else:
            print(f"    -> Not attempted")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

"""
Physics Experiment: Can NSSR learn physics equations from examples?

The system is now float-native — inputs and outputs can be any real number.
Evaluation uses R² score (like any normal DL model).

Experiment plan:

  PART 1 — Integer physics (warm-up, should get R²=1.0)
    Step 1: Check model state, bootstrap ADD/MUL if needed
    Step 2: SQUARE(x) = x*x
    Step 3: FORCE(m,a) = m*a
    Step 4: POWER(m,a,v) = m*a*v  [synthesis test]

  PART 2 — Float physics (the real test)
    Step 5: Bootstrap DIV
    Step 6: KE(m,v) = 0.5*m*v²        [kinetic energy, float output]
    Step 7: PE(m,h) = m*9.81*h         [gravitational PE, uses g=9.81]
    Step 8: TOTAL_E(m,v,h) = KE + PE   [synthesis: compose KE and PE]

Usage:
    1. Start the server:  cd backend && python server.py
    2. Run this script:   python experiment_physics.py
"""

import sys
import time
import requests

BASE = "http://localhost:8000"
MODEL = "physics"


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
    """Create a dataset. Examples: [(inputs, output), ...]"""
    resp = api("POST", "/datasets", {
        "name": name,
        "description": description,
        "examples": [[inp, out] for inp, out in examples],
    })
    if resp:
        print(f"  Dataset '{name}': {resp['num_examples']} examples")
        for inp, out in examples[:5]:
            print(f"    {inp} -> {out}")
        if len(examples) > 5:
            print(f"    ... and {len(examples) - 5} more")
    return resp


def train_function(target_name, dataset_name, max_depth=5, num_epochs=40,
                   max_search_steps=12):
    """Train the model to learn a function."""
    print(f"  Training '{target_name}' from '{dataset_name}'...")
    print(f"  (depth={max_depth}, epochs={num_epochs}, search={max_search_steps})")
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
        r2 = resp.get("r2_score", "?")
        print(f"  Result: {status}  R²={r2}  ({elapsed:.1f}s)")
        print(f"  Vocab size: {resp['vocab_size']}")
    else:
        print(f"  Training failed!")
    return resp


def evaluate(dataset_name):
    """Evaluate the model on a dataset, showing R² scores."""
    resp = api("POST", "/test/eval", {
        "model_name": MODEL,
        "dataset_name": dataset_name,
    })
    if resp:
        print(f"  Exact matches: {resp['correct']}/{resp['total']}")
        print(f"  Best R²: {resp['best_r2']}  (function: {resp['best_function']})")

        # Show all R² scores
        if resp.get("r2_scores"):
            print(f"  R² by function:")
            for fn, r2 in sorted(resp["r2_scores"].items(), key=lambda x: -x[1]):
                bar = "#" * max(0, int(r2 * 20)) if r2 > 0 else ""
                print(f"    {fn:<14} R²={r2:>9.6f}  {bar}")

        # Show per-example detail
        for d in resp["details"]:
            mark = "ok" if d["correct"] else "MISS"
            fns = d["matching_functions"] or []
            match_str = ", ".join(f["name"] for f in fns) if fns else "none"
            print(f"    {d['inputs']} -> expected {d['expected']}  [{mark}]  matched: {match_str}")
    return resp


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------

def main():
    # Preflight
    section("PREFLIGHT")
    health = api("GET", "/health")
    if not health:
        print("Server not reachable. Start it with: cd backend && python server.py")
        sys.exit(1)
    print(f"  Server up. Models: {health['loaded_models']}")

    # ===== PART 1: INTEGER PHYSICS =====

    section("PART 1: INTEGER PHYSICS")

    # Step 1: Check state, bootstrap if needed
    subsection("Step 1: Model state & bootstrap")
    info = api("GET", f"/models/{MODEL}")
    if info:
        func_names = [f["name"] for f in info["functions"]]
        print(f"  Model '{MODEL}': {info['vocab_size']} functions, {info['total_params']:,} params")
        print_functions(info["functions"])

        if "ADD" not in func_names:
            print("\n  Bootstrapping ADD...")
            create_dataset("add_train", "Addition", [
                ([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 2),
                ([2, 3], 5), ([4, 2], 6), ([5, 5], 10), ([7, 3], 10),
            ])
            train_function("ADD", "add_train", max_depth=5, num_epochs=30)

        if "MUL" not in func_names:
            print("\n  Bootstrapping MUL...")
            create_dataset("mul_train", "Multiplication", [
                ([0, 0], 0), ([0, 5], 0), ([1, 5], 5), ([2, 3], 6),
                ([3, 4], 12), ([4, 5], 20), ([5, 5], 25),
            ])
            train_function("MUL", "mul_train", max_depth=5, num_epochs=30)

        # Refresh
        info = api("GET", f"/models/{MODEL}")
        print_functions(info["functions"])

    # Step 2: SQUARE
    section("Step 2: SQUARE(x) = x * x")
    create_dataset("square_train", "x -> x*x", [
        ([1], 1), ([2], 4), ([3], 9), ([4], 16),
        ([5], 25), ([6], 36), ([7], 49),
    ])
    train_function("SQUARE", "square_train", max_depth=5, num_epochs=30)

    create_dataset("square_test", "square holdout", [
        ([8], 64), ([9], 81), ([10], 100),
    ])
    evaluate("square_test")

    # Step 3: FORCE
    section("Step 3: FORCE(m, a) = m * a")
    create_dataset("force_train", "F = m*a", [
        ([1, 1], 1), ([2, 3], 6), ([3, 4], 12),
        ([5, 2], 10), ([4, 5], 20), ([10, 3], 30),
    ])
    train_function("FORCE", "force_train", max_depth=5, num_epochs=40)

    create_dataset("force_test", "F=ma holdout", [
        ([6, 3], 18), ([7, 2], 14), ([3, 8], 24),
    ])
    evaluate("force_test")

    # Step 4: POWER synthesis
    section("Step 4: SYNTHESIS — POWER(m, a, v) = m * a * v")
    print("  3-input function. Must compose existing functions.\n")
    create_dataset("power_train", "P = m*a*v", [
        ([1, 1, 1], 1), ([2, 1, 1], 2), ([1, 2, 1], 2), ([1, 1, 2], 2),
        ([2, 3, 1], 6), ([2, 1, 3], 6), ([1, 2, 3], 6),
        ([2, 3, 4], 24), ([3, 2, 4], 24), ([3, 4, 2], 24),
        ([2, 5, 3], 30), ([4, 3, 2], 24), ([5, 2, 3], 30),
    ])
    train_function("POWER", "power_train", max_depth=5, num_epochs=50,
                   max_search_steps=15)

    create_dataset("power_test", "P=mav holdout", [
        ([3, 3, 3], 27), ([2, 4, 5], 40), ([5, 5, 2], 50),
        ([1, 6, 4], 24), ([4, 1, 7], 28),
    ])
    evaluate("power_test")

    # ===== PART 2: FLOAT PHYSICS =====

    section("PART 2: FLOAT PHYSICS")
    print("  Now the real test: equations with real-valued outputs.")
    print("  KE = 0.5*m*v²,  PE = m*g*h (g=9.81),  E = KE + PE")
    print("  The system works on floats natively — no scaling tricks.\n")

    # Step 5: Make sure DIV is usable
    section("Step 5: Verify DIV primitive")
    create_dataset("div_test", "Division check", [
        ([10, 2], 5.0), ([9, 3], 3.0), ([7, 2], 3.5),
        ([15, 4], 3.75), ([100, 3], 33.333333333333336),
    ])
    evaluate("div_test")

    # Step 6: KE = 0.5 * m * v²
    section("Step 6: KE(m, v) = 0.5 * m * v²")
    print("  Kinetic energy with real-valued outputs.\n")

    ke_train = []
    for m in [1, 2, 3, 4, 5]:
        for v in [1, 2, 3, 4, 5]:
            ke = 0.5 * m * v * v
            ke_train.append(([float(m), float(v)], ke))

    create_dataset("ke_train", "KE = 0.5*m*v^2", ke_train[:18])
    create_dataset("ke_test", "KE holdout", ke_train[18:])

    train_function("KE", "ke_train", max_depth=5, num_epochs=50,
                   max_search_steps=15)

    subsection("KE holdout evaluation")
    evaluate("ke_test")

    # Step 7: PE = m * g * h
    section("Step 7: PE(m, h) = m * 9.81 * h")
    print("  Gravitational potential energy with g = 9.81 m/s².\n")

    g = 9.81
    pe_train = []
    for m in [1, 2, 3, 4, 5]:
        for h in [1, 2, 3, 4, 5, 6]:
            pe = m * g * h
            pe_train.append(([float(m), float(h)], pe))

    create_dataset("pe_train", "PE = m*g*h (g=9.81)", pe_train[:20])
    create_dataset("pe_test", "PE holdout", pe_train[20:])

    train_function("PE", "pe_train", max_depth=5, num_epochs=50,
                   max_search_steps=15)

    subsection("PE holdout evaluation")
    evaluate("pe_test")

    # Step 8: SYNTHESIS — Total Energy = KE + PE
    section("Step 8: SYNTHESIS — E(m, v, h) = 0.5*m*v² + m*9.81*h")
    print("  Can it compose KE and PE into total mechanical energy?\n")

    energy_train = []
    for m in [1, 2, 3, 4]:
        for v in [1, 2, 3, 4]:
            for h in [1, 2, 3]:
                ke = 0.5 * m * v * v
                pe = m * g * h
                energy_train.append(([float(m), float(v), float(h)], ke + pe))

    create_dataset("energy_train", "E = KE + PE", energy_train[:30])
    create_dataset("energy_test", "Total energy holdout", energy_train[30:])

    train_function("TOTAL_E", "energy_train", max_depth=5, num_epochs=60,
                   max_search_steps=20)

    subsection("Total energy holdout evaluation")
    evaluate("energy_test")

    # ===== FINAL SUMMARY =====

    section("FINAL SUMMARY")

    # Print all functions with compositions
    info = api("GET", f"/models/{MODEL}")
    if info:
        print(f"  Model: {info['model_name']}")
        print(f"  Params: {info['total_params']:,}")
        print(f"  Vocab:  {info['vocab_size']}")
        print(f"\n  Functions:")
        for f in info["functions"]:
            comp_str = ""
            if f.get("composition"):
                steps = []
                for c in f["composition"]:
                    args = ",".join(str(a) for a in c["args"])
                    steps.append(f"{c['func_name']}({args})")
                comp_str = f"  = {' -> '.join(steps)}"
            if f.get("constants"):
                mode = f.get("const_mode", "mul")
                comp_str += f"  [k={f['constants']}, {mode}]"
            print(f"    [{f['id']:>2}] {f['name']:<14} arity={f['arity']}  L{f['layer']}{comp_str}")

        print(f"\n  Training history:")
        for h in info["train_history"]:
            status = "ok" if h["success"] else "FAIL"
            r2 = h.get("r2_score", "?")
            print(f"    {h['target']:<14} [{status}]  R²={r2}  {h.get('elapsed_s','')}s")

    # Experiment log
    experiments = api("GET", "/experiments")
    if experiments:
        all_exps = [e for e in experiments["experiments"] if e.get("model_name") == MODEL]
        passed = sum(1 for e in all_exps if e.get("success"))
        print(f"\n  Results: {passed}/{len(all_exps)} passed")
        for e in all_exps:
            r2 = e.get("r2_score", "?")
            mark = "ok" if e.get("success") else "FAIL"
            print(f"    {e['target']:<14} [{mark}]  R²={r2}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

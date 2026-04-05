"""
Noise Column Experiment: Can NSSR ignore irrelevant input columns?

Real-world datasets often have columns that are noisy, redundant, or
completely unrelated to the target. This experiment tests whether the
routing head can learn to ignore dummy columns and still discover the
correct equation.

Experiment plan:

  PART 1 — Dummy columns (zero signal)
    Step 1: Bootstrap ADD/MUL on the "noise" model
    Step 2: SQUARE(x) = x*x, but input has 3 columns [x, junk1, junk2]
    Step 3: FORCE(m,a) = m*a, but input has 4 columns [m, junk, a, junk]

  PART 2 — Noisy columns (weak/misleading signal)
    Step 4: KE(m,v) = 0.5*m*v², input has [m, noise_correlated, v]
            where noise_correlated ~ m + random (multicollinear-ish)
    Step 5: PE(m,h) = m*9.81*h, input has [junk, m, junk, h]

  PART 3 — Synthesis with noise
    Step 6: TOTAL_E(m,v,h) = KE + PE, input has [m, junk, v, junk, h]

Usage:
    1. Start the server:  cd backend && python server.py
    2. Run this script:   python experiment_noise.py
"""

import sys
import time
import random
import requests

BASE = "http://localhost:8000"
MODEL = "noise"

random.seed(42)


# ---------------------------------------------------------------------------
# Helpers (same as experiment_physics.py)
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
                   max_search_steps=15):
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
    resp = api("POST", "/test/eval", {
        "model_name": MODEL,
        "dataset_name": dataset_name,
    })
    if resp:
        print(f"  Exact matches: {resp['correct']}/{resp['total']}")
        print(f"  Best R²: {resp['best_r2']}  (function: {resp['best_function']})")

        if resp.get("r2_scores"):
            print(f"  R² by function:")
            for fn, r2 in sorted(resp["r2_scores"].items(), key=lambda x: -x[1]):
                bar = "#" * max(0, int(r2 * 20)) if r2 > 0 else ""
                print(f"    {fn:<14} R²={r2:>9.6f}  {bar}")

        for d in resp["details"]:
            mark = "ok" if d["correct"] else "MISS"
            fns = d["matching_functions"] or []
            match_str = ", ".join(f["name"] for f in fns) if fns else "none"
            print(f"    {d['inputs']} -> expected {d['expected']}  [{mark}]  matched: {match_str}")
    return resp


def junk():
    """Random junk value that has no relationship to the output."""
    return round(random.uniform(-100, 100), 2)


def noisy_correlated(val):
    """A value loosely correlated with `val` — multicollinearity trap."""
    return round(val + random.uniform(-3, 3), 2)


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

    # ===== PART 1: DUMMY COLUMNS (zero signal) =====

    section("PART 1: DUMMY COLUMNS (zero signal)")
    print("  Can the model ignore columns that are pure random noise?")

    # Step 1: Verify primitives (ADD, MUL, SUB are now built-in)
    subsection("Step 1: Verify primitives")
    info = api("GET", f"/models/{MODEL}")
    if info:
        func_names = [f["name"] for f in info["functions"]]
        print(f"  Model '{MODEL}': {info['vocab_size']} functions")
        print_functions(info["functions"])
        for needed in ("ADD", "MUL", "SUB"):
            status = "ok" if needed in func_names else "MISSING"
            print(f"  {needed}: {status}")

    # Step 2: SQUARE with dummy columns
    # Real equation: output = x * x
    # Input format: [x, junk1, junk2]  — cols 1,2 are irrelevant
    section("Step 2: SQUARE(x) = x*x  with 2 junk columns")
    print("  Input: [x, junk1, junk2]  — model must discover only col 0 matters\n")

    sq_train = []
    for x in range(1, 8):
        sq_train.append(([float(x), junk(), junk()], float(x * x)))
    create_dataset("n_sq_train", "x*x with junk cols", sq_train)
    train_function("SQUARE_N", "n_sq_train", max_depth=5, num_epochs=40,
                   max_search_steps=15)

    sq_test = []
    for x in [8, 9, 10, 11, 12]:
        sq_test.append(([float(x), junk(), junk()], float(x * x)))
    create_dataset("n_sq_test", "square holdout with junk", sq_test)
    evaluate("n_sq_test")

    # Step 3: FORCE with dummy columns scattered between real ones
    # Real equation: output = m * a
    # Input format: [m, junk1, a, junk2]  — cols 1,3 are irrelevant
    section("Step 3: FORCE(m,a) = m*a  with junk columns interleaved")
    print("  Input: [m, junk1, a, junk2]  — real data at cols 0 and 2\n")

    f_train = []
    for m in [1, 2, 3, 4, 5, 6]:
        for a in [1, 2, 3, 4, 5]:
            f_train.append(([float(m), junk(), float(a), junk()], float(m * a)))
    create_dataset("n_force_train", "F=m*a with junk", f_train[:15])
    train_function("FORCE_N", "n_force_train", max_depth=5, num_epochs=50,
                   max_search_steps=20)

    f_test = []
    for m, a in [(7, 3), (3, 8), (6, 6), (9, 2), (4, 7)]:
        f_test.append(([float(m), junk(), float(a), junk()], float(m * a)))
    create_dataset("n_force_test", "F=m*a holdout with junk", f_test)
    evaluate("n_force_test")

    # ===== PART 2: NOISY / CORRELATED COLUMNS =====

    section("PART 2: NOISY / CORRELATED COLUMNS")
    print("  Columns that are loosely correlated with real inputs.")
    print("  The model must not be fooled by multicollinearity.\n")

    # Step 4: KE with a correlated noise column
    # Real equation: KE = 0.5 * m * v²
    # Input: [m, noise~m, v]  — col 1 correlates with col 0 but isn't the signal
    section("Step 4: KE(m,v) = 0.5*m*v²  with correlated noise column")
    print("  Input: [m, noise_correlated_to_m, v]\n")

    ke_train = []
    for m in [1, 2, 3, 4, 5]:
        for v in [1, 2, 3, 4, 5]:
            ke = 0.5 * m * v * v
            ke_train.append(([float(m), noisy_correlated(m), float(v)], ke))
    create_dataset("n_ke_train", "KE with correlated noise", ke_train[:18])

    ke_test = []
    for m in [4, 5, 6]:
        for v in [4, 5, 6]:
            ke = 0.5 * m * v * v
            ke_test.append(([float(m), noisy_correlated(m), float(v)], ke))
    create_dataset("n_ke_test", "KE holdout with noise", ke_test)
    train_function("KE_N", "n_ke_train", max_depth=5, num_epochs=50,
                   max_search_steps=20)

    subsection("KE holdout evaluation")
    evaluate("n_ke_test")

    # Step 5: PE with junk columns on both sides
    # Real equation: PE = m * 9.81 * h
    # Input: [junk, m, junk, h]  — real data at cols 1 and 3
    section("Step 5: PE(m,h) = m*9.81*h  with junk on both sides")
    print("  Input: [junk, m, junk, h]\n")

    g = 9.81
    pe_train = []
    for m in [1, 2, 3, 4, 5]:
        for h in [1, 2, 3, 4, 5, 6]:
            pe = m * g * h
            pe_train.append(([junk(), float(m), junk(), float(h)], pe))
    create_dataset("n_pe_train", "PE with junk cols", pe_train[:20])

    pe_test = []
    for m in [4, 5, 6]:
        for h in [5, 6, 7, 8]:
            pe = m * g * h
            pe_test.append(([junk(), float(m), junk(), float(h)], pe))
    create_dataset("n_pe_test", "PE holdout with junk", pe_test)
    train_function("PE_N", "n_pe_train", max_depth=5, num_epochs=50,
                   max_search_steps=20)

    subsection("PE holdout evaluation")
    evaluate("n_pe_test")

    # ===== PART 3: SYNTHESIS WITH NOISE =====

    section("PART 3: SYNTHESIS WITH NOISE")
    print("  Total energy E = KE + PE, but with junk columns mixed in.")
    print("  Input: [m, junk, v, junk, h]  — real data at cols 0, 2, 4\n")

    # Step 6: TOTAL_E with noise
    section("Step 6: E(m,v,h) = 0.5*m*v² + m*9.81*h  with junk columns")

    energy_train = []
    for m in [1, 2, 3, 4]:
        for v in [1, 2, 3, 4]:
            for h in [1, 2, 3]:
                ke = 0.5 * m * v * v
                pe = m * g * h
                energy_train.append((
                    [float(m), junk(), float(v), junk(), float(h)],
                    ke + pe
                ))
    create_dataset("n_energy_train", "E=KE+PE with junk", energy_train[:30])

    energy_test = []
    for m in [3, 4, 5]:
        for v in [3, 4, 5]:
            for h in [2, 3, 4]:
                ke = 0.5 * m * v * v
                pe = m * g * h
                energy_test.append((
                    [float(m), junk(), float(v), junk(), float(h)],
                    ke + pe
                ))
    create_dataset("n_energy_test", "Total energy holdout with junk",
                   energy_test[:18])
    train_function("TOTAL_E_N", "n_energy_train", max_depth=5, num_epochs=60,
                   max_search_steps=25)

    subsection("Total energy holdout evaluation")
    evaluate("n_energy_test")

    # ===== FINAL SUMMARY =====

    section("FINAL SUMMARY")

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

    experiments = api("GET", "/experiments")
    if experiments:
        all_exps = [e for e in experiments["experiments"]
                    if e.get("model_name") == MODEL]
        passed = sum(1 for e in all_exps if e.get("success"))
        print(f"\n  Results: {passed}/{len(all_exps)} passed")
        for e in all_exps:
            r2 = e.get("r2_score", "?")
            mark = "ok" if e.get("success") else "FAIL"
            print(f"    {e['target']:<14} [{mark}]  R²={r2}")

    print("\nDone.\n")


if __name__ == "__main__":
    main()

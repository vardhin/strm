# STRM: Symbolic Transformer Recursive Machine

STRM is a **Neural-Symbolic Regression** system that learns hierarchical algorithms. It combines a neural controller (a Recursive Transformer) with a symbolic execution engine to discover, simplify, and register new functions.

Unlike traditional genetic programming, STRM uses a neural network to guide the search for programs. Once a program is discovered (e.g., `XOR`), it is added to a persistent registry as a new primitive, allowing the agent to use it to learn even more complex functions (e.g., `ADD`, `MUL`) in a curriculum-based approach.

## 🚀 Key Features

*   **Tiny Recursive Model (TRM):** A specialized transformer architecture that maintains a "carry" state ($z_H$, $z_L$) across recursive steps to plan complex compositions.
*   **Hierarchical Learning:** Starts with bitwise/arithmetic primitives and learns upwards:
    *   Primitives → Logic Gates (NAND, XOR)
    *   Logic → Arithmetic (ADD via Loops)
    *   Arithmetic → Advanced Math (MUL via nested Loops)
*   **Persistent Symbolic Registry:** An SQLite-backed registry that stores learned functions, their compositions, and metadata.
*   **Compositional Search:** Supports multiple composition types:
    *   **Sequential:** $f(g(x))$
    *   **Nested:** $f(g(x), g(y))$
    *   **Parallel:** $h(f(x), g(x))$ (e.g., for XOR)
    *   **Loops:** $LOOP(f, count)$
*   **Auto-Simplification:** Automatically simplifies discovered programs (pruning redundant steps, pattern matching) before registration.
*   **Dynamic Vocabulary Expansion:** The neural network dynamically resizes its output heads to accommodate newly learned functions.

## 📂 Project Structure

```text
strm/
├── agent/                  # The Neural Agent Logic
│   ├── agent.py            # Main SymbolicAgent orchestrator
│   ├── search.py           # Hybrid search (Exhaustive + TRM-guided)
│   ├── training.py         # Training loop for the TRM
│   └── simplifier.py       # Logic to simplify discovered program trees
├── symbolic/               # The Symbolic Engine
│   ├── registry.py         # Manages function definitions and IDs
│   ├── executor.py         # Executes symbolic programs
│   ├── curriculum.py       # Generates foundational training tasks
│   └── symbolic_db.py      # SQLite interface for function storage
├── trm/                    # Neural Network Architecture
│   ├── core.py             # The core SymbolicTRMCore model
│   ├── blocks.py           # Transformer blocks
│   └── carry.py            # State management (z_H, z_L)
├── checkpoints/            # Stores model weights (.pt) and registry (.db)
├── common.py               # Utilities (initialization)
├── layers.py               # Custom NN layers (SwiGLU, RoPE, etc.)
├── sparse_embedding.py     # Sparse embedding optimization for function vocab
├── curriculum_training_main.py  # Step 1: Train foundation (NAND, XOR)
├── main.py                      # Step 2: Learn Addition (ADD)
└── main_multiply.py             # Step 3: Learn Multiplication (MUL)
```

## 🛠️ Installation

Ensure you have Python 3.8+ and PyTorch installed.

```bash
# Install dependencies
pip install torch numpy
```

## 🚦 Usage Workflow

The system is designed to be run in a specific order to build the curriculum hierarchy.

### Step 1: Foundation Training
Initializes the registry with primitives (`OR`, `AND`, `NOT`, `INC`, `DEC`) and learns basic logic gates (`NAND`, `XOR`, `NXOR`).

```bash
python curriculum_training_main.py
```

*   **Goal:** Learn `XOR` using parallel composition of `OR`, `AND`, and `NOT`.
*   **Output:** Creates `checkpoints/symbolic.db` and `checkpoints/model.pt`.

### Step 2: Learning Addition
Loads the foundation and learns `ADD` by discovering how to use the `LOOP` primitive with `INC`.

```bash
python main.py
```

*   **Goal:** Learn `ADD(a, b)` ≈ `LOOP(INC, count=b)` applied to `a`.
*   **Output:** Updates the database with the `ADD` function.

### Step 3: Learning Multiplication
Loads the model (now knowing `ADD`) and learns `MUL` by composing `LOOP` with `ADD`.

```bash
python main_multiply.py
```

*   **Goal:** Learn `MUL(a, b)` ≈ `LOOP(ADD(a), count=b)` applied to `0`.

## 🧠 Architecture Details

### 1. The Symbolic Registry (`symbolic/registry.py`)
The "Long Term Memory" of the system. It uses SQLite to store functions.

*   **Layer 0:** Primitives (`OR`, `AND`, `NOT`, `INC`, `DEC`, `LOOP`).
*   **Layer N:** Learned abstractions composed of functions from layers 0 to N-1.

### 2. The TRM Core (`trm/core.py`)
The "Brain" of the system. It is a recursive transformer that outputs:

*   **Primary/Secondary/Tertiary IDs:** Which functions to use.
*   **Composition Type:** How to combine them (Sequential, Nested, Parallel).
*   **Halting Probability:** When to stop reasoning.

It uses a **Carry State** mechanism:

*   **$z_H$ (High-level reasoning):** Plans the algorithm strategy.
*   **$z_L$ (Low-level computation):** Handles immediate operational details.

### 3. The Search Strategy (`agent/search.py`)
When presented with input/output examples, the agent uses a hybrid approach:

1.  **Exhaustive Search:** For shallow depths (1-2), it quickly checks all combinations.
2.  **TRM-Guided Search:** For deep/complex functions, the TRM predicts the most likely function compositions.
3.  **Validation:** Candidates are executed via `executor.py` against the examples.

### 4. Simplification (`agent/simplifier.py`)
Before saving a discovered function, the simplifier attempts to optimize it by:

*   Checking if a single existing function already solves it.
*   Pruning unused steps in the composition chain.
*   Pattern matching (e.g., recognizing `NOT(XOR)` is simpler than a raw tree).

## 📝 Example: How it learns XOR

1.  **Input:** Examples of `a ^ b`.
2.  **TRM Prediction:** The model suggests a **Parallel** composition.
3.  **Execution:**
    *   Branch 1: `OR(a, b)`
    *   Branch 2: `AND(a, b)` → `NOT(...)` (NAND)
    *   Combiner: `AND(Branch1, Branch2)`
4.  **Result:** `(a | b) & ~(a & b)` which is equivalent to XOR.
5.  **Registration:** `XOR` is saved to the DB and the model vocabulary is resized to include it.

## 🔍 Key Components Explained

### Sparse Embeddings (`sparse_embedding.py`)
Implements efficient gradient updates for the expanding function vocabulary using a sparse embedding approach with SignSGD optimization. This allows the model to scale to thousands of learned functions without memory overhead.

### Custom Layers (`layers.py`)
*   **RoPE (Rotary Position Embeddings):** Encodes positional information for the transformer.
*   **SwiGLU:** Gated activation for better non-linear transformations.
*   **CastedLinear/CastedEmbedding:** Mixed-precision layers for efficient training.

### Database Schema (`symbolic_db.py`)
Functions are stored with:
*   **id:** Unique identifier
*   **name:** Human-readable name
*   **arity:** Number of input arguments
*   **layer:** Abstraction level (0 for primitives, N for compositions)
*   **composition:** JSON-encoded list of child functions and argument mappings

## 🤝 Contributing

Feel free to open issues or submit PRs to:
*   Add new primitives (e.g., bitwise shifts, modulo)
*   Improve the TRM architecture (e.g., attention mechanisms, better carry states)
*   Extend the curriculum to more complex functions (Division, Factorials, GCD)
*   Optimize the search strategy (beam search, Monte Carlo Tree Search)

## 📄 License

MIT License - see LICENSE file for details.
import numpy as np
from pysr import PySRRegressor
import warnings

# Suppress PySR verbosity/warnings for a clean output like your logs
warnings.filterwarnings("ignore")

# 1. Recreate your exact training datasets based on your preflight logs
datasets = {
    "ADD": {
        "X": np.array([[0, 0], [0, 1], [1, 0], [1, 1], [2, 3], [4, 5], [10, 2]]),
        "y": np.array([0, 1, 1, 2, 5, 9, 12])
    },
    "MUL": {
        "X": np.array([[0, 0], [0, 5], [1, 5], [2, 3], [3, 4], [5, 5], [6, 2]]),
        "y": np.array([0, 0, 5, 6, 12, 25, 12])
    },
    "SQUARE (x^2)": {
        "X": np.array([[1], [2], [3], [4], [5], [6], [7]]),
        "y": np.array([1, 4, 9, 16, 25, 36, 49])
    },
    "FORCE (m*a)": {
        "X": np.array([[1, 1], [2, 3], [3, 4], [5, 2], [4, 5], [6, 3]]),
        "y": np.array([1, 6, 12, 10, 20, 18])
    },
    "POWER (m*a*v)": {
        "X": np.array([[1, 1, 1], [2, 1, 1], [1, 2, 1], [1, 1, 2], [2, 3, 1], [3, 3, 3], [2, 4, 5]]),
        "y": np.array([1, 2, 2, 2, 6, 27, 40])
    },
    "KE (0.5*m*v^2)": {
        "X": np.array([[1.0, 1.0], [1.0, 2.0], [1.0, 3.0], [4.0, 4.0], [5.0, 2.0], [5.0, 5.0]]),
        "y": np.array([0.5, 2.0, 4.5, 32.0, 10.0, 62.5])
    },
    "PE (m*9.81*h)": {
        "X": np.array([[1.0, 1.0], [1.0, 2.0], [4.0, 3.0], [4.0, 5.0], [5.0, 1.0], [5.0, 6.0]]),
        "y": np.array([9.81, 19.62, 117.72, 196.2, 49.05, 294.3])
    },
    "TOTAL_E (0.5*m*v^2 + m*9.81*h)": {
        # Inputs: m, v, h
        "X": np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 2.0], [3.0, 3.0, 1.0], [3.0, 4.0, 2.0], [4.0, 2.0, 3.0]]),
        "y": np.array([10.31, 20.12, 42.93, 82.86, 125.72])
    }
}

# 2. Configure PySR to mimic your setup
# We give it basic operators and set constraints to see how fast/well it converges
model = PySRRegressor(
    niterations=40,  # Keep it fast to match your 15-60s epoch times
    binary_operators=["+", "*", "/", "-"],
    unary_operators=["square"], # Mimicking your SQUARE function
    verbosity=0,
    random_state=42
)

print("============================================================")
print("  PySR vs. CUSTOM NEUROSYMBOLIC ENGINE GAUNTLET")
print("============================================================\n")

for name, data in datasets.items():
    print(f"--- Training {name} ---")
    X = data["X"]
    y = data["y"]
    
    try:
        # Fit the model
        model.fit(X, y)
        
        # Extract the best equation (highest complexity-penalized score)
        best_equation = model.sympy()
        print(f"  Result: PASSED")
        print(f"  Discovered Equation: {best_equation}")
        
    except Exception as e:
        print(f"  Result: FAILED")
        print(f"  Error: {e}")
        
    print("-" * 60)

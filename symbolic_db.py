import sqlite3
import json
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class SymbolicDB:
    """SQLite-based storage for symbolic functions with automatic layer tracking."""
    
    def __init__(self, db_path: str = "checkpoints/symbolic.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        self._init_schema()
    
    def _init_schema(self):
        """Create single table for all functions."""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS functions (
                id INTEGER PRIMARY KEY,
                name TEXT UNIQUE NOT NULL,
                arity INTEGER NOT NULL,
                layer INTEGER NOT NULL,
                composition TEXT,  -- JSON: list of [child_id, [arg_indices]]
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_layer ON functions(layer)
        """)
        
        self.conn.commit()
    
    def add_primitive(self, func_id: int, name: str, arity: int) -> None:
        """Add a primitive function (layer 0)."""
        self.conn.execute(
            "INSERT OR REPLACE INTO functions (id, name, arity, layer, composition) VALUES (?, ?, ?, 0, NULL)",
            (func_id, name, arity)
        )
        self.conn.commit()
        print(f"  [DB] Added primitive {name} (id={func_id})")
    
    def add_abstraction(self, func_id: int, name: str, arity: int, 
                       composition: List[Tuple[int, List[int]]]) -> None:
        """Add a learned abstraction with automatic layer calculation.
        
        Args:
            func_id: Unique ID for this function
            name: Function name
            arity: Number of arguments
            composition: List of (child_id, arg_indices) tuples
        """
        # Calculate layer: 1 + max layer of all children
        child_ids = [child_id for child_id, _ in composition]
        
        if not child_ids:
            raise ValueError(f"Cannot add abstraction {name} with no composition")
        
        # Get max layer from children
        placeholders = ','.join('?' * len(child_ids))
        cursor = self.conn.execute(
            f"SELECT MAX(layer) as max_layer FROM functions WHERE id IN ({placeholders})",
            child_ids
        )
        result = cursor.fetchone()
        max_child_layer = result['max_layer'] if result['max_layer'] is not None else -1
        layer = max_child_layer + 1
        
        # Serialize composition to JSON
        comp_json = json.dumps([[child_id, arg_indices] for child_id, arg_indices in composition])
        
        # Insert function with composition
        self.conn.execute(
            "INSERT OR REPLACE INTO functions (id, name, arity, layer, composition) VALUES (?, ?, ?, ?, ?)",
            (func_id, name, arity, layer, comp_json)
        )
        
        # COMMIT THE TRANSACTION
        self.conn.commit()
        print(f"  [DB] Added {name} at layer {layer} with {len(composition)} terms (id={func_id})")
    
    def get_function(self, func_id: int) -> Optional[Dict[str, Any]]:
        """Get function metadata."""
        cursor = self.conn.execute(
            "SELECT * FROM functions WHERE id = ?", (func_id,)
        )
        row = cursor.fetchone()
        if not row:
            return None
        
        result = dict(row)
        # Parse composition if present
        if result['composition']:
            result['composition'] = json.loads(result['composition'])
        return result
    
    def get_composition(self, func_id: int) -> List[Tuple[int, List[int]]]:
        """Get composition for a function."""
        func = self.get_function(func_id)
        if not func or not func['composition']:
            return []
        
        # Convert from [[child_id, [args]], ...] to [(child_id, [args]), ...]
        return [(child_id, args) for child_id, args in func['composition']]
    
    def get_all_functions(self) -> List[Dict[str, Any]]:
        """Get all functions ordered by layer."""
        cursor = self.conn.execute(
            "SELECT * FROM functions ORDER BY layer, id"
        )
        results = []
        for row in cursor:
            result = dict(row)
            if result['composition']:
                result['composition'] = json.loads(result['composition'])
            results.append(result)
        return results
    
    def get_functions_by_layer(self, layer: int) -> List[Dict[str, Any]]:
        """Get all functions at a specific layer."""
        cursor = self.conn.execute(
            "SELECT * FROM functions WHERE layer = ? ORDER BY id",
            (layer,)
        )
        results = []
        for row in cursor:
            result = dict(row)
            if result['composition']:
                result['composition'] = json.loads(result['composition'])
            results.append(result)
        return results
    
    def is_primitive(self, func_id: int) -> bool:
        """Check if function is primitive (layer 0)."""
        func = self.get_function(func_id)
        return func['layer'] == 0 if func else False
    
    def print_summary(self):
        """Print a nice summary of the database."""
        print("\n" + "="*60)
        print("Symbolic Function Database")
        print("="*60)
        
        result = self.conn.execute("SELECT MAX(layer) FROM functions").fetchone()[0]
        max_layer = result if result is not None else -1
        
        if max_layer < 0:
            print("\n  (empty - no functions registered yet)")
            return
        
        for layer in range(max_layer + 1):
            functions = self.get_functions_by_layer(layer)
            if functions:
                layer_type = "PRIMITIVES" if layer == 0 else f"LAYER {layer} COMPOSITIONS"
                print(f"\n{layer_type} ({len(functions)} functions):")
                for func in functions:
                    if layer == 0:
                        print(f"  {func['id']}: {func['name']}(arity={func['arity']})")
                    else:
                        comp = func['composition']
                        child_names = [self.get_function(cid)['name'] for cid, _ in comp]
                        print(f"  {func['id']}: {func['name']}(arity={func['arity']}) = {' → '.join(child_names)}")
        
        print("="*60 + "\n")
    
    def close(self):
        """Close database connection."""
        self.conn.close()
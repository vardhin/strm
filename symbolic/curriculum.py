from typing import List, Tuple, Any, Dict
from .registry import SymbolicRegistry

class CurriculumGenerator:
    """Generates curriculum tasks to teach compositional patterns."""
    
    def __init__(self, registry: SymbolicRegistry, input_dim: int = 32):
        self.registry = registry
        self.input_dim = input_dim  # Add this to know encoding size
        
        # Get function IDs for curriculum
        self.or_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'OR')
        self.and_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'AND')
        self.not_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NOT')
        self.loop_id = registry.loop_id
    
    def generate_curriculum(self) -> List[Dict]:
        """Generate curriculum of tasks with increasing complexity."""
        tasks = []
        
        # Level 1: Identity tasks (learn single operations)
        # Use small integers that fit in input_dim bits
        max_val = min(10, 2**(self.input_dim // 2) - 1)  # Stay within representable range
        
        tasks.append({
            'name': 'OR_identity',
            'examples': [([2, 3], 2|3), ([1, 4], 1|4), ([0, 7], 0|7), ([5, 5], 5|5)],
            'target': {'primary_id': self.or_id, 'secondary_id': None, 
                       'comp_type': 'none', 'loop_count': 1}
        })
        
        tasks.append({
            'name': 'AND_identity',
            'examples': [([2, 3], 2&3), ([1, 4], 1&4), ([7, 3], 7&3), ([5, 5], 5&5)],
            'target': {'primary_id': self.and_id, 'secondary_id': None, 
                       'comp_type': 'none', 'loop_count': 1}
        })
        
        tasks.append({
            'name': 'NOT_identity',
            'examples': [([0], ~0), ([1], ~1), ([5], ~5), ([7], ~7)],
            'target': {'primary_id': self.not_id, 'secondary_id': None, 
                       'comp_type': 'none', 'loop_count': 1}
        })
        
        # Level 2: Sequential compositions
        tasks.append({
            'name': 'NOT_OR',
            'examples': [([2, 3], ~(2|3)), ([1, 4], ~(1|4)), ([0, 7], ~(0|7))],
            'target': {'primary_id': self.or_id, 'secondary_id': self.not_id,
                       'comp_type': 'sequential', 'loop_count': 1}
        })
        
        tasks.append({
            'name': 'NOT_AND',
            'examples': [([2, 3], ~(2&3)), ([1, 4], ~(1&4)), ([7, 3], ~(7&3))],
            'target': {'primary_id': self.and_id, 'secondary_id': self.not_id,
                       'comp_type': 'sequential', 'loop_count': 1}
        })
        
        # Level 3: Parallel compositions
        tasks.append({
            'name': 'parallel_identity',
            'examples': [
                ([2, 3], (2|3) & (2&3)),
                ([1, 4], (1|4) & (1&4)),
                ([7, 3], (7|3) & (7&3))
            ],
            'target': {'primary_id': self.or_id, 'secondary_id': self.and_id,
                       'tertiary_id': self.and_id, 'comp_type': 'parallel', 'loop_count': 1}
        })
        
        tasks.append({
            'name': 'parallel_or_combiner',
            'examples': [
                ([2, 3], (2&3) | (2&3)),
                ([1, 4], (1&4) | (1&4)),
                ([7, 3], (7&3) | (7&3))
            ],
            'target': {'primary_id': self.and_id, 'secondary_id': self.and_id,
                       'tertiary_id': self.or_id, 'comp_type': 'parallel', 'loop_count': 1}
        })
        
        # Level 4: LOOP compositions
        # LOOP(NOT, 2) on integers: ~~x = x (identity)
        tasks.append({
            'name': 'LOOP_NOT_2',
            'examples': [([0], ~~0), ([1], ~~1), ([5], ~~5), ([7], ~~7)],
            'target': {'primary_id': self.not_id, 'secondary_id': self.loop_id,
                       'comp_type': 'sequential', 'loop_count': 2}
        })
        
        # LOOP(NOT, 3) = NOT (since ~~~x = ~x)
        tasks.append({
            'name': 'LOOP_NOT_3',
            'examples': [([0], ~~~0), ([1], ~~~1), ([5], ~~~5), ([7], ~~~7)],
            'target': {'primary_id': self.not_id, 'secondary_id': self.loop_id,
                       'comp_type': 'sequential', 'loop_count': 3}
        })
        
        # LOOP(OR, 3) - iterated OR (accumulates bits)
        tasks.append({
            'name': 'LOOP_OR_accumulate',
            'examples': [([2, 3], 2|3|3), ([1, 4], 1|4|4), ([0, 7], 0|7|7)],
            'target': {'primary_id': self.or_id, 'secondary_id': self.loop_id,
                       'comp_type': 'sequential', 'loop_count': 3}
        })
        
        return tasks
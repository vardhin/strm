from typing import List, Tuple, Any, Dict
from .registry import SymbolicRegistry

class CurriculumGenerator:
    """Generates curriculum tasks to teach compositional patterns."""
    
    def __init__(self, registry: SymbolicRegistry, input_dim: int = 32):
        self.registry = registry
        self.input_dim = input_dim
        
        # Get function IDs for curriculum
        self.or_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'OR')
        self.and_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'AND')
        self.not_id = next(fid for fid, m in registry.metadata.items() if m['name'] == 'NOT')
        self.loop_id = registry.loop_id
    
    def generate_curriculum(self) -> List[Dict]:
        """Generate curriculum of tasks with increasing complexity."""
        tasks = []
        
        # Level 1: Identity tasks (learn single operations only)
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
        
        return tasks
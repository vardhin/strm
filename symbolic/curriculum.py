from typing import List, Tuple, Any, Dict
from .registry import SymbolicRegistry

class CurriculumGenerator:
    """Generates curriculum tasks to teach compositional patterns."""
    
    def __init__(self, registry: SymbolicRegistry):
        self.registry = registry
    
    def generate_tasks(self) -> List[Tuple[str, List[Tuple[List[int], Any]], Dict]]:
        """Generate synthetic tasks to teach compositional search."""
        tasks = []
        
        # Task 1: Basic single-function usage
        tasks.append((
            "OR_identity",
            [([0, 0], 0), ([0, 1], 1), ([1, 0], 1), ([1, 1], 1)],
            {'primary_id': 0, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        tasks.append((
            "AND_identity", 
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 1, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        tasks.append((
            "NOT_identity",
            [([0], 1), ([1], 0)],
            {'primary_id': 2, 'secondary_id': 0, 'tertiary_id': None, 'comp_type': 'none', 'step': 0}
        ))
        
        # Task 2: Sequential composition
        tasks.append((
            "NOT_OR",  # NOR
            [([0, 0], 1), ([0, 1], 0), ([1, 0], 0), ([1, 1], 0)],
            {'primary_id': 0, 'secondary_id': 2, 'tertiary_id': None, 'comp_type': 'sequential', 'step': 0}
        ))
        
        tasks.append((
            "NOT_AND",  # NAND
            [([0, 0], 1), ([0, 1], 1), ([1, 0], 1), ([1, 1], 0)],
            {'primary_id': 1, 'secondary_id': 2, 'tertiary_id': None, 'comp_type': 'sequential', 'step': 0}
        ))
        
        # Task 3: Parallel composition with AND combiner
        tasks.append((
            "parallel_identity",
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 0, 'secondary_id': 1, 'tertiary_id': 1, 'comp_type': 'parallel', 'step': 0}
        ))
        
        # Task 4: Parallel composition with OR combiner
        tasks.append((
            "parallel_or_combiner",
            [([0, 0], 0), ([0, 1], 0), ([1, 0], 0), ([1, 1], 1)],
            {'primary_id': 1, 'secondary_id': 1, 'tertiary_id': 0, 'comp_type': 'parallel', 'step': 0}
        ))
        
        return tasks
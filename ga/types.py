# ============================================================================
# GA DATA TYPES
# ============================================================================
# Shared data types for Genetic Algorithm modules
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from dataclasses import dataclass
from typing import Dict, Any


# ============================================================================
# INDIVIDUAL CLASS
# ============================================================================

@dataclass
class Individual: 
    """Represents a single solution (chromosome)."""
    
    chromosome: Dict[str, Any]
    fitness: float = -np.inf
    age: int = 0
    generation: int = 0
    evaluated: bool = False
    
    def __hash__(self):
        """Make individual hashable for caching."""
        import json
        
        # Convert NumPy types to native Python types
        def convert(val):
            if isinstance(val, (np.integer, np.int32, np.int64)):
                return int(val)
            elif isinstance(val, (np.floating, np.float32, np.float64)):
                return float(val)
            elif isinstance(val, np.ndarray):
                return val.tolist()
            return val
        
        serializable = {k: convert(v) for k, v in self.chromosome.items()}
        return hash(json.dumps(serializable, sort_keys=True))


# ============================================================================
# GENERATION STATS CLASS
# ============================================================================

@dataclass
class GenerationStats: 
    """Statistics for a single generation."""
    
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    diversity: float
    evaluation_time: float
    total_time: float
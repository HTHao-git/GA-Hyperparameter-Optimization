# ============================================================================
# TIMING HELPER
# ============================================================================
# Helper functions for consistent timing across optimizers
# ============================================================================

import time
from typing import Dict, Any, Callable
from functools import wraps


def time_function(func: Callable) -> Callable:
    """Decorator to time a function execution."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        
        # If result is a tuple, add timing to last element if it's a dict
        if isinstance(result, tuple) and isinstance(result[-1], dict):
            result_list = list(result)
            result_list[-1]['elapsed_time'] = elapsed
            return tuple(result_list)
        
        return result, {'elapsed_time': elapsed}
    
    return wrapper


def format_time(seconds: float) -> str:
    """Format seconds into human-readable string."""
    if seconds < 60:
        return f"{seconds:.2f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.2f}m ({seconds:.1f}s)"
    else:
        hours = seconds / 3600
        return f"{hours:.2f}h ({seconds:.1f}s)"


def add_timing_to_results(results: Dict[str, Any], 
                          optimization_time: float,
                          evaluation_time: float) -> Dict[str, Any]: 
    """Add timing information to results dictionary."""
    
    results['timing'] = {
        'optimization_time_seconds': float(optimization_time),
        'evaluation_time_seconds': float(evaluation_time),
        'total_time_seconds':  float(optimization_time + evaluation_time),
        'optimization_time_formatted': format_time(optimization_time),
        'evaluation_time_formatted': format_time(evaluation_time),
        'total_time_formatted': format_time(optimization_time + evaluation_time)
    }
    
    return results
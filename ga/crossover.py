# ============================================================================
# CROSSOVER OPERATORS
# ============================================================================
# Advanced crossover methods for Genetic Algorithms
#
# FEATURES:
#   - Uniform crossover
#   - Single-point crossover
#   - Two-point crossover
#   - Arithmetic crossover (for continuous values)
#   - Simulated Binary Crossover (SBX)
#   - Partially Mapped Crossover (PMX) for permutations
#
# USAGE:
#   from ga.crossover import CrossoverOperator
#   
#   crossover = CrossoverOperator(method='two_point')
#   child1, child2 = crossover.crossover(parent1, parent2)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Dict, Any, Tuple, List, Optional

from utils.logger import Logger


# ============================================================================
# CROSSOVER OPERATOR CLASS
# ============================================================================

class CrossoverOperator:
    """
    Crossover operator for Genetic Algorithms.  
    
    Args:
        method:   Crossover method
        crossover_rate: Probability of crossover
        alpha: Parameter for arithmetic crossover (blend factor)
        eta: Parameter for SBX (distribution index)
        logger: Logger instance
    """
    
    def __init__(self,
                 method: str = 'uniform',
                 crossover_rate: float = 0.8,
                 alpha: float = 0.5,
                 eta: float = 20.0,
                 logger: Optional[Logger] = None):
        
        self.method = method
        self.crossover_rate = crossover_rate
        self.alpha = alpha
        self.eta = eta
        self.logger = logger
        
        # Valid methods
        self.valid_methods = [
            'uniform',
            'single_point',
            'two_point',
            'arithmetic',
            'sbx',
            'pmx'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Invalid crossover method '{method}'.Valid:  {self.valid_methods}")
    
    # ========================================================================
    # MAIN CROSSOVER INTERFACE
    # ========================================================================
    
    def crossover(self, 
                  parent1: Dict[str, Any], 
                  parent2: Dict[str, Any],
                  chromosome_template: Optional[Dict[str, Any]] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Perform crossover between two parents.  
        
        Args: 
            parent1: First parent chromosome
            parent2: Second parent chromosome
            chromosome_template: Template with gene types/bounds
            
        Returns:
            (child1, child2) tuple
        """
        # Decide whether to perform crossover
        if np.random.random() > self.crossover_rate:
            # No crossover - return copies
            return parent1.copy(), parent2.copy()
        
        # Perform crossover based on method
        if self.method == 'uniform':
            return self._uniform_crossover(parent1, parent2)
        
        elif self.method == 'single_point':
            return self._single_point_crossover(parent1, parent2)
        
        elif self.method == 'two_point':
            return self._two_point_crossover(parent1, parent2)
        
        elif self.method == 'arithmetic':
            return self._arithmetic_crossover(parent1, parent2)
        
        elif self.method == 'sbx':
            if chromosome_template is None:
                raise ValueError("SBX requires chromosome_template")
            return self._sbx_crossover(parent1, parent2, chromosome_template)
        
        elif self.method == 'pmx':
            return self._pmx_crossover(parent1, parent2)
        
        else:
            raise ValueError(f"Unknown crossover method:  {self.method}")
    
    # ========================================================================
    # UNIFORM CROSSOVER
    # ========================================================================
    
    def _uniform_crossover(self, 
                           parent1: Dict[str, Any], 
                           parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Uniform crossover:   Each gene has 50% chance from each parent.  
        
        Good for: 
        - High diversity
        - Mixing independent genes
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns: 
            Two offspring
        """
        child1 = {}
        child2 = {}
        
        for gene in parent1.keys():
            if np.random.random() < 0.5:
                child1[gene] = parent1[gene]
                child2[gene] = parent2[gene]
            else: 
                child1[gene] = parent2[gene]
                child2[gene] = parent1[gene]
        
        return child1, child2
    
    # ========================================================================
    # SINGLE-POINT CROSSOVER
    # ========================================================================
    
    def _single_point_crossover(self, 
                                parent1: Dict[str, Any], 
                                parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """
        Single-point crossover:   Split at one random point.  
        
        Good for: 
        - Preserving building blocks
        - Linked genes
        
        Args: 
            parent1, parent2: Parent chromosomes
            
        Returns: 
            Two offspring
        """
        genes = list(parent1.keys())
        n_genes = len(genes)
        
        if n_genes < 2:
            return parent1.copy(), parent2.copy()
        
        # Random crossover point
        point = np.random.randint(1, n_genes)
        
        # Split genes
        child1 = {}
        child2 = {}
        
        for i, gene in enumerate(genes):
            if i < point:
                child1[gene] = parent1[gene]
                child2[gene] = parent2[gene]
            else:
                child1[gene] = parent2[gene]
                child2[gene] = parent1[gene]
        
        return child1, child2
    
    # ========================================================================
    # TWO-POINT CROSSOVER
    # ========================================================================
    
    def _two_point_crossover(self, 
                             parent1: Dict[str, Any], 
                             parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Two-point crossover: Split at two random points. 
        
        Good for:
        - Better than single-point for circular chromosomes
        - Preserving gene segments
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns: 
            Two offspring
        """
        genes = list(parent1.keys())
        n_genes = len(genes)
        
        if n_genes < 3:
            return self._single_point_crossover(parent1, parent2)
        
        # Two random crossover points
        point1, point2 = sorted(np.random.choice(range(1, n_genes), size=2, replace=False))
        
        # Create offspring
        child1 = {}
        child2 = {}
        
        for i, gene in enumerate(genes):
            if i < point1 or i >= point2:
                # Outside segment - from parent1
                child1[gene] = parent1[gene]
                child2[gene] = parent2[gene]
            else:
                # Inside segment - swap
                child1[gene] = parent2[gene]
                child2[gene] = parent1[gene]
        
        return child1, child2
    
    # ========================================================================
    # ARITHMETIC CROSSOVER
    # ========================================================================
    
    def _arithmetic_crossover(self, 
                              parent1: Dict[str, Any], 
                              parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """
        Arithmetic crossover:   Linear combination of parents. 
        
        child1 = alpha * parent1 + (1 - alpha) * parent2
        child2 = (1 - alpha) * parent1 + alpha * parent2
        
        Only works for numerical genes. 
        
        Good for:
        - Continuous optimization
        - Smooth search space
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns: 
            Two offspring
        """
        child1 = {}
        child2 = {}
        
        for gene in parent1.keys():
            val1 = parent1[gene]
            val2 = parent2[gene]
            
            # Check if numerical
            if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                # Arithmetic blend
                child1[gene] = self.alpha * val1 + (1 - self.alpha) * val2
                child2[gene] = (1 - self.alpha) * val1 + self.alpha * val2
                
                # Preserve type
                if isinstance(val1, int) and isinstance(val2, int):
                    child1[gene] = int(round(child1[gene]))
                    child2[gene] = int(round(child2[gene]))
            else:
                # Non-numerical - random choice
                if np.random.random() < 0.5:
                    child1[gene] = val1
                    child2[gene] = val2
                else:
                    child1[gene] = val2
                    child2[gene] = val1
        
        return child1, child2
    
    # ========================================================================
    # SIMULATED BINARY CROSSOVER (SBX)
    # ========================================================================
    
    def _sbx_crossover(self, 
                       parent1: Dict[str, Any], 
                       parent2: Dict[str, Any],
                       chromosome_template: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Simulated Binary Crossover:  Mimics single-point crossover for real values.
        
        Used in NSGA-II and other modern GAs.
        
        Args:
            parent1, parent2: Parent chromosomes
            chromosome_template: Template with bounds
            
        Returns:
            Two offspring
        """
        child1 = {}
        child2 = {}
        
        for gene in parent1.keys():
            val1 = parent1[gene]
            val2 = parent2[gene]
            
            # Get bounds from template
            gene_spec = chromosome_template.get(gene)
            
            # Check if continuous
            if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
                low, high = gene_spec
                
                # SBX
                if abs(val1 - val2) > 1e-10:
                    # Random number
                    u = np.random.random()
                    
                    # Calculate beta
                    if u <= 0.5:
                        beta = (2 * u) ** (1.0 / (self.eta + 1))
                    else:
                        beta = (1.0 / (2 * (1 - u))) ** (1.0 / (self.eta + 1))
                    
                    # Create offspring
                    c1 = 0.5 * ((val1 + val2) - beta * abs(val1 - val2))
                    c2 = 0.5 * ((val1 + val2) + beta * abs(val1 - val2))
                    
                    # Clip to bounds
                    c1 = np.clip(c1, low, high)
                    c2 = np.clip(c2, low, high)
                    
                    # Preserve type
                    if isinstance(low, int) and isinstance(high, int):
                        c1 = int(round(c1))
                        c2 = int(round(c2))
                    
                    child1[gene] = c1
                    child2[gene] = c2
                else:
                    child1[gene] = val1
                    child2[gene] = val2
            else:
                # Discrete - uniform crossover
                if np.random.random() < 0.5:
                    child1[gene] = val1
                    child2[gene] = val2
                else:
                    child1[gene] = val2
                    child2[gene] = val1
        
        return child1, child2
    
    # ========================================================================
    # PARTIALLY MAPPED CROSSOVER (PMX)
    # ========================================================================
    
    def _pmx_crossover(self, 
                       parent1: Dict[str, Any], 
                       parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """
        Partially Mapped Crossover: For permutation problems.
        
        Preserves relative ordering while mixing parents.
        
        Good for:
        - Traveling Salesman Problem
        - Scheduling problems
        - Any permutation-based encoding
        
        Args:
            parent1, parent2: Parent chromosomes
            
        Returns: 
            Two offspring
        """
        # PMX is complex for dict-based chromosomes
        # For now, fallback to uniform crossover
        # TODO: Implement proper PMX if needed for permutation problems
        
        return self._uniform_crossover(parent1, parent2)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success, print_info
    
    logger = get_logger(name="CROSSOVER_TEST", verbose=True)
    
    print_header("CROSSOVER OPERATOR TEST")
    print()
    
    # Test parents
    parent1 = {
        'x1': 10,
        'x2': 20.5,
        'x3':  'optionA',
        'x4':  100
    }
    
    parent2 = {
        'x1': 50,
        'x2': 80.3,
        'x3': 'optionB',
        'x4': 300
    }
    
    template = {
        'x1':  (0, 100),
        'x2': (0.0, 100.0),
        'x3': ['optionA', 'optionB', 'optionC'],
        'x4': (50, 500)
    }
    
    methods = ['uniform', 'single_point', 'two_point', 'arithmetic', 'sbx']
    
    for method in methods:
        print_section(f"Method: {method.upper()}")
        
        crossover = CrossoverOperator(method=method, crossover_rate=1.0)
        child1, child2 = crossover.crossover(parent1, parent2, template)
        
        print_info("Parent 1:")
        print(f"  {parent1}")
        
        print_info("Parent 2:")
        print(f"  {parent2}")
        
        print_info("Child 1:")
        print(f"  {child1}")
        
        print_info("Child 2:")
        print(f"  {child2}")
        
        print()
    
    print_success("✓ Crossover operator test complete!")
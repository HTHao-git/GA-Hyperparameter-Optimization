# ============================================================================
# MUTATION OPERATORS
# ============================================================================
# Advanced mutation methods for Genetic Algorithms
#
# FEATURES:
#   - Uniform mutation
#   - Gaussian mutation
#   - Polynomial mutation
#   - Adaptive mutation
#   - Boundary mutation
#   - Non-uniform mutation
#
# USAGE:
#   from ga.mutation import MutationOperator
#   
#   mutator = MutationOperator(method='gaussian')
#   mutated = mutator.mutate(chromosome, template)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Dict, Any, Optional

from utils.logger import Logger


# ============================================================================
# MUTATION OPERATOR CLASS
# ============================================================================

class MutationOperator: 
    """
    Mutation operator for Genetic Algorithms. 
    
    Args:
        method:  Mutation method
        mutation_rate: Probability of mutation
        sigma: Standard deviation for Gaussian mutation
        eta: Distribution index for polynomial mutation
        generation:  Current generation (for adaptive methods)
        max_generations: Maximum generations (for adaptive methods)
        logger: Logger instance
    """
    
    def __init__(self,
                 method: str = 'gaussian',
                 mutation_rate:  float = 0.1,
                 sigma: float = 0.1,
                 eta: float = 20.0,
                 generation: int = 0,
                 max_generations: int = 100,
                 logger: Optional[Logger] = None):
        
        self.method = method
        self.mutation_rate = mutation_rate
        self.sigma = sigma
        self.eta = eta
        self.generation = generation
        self.max_generations = max_generations
        self.logger = logger
        
        # Valid methods
        self.valid_methods = [
            'uniform',
            'gaussian',
            'polynomial',
            'adaptive',
            'boundary',
            'non_uniform'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Invalid mutation method '{method}'.Valid:  {self.valid_methods}")
    
    # ========================================================================
    # MAIN MUTATION INTERFACE
    # ========================================================================
    
    def mutate(self, 
               chromosome: Dict[str, Any],
               chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate a chromosome. 
        
        Args:
            chromosome:  Chromosome to mutate
            chromosome_template: Template with gene types/bounds
            
        Returns:
            Mutated chromosome
        """
        # Decide whether to mutate
        if np.random.random() > self.mutation_rate:
            return chromosome.copy()
        
        # Perform mutation based on method
        if self.method == 'uniform':
            return self._uniform_mutation(chromosome, chromosome_template)
        
        elif self.method == 'gaussian':
            return self._gaussian_mutation(chromosome, chromosome_template)
        
        elif self.method == 'polynomial':
            return self._polynomial_mutation(chromosome, chromosome_template)
        
        elif self.method == 'adaptive':
            return self._adaptive_mutation(chromosome, chromosome_template)
        
        elif self.method == 'boundary':
            return self._boundary_mutation(chromosome, chromosome_template)
        
        elif self.method == 'non_uniform':
            return self._non_uniform_mutation(chromosome, chromosome_template)
        
        else:
            raise ValueError(f"Unknown mutation method: {self.method}")
    
    def update_generation(self, generation: int):
        """Update current generation for adaptive methods."""
        self.generation = generation
    
    # ========================================================================
    # UNIFORM MUTATION
    # ========================================================================
    
    def _uniform_mutation(self, 
                          chromosome: Dict[str, Any],
                          chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Uniform mutation:  Replace gene with random value from valid range.
        
        Good for: 
        - Exploration
        - Escaping local optima
        
        Args:
            chromosome: Original chromosome
            chromosome_template: Template with valid values
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Select random gene to mutate
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        
        if isinstance(gene_spec, list):
            # Discrete:  random choice
            mutated[gene] = np.random.choice(gene_spec)
        
        elif isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            # Continuous: random value in range
            low, high = gene_spec
            
            if isinstance(low, int) and isinstance(high, int):
                mutated[gene] = np.random.randint(low, high + 1)
            else:
                mutated[gene] = np.random.uniform(low, high)
        
        return mutated
    
    # ========================================================================
    # GAUSSIAN MUTATION
    # ========================================================================
    
    def _gaussian_mutation(self, 
                           chromosome: Dict[str, Any],
                           chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Gaussian mutation: Add Gaussian noise to continuous genes.
        
        mutation = current_value + N(0, sigma * range)
        
        Good for: 
        - Fine-tuning
        - Local search
        
        Args:
            chromosome: Original chromosome
            chromosome_template: Template with bounds
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Select random gene
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        current_value = mutated[gene]
        
        if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            # Continuous gene
            low, high = gene_spec
            gene_range = high - low
            
            # Add Gaussian noise
            noise = np.random.normal(0, self.sigma * gene_range)
            new_value = current_value + noise
            
            # Clip to bounds
            new_value = np.clip(new_value, low, high)
            
            # Preserve type
            if isinstance(low, int) and isinstance(high, int):
                new_value = int(round(new_value))
            
            mutated[gene] = new_value
        
        else:
            # Discrete gene - fallback to uniform
            if isinstance(gene_spec, list):
                mutated[gene] = np.random.choice(gene_spec)
        
        return mutated
    
    # ========================================================================
    # POLYNOMIAL MUTATION
    # ========================================================================
    
    def _polynomial_mutation(self, 
                             chromosome: Dict[str, Any],
                             chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Polynomial mutation: Used in NSGA-II.
        
        Creates small changes with high probability, 
        large changes with low probability.
        
        Args:
            chromosome: Original chromosome
            chromosome_template: Template with bounds
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Select random gene
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        
        if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            low, high = gene_spec
            current_value = mutated[gene]
            
            # Random number
            u = np.random.random()
            
            # Calculate delta
            if u < 0.5:
                delta = (2 * u) ** (1.0 / (self.eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1.0 / (self.eta + 1))
            
            # Apply mutation
            new_value = current_value + delta * (high - low)
            new_value = np.clip(new_value, low, high)
            
            # Preserve type
            if isinstance(low, int) and isinstance(high, int):
                new_value = int(round(new_value))
            
            mutated[gene] = new_value
        
        else:
            # Discrete - uniform
            if isinstance(gene_spec, list):
                mutated[gene] = np.random.choice(gene_spec)
        
        return mutated
    
    # ========================================================================
    # ADAPTIVE MUTATION
    # ========================================================================
    
    def _adaptive_mutation(self, 
                           chromosome: Dict[str, Any],
                           chromosome_template:  Dict[str, Any]) -> Dict[str, Any]:
        """
        Adaptive mutation: Mutation strength decreases over generations.
        
        Early generations: Large mutations (exploration)
        Late generations: Small mutations (exploitation)
        
        sigma_adaptive = sigma * (1 - generation / max_generations)
        
        Args:
            chromosome: Original chromosome
            chromosome_template: Template with bounds
            
        Returns:
            Mutated chromosome
        """
        # Calculate adaptive sigma
        progress = self.generation / max(self.max_generations, 1)
        adaptive_sigma = self.sigma * (1 - progress)
        
        # Use Gaussian mutation with adaptive sigma
        mutated = chromosome.copy()
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        
        if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            low, high = gene_spec
            current_value = mutated[gene]
            gene_range = high - low
            
            # Adaptive Gaussian noise
            noise = np.random.normal(0, adaptive_sigma * gene_range)
            new_value = current_value + noise
            new_value = np.clip(new_value, low, high)
            
            if isinstance(low, int) and isinstance(high, int):
                new_value = int(round(new_value))
            
            mutated[gene] = new_value
        
        else:
            if isinstance(gene_spec, list):
                mutated[gene] = np.random.choice(gene_spec)
        
        return mutated
    
    # ========================================================================
    # BOUNDARY MUTATION
    # ========================================================================
    
    def _boundary_mutation(self, 
                           chromosome:  Dict[str, Any],
                           chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Boundary mutation: Set gene to min or max value.
        
        Good for:
        - Exploring extreme values
        - Constraint boundaries
        
        Args:
            chromosome: Original chromosome
            chromosome_template: Template with bounds
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        
        if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            low, high = gene_spec
            
            # Set to boundary (50% chance each)
            if np.random.random() < 0.5:
                mutated[gene] = low
            else:
                mutated[gene] = high
        
        elif isinstance(gene_spec, list):
            # Discrete - choose first or last
            if np.random.random() < 0.5:
                mutated[gene] = gene_spec[0]
            else:
                mutated[gene] = gene_spec[-1]
        
        return mutated
    
    # ========================================================================
    # NON-UNIFORM MUTATION
    # ========================================================================
    
    def _non_uniform_mutation(self, 
                              chromosome: Dict[str, Any],
                              chromosome_template: Dict[str, Any]) -> Dict[str, Any]:
        """
        Non-uniform mutation: Mutation range decreases non-linearly.
        
        delta = (high - low) * (1 - r^((1 - t/T)^b))
        where:
        - r:  random [0,1]
        - t: current generation
        - T: max generations
        - b: parameter (usually 2-5)
        
        Args: 
            chromosome: Original chromosome
            chromosome_template: Template with bounds
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        gene = np.random.choice(list(mutated.keys()))
        gene_spec = chromosome_template[gene]
        
        if isinstance(gene_spec, tuple) and len(gene_spec) == 2:
            low, high = gene_spec
            current_value = mutated[gene]
            
            # Non-uniform parameter
            b = 3.0
            
            # Progress
            progress = self.generation / max(self.max_generations, 1)
            
            # Random value
            r = np.random.random()
            
            # Calculate delta
            delta = (high - low) * (1 - r ** ((1 - progress) ** b))
            
            # Apply mutation (+ or -)
            if np.random.random() < 0.5:
                new_value = current_value + delta
            else:
                new_value = current_value - delta
            
            new_value = np.clip(new_value, low, high)
            
            if isinstance(low, int) and isinstance(high, int):
                new_value = int(round(new_value))
            
            mutated[gene] = new_value
        
        else:
            if isinstance(gene_spec, list):
                mutated[gene] = np.random.choice(gene_spec)
        
        return mutated


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success, print_info
    
    logger = get_logger(name="MUTATION_TEST", verbose=True)
    
    print_header("MUTATION OPERATOR TEST")
    print()
    
    # Test chromosome
    chromosome = {
        'x1': 50,
        'x2': 50.0,
        'x3': 'optionB',
        'x4': 200
    }
    
    template = {
        'x1':  (0, 100),
        'x2': (0.0, 100.0),
        'x3': ['optionA', 'optionB', 'optionC'],
        'x4': (50, 500)
    }
    
    methods = ['uniform', 'gaussian', 'polynomial', 'adaptive', 'boundary', 'non_uniform']
    
    for method in methods:
        print_section(f"Method: {method.upper()}")
        
        mutator = MutationOperator(
            method=method, 
            mutation_rate=1.0,  # Always mutate for testing
            generation=50,
            max_generations=100
        )
        
        print_info("Original:")
        print(f"  {chromosome}")
        
        # Mutate 5 times to show variation
        print_info("Mutations:")
        for i in range(5):
            mutated = mutator.mutate(chromosome, template)
            print(f"  {i+1}.{mutated}")
        
        print()
    
    print_success("✓ Mutation operator test complete!")
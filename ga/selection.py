# ============================================================================
# SELECTION OPERATORS
# ============================================================================
# Advanced selection methods for Genetic Algorithms
#
# FEATURES:
#   - Tournament selection
#   - Roulette wheel selection
#   - Rank-based selection
#   - Boltzmann selection
#   - Stochastic Universal Sampling (SUS)
#   - Truncation selection
#   - Linear ranking selection
#
# USAGE:
#   from ga.selection import SelectionOperator
#   
#   selector = SelectionOperator(method='tournament')
#   parent = selector.select(population)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import List, Optional
from dataclasses import dataclass
from utils.colors import print_info

from ga.types import Individual
from utils.logger import Logger


# ============================================================================
# SELECTION OPERATOR CLASS
# ============================================================================

class SelectionOperator:
    """
    Selection operator for Genetic Algorithms. 
    
    Args:
        method:  Selection method
        tournament_size: Size of tournament (for tournament selection)
        temperature: Temperature parameter (for Boltzmann selection)
        truncation_threshold: Percentage to keep (for truncation selection)
        pressure: Selection pressure (for linear ranking)
        logger: Logger instance
    """
    
    def __init__(self,
                 method: str = 'tournament',
                 tournament_size: int = 3,
                 temperature: float = 1.0,
                 truncation_threshold: float = 0.5,
                 pressure: float = 1.5,
                 logger: Optional[Logger] = None):
        
        self.method = method
        self.tournament_size = tournament_size
        self.temperature = temperature
        self.truncation_threshold = truncation_threshold
        self.pressure = pressure
        self.logger = logger
        
        # Valid methods
        self.valid_methods = [
            'tournament',
            'roulette',
            'rank',
            'boltzmann',
            'sus',
            'truncation',
            'linear_ranking'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Invalid selection method '{method}'.Valid:  {self.valid_methods}")
    
    # ========================================================================
    # MAIN SELECTION INTERFACE
    # ========================================================================
    
    def select(self, population: List[Individual]) -> Individual:
        """
        Select one individual from population. 
        
        Args:
            population: List of individuals
            
        Returns:
            Selected individual
        """
        if self.method == 'tournament': 
            return self._tournament_selection(population)
        
        elif self.method == 'roulette':
            return self._roulette_selection(population)
        
        elif self.method == 'rank':
            return self._rank_selection(population)
        
        elif self.method == 'boltzmann': 
            return self._boltzmann_selection(population)
        
        elif self.method == 'sus':
            # SUS selects multiple, but we return one
            return self._sus_selection(population, n_select=1)[0]
        
        elif self.method == 'truncation': 
            return self._truncation_selection(population)
        
        elif self.method == 'linear_ranking':
            return self._linear_ranking_selection(population)
        
        else:
            raise ValueError(f"Unknown selection method: {self.method}")
    
    def select_multiple(self, population: List[Individual], n:  int) -> List[Individual]: 
        """
        Select multiple individuals. 
        
        Args:
            population: List of individuals
            n: Number to select
            
        Returns: 
            List of selected individuals
        """
        if self.method == 'sus':
            return self._sus_selection(population, n)
        else:
            # Use single selection repeatedly
            return [self.select(population) for _ in range(n)]
    
    # ========================================================================
    # TOURNAMENT SELECTION
    # ========================================================================
    
    def _tournament_selection(self, population: List[Individual]) -> Individual:
        """
        Tournament selection:  Select best from random subset.
        
        Advantages:
        - Efficient
        - No need to sort entire population
        - Selection pressure controlled by tournament size
        
        Args: 
            population: Population to select from
            
        Returns:
            Winner of tournament
        """
        tournament = np.random.choice(
            population,
            size=min(self.tournament_size, len(population)),
            replace=False
        )
        
        return max(tournament, key=lambda x:  x.fitness)
    
    # ========================================================================
    # ROULETTE WHEEL SELECTION
    # ========================================================================
    
    def _roulette_selection(self, population:  List[Individual]) -> Individual:
        """
        Roulette wheel selection:  Probability proportional to fitness.
        
        Advantages:
        - Simple
        - All individuals have chance
        
        Disadvantages:
        - Can have premature convergence if one individual dominates
        - Doesn't work well with negative fitness
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        fitnesses = np.array([ind.fitness for ind in population])
        
        # Shift to positive if needed
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min()
        
        # Handle all-zero case
        if fitnesses.sum() == 0:
            return np.random.choice(population)
        
        # Calculate probabilities
        probabilities = fitnesses / fitnesses.sum()
        
        return np.random.choice(population, p=probabilities)
    
    # ========================================================================
    # RANK-BASED SELECTION
    # ========================================================================
    
    def _rank_selection(self, population:  List[Individual]) -> Individual:
        """
        Rank-based selection:  Probability based on rank, not raw fitness.
        
        Advantages:
        - Works with negative fitness
        - Prevents premature convergence
        - More stable than roulette
        
        Args: 
            population: Population to select from
            
        Returns:
            Selected individual
        """
        # Sort by fitness
        sorted_pop = sorted(population, key=lambda x: x.fitness)
        
        # Assign ranks (1 to N)
        ranks = np.arange(1, len(sorted_pop) + 1)
        probabilities = ranks / ranks.sum()
        
        return np.random.choice(sorted_pop, p=probabilities)
    
    # ========================================================================
    # BOLTZMANN SELECTION
    # ========================================================================
    
    def _boltzmann_selection(self, population: List[Individual]) -> Individual:
        """
        Boltzmann selection:  Temperature-based selection.
        
        Uses Boltzmann distribution: 
        P(i) ∝ exp(fitness_i / T)
        
        Temperature controls selection pressure:
        - High T: More random (exploration)
        - Low T: More greedy (exploitation)
        
        Typically start with high T and decrease over time (simulated annealing).
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        fitnesses = np.array([ind.fitness for ind in population])
        
        # Shift to avoid numerical issues
        fitnesses = fitnesses - fitnesses.min()
        
        # Boltzmann probabilities
        exp_values = np.exp(fitnesses / self.temperature)
        probabilities = exp_values / exp_values.sum()
        
        # Handle numerical errors
        if np.any(np.isnan(probabilities)):
            # Fallback to uniform
            probabilities = np.ones(len(population)) / len(population)
        
        return np.random.choice(population, p=probabilities)
    
    # ========================================================================
    # STOCHASTIC UNIVERSAL SAMPLING (SUS)
    # ========================================================================
    
    def _sus_selection(self, population: List[Individual], n_select: int) -> List[Individual]:
        """
        Stochastic Universal Sampling: Improved roulette wheel. 
        
        Uses equally-spaced pointers instead of multiple spins.
        
        Advantages:
        - No bias (unlike roulette which can select same individual multiple times)
        - More efficient
        - Better distribution
        
        Args: 
            population: Population to select from
            n_select: Number of individuals to select
            
        Returns:
            List of selected individuals
        """
        fitnesses = np.array([ind.fitness for ind in population])
        
        # Shift to positive
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min()
        
        # Handle all-zero
        if fitnesses.sum() == 0:
            return list(np.random.choice(population, size=n_select, replace=True))
        
        # Calculate cumulative fitness
        total_fitness = fitnesses.sum()
        cumulative_fitness = np.cumsum(fitnesses)
        
        # Calculate pointer distance
        pointer_distance = total_fitness / n_select
        
        # Random start point
        start = np.random.uniform(0, pointer_distance)
        
        # Generate pointers
        pointers = [start + i * pointer_distance for i in range(n_select)]
        
        # Select individuals
        selected = []
        
        for pointer in pointers:
            # Find first individual whose cumulative fitness >= pointer
            for i, cum_fit in enumerate(cumulative_fitness):
                if pointer <= cum_fit:
                    selected.append(population[i])
                    break
        
        return selected
    
    # ========================================================================
    # TRUNCATION SELECTION
    # ========================================================================
    
    def _truncation_selection(self, population: List[Individual]) -> Individual:
        """
        Truncation selection:  Only select from top X%. 
        
        Very high selection pressure - can cause premature convergence.
        
        Args:
            population: Population to select from
            
        Returns:
            Randomly selected individual from top X%
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Select top X%
        n_keep = max(1, int(len(sorted_pop) * self.truncation_threshold))
        elite = sorted_pop[:n_keep]
        
        # Random choice from elite
        return np.random.choice(elite)
    
    # ========================================================================
    # LINEAR RANKING SELECTION
    # ========================================================================
    
    def _linear_ranking_selection(self, population: List[Individual]) -> Individual:
        """
        Linear ranking selection: Probability is linear function of rank.
        
        P(i) = (1/N) * (pressure - (2*pressure - 2) * (rank - 1) / (N - 1))
        
        Pressure ∈ [1, 2]: 
        - 1.0: Uniform selection
        - 2.0: Maximum selection pressure
        
        Args:
            population: Population to select from
            
        Returns:
            Selected individual
        """
        N = len(population)
        
        # Sort by fitness (ascending)
        sorted_pop = sorted(population, key=lambda x:  x.fitness)
        
        # Calculate probabilities using linear ranking
        probabilities = []
        
        for rank in range(1, N + 1):
            p = (1.0 / N) * (self.pressure - (2 * self.pressure - 2) * (rank - 1) / (N - 1))
            probabilities.append(p)
        
        probabilities = np.array(probabilities)
        probabilities = probabilities / probabilities.sum()  # Normalize
        
        return np.random.choice(sorted_pop, p=probabilities)


# ============================================================================
# COMPARISON UTILITY
# ============================================================================

def compare_selection_methods(population: List[Individual], 
                              n_trials: int = 1000,
                              logger: Optional[Logger] = None):
    """
    Compare different selection methods. 
    
    Args:
        population: Test population
        n_trials: Number of selection trials
        logger: Logger instance
    """
    from collections import Counter
    from utils.colors import print_header, print_section, print_info
    
    methods = ['tournament', 'roulette', 'rank', 'boltzmann', 'sus', 'truncation', 'linear_ranking']
    
    print_header("SELECTION METHOD COMPARISON")
    print()
    
    for method in methods:
        print_section(f"Method: {method.upper()}")
        
        selector = SelectionOperator(method=method, logger=logger)
        
        # Run trials
        selected_indices = []
        
        for _ in range(n_trials):
            if method == 'sus':
                # SUS selects multiple at once
                selected = selector.select_multiple(population, 1)
                selected_ind = selected[0]
            else:
                selected_ind = selector.select(population)
            
            # Find index
            idx = population.index(selected_ind)
            selected_indices.append(idx)
        
        # Analyze distribution
        counter = Counter(selected_indices)
        
        print_info(f"Selection distribution ({n_trials} trials):")
        
        # Sort by individual fitness
        sorted_pop_indices = sorted(range(len(population)), 
                                   key=lambda i: population[i].fitness, 
                                   reverse=True)
        
        for rank, idx in enumerate(sorted_pop_indices[: 10]):
            count = counter.get(idx, 0)
            pct = (count / n_trials) * 100
            fitness = population[idx].fitness
            print(f"  Rank {rank+1:2} (fitness={fitness:.3f}): {count:4} times ({pct:5.1f}%)")
        
        print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_success
    
    logger = get_logger(name="SELECTION_TEST", verbose=True)
    
    # Create test population with varying fitness
    np.random.seed(42)
    population = []
    
    for i in range(20):
        ind = Individual(
            chromosome={'x': i},
            fitness=np.random.exponential(scale=2.0),  # Exponential distribution
            evaluated=True
        )
        population.append(ind)
    
    print_info(f"Created test population:  {len(population)} individuals")
    print_info(f"Fitness range: {min(ind.fitness for ind in population):.3f} - {max(ind.fitness for ind in population):.3f}")
    print()
    
    # Compare all selection methods
    compare_selection_methods(population, n_trials=1000, logger=logger)
    
    print_success("✓ Selection operator test complete!")
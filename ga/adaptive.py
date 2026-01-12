# ============================================================================
# ADAPTIVE GENETIC ALGORITHM PARAMETERS
# ============================================================================
# Self-adapting GA parameters during evolution
#
# FEATURES:
#   - Adaptive mutation rate (based on diversity, convergence)
#   - Adaptive crossover rate
#   - Adaptive population size
#   - Self-adaptive parameters (encoded in chromosome)
#
# USAGE:
#   from ga.adaptive import AdaptiveParameterManager
#   
#   adapter = AdaptiveParameterManager(method='diversity_based')
#   new_mutation_rate = adapter.adapt_mutation_rate(population, generation)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import List, Optional, Tuple
from dataclasses import dataclass

from ga.types import Individual
from utils.logger import Logger
from utils.colors import print_info


# ============================================================================
# ADAPTIVE PARAMETER MANAGER
# ============================================================================

class AdaptiveParameterManager: 
    """
    Manages adaptive GA parameters. 
    
    Args:
        method:  Adaptation method ('diversity_based', 'fitness_based', 'schedule', 'self_adaptive')
        initial_mutation_rate: Starting mutation rate
        initial_crossover_rate: Starting crossover rate
        min_mutation_rate: Minimum mutation rate
        max_mutation_rate: Maximum mutation rate
        min_crossover_rate: Minimum crossover rate
        max_crossover_rate: Maximum crossover rate
        diversity_threshold_low: Low diversity threshold
        diversity_threshold_high: High diversity threshold
        logger: Logger instance
    """
    
    def __init__(self,
                 method: str = 'diversity_based',
                 initial_mutation_rate: float = 0.1,
                 initial_crossover_rate: float = 0.8,
                 min_mutation_rate: float = 0.01,
                 max_mutation_rate: float = 0.5,
                 min_crossover_rate:  float = 0.5,
                 max_crossover_rate: float = 0.95,
                 diversity_threshold_low: float = 0.05,
                 diversity_threshold_high: float = 0.3,
                 logger: Optional[Logger] = None):
        
        self.method = method
        self.initial_mutation_rate = initial_mutation_rate
        self.initial_crossover_rate = initial_crossover_rate
        self.min_mutation_rate = min_mutation_rate
        self.max_mutation_rate = max_mutation_rate
        self.min_crossover_rate = min_crossover_rate
        self.max_crossover_rate = max_crossover_rate
        self.diversity_threshold_low = diversity_threshold_low
        self.diversity_threshold_high = diversity_threshold_high
        self.logger = logger
        
        # Current rates
        self.current_mutation_rate = initial_mutation_rate
        self.current_crossover_rate = initial_crossover_rate
        
        # History
        self.mutation_rate_history = []
        self.crossover_rate_history = []
        
        # Valid methods
        self.valid_methods = [
            'diversity_based',
            'fitness_based',
            'schedule',
            'self_adaptive',
            'none'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Invalid adaptive method '{method}'.Valid: {self.valid_methods}")
    
    # ========================================================================
    # ADAPTATION INTERFACE
    # ========================================================================
    
    def adapt(self,
              population: List[Individual],
              generation: int,
              max_generations: int,
              diversity:  float = None) -> Tuple[float, float]:
        """
        Adapt mutation and crossover rates. 
        
        Args:
            population: Current population
            generation:  Current generation
            max_generations:  Maximum generations
            diversity: Population diversity (optional)
            
        Returns: 
            (mutation_rate, crossover_rate) tuple
        """
        if self.method == 'none':
            return self.current_mutation_rate, self.current_crossover_rate
        
        elif self.method == 'diversity_based':
            mutation_rate, crossover_rate = self._adapt_diversity_based(
                population, diversity
            )
        
        elif self.method == 'fitness_based':
            mutation_rate, crossover_rate = self._adapt_fitness_based(population)
        
        elif self.method == 'schedule':
            mutation_rate, crossover_rate = self._adapt_schedule(
                generation, max_generations
            )
        
        elif self.method == 'self_adaptive':
            # Self-adaptive uses parameters encoded in chromosome
            mutation_rate = self.current_mutation_rate
            crossover_rate = self.current_crossover_rate
        
        else:
            raise ValueError(f"Unknown adaptive method: {self.method}")
        
        # Update current rates
        self.current_mutation_rate = mutation_rate
        self.current_crossover_rate = crossover_rate
        
        # Record history
        self.mutation_rate_history.append(mutation_rate)
        self.crossover_rate_history.append(crossover_rate)
        
        return mutation_rate, crossover_rate
    
    # ========================================================================
    # DIVERSITY-BASED ADAPTATION
    # ========================================================================
    
    def _adapt_diversity_based(self,
                               population: List[Individual],
                               diversity: Optional[float] = None) -> Tuple[float, float]:
        """
        Adapt based on population diversity.
        
        Low diversity → Increase mutation (exploration)
        High diversity → Decrease mutation (exploitation)
        
        Args:
            population: Current population
            diversity:  Diversity metric (if None, will calculate)
            
        Returns: 
            (mutation_rate, crossover_rate)
        """
        # Calculate diversity if not provided
        if diversity is None:
            diversity = self._calculate_diversity(population)
        
        # Adaptive mutation rate
        if diversity < self.diversity_threshold_low:
            # Low diversity → high mutation
            mutation_rate = self.max_mutation_rate
        elif diversity > self.diversity_threshold_high:
            # High diversity → low mutation
            mutation_rate = self.min_mutation_rate
        else:
            # Linear interpolation
            ratio = (diversity - self.diversity_threshold_low) / \
                   (self.diversity_threshold_high - self.diversity_threshold_low)
            mutation_rate = self.max_mutation_rate - ratio * (self.max_mutation_rate - self.min_mutation_rate)
        
        # Adaptive crossover rate (inverse of mutation)
        crossover_rate = self.max_crossover_rate - (mutation_rate - self.min_mutation_rate) / \
                        (self.max_mutation_rate - self.min_mutation_rate) * \
                        (self.max_crossover_rate - self.min_crossover_rate)
        
        return mutation_rate, crossover_rate
    
    # ========================================================================
    # FITNESS-BASED ADAPTATION
    # ========================================================================
    
    def _adapt_fitness_based(self, population: List[Individual]) -> Tuple[float, float]:
        """
        Adapt based on fitness improvement.
        
        Stagnation → Increase mutation
        Improvement → Decrease mutation
        
        Args:
            population: Current population
            
        Returns:
            (mutation_rate, crossover_rate)
        """
        # Get current best fitness
        current_best = max(ind.fitness for ind in population)
        
        # Check if we have history
        if not hasattr(self, 'best_fitness_history'):
            self.best_fitness_history = []
        
        self.best_fitness_history.append(current_best)
        
        # Need at least 5 generations to detect stagnation
        if len(self.best_fitness_history) < 5:
            return self.current_mutation_rate, self.current_crossover_rate
        
        # Check improvement over last 5 generations
        recent_improvement = self.best_fitness_history[-1] - self.best_fitness_history[-5]
        
        if recent_improvement < 0.001: 
            # Stagnation → increase mutation
            mutation_rate = min(self.current_mutation_rate * 1.2, self.max_mutation_rate)
        else:
            # Improvement → decrease mutation
            mutation_rate = max(self.current_mutation_rate * 0.9, self.min_mutation_rate)
        
        # Crossover inverse
        crossover_rate = self.max_crossover_rate - (mutation_rate - self.min_mutation_rate) / \
                        (self.max_mutation_rate - self.min_mutation_rate) * \
                        (self.max_crossover_rate - self.min_crossover_rate)
        
        return mutation_rate, crossover_rate
    
    # ========================================================================
    # SCHEDULE-BASED ADAPTATION
    # ========================================================================
    
    def _adapt_schedule(self, generation:  int, max_generations: int) -> Tuple[float, float]: 
        """
        Adapt based on generation schedule.
        
        Early generations:  High mutation (exploration)
        Late generations: Low mutation (exploitation)
        
        Args:
            generation: Current generation
            max_generations: Maximum generations
            
        Returns:
            (mutation_rate, crossover_rate)
        """
        # Progress ratio
        progress = generation / max(max_generations, 1)
        
        # Linear decrease in mutation rate
        mutation_rate = self.max_mutation_rate - progress * (self.max_mutation_rate - self.min_mutation_rate)
        
        # Linear increase in crossover rate
        crossover_rate = self.min_crossover_rate + progress * (self.max_crossover_rate - self.min_crossover_rate)
        
        return mutation_rate, crossover_rate
    
    # ========================================================================
    # SELF-ADAPTIVE PARAMETERS
    # ========================================================================
    
    def encode_self_adaptive_params(self, chromosome: dict) -> dict:
        """
        Add self-adaptive parameters to chromosome. 
        
        Args:
            chromosome: Original chromosome
            
        Returns: 
            Chromosome with adaptive parameters
        """
        chromosome['_mutation_rate'] = self.initial_mutation_rate
        chromosome['_crossover_rate'] = self.initial_crossover_rate
        return chromosome
    
    def get_individual_rates(self, chromosome: dict) -> Tuple[float, float]:
        """
        Get mutation/crossover rates from self-adaptive chromosome.
        
        Args:
            chromosome: Chromosome with adaptive parameters
            
        Returns: 
            (mutation_rate, crossover_rate)
        """
        mutation_rate = chromosome.  get('_mutation_rate', self.current_mutation_rate)
        crossover_rate = chromosome. get('_crossover_rate', self.current_crossover_rate)
        
        return mutation_rate, crossover_rate
    
    # ========================================================================
    # UTILITIES
    # ========================================================================
    
    def _calculate_diversity(self, population: List[Individual]) -> float:
        """Calculate population diversity (fitness std)."""
        fitnesses = [ind.fitness for ind in population]
        return float(np.std(fitnesses))
    
    def get_history(self) -> dict:
        """Get parameter adaptation history."""
        return {
            'mutation_rate':  self.mutation_rate_history. copy(),
            'crossover_rate': self.crossover_rate_history.copy()
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success
    
    logger = get_logger(name="ADAPTIVE_TEST", verbose=True)
    
    print_header("ADAPTIVE PARAMETERS TEST")
    print()
    
    # Simulate evolution with changing diversity
    np.random.seed(42)
    
    methods = ['diversity_based', 'fitness_based', 'schedule']
    
    for method in methods:
        print_section(f"Method: {method. upper()}")
        
        adapter = AdaptiveParameterManager(
            method=method,
            logger=logger
        )
        
        # Simulate 20 generations
        print_info("Generation | Diversity | Mutation Rate | Crossover Rate")
        print_info("-" * 60)
        
        for gen in range(20):
            # Simulate population with varying diversity
            if gen < 5:
                diversity = 0.3  # High diversity (early)
            elif gen < 15:
                diversity = 0.1  # Medium diversity
            else: 
                diversity = 0.02  # Low diversity (converging)
            
            # Create fake population
            population = [
                Individual(
                    chromosome={'x':  i},
                    fitness=0.8 + gen * 0.01 + np.random.random() * diversity,
                    evaluated=True
                )
                for i in range(20)
            ]
            
            # Adapt parameters
            mutation_rate, crossover_rate = adapter.adapt(
                population, gen, 20, diversity
            )
            
            print(f"  Gen {gen:2d}   | {diversity: 9.3f} | {mutation_rate:13.3f} | {crossover_rate:14.3f}")
        
        print()

    print_success("✓ Adaptive parameters test complete!")
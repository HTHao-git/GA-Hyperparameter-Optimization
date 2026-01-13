# ============================================================================
# GENETIC ALGORITHM CORE
# ============================================================================
# Main GA implementation with support for:
#   - Configurable operators (selection, crossover, mutation)
#   - Adaptive parameters
#   - Early stopping
#   - Fitness caching
#   - Comprehensive logging
#
# Last updated: 2026-01-12
# ============================================================================

import numpy as np
import random
import time
import json
from typing import Dict, Any, List, Callable, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path

from utils.logger import Logger
from utils. colors import print_header, print_section, print_info, print_success, print_warning


# ============================================================================
# CONFIGURATION
# ============================================================================

@dataclass
class GAConfig: 
    """Configuration for Genetic Algorithm."""
    
    # ========================================================================
    # POPULATION & GENERATIONS
    # ========================================================================
    population_size: int = 50
    num_generations: int = 100
    
    # ========================================================================
    # OPERATOR RATES
    # ========================================================================
    crossover_rate: float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    
    # ========================================================================
    # OPERATOR METHODS
    # ========================================================================
    selection_method: str = 'tournament'  # 'tournament', 'roulette', 'rank', 'sus', 'boltzmann'
    crossover_method: str = 'uniform'     # 'single_point', 'two_point', 'uniform', 'arithmetic', 'blx_alpha', 'sbx'
    mutation_method: str = 'adaptive'     # 'uniform', 'gaussian', 'polynomial', 'adaptive', 'boundary', 'non_uniform'
    
    # ========================================================================
    # ADVANCED MUTATION SETTINGS
    # ========================================================================
    mutation_strength: str = 'medium'     # 'small', 'medium', 'large'
    adaptive_mutation: bool = True        # Enable adaptive mutation rate
    adaptive_method: str = 'diversity_based'  # 'diversity_based', 'fitness_based', 'schedule'
    
    # ========================================================================
    # SELECTION PARAMETERS
    # ========================================================================
    tournament_size: int = 3              # For tournament selection
    
    # ========================================================================
    # DIVERSITY MAINTENANCE
    # ========================================================================
    diversity_maintenance:  bool = False   # Enable diversity maintenance
    diversity_method: str = 'fitness_sharing'  # 'fitness_sharing', 'crowding'
    sharing_radius: float = 0.1          # For fitness sharing
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    early_stopping: bool = True
    patience: int = 10
    diversity_threshold: float = 0.0      # Min diversity to continue (0.0 = disabled)
    
    # ========================================================================
    # PERFORMANCE
    # ========================================================================
    cache_fitness: bool = False           # Cache fitness evaluations
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    verbose: int = 1                      # 0=silent, 1=normal, 2=detailed
    
    # ========================================================================
    # REPRODUCIBILITY
    # ========================================================================
    random_state: int = 42


# ============================================================================
# INDIVIDUAL
# ============================================================================

@dataclass
class Individual: 
    """Represents a single solution (individual) in the population."""
    chromosome: Dict[str, Any]
    fitness: float = 0.0
    evaluated: bool = False
    
    def __hash__(self):
        """Make Individual hashable for fitness caching."""
        # Convert chromosome to JSON string for hashing
        def make_serializable(obj):
            """Convert NumPy types to Python natives."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float32, np. float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            else:
                return obj
        
        serializable = {k: make_serializable(v) for k, v in self.chromosome.items()}
        return hash(json.dumps(serializable, sort_keys=True))
    
    def __eq__(self, other):
        """Equality comparison for caching."""
        if not isinstance(other, Individual):
            return False
        return self.chromosome == other.chromosome


@dataclass  
class GenerationStats:
    """Statistics for a generation."""
    generation: int
    best_fitness: float
    mean_fitness: float
    std_fitness: float
    diversity: float
    time:  float = 0.0


# ============================================================================
# GENETIC ALGORITHM
# ============================================================================

class GeneticAlgorithm: 
    """
    Genetic Algorithm for hyperparameter optimization.
    
    Args:
        config: GA configuration
        fitness_function: Function to evaluate individuals
        chromosome_template:  Hyperparameter search space
        logger: Logger instance
    """
    
    def __init__(self,
                 config: GAConfig,
                 fitness_function:  Callable[[Dict[str, Any]], float],
                 chromosome_template: Dict[str, Any],
                 logger: Optional[Logger] = None):
        
        self.config = config
        self.fitness_function = fitness_function
        self.chromosome_template = chromosome_template
        self.logger = logger
        
        # Set random seeds
        random.seed(config.random_state)
        np.random.seed(config.random_state)
        
        # Population
        self.population: List[Individual] = []
        self. best_individual: Optional[Individual] = None
        
        # History
        self.history: List[GenerationStats] = []
        
        # Fitness cache
        self.fitness_cache: Dict[int, float] = {}
        
        # Early stopping
        self.generations_without_improvement = 0
        self.best_fitness_so_far = -np.inf
        
        # Initialize advanced operators (with safe fallbacks)
        self._initialize_operators()
        
        if self.logger:
            self.logger.info("Genetic Algorithm initialized")
            self.logger.info(f"  Population size: {config.population_size}")
            self.logger.info(f"  Generations: {config.num_generations}")
            self.logger.info(f"  Crossover rate: {config.crossover_rate}")
            self.logger.info(f"  Mutation rate: {config.mutation_rate}")
    
    def _initialize_operators(self):
        """Initialize advanced operators with safe fallbacks."""
        
        # Selection operator
        try: 
            from ga.selection import SelectionOperator
            self.selection_operator = SelectionOperator(
                method=self.config. selection_method,
                logger=self.logger
            )
            # Set tournament size separately if supported
            if hasattr(self.selection_operator, 'tournament_size'):
                self.selection_operator.tournament_size = self.config.tournament_size
        except (ImportError, AttributeError) as e:
            if self.logger and self.config.verbose >= 2:
                self.logger.warning(f"Could not load SelectionOperator: {e}")
            self.selection_operator = None
        
        # Crossover operator
        try:
            from ga.crossover import CrossoverOperator
            self.crossover_operator = CrossoverOperator(
                method=self.config.crossover_method,
                logger=self.logger
            )
        except (ImportError, AttributeError) as e:
            if self.logger and self.config.verbose >= 2:
                self. logger.warning(f"Could not load CrossoverOperator: {e}")
            self.crossover_operator = None
        
        # Mutation operator
        try: 
            from ga.mutation import MutationOperator
            self.mutation_operator = MutationOperator(
                method=self.config.mutation_method,
                mutation_rate=self.config.mutation_rate,
                logger=self.logger
            )
            # Set strength separately if supported
            if hasattr(self. mutation_operator, 'strength'):
                self.mutation_operator. strength = self.config.mutation_strength
        except (ImportError, AttributeError) as e:
            if self.logger and self. config.verbose >= 2:
                self.logger.warning(f"Could not load MutationOperator: {e}")
            self.mutation_operator = None
        
        # Adaptive parameter manager
        if self.config.adaptive_mutation:
            try:
                from ga.adaptive import AdaptiveParameterManager
                self.adaptive_manager = AdaptiveParameterManager(
                    method=self.config.adaptive_method,
                    initial_mutation_rate=self.config.mutation_rate,
                    initial_crossover_rate=self.config.crossover_rate,
                    logger=self.logger
                )
            except (ImportError, AttributeError) as e:
                if self.logger and self.config.verbose >= 2:
                    self. logger.warning(f"Could not load AdaptiveParameterManager: {e}")
                self.adaptive_manager = None
        else:
            self.adaptive_manager = None
        
        if self.logger and self.config.verbose >= 2:
            self.logger. debug("Using basic GA operators (selection, crossover, mutation)")
        else: 
            self.adaptive_manager = None
    
    # ========================================================================
    # MAIN EVOLUTION LOOP
    # ========================================================================
    
    def run(self) -> Individual:
        """
        Run the genetic algorithm. 
        
        Returns:
            Best individual found
        """
        if self.logger and self.config. verbose >= 1:
            print_header("GENETIC ALGORITHM OPTIMIZATION")
            print()
        
        # Initialize population
        self._initialize_population()
        
        # Evolution loop
        for generation in range(self.config.num_generations):
            generation_start = time.time()
            
            # Evaluate population
            self._evaluate_population()
            
            # Calculate statistics
            stats = self._calculate_stats(generation)
            self.history.append(stats)
            
            # Update best individual
            current_best = max(self.population, key=lambda ind: ind.fitness)
            if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
                self.best_individual = Individual(
                    chromosome=current_best.chromosome. copy(),
                    fitness=current_best.fitness,
                    evaluated=True
                )
            
            # Record time
            stats.time = time.time() - generation_start
            
            # Logging
            if self.logger and self.config.verbose >= 1:
                self._log_generation(stats)
            
            # Adaptive parameters
            if self.adaptive_manager:
                new_mutation_rate, new_crossover_rate = self. adaptive_manager.adapt(
                    self.population,
                    generation,
                    self.config.num_generations,
                    stats.diversity
                )
                self.config.mutation_rate = new_mutation_rate
                self. config.crossover_rate = new_crossover_rate
                
                if self.mutation_operator:
                    self. mutation_operator.mutation_rate = new_mutation_rate
                
                if self.logger and self.config.verbose >= 2:
                    self. logger.debug(f"  Adapted rates: mutation={new_mutation_rate:.3f}, crossover={new_crossover_rate:.3f}")
            
            # Early stopping checks
            if self._should_stop(stats):
                break
            
            # Create next generation (except on last iteration)
            if generation < self. config.num_generations - 1:
                self. population = self._create_next_generation()
        
        # Final summary
        if self.logger and self.config. verbose >= 1:
            self._print_summary()
        
        return self.best_individual
    
    # ========================================================================
    # POPULATION INITIALIZATION
    # ========================================================================
    
    def _initialize_population(self):
        """Initialize random population."""
        if self.logger and self.config.verbose >= 1:
            print_info("Initializing population...")
        
        self.population = []
        for _ in range(self.config.population_size):
            individual = self._initialize_individual()
            self.population.append(individual)
        
        if self.logger and self.config.verbose >= 1:
            print_success(f"Population initialized:  {len(self.population)} individuals")
    
    def _initialize_individual(self) -> Individual:
        """Create a random individual."""
        chromosome = {}
        
        for gene, possible_values in self.chromosome_template.items():
            if isinstance(possible_values, tuple) and len(possible_values) == 2:
                # Continuous range
                low, high = possible_values
                chromosome[gene] = random.uniform(low, high)
            
            elif isinstance(possible_values, list):
                # Discrete choice - handle mixed types (e.g., gamma:  ['scale', 'auto', (0.001, 10. 0)])
                has_tuples = any(isinstance(v, tuple) for v in possible_values)
                
                if has_tuples:
                    # Mix of discrete values and continuous ranges
                    non_tuple_values = [v for v in possible_values if not isinstance(v, tuple)]
                    tuple_values = [v for v in possible_values if isinstance(v, tuple)]
                    
                    if non_tuple_values and tuple_values and random.random() < 0.5:
                        # Choose from discrete values
                        chromosome[gene] = random.choice(non_tuple_values)
                    elif tuple_values:
                        # Choose from continuous range
                        low, high = random.choice(tuple_values)
                        chromosome[gene] = random.uniform(low, high)
                    else:
                        # Fallback to discrete
                        chromosome[gene] = random.choice(non_tuple_values)
                else:
                    # All discrete values
                    chromosome[gene] = random.choice(possible_values)
            
            else:
                # Single value (constant)
                chromosome[gene] = possible_values
        
        return Individual(chromosome=chromosome, evaluated=False)
    
    # ========================================================================
    # FITNESS EVALUATION
    # ========================================================================
    
    def _evaluate_population(self):
        """Evaluate all unevaluated individuals."""
        for individual in self.population:
            if not individual.evaluated:
                # Check cache
                if self.config.cache_fitness:
                    ind_hash = hash(individual)
                    if ind_hash in self.fitness_cache:
                        individual.fitness = self.fitness_cache[ind_hash]
                        individual.evaluated = True
                        continue
                
                # Evaluate
                individual.fitness = self. fitness_function(individual.chromosome)
                individual.evaluated = True
                
                # Cache result
                if self.config.cache_fitness:
                    self.fitness_cache[ind_hash] = individual. fitness
    
    # ========================================================================
    # SELECTION
    # ========================================================================
    
    def _select_parent(self) -> Individual:
        """Select a parent using configured selection method."""
        if self. selection_operator:
            # Use advanced selection operator
            return self.selection_operator.select(self.population)
        else:
            # Fallback:  tournament selection
            tournament_size = self.config.tournament_size
            tournament = random.sample(self.population, tournament_size)
            return max(tournament, key=lambda ind: ind.fitness)
    
    # ========================================================================
    # CROSSOVER
    # ========================================================================
    
    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[Individual, Individual]:
        """Perform crossover."""
        if self.crossover_operator:
            # Use advanced crossover operator
            child1_chromo, child2_chromo = self. crossover_operator.crossover(
                parent1.chromosome,
                parent2.chromosome,
                self.chromosome_template
            )
        else:
            # Fallback:  uniform crossover
            child1_chromo = {}
            child2_chromo = {}
            
            for gene in parent1.chromosome. keys():
                if random.random() < 0.5:
                    child1_chromo[gene] = parent1.chromosome[gene]
                    child2_chromo[gene] = parent2.chromosome[gene]
                else:
                    child1_chromo[gene] = parent2.chromosome[gene]
                    child2_chromo[gene] = parent1.chromosome[gene]
        
        return Individual(chromosome=child1_chromo, evaluated=False), \
               Individual(chromosome=child2_chromo, evaluated=False)
    
    # ========================================================================
    # MUTATION
    # ========================================================================
    
    def _mutate(self, individual:  Individual) -> Individual:
        """Apply mutation."""
        if self.mutation_operator:
            # Use advanced mutation operator
            mutated_chromosome = self.mutation_operator.mutate(
                individual.chromosome,
                self.chromosome_template
            )
        else:
            # Fallback: basic uniform mutation
            mutated_chromosome = individual.chromosome.copy()
            
            for gene, value in mutated_chromosome.items():
                if random.random() < self.config.mutation_rate:
                    possible_values = self.chromosome_template[gene]
                    
                    if isinstance(possible_values, tuple) and len(possible_values) == 2:
                        low, high = possible_values
                        mutated_chromosome[gene] = random.uniform(low, high)
                    elif isinstance(possible_values, list):
                        mutated_chromosome[gene] = random.choice(possible_values)
        
        return Individual(chromosome=mutated_chromosome, evaluated=False)
    
    # ========================================================================
    # NEXT GENERATION
    # ========================================================================
    
    def _create_next_generation(self) -> List[Individual]:
        """Create next generation using selection, crossover, and mutation."""
        next_generation = []
        
        # Elitism: preserve best individuals
        n_elites = int(self.config.elitism_rate * self.config.population_size)
        if n_elites > 0:
            elites = sorted(self.population, key=lambda ind: ind.fitness, reverse=True)[:n_elites]
            next_generation.extend(elites)
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Selection
            parent1 = self._select_parent()
            parent2 = self._select_parent()
            
            # Crossover
            if random.random() < self.config.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1 = Individual(chromosome=parent1.chromosome.copy(), evaluated=False)
                child2 = Individual(chromosome=parent2.chromosome.copy(), evaluated=False)
            
            # Mutation
            if random.random() < self.config.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.config.mutation_rate:
                child2 = self._mutate(child2)
            
            next_generation.append(child1)
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        return next_generation[: self.config.population_size]
    
    # ========================================================================
    # STATISTICS
    # ========================================================================
    
    def _calculate_stats(self, generation: int) -> GenerationStats:
        """Calculate generation statistics."""
        fitnesses = [ind.fitness for ind in self.population]
        
        best_fitness = max(fitnesses)
        mean_fitness = np. mean(fitnesses)
        std_fitness = np.std(fitnesses)
        diversity = self._calculate_diversity()
        
        return GenerationStats(
            generation=generation,
            best_fitness=best_fitness,
            mean_fitness=mean_fitness,
            std_fitness=std_fitness,
            diversity=diversity
        )
    
    def _calculate_diversity(self) -> float:
        """Calculate population diversity (normalized standard deviation of fitness)."""
        fitnesses = [ind.fitness for ind in self.population]
        
        if len(fitnesses) == 0:
            return 0.0
        
        mean_fitness = np.mean(fitnesses)
        
        if mean_fitness == 0:
            return 0.0
        
        std_fitness = np.std(fitnesses)
        diversity = std_fitness / (abs(mean_fitness) + 1e-10)
        
        return diversity
    
    # ========================================================================
    # EARLY STOPPING
    # ========================================================================
    
    def _should_stop(self, stats: GenerationStats) -> bool:
        """Check if should stop early."""
        
        # Check improvement
        if stats.best_fitness > self.best_fitness_so_far:
            self.best_fitness_so_far = stats. best_fitness
            self.generations_without_improvement = 0
        else:
            self.generations_without_improvement += 1
        
        # Patience-based stopping
        if self.config.early_stopping and self.generations_without_improvement >= self.config.patience:
            if self.logger and self.config.verbose >= 1:
                print_success(f"✓ Converged at generation {stats.generation}")
            return True
        
        # Diversity-based stopping
        if self.config.diversity_threshold > 0 and stats.diversity < self.config.diversity_threshold:
            if self.logger and self.config. verbose >= 1:
                print_warning("Low diversity detected")
            
            # Only stop if also no improvement
            if self.generations_without_improvement >= 2:
                return True
        
        return False
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def _log_generation(self, stats: GenerationStats):
        """Log generation statistics."""
        msg = (f"Gen {int(stats.generation):3d} | "
               f"Best:  {stats.best_fitness:.4f} | "
               f"Mean: {stats.mean_fitness:.4f}±{stats. std_fitness:.4f} | "
               f"Diversity:  {stats.diversity:.4f} | "
               f"Time:  {stats.time:.2f}s")
        
        print_info(msg)
    
    def _print_summary(self):
        """Print optimization summary."""
        print()
        print_header("GA OPTIMIZATION COMPLETE")
        print()
        
        print_success(f"Best fitness: {self.best_individual.fitness:.4f}")
        best_gen_idx = max(range(len(self.history)), key=lambda i: self.history[i].best_fitness)
        print_info(f"Found at generation:  {int(self.history[best_gen_idx].generation)}")
        print_info(f"Total generations: {len(self.history)}")
        print_info(f"Total evaluations: {sum(len(self.population) for _ in self.history)}")
        
        total_time = sum(stats.time for stats in self.history)
        print_info(f"Total time: {total_time:.2f}s")
        print_info(f"Avg time per generation: {total_time/len(self.history):.2f}s")
        
        print()
        print_section("Best Chromosome")
        for gene, value in self.best_individual.chromosome.items():
            print(f"  {gene}:  {value}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    
    logger = get_logger(name="GA_TEST", verbose=True)
    
    # Simple test:  optimize a mathematical function
    def sphere_function(chromosome:  Dict[str, Any]) -> float:
        """Sphere function:  minimize sum of squares."""
        x = chromosome['x']
        y = chromosome['y']
        # Return negative (since GA maximizes)
        return -(x**2 + y**2)
    
    # Chromosome template
    template = {
        'x': (-5.0, 5.0),
        'y': (-5.0, 5.0)
    }
    
    # GA configuration
    config = GAConfig(
        population_size=20,
        num_generations=30,
        mutation_rate=0.2,
        verbose=1
    )
    
    # Run GA
    ga = GeneticAlgorithm(
        config=config,
        fitness_function=sphere_function,
        chromosome_template=template,
        logger=logger
    )
    
    best = ga.run()
    
    print()
    print_success("✓ GA test complete!")
    print_info(f"  Best solution: x={best.chromosome['x']:.4f}, y={best.chromosome['y']:. 4f}")
    print_info(f"  Fitness: {best.fitness:.4f}")
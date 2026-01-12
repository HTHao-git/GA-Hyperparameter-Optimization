# ============================================================================
# GENETIC ALGORITHM CORE FRAMEWORK
# ============================================================================
# Main GA implementation for hyperparameter optimization
#
# FEATURES:
#   - Population management
#   - Generation evolution
#   - Elitism preservation
#   - Convergence tracking
#   - Progress monitoring
#   - Result persistence
#
# USAGE: 
#   from ga.genetic_algorithm import GeneticAlgorithm
#   
#   ga = GeneticAlgorithm(config)
#   best_solution = ga.run()
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import time
from typing import Dict, Any, List, Tuple, Optional, Callable
from pathlib import Path
import json
from dataclasses import dataclass, asdict

from utils.logger import Logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning
from ga.types import Individual, GenerationStats
from ga.selection import SelectionOperator
from ga.crossover import CrossoverOperator
from ga.mutation import MutationOperator
from ga.adaptive import AdaptiveParameterManager

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class GAConfig:
    """Configuration for Genetic Algorithm."""
    
    # Population parameters
    population_size: int = 50
    num_generations: int = 100
    
    # Genetic operators
    crossover_rate:  float = 0.8
    mutation_rate: float = 0.1
    elitism_rate: float = 0.1
    
    # Selection
    tournament_size: int = 3
    selection_method: str = 'tournament'  # Can use:  tournament, roulette, rank, boltzmann, sus, truncation, linear_ranking
    
    # Crossover
    crossover_method: str = 'uniform'  # uniform, single_point, two_point, arithmetic, sbx
    
    # Mutation
    mutation_method: str = 'gaussian'  # uniform, gaussian, polynomial, adaptive, boundary, non_uniform
    mutation_sigma: float = 0.1        # For Gaussian/Adaptive
    mutation_eta: float = 20.0         # For Polynomial
    
    # Diversity
    diversity_threshold: float = 0.1
    niching_enabled: bool = False
    
    # Convergence
    early_stopping:  bool = True
    patience: int = 10  # Generations without improvement
    min_improvement: float = 0.001
    
    # Performance
    n_jobs: int = 1  # Parallel fitness evaluation
    cache_fitness: bool = True
    
    # Logging
    verbose: int = 1  # 0: silent, 1: normal, 2: detailed
    save_history: bool = True
    
    # Random state
    random_state: int = 42


@dataclass
class Individual:
    """Represents a single solution (chromosome)."""
    
    chromosome: Dict[str, Any]
    fitness: float = -np.inf
    age: int = 0
    generation: int = 0
    evaluated: bool = False
    
    def __hash__(self):
        """Make Individual hashable for fitness caching."""
        # Convert chromosome to JSON string for hashing
        def make_serializable(obj):
            """Convert NumPy types to Python natives."""
            if isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np. floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.bool_, bool)): 
                return bool(obj)
            elif isinstance(obj, np. ndarray):
                return obj. tolist()
            else:
                return obj
        
        serializable = {k: make_serializable(v) for k, v in self.chromosome.items()}
        return hash(json.dumps(serializable, sort_keys=True))

@dataclass
class GenerationStats:
    """Statistics for a single generation."""
    
    generation: int
    best_fitness:  float
    mean_fitness: float
    std_fitness: float
    worst_fitness: float
    diversity: float
    evaluation_time: float
    total_time: float


# ============================================================================
# GENETIC ALGORITHM CLASS
# ============================================================================

class GeneticAlgorithm:
    """
    Genetic Algorithm for hyperparameter optimization. 
    
    Args:
        config: GA configuration
        fitness_function: Function to evaluate fitness (chromosome -> float)
        chromosome_template: Template defining chromosome structure
        logger: Logger instance
    """
    
    def __init__(self,
                 config: GAConfig,
                 fitness_function:  Callable[[Dict[str, Any]], float],
                 chromosome_template: Dict[str, List[Any]],
                 logger: Optional[Logger] = None):
        
        self.config = config
        self.fitness_function = fitness_function
        self.chromosome_template = chromosome_template
        self.logger = logger
        
        # Population
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.generation = 0
        
        # History
        self.history: List[GenerationStats] = []
        self.fitness_cache: Dict[int, float] = {}
        
        # Convergence tracking
        self.generations_without_improvement = 0
        self.best_fitness_so_far = -np.inf
        
        # Timing
        self.start_time = None
        self.total_evaluations = 0
        
        # Set random seed
        np.random.seed(config.random_state)
        
        if self.logger:
            self.logger.info("Genetic Algorithm initialized")
            self.logger.info(f"  Population size: {config.population_size}")
            self.logger.info(f"  Generations:  {config.num_generations}")
            self.logger.info(f"  Crossover rate: {config.crossover_rate}")
            self.logger.info(f"  Mutation rate: {config.mutation_rate}")

        # Create selection operator
        self.selector = SelectionOperator(
            method=config.selection_method,
            tournament_size=config.tournament_size,
            logger=logger
        )
        
        # Create crossover operator
        self. crossover_op = CrossoverOperator(
            method=config.crossover_method,
            crossover_rate=config.crossover_rate,
            logger=logger
        )
                
        # Create mutation operator
        self. mutator = MutationOperator(
            method=config.mutation_method,
            mutation_rate=config.mutation_rate,
            sigma=config.mutation_sigma,
            eta=config.mutation_eta,
            max_generations=config.num_generations,
            logger=logger
        )
    
    # ========================================================================
    # MAIN GA LOOP
    # ========================================================================
    
    def run(self) -> Individual:
        """
        Run the genetic algorithm. 
        
        Returns:
            Best individual found
        """
        if self.logger:
            print_header("GENETIC ALGORITHM OPTIMIZATION")
            print()
        
        self.start_time = time.time()
        
        # Initialize population
        self._initialize_population()
        
        # Main evolution loop
        for gen in range(self.config.num_generations):
            self.generation = gen
            
            gen_start_time = time.time()
            
            # Evaluate fitness
            eval_start = time.time()
            self._evaluate_population()
            eval_time = time.time() - eval_start
            
            # Update best individual
            self._update_best()
            
            # Calculate statistics
            stats = self._calculate_statistics(eval_time, time.time() - gen_start_time)
            self.history.append(stats)
            
            # Log progress
            self._log_generation(stats)
            
            # Check convergence
            if self._check_convergence():
                if self.logger:
                    print_success(f"✓ Converged at generation {gen}")
                break
            
            # Create next generation
            if gen < self.config.num_generations - 1:
                self._evolve()
        
        # Final summary
        self._print_summary()
        
        return self.best_individual
    
    # ========================================================================
    # POPULATION INITIALIZATION
    # ========================================================================
    
    def _initialize_population(self):
        """Create initial random population."""
        if self.logger and self.config.verbose >= 1:
            self.logger.info("Initializing population...")
        
        self.population = []
        
        for i in range(self.config.population_size):
            chromosome = self._random_chromosome()
            individual = Individual(
                chromosome=chromosome,
                generation=0
            )
            self.population.append(individual)
        
        if self.logger and self.config.verbose >= 1:
            self.logger.success(f"Population initialized: {len(self.population)} individuals")
    
    def _random_chromosome(self) -> Dict[str, Any]:
        """Generate random chromosome from template."""
        chromosome = {}
        
        for gene, possible_values in self.chromosome_template.items():
            if isinstance(possible_values, list):
                # Discrete choice
                # Handle mixed types (e.g., gamma:  ['scale', 'auto', (0.001, 10.0)])
                if isinstance(possible_values, list):
                    # Check if list contains tuples (continuous ranges)
                    has_tuples = any(isinstance(v, tuple) for v in possible_values)
                    if has_tuples:
                        # Randomly choose between discrete values and continuous range
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
                    chromosome[gene] = np.random.choice(possible_values)
            elif isinstance(possible_values, tuple) and len(possible_values) == 2:
                # Continuous range (min, max)
                low, high = possible_values
                if isinstance(low, int) and isinstance(high, int):
                    chromosome[gene] = np.random.randint(low, high + 1)
                else:
                    chromosome[gene] = np.random.uniform(low, high)
            else:
                raise ValueError(f"Invalid gene specification for {gene}:  {possible_values}")
        
        return chromosome
    
    # ========================================================================
    # FITNESS EVALUATION
    # ========================================================================
    
    def _evaluate_population(self):
        """Evaluate fitness for all individuals."""
        for individual in self.population:
            if not individual.evaluated:
                # Check cache
                ind_hash = hash(individual)
                
                if self.config.cache_fitness and ind_hash in self.fitness_cache:
                    individual.fitness = self.fitness_cache[ind_hash]
                else:
                    # Evaluate
                    individual.fitness = self.fitness_function(individual.chromosome)
                    self.total_evaluations += 1
                    
                    # Cache result
                    if self.config.cache_fitness:
                        self.fitness_cache[ind_hash] = individual.fitness
                
                individual.evaluated = True
    
    # ========================================================================
    # EVOLUTION
    # ========================================================================
    
    def _evolve(self):
        """Create next generation."""
        next_generation = []
        
        # Elitism:  preserve best individuals
        elite_count = int(self.config.population_size * self.config.elitism_rate)
        if elite_count > 0:
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            elites = sorted_pop[:elite_count]
            
            # Age elites
            for elite in elites:
                elite.age += 1
                elite.generation = self.generation + 1
            
            next_generation.extend(elites)
        
        # Generate offspring
        while len(next_generation) < self.config.population_size:
            # Selection
            parent1 = self._select()
            parent2 = self._select()
            
            # Crossover
            child1_chromosome, child2_chromosome = self.crossover_op.crossover(
                parent1.chromosome,
                parent2.chromosome,
                self.chromosome_template
            )
            
            # Mutation
            self.mutator.update_generation(self.generation)  # For adaptive methods
            child1_chromosome = self.mutator.mutate(child1_chromosome, self.chromosome_template)
            child2_chromosome = self.mutator.mutate(child2_chromosome, self.chromosome_template)
            
            # Create offspring
            child1 = Individual(
                chromosome=child1_chromosome,
                generation=self.generation + 1
            )
            
            child2 = Individual(
                chromosome=child2_chromosome,
                generation=self.generation + 1
            )
            
            next_generation.append(child1)
            
            if len(next_generation) < self.config.population_size:
                next_generation.append(child2)
        
        self.population = next_generation
    
    # ========================================================================
    # SELECTION
    # ========================================================================
    
    def _select(self) -> Individual:
        """Select an individual for reproduction."""
        return self.selector.select(self.population)
    
    def _tournament_selection(self) -> Individual:
        """Tournament selection."""
        tournament = np.random.choice(
            self.population,
            size=min(self.config.tournament_size, len(self.population)),
            replace=False
        )
        
        return max(tournament, key=lambda x:  x.fitness)
    
    def _roulette_selection(self) -> Individual:
        """Roulette wheel selection."""
        fitnesses = np.array([ind.fitness for ind in self.population])
        
        # Shift to positive if needed
        if fitnesses.min() < 0:
            fitnesses = fitnesses - fitnesses.min()
        
        # Normalize
        total_fitness = fitnesses.sum()
        
        if total_fitness == 0:
            # All equal, random selection
            return np.random.choice(self.population)
        
        probabilities = fitnesses / total_fitness
        
        return np.random.choice(self.population, p=probabilities)
    
    def _rank_selection(self) -> Individual:
        """Rank-based selection."""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness)
        
        # Assign ranks (1 to N)
        ranks = np.arange(1, len(sorted_pop) + 1)
        probabilities = ranks / ranks.sum()
        
        return np.random.choice(sorted_pop, p=probabilities)
    
    # ========================================================================
    # CROSSOVER
    # ========================================================================
    
    def _crossover(self, 
                   parent1: Dict[str, Any], 
                   parent2: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]: 
        """
        Uniform crossover. 
        
        Args:
            parent1: First parent chromosome
            parent2: Second parent chromosome
            
        Returns:
            Two offspring chromosomes
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
    # MUTATION
    # ========================================================================
    
    def _mutate(self, chromosome: Dict[str, Any]) -> Dict[str, Any]:
        """
        Mutate chromosome.
        
        Args:
            chromosome: Chromosome to mutate
            
        Returns:
            Mutated chromosome
        """
        mutated = chromosome.copy()
        
        # Select random gene to mutate
        gene = np.random.choice(list(mutated.keys()))
        
        # Get possible values
        possible_values = self.chromosome_template[gene]
        
        if isinstance(possible_values, list):
            # Discrete:  random choice
            mutated[gene] = np.random.choice(possible_values)
        
        elif isinstance(possible_values, tuple) and len(possible_values) == 2:
            # Continuous:  Gaussian perturbation
            low, high = possible_values
            current = mutated[gene]
            
            if isinstance(low, int) and isinstance(high, int):
                # Integer range
                mutation = np.random.randint(-2, 3)  # Small perturbation
                mutated[gene] = np.clip(current + mutation, low, high)
            else:
                # Float range
                sigma = (high - low) * 0.1  # 10% of range
                mutation = np.random.normal(0, sigma)
                mutated[gene] = np.clip(current + mutation, low, high)
        
        return mutated
    
    # ========================================================================
    # STATISTICS & TRACKING
    # ========================================================================
    
    def _update_best(self):
        """Update best individual found so far."""
        current_best = max(self.population, key=lambda x: x.fitness)
        
        if self.best_individual is None or current_best.fitness > self.best_individual.fitness:
            self.best_individual = current_best
            
            # Check for improvement
            if current_best.fitness > self.best_fitness_so_far + self.config.min_improvement:
                self.best_fitness_so_far = current_best.fitness
                self.generations_without_improvement = 0
            else:
                self.generations_without_improvement += 1
        else:
            self.generations_without_improvement += 1
    
    def _calculate_statistics(self, eval_time: float, total_time: float) -> GenerationStats:
        """Calculate generation statistics."""
        fitnesses = [ind.fitness for ind in self.population]
        
        # Diversity (standard deviation of fitnesses)
        diversity = np.std(fitnesses)
        
        return GenerationStats(
            generation=self.generation,
            best_fitness=max(fitnesses),
            mean_fitness=np.mean(fitnesses),
            std_fitness=np.std(fitnesses),
            worst_fitness=min(fitnesses),
            diversity=diversity,
            evaluation_time=eval_time,
            total_time=total_time
        )
    
    def _check_convergence(self) -> bool:
        """Check if algorithm has converged."""
        if not self.config.early_stopping:
            return False
        
        # Check patience
        if self.generations_without_improvement >= self.config.patience:
            return True
        
        # Check diversity
        if len(self.history) > 0:
            if self.history[-1].diversity < self.config.diversity_threshold:
                if self.logger and self.config.verbose >= 1:
                    self.logger.warning("Low diversity detected")
                return True
        
        return False
    
    # ========================================================================
    # LOGGING
    # ========================================================================
    
    def _log_generation(self, stats: GenerationStats):
        """Log generation progress."""
        if not self.logger or self.config.verbose == 0:
            return
        
        if self.config.verbose >= 1:
            # Standard logging (every 10 generations or first/last)
            if stats.generation % 10 == 0 or stats.generation == 0 or stats.generation == self.config.num_generations - 1:
                print_info(
                    f"Gen {stats.generation:3d} | "
                    f"Best: {stats.best_fitness:.4f} | "
                    f"Mean: {stats.mean_fitness:.4f}±{stats.std_fitness:.4f} | "
                    f"Diversity: {stats.diversity:.4f} | "
                    f"Time: {stats.total_time:.2f}s"
                )
        
        if self.config.verbose >= 2:
            # Detailed logging (every generation)
            self.logger.debug(
                f"Gen {stats.generation}:  "
                f"Best={stats.best_fitness:.4f}, "
                f"Worst={stats.worst_fitness:.4f}, "
                f"Evals={self.total_evaluations}"
            )
    
    def _print_summary(self):
        """Print final summary."""
        if not self.logger:
            return
        
        total_time = time.time() - self.start_time
        
        print()
        print_header("GA OPTIMIZATION COMPLETE")
        print()
        
        print_success(f"Best fitness: {self.best_individual.fitness:.4f}")
        print_info(f"Found at generation:  {self.best_individual.generation}")
        print_info(f"Total generations: {self.generation + 1}")
        print_info(f"Total evaluations: {self.total_evaluations}")
        print_info(f"Total time: {total_time:.2f}s")
        print_info(f"Avg time per generation: {total_time / (self.generation + 1):.2f}s")
        
        print()
        print_section("Best Chromosome")
        for gene, value in self.best_individual.chromosome.items():
            print(f"  {gene}:  {value}")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_results(self, filepath: Path):
        """Save GA results to file."""
        results = {
            'config': asdict(self.config),
            'best_individual': {
                'chromosome': self.best_individual.chromosome,
                'fitness': float(self.best_individual.fitness),
                'generation': int(self.best_individual.generation)
            },
            'history': [asdict(stats) for stats in self.history],
            'total_evaluations': int(self.total_evaluations),
            'total_time': float(time.time() - self.start_time)
        }
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Results saved to:  {filepath}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    
    logger = get_logger(name="GA_TEST", verbose=True)
    
    # Simple test:  maximize sum of values
    def fitness_function(chromosome):
        """Simple fitness:  sum of all values."""
        time.sleep(0.01)  # Simulate evaluation time
        return sum(chromosome.values())
    
    # Chromosome template
    template = {
        'x1': list(range(10)),
        'x2': list(range(10)),
        'x3': list(range(10))
    }
    
    # Configuration
    config = GAConfig(
        population_size=20,
        num_generations=50,
        crossover_rate=0.8,
        mutation_rate=0.1,
        elitism_rate=0.1,
        verbose=1
    )
    
    # Run GA
    ga = GeneticAlgorithm(config, fitness_function, template, logger)
    best = ga.run()
    
    print()
    print_success(f"Best solution found: {best.chromosome}")
    print_info(f"Fitness: {best.fitness}")
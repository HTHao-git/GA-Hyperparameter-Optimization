# ============================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================
# Multi-objective GA using NSGA-II algorithm
#
# FEATURES:
#   - Pareto ranking
#   - Crowding distance
#   - Non-dominated sorting
#   - Multi-objective fitness evaluation
#
# USAGE:
#   from ga.multi_objective import MultiObjectiveGA
#   
#   ga = MultiObjectiveGA(objectives=['accuracy', 'speed'])
#   pareto_front = ga.optimize()
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import List, Dict, Any, Tuple, Callable, Optional
from dataclasses import dataclass, field

from ga.types import Individual
from utils.logger import Logger
from utils.colors import print_info, print_success


# ============================================================================
# MULTI-OBJECTIVE INDIVIDUAL
# ============================================================================

@dataclass
class MultiObjectiveIndividual(Individual):
    """Individual with multiple objectives."""
    
    objectives: Dict[str, float] = field(default_factory=dict)  # Multiple fitness values
    rank: int = -1  # Pareto rank (0 = non-dominated)
    crowding_distance: float = 0.0  # Diversity metric
    dominated_by: List['MultiObjectiveIndividual'] = field(default_factory=list)
    dominates: List['MultiObjectiveIndividual'] = field(default_factory=list)


# ============================================================================
# MULTI-OBJECTIVE UTILITIES
# ============================================================================

class MultiObjectiveUtils:
    """Utilities for multi-objective optimization."""
    
    @staticmethod
    def dominates(ind1: MultiObjectiveIndividual, 
                  ind2: MultiObjectiveIndividual,
                  minimize: List[str] = []) -> bool:
        """
        Check if ind1 dominates ind2.
        
        ind1 dominates ind2 if:
        - ind1 is no worse than ind2 in all objectives
        - ind1 is strictly better than ind2 in at least one objective
        
        Args:
            ind1, ind2: Individuals to compare
            minimize: List of objectives to minimize (others are maximized)
            
        Returns:
            True if ind1 dominates ind2
        """
        better_in_any = False
        
        for obj_name, obj_value1 in ind1.objectives.items():
            obj_value2 = ind2.objectives[obj_name]
            
            if obj_name in minimize:
                # Minimize this objective (lower is better)
                if obj_value1 > obj_value2:
                    return False  # ind1 is worse
                elif obj_value1 < obj_value2:
                    better_in_any = True
            else:
                # Maximize this objective (higher is better)
                if obj_value1 < obj_value2:
                    return False  # ind1 is worse
                elif obj_value1 > obj_value2:
                    better_in_any = True
        
        return better_in_any
    
    @staticmethod
    def fast_non_dominated_sort(population: List[MultiObjectiveIndividual],
                                minimize: List[str] = []) -> List[List[MultiObjectiveIndividual]]:
        """
        NSGA-II fast non-dominated sorting.
        
        Assigns Pareto ranks to individuals. 
        Rank 0 = non-dominated (Pareto front)
        Rank 1 = dominated only by rank 0
        etc.
        
        Args:
            population: Population to sort
            minimize:  Objectives to minimize
            
        Returns:
            List of fronts (each front is a list of individuals)
        """
        # Reset domination info
        for ind in population:
            ind.dominated_by = []
            ind.dominates = []
            ind.rank = -1
        
        # Calculate domination
        for i, ind1 in enumerate(population):
            for ind2 in population[i+1:]:
                if MultiObjectiveUtils.dominates(ind1, ind2, minimize):
                    ind1.dominates.append(ind2)
                    ind2.dominated_by.append(ind1)
                elif MultiObjectiveUtils.dominates(ind2, ind1, minimize):
                    ind2.dominates.append(ind1)
                    ind1.dominated_by.append(ind2)
        
        # Build fronts
        fronts = []
        current_front = []
        
        # First front: non-dominated individuals
        for ind in population: 
            if len(ind.dominated_by) == 0:
                ind.rank = 0
                current_front.append(ind)
        
        fronts.append(current_front)
        
        # Subsequent fronts
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            
            for ind1 in fronts[i]: 
                for ind2 in ind1.dominates:
                    ind2.dominated_by.remove(ind1)
                    
                    if len(ind2.dominated_by) == 0:
                        ind2.rank = i + 1
                        next_front.append(ind2)
            
            i += 1
            fronts.append(next_front)
        
        # Remove empty last front
        if len(fronts[-1]) == 0:
            fronts.pop()
        
        return fronts
    
    @staticmethod
    def calculate_crowding_distance(front: List[MultiObjectiveIndividual],
                                    minimize: List[str] = []):
        """
        Calculate crowding distance for a front.
        
        Crowding distance measures how close an individual is to its neighbors. 
        Higher distance = more isolated = more diverse.
        
        Args:
            front: List of individuals in the same front
            minimize: Objectives to minimize
        """
        if len(front) == 0:
            return
        
        # Reset crowding distance
        for ind in front:
            ind.crowding_distance = 0.0
        
        # Number of objectives
        n_objectives = len(front[0].objectives)
        
        # For each objective
        for obj_name in front[0].objectives.keys():
            # Sort by this objective
            front_sorted = sorted(front, key=lambda x: x.objectives[obj_name])
            
            # Boundary individuals get infinite distance
            front_sorted[0].crowding_distance = float('inf')
            front_sorted[-1].crowding_distance = float('inf')
            
            # Get objective range
            obj_min = front_sorted[0].objectives[obj_name]
            obj_max = front_sorted[-1].objectives[obj_name]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            # Calculate crowding distance for middle individuals
            for i in range(1, len(front_sorted) - 1):
                distance = (front_sorted[i+1].objectives[obj_name] - 
                           front_sorted[i-1].objectives[obj_name]) / obj_range
                front_sorted[i].crowding_distance += distance
    
    @staticmethod
    def crowding_distance_selection(front: List[MultiObjectiveIndividual],
                                    n_select: int) -> List[MultiObjectiveIndividual]: 
        """
        Select individuals from front based on crowding distance.
        
        Args:
            front: Front to select from
            n_select:  Number to select
            
        Returns: 
            Selected individuals (most diverse)
        """
        # Sort by crowding distance (descending)
        sorted_front = sorted(front, key=lambda x: x.crowding_distance, reverse=True)
        
        return sorted_front[:n_select]

# ============================================================================
# NSGA-II ALGORITHM
# ============================================================================

class NSGAII: 
    """
    NSGA-II (Non-dominated Sorting Genetic Algorithm II) implementation.
    
    Multi-objective optimization using Pareto ranking and crowding distance.
    
    Args:
        population_size: Number of individuals
        num_generations: Maximum generations
        objectives: List of objective types ('maximize' or 'minimize')
        crossover_rate: Crossover probability
        mutation_rate: Mutation probability
        random_state: Random seed
        logger: Logger instance
    
    Example:
        >>> def fitness_func(config):
        ...     accuracy = train_model(config)
        ...     speed = 1.0 / training_time
        ...     return [accuracy, speed]  # Return list of objectives
        
        >>> nsga2 = NSGAII(
        ...     population_size=30,
        ...     num_generations=20,
        ...     objectives=['maximize', 'maximize']
        ... )
        
        >>> pareto_front = nsga2.optimize(fitness_func, chromosome_template)
    """
    
    def __init__(self,
                 population_size: int = 50,
                 num_generations:  int = 100,
                 objectives: List[str] = ['maximize', 'maximize'],
                 crossover_rate: float = 0.8,
                 mutation_rate: float = 0.1,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        
        self.population_size = population_size
        self.num_generations = num_generations
        self. objectives = objectives  # List of 'maximize' or 'minimize'
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.random_state = random_state
        self.logger = logger
        
        # Which objectives to minimize
        self.minimize_objectives = [
            i for i, obj in enumerate(objectives) if obj == 'minimize'
        ]
        
        # Set random seed
        np.random.seed(random_state)
        
        if self.logger:
            self.logger.info(f"NSGA-II initialized")
            self.logger.info(f"  Population:  {population_size}")
            self.logger.info(f"  Generations: {num_generations}")
            self.logger.info(f"  Objectives: {len(objectives)}")
    
    def optimize(self,
                 fitness_function:  Callable,
                 chromosome_template: Dict[str, Any]) -> List[MultiObjectiveIndividual]:
        """
        Run NSGA-II optimization.
        
        Args:
            fitness_function: Function that takes config, returns list of objective values
            chromosome_template:  Hyperparameter search space
            
        Returns: 
            Pareto front (list of non-dominated solutions)
        """
        from ga.genetic_algorithm import GeneticAlgorithm
        
        # Store template for mutation/crossover
        self.chromosome_template = chromosome_template
        self.fitness_function = fitness_function
        
        # Initialize population
        if self.logger:
            print_info("Initializing population...")
        
        population = self._initialize_population()
        
        if self.logger:
            print_success(f"Population initialized:  {len(population)} individuals")
        
        # Evaluate initial population
        self._evaluate_population(population)
        
        # Evolution loop
        for generation in range(self.num_generations):
            import time
            gen_start = time.time()
            
            # Create offspring
            offspring = self._create_offspring(population)
            
            # Evaluate offspring
            self._evaluate_population(offspring)
            
            # Combine parent and offspring
            combined = population + offspring
            
            # Non-dominated sorting
            fronts = MultiObjectiveUtils.fast_non_dominated_sort(
                combined, 
                minimize=[f"obj_{i}" for i in self.minimize_objectives]
            )
            
            # Calculate crowding distance for each front
            for front in fronts:
                MultiObjectiveUtils.calculate_crowding_distance(
                    front,
                    minimize=[f"obj_{i}" for i in self.minimize_objectives]
                )
            
            # Select next generation
            population = self._environmental_selection(fronts)
            
            gen_time = time.time() - gen_start
            
            # Logging
            if self.logger:
                best_front = fronts[0]
                avg_objectives = {
                    f"obj_{i}": np.mean([ind. objectives[f"obj_{i}"] for ind in best_front])
                    for i in range(len(self.objectives))
                }
                
                obj_str = ", ".join([f"{v:.4f}" for v in avg_objectives.values()])
                
                print_info(f"Gen {generation: 3d} | Pareto size: {len(best_front):3d} | "
                          f"Avg objectives: [{obj_str}] | Time: {gen_time:.2f}s")
        
        # Return final Pareto front
        final_fronts = MultiObjectiveUtils.fast_non_dominated_sort(
            population,
            minimize=[f"obj_{i}" for i in self.minimize_objectives]
        )
        
        pareto_front = final_fronts[0] if final_fronts else []
        
        if self.logger:
            print()
            print_success(f"✓ NSGA-II complete!")
            print_info(f"  Pareto front size: {len(pareto_front)}")
        
        return pareto_front
    
    def _initialize_population(self) -> List[MultiObjectiveIndividual]: 
        """Initialize random population."""
        from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
        
        # Create temporary GA config
        temp_config = GAConfig(
            population_size=self.population_size,
            random_state=self.random_state
        )
        
        # Create temporary GA instance to use its initialization
        temp_ga = GeneticAlgorithm(
            config=temp_config,
            fitness_function=lambda x: 0.0,
            chromosome_template=self.chromosome_template,
            logger=None
        )
        
        population = []
        for _ in range(self.population_size):
            # Use GA's initialization method
            individual = temp_ga._initialize_individual()
            
            # Convert to MultiObjectiveIndividual
            mo_individual = MultiObjectiveIndividual(
                chromosome=individual. chromosome,
                evaluated=False
            )
            population.append(mo_individual)
        
        return population
    
    def _evaluate_population(self, population: List[MultiObjectiveIndividual]):
        """Evaluate all unevaluated individuals."""
        for individual in population:
            if not individual.evaluated:
                # Get objective values
                objective_values = self.fitness_function(individual.chromosome)
                
                # Store as dictionary
                individual.objectives = {
                    f"obj_{i}": float(val)
                    for i, val in enumerate(objective_values)
                }
                
                # Also store in fitness for compatibility
                individual.fitness = objective_values
                individual.evaluated = True
    
    def _create_offspring(self, population: List[MultiObjectiveIndividual]) -> List[MultiObjectiveIndividual]:
        """Create offspring through tournament selection, crossover, and mutation."""
        offspring = []
        
        while len(offspring) < self.population_size:
            # Tournament selection
            parent1 = self._tournament_selection(population)
            parent2 = self._tournament_selection(population)
            
            # Crossover
            if np.random.random() < self.crossover_rate:
                child1_chromo, child2_chromo = self._crossover(parent1.chromosome, parent2.chromosome)
            else:
                child1_chromo = parent1.chromosome. copy()
                child2_chromo = parent2.chromosome.copy()
            
            # Mutation
            if np.random.random() < self.mutation_rate:
                child1_chromo = self._mutate(child1_chromo)
            if np.random.random() < self.mutation_rate:
                child2_chromo = self._mutate(child2_chromo)
            
            # Create offspring individuals
            offspring.append(MultiObjectiveIndividual(chromosome=child1_chromo, evaluated=False))
            if len(offspring) < self.population_size:
                offspring.append(MultiObjectiveIndividual(chromosome=child2_chromo, evaluated=False))
        
        return offspring[: self.population_size]
    
    def _tournament_selection(self, population: List[MultiObjectiveIndividual], 
                              tournament_size: int = 2) -> MultiObjectiveIndividual:
        """Binary tournament selection based on rank and crowding distance."""
        # Select random individuals
        candidates = np.random.choice(population, size=tournament_size, replace=False)
        
        # Compare based on rank first, then crowding distance
        best = candidates[0]
        for candidate in candidates[1:]:
            if candidate.rank < best.rank:
                best = candidate
            elif candidate.rank == best.rank and candidate.crowding_distance > best.crowding_distance:
                best = candidate
        
        return best
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Tuple[Dict, Dict]:
        """Uniform crossover."""
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
    
    def _mutate(self, chromosome:  Dict) -> Dict:
        """Mutate chromosome."""
        mutated = chromosome.copy()
        
        for gene, value in mutated.items():
            if np.random.random() < 0.3:  # 30% chance per gene
                possible_values = self. chromosome_template[gene]
                
                if isinstance(possible_values, tuple) and len(possible_values) == 2:
                    # Continuous range
                    low, high = possible_values
                    mutated[gene] = np.random.uniform(low, high)
                elif isinstance(possible_values, list):
                    # Discrete values
                    mutated[gene] = np.random.choice(possible_values)
        
        return mutated
    
    def _environmental_selection(self, fronts: List[List[MultiObjectiveIndividual]]) -> List[MultiObjectiveIndividual]:
        """Select next generation from fronts."""
        next_population = []
        
        for front in fronts:
            if len(next_population) + len(front) <= self.population_size:
                # Add entire front
                next_population.extend(front)
            else:
                # Add part of front based on crowding distance
                remaining = self.population_size - len(next_population)
                selected = MultiObjectiveUtils.crowding_distance_selection(front, remaining)
                next_population.extend(selected)
                break
        
        return next_population[: self.population_size]

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section
    
    logger = get_logger(name="MULTI_OBJ_TEST", verbose=True)
    
    print_header("MULTI-OBJECTIVE OPTIMIZATION TEST")
    print()
    
    # Create test population
    population = []
    
    # Generate random individuals with 2 objectives
    np.random.seed(42)
    for i in range(20):
        ind = MultiObjectiveIndividual(
            chromosome={'x': i},
            objectives={
                'accuracy': np.random.random(),  # Maximize
                'speed': np.random.random()       # Maximize
            }
        )
        population.append(ind)
    
    print_section("Original Population")
    for ind in population[: 5]:
        print(f"  {ind.chromosome} → Acc: {ind.objectives['accuracy']:.3f}, Speed: {ind.objectives['speed']:.3f}")
    print(f"  ... and {len(population)-5} more")
    print()
    
    # Perform non-dominated sorting
    print_section("Non-Dominated Sorting")
    fronts = MultiObjectiveUtils.fast_non_dominated_sort(population)
    
    print_info(f"Found {len(fronts)} Pareto fronts:")
    for i, front in enumerate(fronts):
        print(f"  Front {i} (rank {i}): {len(front)} individuals")
    print()
    
    # Calculate crowding distance for first front
    print_section("Crowding Distance (Front 0)")
    MultiObjectiveUtils.calculate_crowding_distance(fronts[0])
    
    for ind in sorted(fronts[0], key=lambda x: x.crowding_distance, reverse=True)[:5]:
        print(f"  Acc: {ind.objectives['accuracy']:.3f}, Speed: {ind.objectives['speed']:.3f}, Distance: {ind.crowding_distance:.3f}")
    print()
    
    print_success("✓ Multi-objective test complete!")
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
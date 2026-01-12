# ============================================================================
# DIVERSITY MAINTENANCE & NICHING
# ============================================================================
# Diversity preservation mechanisms for Genetic Algorithms
#
# FEATURES:
#   - Fitness sharing (penalize crowded regions)
#   - Niching (maintain diverse subpopulations)
#   - Crowding (replace similar individuals)
#   - Diversity metrics
#   - Speciation
#
# USAGE:
#   from ga.diversity import DiversityManager
#   
#   diversity_mgr = DiversityManager(method='fitness_sharing')
#   adjusted_fitness = diversity_mgr.apply(population)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass

from ga.types import Individual
from utils.logger import Logger
from utils.colors import print_info, print_warning


# ============================================================================
# DIVERSITY MANAGER CLASS
# ============================================================================

class DiversityManager:
    """
    Manages population diversity using various techniques. 
    
    Args:
        method:  Diversity method ('fitness_sharing', 'crowding', 'speciation', 'none')
        sigma_share:  Sharing radius for fitness sharing
        alpha: Sharing function exponent
        crowding_factor: Replacement factor for crowding
        species_threshold: Distance threshold for speciation
        logger: Logger instance
    """
    
    def __init__(self,
                 method: str = 'fitness_sharing',
                 sigma_share: float = 0.1,
                 alpha: float = 1.0,
                 crowding_factor: int = 3,
                 species_threshold:  float = 0.1,
                 logger: Optional[Logger] = None):
        
        self.method = method
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.crowding_factor = crowding_factor
        self.species_threshold = species_threshold
        self.logger = logger
        
        # Valid methods
        self.valid_methods = [
            'fitness_sharing',
            'crowding',
            'speciation',
            'clearing',
            'none'
        ]
        
        if method not in self.valid_methods:
            raise ValueError(f"Invalid diversity method '{method}'.Valid:  {self.valid_methods}")
    
    # ========================================================================
    # MAIN DIVERSITY INTERFACE
    # ========================================================================
    
    def apply(self, population: List[Individual]) -> List[Individual]:
        """
        Apply diversity maintenance to population.
        
        Args:
            population: Population to process
            
        Returns:
            Population with adjusted fitness/diversity
        """
        if self.method == 'none':
            return population
        
        elif self.method == 'fitness_sharing':
            return self._apply_fitness_sharing(population)
        
        elif self.method == 'crowding':
            return self._apply_crowding(population)
        
        elif self.method == 'speciation': 
            return self._apply_speciation(population)
        
        elif self.method == 'clearing': 
            return self._apply_clearing(population)
        
        else:
            raise ValueError(f"Unknown diversity method: {self.method}")
    
    def calculate_diversity(self, population: List[Individual]) -> float:
        """
        Calculate population diversity metric.
        
        Args:
            population: Population to measure
            
        Returns:
            Diversity score (higher = more diverse)
        """
        if len(population) < 2:
            return 0.0
        
        # Calculate average pairwise distance
        total_distance = 0.0
        count = 0
        
        for i, ind1 in enumerate(population):
            for ind2 in population[i+1:]:
                distance = self._chromosome_distance(ind1.chromosome, ind2.chromosome)
                total_distance += distance
                count += 1
        
        if count == 0:
            return 0.0
        
        return total_distance / count
    
    # ========================================================================
    # FITNESS SHARING
    # ========================================================================
    
    def _apply_fitness_sharing(self, population: List[Individual]) -> List[Individual]:
        """
        Fitness sharing:  Reduce fitness of individuals in crowded regions.
        
        Shared fitness = Original fitness / Niche count
        
        Where niche count is the sum of sharing function values with all other individuals. 
        
        Args:
            population: Population to process
            
        Returns: 
            Population with shared fitness
        """
        n = len(population)
        
        # Calculate niche counts
        niche_counts = np.zeros(n)
        
        for i, ind1 in enumerate(population):
            for j, ind2 in enumerate(population):
                distance = self._chromosome_distance(ind1.chromosome, ind2.chromosome)
                niche_counts[i] += self._sharing_function(distance)
        
        # Adjust fitness
        for i, ind in enumerate(population):
            if niche_counts[i] > 0:
                # Store original fitness
                if not hasattr(ind, 'original_fitness'):
                    ind.original_fitness = ind.fitness
                
                # Apply sharing
                ind.fitness = ind.original_fitness / niche_counts[i]
        
        return population
    
    def _sharing_function(self, distance: float) -> float:
        """
        Sharing function: sh(d).
        
        sh(d) = 1 - (d / sigma_share)^alpha   if d < sigma_share
        sh(d) = 0                             otherwise
        
        Args:
            distance: Distance between individuals
            
        Returns:
            Sharing value [0, 1]
        """
        if distance < self.sigma_share:
            return 1.0 - (distance / self.sigma_share) ** self.alpha
        else:
            return 0.0
    
    # ========================================================================
    # CROWDING
    # ========================================================================
    
    def _apply_crowding(self, population: List[Individual]) -> List[Individual]:
        """
        Deterministic crowding:  Offspring replace most similar parents.
        
        Maintains diversity by ensuring offspring don't cluster. 
        
        Note: This is typically applied during evolution, not as a post-process.
        For now, we'll just mark it as a placeholder.
        
        Args:
            population: Population to process
            
        Returns:
            Population (unchanged - crowding happens during evolution)
        """
        # Crowding is applied during evolution in _evolve_with_crowding
        # This is a placeholder for consistency
        return population
    
    # ========================================================================
    # SPECIATION
    # ========================================================================
    
    def _apply_speciation(self, population: List[Individual]) -> List[Individual]: 
        """
        Speciation: Group individuals into species based on similarity.
        
        Individuals only compete within their species. 
        
        Args:
            population: Population to process
            
        Returns:
            Population with species assignments
        """
        # Cluster individuals into species
        species = self._form_species(population)
        
        if self.logger:
            print_info(f"Formed {len(species)} species")
        
        # Adjust fitness within each species
        for species_members in species:
            if len(species_members) > 0:
                # Share fitness within species
                for ind in species_members:
                    if not hasattr(ind, 'original_fitness'):
                        ind.original_fitness = ind.fitness
                    
                    # Divide by species size
                    ind.fitness = ind.original_fitness / len(species_members)
        
        return population
    
    def _form_species(self, population: List[Individual]) -> List[List[Individual]]:
        """
        Form species using distance-based clustering.
        
        Args:
            population: Population to cluster
            
        Returns: 
            List of species (each species is a list of individuals)
        """
        species = []
        assigned = [False] * len(population)
        
        for i, ind in enumerate(population):
            if assigned[i]:
                continue
            
            # Create new species with this individual
            current_species = [ind]
            assigned[i] = True
            
            # Find similar individuals
            for j, other_ind in enumerate(population):
                if assigned[j]:
                    continue
                
                distance = self._chromosome_distance(ind.chromosome, other_ind.chromosome)
                
                if distance < self.species_threshold:
                    current_species.append(other_ind)
                    assigned[j] = True
            
            species.append(current_species)
        
        return species
    
    # ========================================================================
    # CLEARING
    # ========================================================================
    
    def _apply_clearing(self, population: List[Individual]) -> List[Individual]:
        """
        Clearing: Keep only the best individual in each niche.
        
        Within each niche (defined by sigma_share), only the fittest individual
        retains its fitness. Others are set to 0.
        
        Args:
            population: Population to process
            
        Returns: 
            Population with cleared fitness
        """
        # Sort by fitness (descending)
        sorted_pop = sorted(population, key=lambda x: x.fitness, reverse=True)
        
        # Track which individuals have been cleared
        cleared = [False] * len(sorted_pop)
        
        for i, ind in enumerate(sorted_pop):
            if cleared[i]:
                continue
            
            # This individual is the winner in its niche
            # Clear all nearby individuals
            for j, other_ind in enumerate(sorted_pop[i+1:], start=i+1):
                if cleared[j]:
                    continue
                
                distance = self._chromosome_distance(ind.chromosome, other_ind.chromosome)
                
                if distance < self.sigma_share:
                    # Clear this individual
                    if not hasattr(other_ind, 'original_fitness'):
                        other_ind.original_fitness = other_ind.fitness
                    
                    other_ind.fitness = 0.0
                    cleared[j] = True
        
        return population
    
    # ========================================================================
    # DISTANCE METRICS
    # ========================================================================
    
    def _chromosome_distance(self, chrom1: Dict[str, Any], chrom2: Dict[str, Any]) -> float:
        """
        Calculate normalized distance between two chromosomes.
        
        Uses Euclidean distance for numerical genes, 
        Hamming distance for categorical genes. 
        
        Args:
            chrom1, chrom2: Chromosomes to compare
            
        Returns: 
            Normalized distance [0, 1]
        """
        if set(chrom1.keys()) != set(chrom2.keys()):
            raise ValueError("Chromosomes have different genes")
        
        total_distance = 0.0
        n_genes = len(chrom1)
        
        for gene in chrom1.keys():
            val1 = chrom1[gene]
            val2 = chrom2[gene]
            
            if isinstance(val1, (int, float, np.number)) and isinstance(val2, (int, float, np.number)):
                # Numerical:  normalized absolute difference
                # Assume range [0, 1] or normalize if needed
                gene_distance = abs(float(val1) - float(val2))
            else:
                # Categorical:  Hamming distance
                gene_distance = 0.0 if val1 == val2 else 1.0
            
            total_distance += gene_distance
        
        # Normalize by number of genes
        return total_distance / n_genes if n_genes > 0 else 0.0
    
    # ========================================================================
    # DIVERSITY METRICS
    # ========================================================================
    
    def calculate_genotypic_diversity(self, population:  List[Individual]) -> float:
        """
        Calculate genotypic diversity (average chromosome distance).
        
        Args:
            population: Population to measure
            
        Returns:
            Diversity score
        """
        return self.calculate_diversity(population)
    
    def calculate_phenotypic_diversity(self, population: List[Individual]) -> float:
        """
        Calculate phenotypic diversity (fitness variance).
        
        Args:
            population: Population to measure
            
        Returns:
            Diversity score (standard deviation of fitness)
        """
        fitnesses = [ind.fitness for ind in population]
        return float(np.std(fitnesses))


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success
    
    logger = get_logger(name="DIVERSITY_TEST", verbose=True)
    
    print_header("DIVERSITY MAINTENANCE TEST")
    print()
    
    # Create test population with some similar individuals
    np.random.seed(42)
    population = []
    
    # Create clusters
    for cluster in range(3):
        for i in range(5):
            # Add noise to create similar individuals
            base_value = cluster * 30
            ind = Individual(
                chromosome={
                    'x1': base_value + np.random.randint(0, 10),
                    'x2': base_value + np.random.randint(0, 10),
                    'x3': f'option{cluster}'
                },
                fitness=np.random.uniform(0.5, 1.0),
                evaluated=True
            )
            population.append(ind)
    
    print_section("Original Population")
    print_info(f"Population size: {len(population)}")
    
    for i, ind in enumerate(population[: 6]):
        print(f"  {i+1}.{ind.chromosome} → Fitness: {ind.fitness:.3f}")
    print(f"  ... and {len(population)-6} more")
    print()
    
    # Test different diversity methods
    methods = ['fitness_sharing', 'speciation', 'clearing']
    
    for method in methods:
        print_section(f"Method: {method.upper()}")
        
        # Create fresh copy
        test_pop = [Individual(
            chromosome=ind.chromosome.copy(),
            fitness=ind.fitness,
            evaluated=True
        ) for ind in population]
        
        diversity_mgr = DiversityManager(
            method=method,
            sigma_share=15.0,  # Larger radius to capture clusters
            species_threshold=15.0,
            logger=logger
        )
        
        # Calculate initial diversity
        initial_diversity = diversity_mgr.calculate_diversity(test_pop)
        print_info(f"Initial diversity: {initial_diversity:.3f}")
        
        # Apply diversity mechanism
        adjusted_pop = diversity_mgr.apply(test_pop)
        
        # Show fitness changes
        print_info("Fitness adjustments (first 6):")
        for i, ind in enumerate(adjusted_pop[:6]):
            original = getattr(ind, 'original_fitness', ind.fitness)
            print(f"  {i+1}.Original: {original:.3f} → Adjusted: {ind.fitness:.3f}")
        
        # Calculate final diversity
        final_diversity = diversity_mgr.calculate_phenotypic_diversity(adjusted_pop)
        print_info(f"Phenotypic diversity: {final_diversity:.3f}")
        
        print()
    
    print_success("✓ Diversity maintenance test complete!")
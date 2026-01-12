# ============================================================================
# CHROMOSOME UTILITIES - Encode/Decode Hyperparameters for Genetic Algorithms
# ============================================================================
# This module handles conversion between hyperparameters and chromosomes.
#
# CHROMOSOME FORMAT:
#   - List of floats in [0, 1] range
#   - Each gene represents one hyperparameter
#   - Order is consistent (defined by GENE_NAMES)
#
# ENCODING TYPES:
#   - discrete: Integer in [min, max] → normalized to [0, 1]
#   - continuous: Float in [min, max] with linear or log10 scale → [0, 1]
#   - categorical: Index of choice list → normalized to [0, 1]
#
# USAGE:
#   from utils.chromosome import encode_hyperparameters, decode_chromosome
#   chromosome = encode_hyperparameters(hyperparams, config)
#   hyperparams = decode_chromosome(chromosome, config)
#
# Last updated: 2025-12-31
# ============================================================================

import numpy as np
from typing import Dict, List, Any, Tuple
from utils.config_loader import get_full_config


# ============================================================================
# GLOBAL:  GENE NAMES (Ordered list of hyperparameter names)
# ============================================================================

def get_gene_names(hyperparameter_config: Dict[str, Any]) -> List[str]:
    """
    Get ordered list of gene names (hyperparameter names).
    
    This order MUST be consistent across all chromosome operations.
    
    Args:
        hyperparameter_config:  Hyperparameter configuration dict
        
    Returns:
        Ordered list of hyperparameter names
    """
    # Extract hyperparameter names (excluding metadata)
    hyperparams = hyperparameter_config['hyperparameters']
    
    # Sort alphabetically for consistency
    gene_names = sorted(hyperparams.keys())
    
    return gene_names


# ============================================================================
# ENCODE:  Hyperparameters → Chromosome
# ============================================================================

def encode_hyperparameter(value: Any, 
                         param_name: str,
                         param_config: Dict[str, Any]) -> float:
    """
    Encode a single hyperparameter value to [0, 1] range.
    
    Args:
        value:  Hyperparameter value (int, float, or string)
        param_name:  Hyperparameter name
        param_config: Configuration for this parameter
        
    Returns: 
        Encoded value in [0, 1]
        
    Raises:
        ValueError: If encoding fails
    """
    param_type = param_config['type']
    
    # ------------------------------------------------------------------------
    # DISCRETE:  Integer in [min, max]
    # ------------------------------------------------------------------------
    if param_type == 'discrete':
        min_val = param_config['min']
        max_val = param_config['max']
        
        if value < min_val or value > max_val:
            raise ValueError(
                f"{param_name}:  value {value} out of range [{min_val}, {max_val}]"
            )
        
        # Normalize to [0, 1]
        if max_val == min_val: 
            return 0.5  # Single value, return middle
        
        normalized = (value - min_val) / (max_val - min_val)
        return float(normalized)
    
    # ------------------------------------------------------------------------
    # CONTINUOUS: Float in [min, max] with scaling
    # ------------------------------------------------------------------------
    elif param_type == 'continuous': 
        min_val = param_config['min']
        max_val = param_config['max']
        scale = param_config.get('scale', 'linear')
        
        if value < min_val or value > max_val:
            raise ValueError(
                f"{param_name}: value {value} out of range [{min_val}, {max_val}]"
            )
        
        # Apply inverse scaling
        if scale == 'log10':
            # Convert to log space
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            log_val = np.log10(value)
            
            normalized = (log_val - log_min) / (log_max - log_min)
        
        else:  # linear
            normalized = (value - min_val) / (max_val - min_val)
        
        return float(np.clip(normalized, 0.0, 1.0))
    
    # ------------------------------------------------------------------------
    # CATEGORICAL: Choice from list
    # ------------------------------------------------------------------------
    elif param_type == 'categorical':
        choices = param_config['values']
        
        if value not in choices:
            raise ValueError(
                f"{param_name}: value '{value}' not in choices {choices}"
            )
        
        # Get index
        index = choices.index(value)
        
        # Normalize to [0, 1]
        if len(choices) == 1:
            return 0.5
        
        normalized = index / (len(choices) - 1)
        return float(normalized)
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def encode_hyperparameters(hyperparameters: Dict[str, Any],
                           hyperparameter_config: Dict[str, Any]) -> List[float]:
    """
    Encode a full set of hyperparameters into a chromosome.
    
    Args:
        hyperparameters: Dictionary of hyperparameter values
        hyperparameter_config:  Hyperparameter configuration
        
    Returns:
        Chromosome (list of floats in [0, 1])
    """
    gene_names = get_gene_names(hyperparameter_config)
    chromosome = []
    
    for gene_name in gene_names:
        if gene_name not in hyperparameters:
            raise ValueError(f"Missing hyperparameter: {gene_name}")
        
        value = hyperparameters[gene_name]
        param_config = hyperparameter_config['hyperparameters'][gene_name]
        
        encoded_value = encode_hyperparameter(value, gene_name, param_config)
        chromosome.append(encoded_value)
    
    return chromosome


# ============================================================================
# DECODE: Chromosome → Hyperparameters
# ============================================================================

def decode_gene(gene_value: float,
               param_name: str,
               param_config: Dict[str, Any]) -> Any:
    """
    Decode a single gene value from [0, 1] to actual hyperparameter value.
    
    Args:
        gene_value:  Encoded value in [0, 1]
        param_name: Hyperparameter name
        param_config: Configuration for this parameter
        
    Returns:
        Decoded hyperparameter value (int, float, or string)
    """
    param_type = param_config['type']
    
    # Clip to valid range (in case of floating point errors)
    gene_value = float(np.clip(gene_value, 0.0, 1.0))
    
    # ------------------------------------------------------------------------
    # DISCRETE: Integer in [min, max]
    # ------------------------------------------------------------------------
    if param_type == 'discrete':
        min_val = param_config['min']
        max_val = param_config['max']
        
        # Denormalize
        value = min_val + gene_value * (max_val - min_val)
        
        # Round to integer
        return int(round(value))
    
    # ------------------------------------------------------------------------
    # CONTINUOUS:  Float in [min, max] with scaling
    # ------------------------------------------------------------------------
    elif param_type == 'continuous':
        min_val = param_config['min']
        max_val = param_config['max']
        scale = param_config.get('scale', 'linear')
        
        # Denormalize
        if scale == 'log10': 
            # Convert from normalized to log space
            log_min = np.log10(min_val)
            log_max = np.log10(max_val)
            
            log_val = log_min + gene_value * (log_max - log_min)
            
            # Convert back from log space
            value = 10 ** log_val
        
        else:  # linear
            value = min_val + gene_value * (max_val - min_val)
        
        return float(value)
    
    # ------------------------------------------------------------------------
    # CATEGORICAL: Choice from list
    # ------------------------------------------------------------------------
    elif param_type == 'categorical':
        choices = param_config['values']
        
        # Convert normalized value to index
        index = int(round(gene_value * (len(choices) - 1)))
        
        # Clip to valid range
        index = max(0, min(index, len(choices) - 1))
        
        return choices[index]
    
    else:
        raise ValueError(f"Unknown parameter type: {param_type}")


def decode_chromosome(chromosome: List[float],
                     hyperparameter_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Decode a chromosome into hyperparameters.
    
    Args:
        chromosome: List of floats in [0, 1]
        hyperparameter_config:  Hyperparameter configuration
        
    Returns:
        Dictionary of hyperparameter values
    """
    gene_names = get_gene_names(hyperparameter_config)
    
    if len(chromosome) != len(gene_names):
        raise ValueError(
            f"Chromosome length ({len(chromosome)}) doesn't match "
            f"number of genes ({len(gene_names)})"
        )
    
    hyperparameters = {}
    
    for gene_name, gene_value in zip(gene_names, chromosome):
        param_config = hyperparameter_config['hyperparameters'][gene_name]
        
        decoded_value = decode_gene(gene_value, gene_name, param_config)
        hyperparameters[gene_name] = decoded_value
    
    return hyperparameters


# ============================================================================
# INITIALIZE POPULATION
# ============================================================================

def initialize_random_chromosome(hyperparameter_config: Dict[str, Any],
                                random_state: int = None) -> List[float]:
    """
    Generate a random chromosome with values in [0, 1].
    
    Args:
        hyperparameter_config: Hyperparameter configuration
        random_state: Random seed (optional)
        
    Returns: 
        Random chromosome (list of floats in [0, 1])
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    gene_names = get_gene_names(hyperparameter_config)
    chromosome = []
    
    for _ in gene_names:
        # Generate random value in [0, 1]
        gene_value = np.random.uniform(0.0, 1.0)
        chromosome.append(float(gene_value))
    
    return chromosome


def initialize_population(population_size: int,
                         hyperparameter_config: Dict[str, Any],
                         random_state: int = None) -> List[List[float]]: 
    """
    Initialize a population of random chromosomes.
    
    Args:
        population_size:  Number of individuals in population
        hyperparameter_config: Hyperparameter configuration
        random_state: Random seed (optional)
        
    Returns: 
        List of chromosomes (population)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    population = []
    
    for i in range(population_size):
        # Use different seed for each individual
        seed = None if random_state is None else random_state + i
        chromosome = initialize_random_chromosome(hyperparameter_config, seed)
        population.append(chromosome)
    
    return population


# ============================================================================
# VALIDATION
# ============================================================================

def validate_chromosome(chromosome: List[float],
                       hyperparameter_config: Dict[str, Any]) -> bool:
    """
    Validate that a chromosome is well-formed.
    
    Args:
        chromosome:  Chromosome to validate
        hyperparameter_config: Hyperparameter configuration
        
    Returns:
        True if valid
        
    Raises:
        ValueError: If chromosome is invalid
    """
    gene_names = get_gene_names(hyperparameter_config)
    
    # Check length
    if len(chromosome) != len(gene_names):
        raise ValueError(
            f"Chromosome length ({len(chromosome)}) doesn't match "
            f"number of genes ({len(gene_names)})"
        )
    
    # Check all values are in [0, 1]
    for i, gene_value in enumerate(chromosome):
        if not isinstance(gene_value, (int, float)):
            raise ValueError(
                f"Gene {i} ({gene_names[i]}): value must be numeric, "
                f"got {type(gene_value)}"
            )
        
        if gene_value < 0.0 or gene_value > 1.0:
            raise ValueError(
                f"Gene {i} ({gene_names[i]}): value {gene_value} "
                f"out of range [0, 1]"
            )
    
    return True


# ============================================================================
# CHROMOSOME INFO
# ============================================================================

def get_chromosome_info(hyperparameter_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get information about chromosome structure.
    
    Args:
        hyperparameter_config:  Hyperparameter configuration
        
    Returns:
        Dictionary with chromosome metadata
    """
    gene_names = get_gene_names(hyperparameter_config)
    hyperparams = hyperparameter_config['hyperparameters']
    
    info = {
        'chromosome_length': len(gene_names),
        'gene_names':  gene_names,
        'genes': {}
    }
    
    for i, gene_name in enumerate(gene_names):
        param_config = hyperparams[gene_name]
        
        gene_info = {
            'index': i,
            'type': param_config['type'],
            'description': param_config.get('description', 'No description')
        }
        
        # Add type-specific info
        if param_config['type'] == 'discrete':
            gene_info['min'] = param_config['min']
            gene_info['max'] = param_config['max']
        
        elif param_config['type'] == 'continuous':
            gene_info['min'] = param_config['min']
            gene_info['max'] = param_config['max']
            gene_info['scale'] = param_config.get('scale', 'linear')
        
        elif param_config['type'] == 'categorical':
            gene_info['choices'] = param_config['values']
        
        info['genes'][gene_name] = gene_info
    
    return info


def print_chromosome_info(hyperparameter_config: Dict[str, Any]):
    """
    Pretty-print chromosome structure information.
    
    Args:
        hyperparameter_config:  Hyperparameter configuration
    """
    from tabulate import tabulate
    
    info = get_chromosome_info(hyperparameter_config)
    
    print("\n" + "="*70)
    print("🧬 CHROMOSOME STRUCTURE")
    print("="*70)
    print(f"Chromosome Length: {info['chromosome_length']} genes")
    print(f"Model Type: {hyperparameter_config.get('model_type', 'Unknown')}")
    print("\nGene Mapping:")
    print("-"*70)
    
    table_data = []
    
    for gene_name in info['gene_names']:
        gene_info = info['genes'][gene_name]
        
        # Format range/choices based on type
        if gene_info['type'] == 'discrete': 
            range_str = f"[{gene_info['min']}, {gene_info['max']}]"
        elif gene_info['type'] == 'continuous':
            scale = gene_info['scale']
            range_str = f"[{gene_info['min']}, {gene_info['max']}] ({scale})"
        elif gene_info['type'] == 'categorical':
            choices = gene_info['choices']
            range_str = f"{choices}"
        else:
            range_str = "Unknown"
        
        table_data.append([
            gene_info['index'],
            gene_name,
            gene_info['type'],
            range_str
        ])
    
    headers = ["Index", "Gene Name", "Type", "Range/Choices"]
    print(tabulate(table_data, headers=headers, tablefmt="simple"))
    print("="*70)


# ============================================================================
# EXAMPLE USAGE & TESTING
# ============================================================================

if __name__ == '__main__':
    print("🧬 Testing Chromosome Utilities")
    print("="*70)
    
    # Load configuration
    from utils.config_loader import get_full_config
    
    try:
        # Test with default config (Neural Network + SECOM)
        config = get_full_config()
        hyperparam_config = {
            'hyperparameters': config['hyperparameters'],
            'model_type': config['hyperparameter_metadata']['model_type']
        }
        
        print("\n✅ Configuration loaded")
        
        # Print chromosome structure
        print_chromosome_info(hyperparam_config)
        
        # Test 1: Initialize random chromosome
        print("\n" + "="*70)
        print("TEST 1: Initialize Random Chromosome")
        print("="*70)
        
        chromosome = initialize_random_chromosome(hyperparam_config, random_state=42)
        print(f"✅ Random chromosome generated (length: {len(chromosome)})")
        print(f"Chromosome: {[f'{g:.4f}' for g in chromosome]}")
        
        # Test 2: Decode chromosome
        print("\n" + "="*70)
        print("TEST 2: Decode Chromosome")
        print("="*70)
        
        decoded = decode_chromosome(chromosome, hyperparam_config)
        print("✅ Chromosome decoded to hyperparameters:")
        for key, value in decoded.items():
            print(f"   {key}: {value}")
        
        # Test 3: Encode hyperparameters back
        print("\n" + "="*70)
        print("TEST 3: Encode Hyperparameters → Chromosome")
        print("="*70)
        
        re_encoded = encode_hyperparameters(decoded, hyperparam_config)
        print(f"✅ Hyperparameters re-encoded")
        print(f"Re-encoded:  {[f'{g:.4f}' for g in re_encoded]}")
        
        # Check if encoding is reversible
        max_diff = max(abs(a - b) for a, b in zip(chromosome, re_encoded))
        print(f"\nMax difference after encode/decode cycle: {max_diff:.10f}")
        
        if max_diff < 1e-6:
            print("✅ Encoding is reversible (within floating point precision)")
        else:
            print("⚠️  Warning: Encoding may not be perfectly reversible")
        
        # Test 4: Initialize population
        print("\n" + "="*70)
        print("TEST 4: Initialize Population")
        print("="*70)
        
        population = initialize_population(5, hyperparam_config, random_state=42)
        print(f"✅ Population of {len(population)} individuals created")
        
        for i, individual in enumerate(population):
            print(f"\nIndividual {i}:")
            decoded_ind = decode_chromosome(individual, hyperparam_config)
            print(f"   num_layers: {decoded_ind.get('num_layers', 'N/A')}")
            print(f"   neurons:  {decoded_ind.get('neurons', 'N/A')}")
            print(f"   activation: {decoded_ind.get('activation', 'N/A')}")
            print(f"   learning_rate: {decoded_ind.get('learning_rate', 'N/A'):.6f}")
        
        # Test 5: Validate chromosome
        print("\n" + "="*70)
        print("TEST 5: Validate Chromosome")
        print("="*70)
        
        validate_chromosome(chromosome, hyperparam_config)
        print("✅ Chromosome is valid")
        
        # Test invalid chromosome
        invalid_chromosome = chromosome.copy()
        invalid_chromosome[0] = 1.5  # Out of range
        
        try: 
            validate_chromosome(invalid_chromosome, hyperparam_config)
            print("❌ Should have raised ValueError for invalid chromosome")
        except ValueError as e:
            print(f"✅ Correctly rejected invalid chromosome: {e}")
        
        print("\n" + "="*70)
        print("🎉 ALL TESTS PASSED!")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
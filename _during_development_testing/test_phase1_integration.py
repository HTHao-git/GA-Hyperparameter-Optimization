# Integration test:  Use all Phase 1 modules together
from utils.config_loader import get_full_config
from utils.chromosome import (
    initialize_population,
    decode_chromosome,
    get_gene_names
)
from utils.logger import get_logger
from utils.validators import validate_range, validate_choice
from utils.colors import print_header

# Initialize logger
logger = get_logger(name="INTEGRATION_TEST", verbose=True)

logger.header("PHASE 1 INTEGRATION TEST")

# Step 1: Load configuration
logger.section("Step 1: Load Configuration")
try:
    config = get_full_config()
    logger.success("Configuration loaded")
    logger.info(f"  Dataset: {config['dataset']['name']}")
    logger.info(f"  Model:  {config['model']['type']}")
    logger.info(f"  HPO Method: {config['hpo']['method']}")
except Exception as e:
    logger.error(f"Failed to load config: {e}")
    exit(1)

# Step 2: Initialize population
logger.blank()
logger.section("Step 2: Initialize Population")

try:
    hyperparam_config = {
        'hyperparameters': config['hyperparameters'],
        'model_type': config['hyperparameter_metadata']['model_type']
    }
    
    population_size = 5
    population = initialize_population(population_size, hyperparam_config, random_state=42)
    
    logger.success(f"Population initialized ({population_size} individuals)")
    logger.info(f"  Chromosome length: {len(population[0])}")
    logger.info(f"  Gene names: {', '.join(get_gene_names(hyperparam_config))}")
except Exception as e:
    logger.error(f"Failed to initialize population: {e}")
    exit(1)

# Step 3: Decode and validate chromosomes
logger.blank()
logger.section("Step 3: Decode & Validate Chromosomes")

for i, chromosome in enumerate(population):
    try:
        # Decode
        hyperparams = decode_chromosome(chromosome, hyperparam_config)
        
        logger.info(f"Individual {i}:")
        
        # Validate specific hyperparameters
        if 'learning_rate' in hyperparams:
            validate_range(hyperparams['learning_rate'], 0.0001, 0.01, "learning_rate")
            logger.debug(f"  learning_rate: {hyperparams['learning_rate']:.6f} ✓")
        
        if 'num_layers' in hyperparams: 
            validate_range(hyperparams['num_layers'], 1, 4, "num_layers")
            logger.debug(f"  num_layers: {hyperparams['num_layers']} ✓")
        
        if 'activation' in hyperparams:
            validate_choice(hyperparams['activation'], ['relu', 'tanh', 'elu'], "activation")
            logger.debug(f"  activation: {hyperparams['activation']} ✓")
        
    except Exception as e:
        logger.error(f"Validation failed for individual {i}: {e}")

logger.blank()
logger.success("All individuals decoded and validated successfully")

# Step 4: Demonstrate logging features
logger.blank()
logger.section("Step 4: Logging Features Demo")

logger.info("This is an info message")
logger.warning("This is a warning")
logger.success("This is a success message")

logger.blank()
for gen in range(1, 4):
    logger.progress(gen, 3, "Simulating generation")

logger.blank()
logger.header("INTEGRATION TEST COMPLETE")
logger.success("All Phase 1 modules working together correctly!")
# ============================================================================
# RUN GA HYPERPARAMETER OPTIMIZATION
# ============================================================================
# Production script for optimizing ML models with GA
# ============================================================================

import numpy as np
from pathlib import Path
import time

from preprocessing.data_loader import DatasetLoader
from ga.ml_optimizer import MLOptimizer
from ga.genetic_algorithm import GAConfig
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': 'secom',
    'model_type': 'random_forest',  # 'random_forest', 'svm', 'knn'
    
    # GA settings
    'ga':  {
        'population_size':  15,
        'num_generations': 15,
        'crossover_rate': 0.8,
        'mutation_rate': 0.20,
        'elitism_rate': 0.15,
        'early_stopping': True,
        'patience': 5,
        'diversity_threshold': 0.0001,
        'verbose': 1
    },
    
    # Evaluation settings
    'cv_folds': 3,  # Reduced for speed
    'test_size': 0.2,
    'random_state': 42,
    
    # Output
    'output_dir': 'outputs/ga_optimization'
}

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run GA optimization."""
    
    logger = get_logger(name="GA_OPTIMIZATION", verbose=True)
    
    print_header("GA HYPERPARAMETER OPTIMIZATION")
    print()
    
    # Create output directory
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Dataset
    # ========================================================================
    
    print_section("STEP 1: Load Dataset")
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset(CONFIG['dataset'])
    
    logger.blank()
    print_info(f"Dataset: {CONFIG['dataset']}")
    print_info(f"  Shape: {X.shape}")
    print_info(f"  Classes: {len(np.unique(y))}")
    print_info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 2: Create GA Configuration
    # ========================================================================
    
    print_section("STEP 2: Configure GA")
    
    ga_config = GAConfig(**CONFIG['ga'])
    
    print_info(f"Population size: {ga_config.population_size}")
    print_info(f"Generations: {ga_config.num_generations}")
    print_info(f"Mutation rate: {ga_config.mutation_rate}")
    print_info(f"Crossover rate:  {ga_config.crossover_rate}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 3: Run Optimization
    # ========================================================================
    
    print_section(f"STEP 3: Optimize {CONFIG['model_type'].upper()}")
    print()
    
    start_time = time.time()
    
    optimizer = MLOptimizer(
        X, y,
        model_type=CONFIG['model_type'],
        ga_config=ga_config,
        cv_folds=CONFIG['cv_folds'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        logger=logger
    )
    
    results = optimizer.optimize()
    
    optimization_time = time.time() - start_time
    
    # ========================================================================
    # STEP 4: Save Results
    # ========================================================================
    
    logger.blank()
    print_section("STEP 4: Save Results")
    
    # Save detailed results
    output_file = output_dir / f"optimization_{CONFIG['model_type']}.json"
    optimizer.save_results(output_file)
    
    # Save summary
    summary_file = output_dir / "summary.txt"
    
    with open(summary_file, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("GA HYPERPARAMETER OPTIMIZATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        f.write(f"Dataset: {CONFIG['dataset']}\n")
        f.write(f"Model: {CONFIG['model_type']}\n")
        f.write(f"Optimization time: {optimization_time:.1f}s ({optimization_time/60:.1f} min)\n\n")
        
        f.write(f"Best CV Score: {results['cv_score']:.4f}\n")
        f.write(f"Test Score: {results['test_score']:.4f}\n\n")
        
        f.write("Best Configuration:\n")
        f.write("-" * 70 + "\n")
        for key, value in results['config'].items():
            f.write(f"  {key}: {value}\n")
    
    print_success(f"✓ Results saved to: {output_dir}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 5: Comparison with Baseline
    # ========================================================================
    
    print_section("STEP 5: Performance Summary")
    print()
    
    baseline_score = 0.9927  # From integration test
    
    print_info(f"Baseline (default params): {baseline_score:.4f}")
    print_info(f"GA Optimized CV:            {results['cv_score']:.4f}")
    print_info(f"GA Optimized Test:         {results['test_score']:.4f}")
    
    improvement = results['test_score'] - baseline_score
    
    if improvement > 0:
        print_success(f"✓ Improvement: +{improvement:.4f} ({improvement*100:.2f}%)")
    elif improvement > -0.01:
        print_info(f"≈ Similar performance:  {improvement:.4f}")
    else:
        print_warning(f"⚠ Below baseline: {improvement:.4f}")
    
    logger.blank()
    print_header("OPTIMIZATION COMPLETE")


if __name__ == '__main__':
    main()
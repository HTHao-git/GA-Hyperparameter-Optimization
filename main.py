# ============================================================================
# MAIN EXECUTION FILE
# ============================================================================
# Central control panel for GA hyperparameter optimization
#
# USAGE: 
#   1. Configure settings below (dataset, model, metrics, GA parameters)
#   2. Enable/disable experiments (set True/False)
#   3. Run:   python main.py
#
# Last updated: 2026-01-12
# ============================================================================

import numpy as np
from pathlib import Path
import json
import time
from datetime import datetime

from preprocessing.data_loader import DatasetLoader
from ga.unified_optimizer import UnifiedOptimizer
from ga.genetic_algorithm import GAConfig
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning


# ============================================================================
# CONFIGURATION SECTION
# ============================================================================
# MODIFY THESE SETTINGS TO CUSTOMIZE YOUR EXPERIMENTS
# ============================================================================

# ----------------------------------------------------------------------------
# EXPERIMENT CONTROL - Enable/Disable What to Run
# ----------------------------------------------------------------------------

RUN_EXPERIMENTS = {
    'single_objective': True,        # Single objective optimization (GA)
    'multi_objective':  True,         # Multi-objective optimization (NSGA-II)
    'model_comparison': False,       # Compare multiple models (slower)
    'metric_comparison': False,      # Compare different fitness metrics
    'full_analysis': False,          # Comprehensive analysis (very slow)
}

# ----------------------------------------------------------------------------
# DATASET CONFIGURATION
# ----------------------------------------------------------------------------

DATASET_CONFIG = {
    'name': 'isolet',  # Options: 'secom', 'fashion_mnist', 'isolet', 'steel_plates'
    'test_size': 0.2,
    'random_state': 42
}

# ----------------------------------------------------------------------------
# MODEL CONFIGURATION
# ----------------------------------------------------------------------------

MODEL_CONFIG = {
    'primary_model': 'neural_network',  # Main model to optimize
    
    # For model comparison (if enabled)
    'comparison_models': ['random_forest', 'xgboost', 'lightgbm'],
    
    # Options: 'random_forest', 'xgboost', 'lightgbm', 'neural_network', 'svm'
}

# ----------------------------------------------------------------------------
# FITNESS METRICS CONFIGURATION
# ----------------------------------------------------------------------------

METRICS_CONFIG = {
    # Single objective
    'single_metric': 'accuracy',  
    # Options: 'accuracy', 'f1_macro', 'f1_weighted', 'balanced_accuracy', 
    #          'recall_macro', 'recall_minority', 'precision_macro'
    
    # Multi-objective
    'multi_metrics': ['accuracy', 'recall_minority'],
    # Can be 2-5 metrics (more = slower)
    
    # For metric comparison (if enabled)
    'comparison_metrics': ['accuracy', 'f1_macro', 'balanced_accuracy']
}

# ----------------------------------------------------------------------------
# GENETIC ALGORITHM CONFIGURATION
# ----------------------------------------------------------------------------

GA_CONFIG = {
    # Population & Generations
    'population_size':  20,       # Number of individuals (10-50)
    'num_generations': 40,       # Max generations (5-30)
    
    # Operator Rates
    'crossover_rate': 0.8,       # Crossover probability (0.7-0.95)
    'mutation_rate':  0.1,       # Mutation probability (0.1-0.3)
    'elitism_rate': 0.20,        # Elite preservation (0.05-0.2)
    
    # Advanced Mutation
    'adaptive_mutation': True,   # Enable adaptive mutation
    'mutation_method': 'adaptive',  # 'uniform', 'gaussian', 'polynomial', 'adaptive'
    'mutation_strength': 'large',   # 'small', 'medium', 'large'
    'adaptive_method': 'diversity_based',  # 'diversity_based', 'fitness_based', 'schedule'
    
    # Early Stopping
    'early_stopping':  True,      # Stop if no improvement
    'patience': 8,               # Generations to wait
    'diversity_threshold': 0.0,  # Min diversity (0.0 = disabled)
    
    # Other
    'cv_folds': 5,               # Cross-validation folds
    'cache_fitness': False,      # Cache evaluations (uses more memory)
    'verbose': 1,                # Logging level (0=silent, 1=normal, 2=detailed)
    'random_state': 42           # Reproducibility
}

# ----------------------------------------------------------------------------
# OUTPUT CONFIGURATION
# ----------------------------------------------------------------------------

OUTPUT_CONFIG = {
    'base_dir': 'outputs/experiments',
    'save_json': True,           # Save results as JSON
    'save_plots': True,          # Generate visualizations
    'create_report': True,       # Generate HTML report
    'timestamp': True            # Add timestamp to output folder
}


# ============================================================================
# EXPERIMENT FUNCTIONS
# ============================================================================

def setup_output_directory():
    """Create timestamped output directory."""
    base_dir = Path(OUTPUT_CONFIG['base_dir'])
    
    if OUTPUT_CONFIG['timestamp']:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = base_dir / f"run_{timestamp}"
    else: 
        output_dir = base_dir / 'latest'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def load_dataset(logger):
    """Load configured dataset."""
    print_section(f"Loading Dataset: {DATASET_CONFIG['name'].upper()}")
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset(DATASET_CONFIG['name'])
    
    logger.blank()
    print_info(f"✓ Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    print_info(f"  Classes: {len(np.unique(y))}")
    print_info(f"  Distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    logger.blank()
    
    return X, y, metadata


def create_ga_config():
    """Create GA configuration from settings."""
    return GAConfig(
        population_size=GA_CONFIG['population_size'],
        num_generations=GA_CONFIG['num_generations'],
        crossover_rate=GA_CONFIG['crossover_rate'],
        mutation_rate=GA_CONFIG['mutation_rate'],
        elitism_rate=GA_CONFIG['elitism_rate'],
        
        adaptive_mutation=GA_CONFIG['adaptive_mutation'],
        mutation_method=GA_CONFIG['mutation_method'],
        mutation_strength=GA_CONFIG['mutation_strength'],
        adaptive_method=GA_CONFIG['adaptive_method'],
        
        early_stopping=GA_CONFIG['early_stopping'],
        patience=GA_CONFIG['patience'],
        diversity_threshold=GA_CONFIG['diversity_threshold'],
        
        cache_fitness=GA_CONFIG['cache_fitness'],
        verbose=GA_CONFIG['verbose'],
        random_state=GA_CONFIG['random_state']
    )


# ============================================================================
# EXPERIMENT 1: SINGLE OBJECTIVE OPTIMIZATION
# ============================================================================

def run_single_objective(X, y, output_dir, logger):
    """Run single-objective optimization."""
    
    print_header("EXPERIMENT 1: Single Objective Optimization")
    print_info(f"Model: {MODEL_CONFIG['primary_model']}")
    print_info(f"Metric: {METRICS_CONFIG['single_metric']}")
    print()
    
    # Create optimizer
    optimizer = UnifiedOptimizer(
        X, y,
        model_type=MODEL_CONFIG['primary_model'],
        fitness_metrics=METRICS_CONFIG['single_metric'],
        ga_config=create_ga_config(),
        cv_folds=GA_CONFIG['cv_folds'],
        test_size=DATASET_CONFIG['test_size'],
        random_state=DATASET_CONFIG['random_state'],
        logger=logger
    )
    
    # Optimize
    start_time = time.time()
    results = optimizer.optimize()
    total_time = time.time() - start_time
    
    # Save results
    if OUTPUT_CONFIG['save_json']: 
        output_file = output_dir / 'single_objective_results.json'
        with open(output_file, 'w') as f:
            json.dump({
                'experiment':  'single_objective',
                'dataset': DATASET_CONFIG['name'],
                'model': MODEL_CONFIG['primary_model'],
                'metric': METRICS_CONFIG['single_metric'],
                'config': results.get('best_config'),
                'cv_score': results.get('best_score'),
                'test_accuracy': results.get('test_accuracy'),
                'test_f1_score': results.get('test_f1_score'),
                'optimization_time': results.get('optimization_time', total_time),
                'total_time': total_time
            }, f, indent=2, default=str)
        
        print_success(f"✓ Results saved:  {output_file}")
    
    logger.blank()
    return results


# ============================================================================
# EXPERIMENT 2: MULTI-OBJECTIVE OPTIMIZATION
# ============================================================================

def run_multi_objective(X, y, output_dir, logger):
    """Run multi-objective optimization."""
    
    print_header("EXPERIMENT 2: Multi-Objective Optimization")
    print_info(f"Model: {MODEL_CONFIG['primary_model']}")
    print_info(f"Metrics: {', '.join(METRICS_CONFIG['multi_metrics'])}")
    print()
    
    # Create optimizer
    optimizer = UnifiedOptimizer(
        X, y,
        model_type=MODEL_CONFIG['primary_model'],
        fitness_metrics=METRICS_CONFIG['multi_metrics'],
        ga_config=create_ga_config(),
        cv_folds=GA_CONFIG['cv_folds'],
        test_size=DATASET_CONFIG['test_size'],
        random_state=DATASET_CONFIG['random_state'],
        logger=logger
    )
    
    # Optimize
    start_time = time.time()
    results = optimizer.optimize()
    total_time = time.time() - start_time
    
    # Save Pareto front
    if OUTPUT_CONFIG['save_json'] and 'pareto_front' in results:
        pareto_data = []
        for solution in results['pareto_front']: 
            pareto_data.append({
                'config': solution.chromosome,
                'fitness': {
                    metric: float(score)
                    for metric, score in zip(METRICS_CONFIG['multi_metrics'], solution.fitness)
                }
            })
        
        output_file = output_dir / 'multi_objective_pareto.json'
        with open(output_file, 'w') as f:
            json.dump({
                'experiment': 'multi_objective',
                'dataset': DATASET_CONFIG['name'],
                'model': MODEL_CONFIG['primary_model'],
                'metrics':  METRICS_CONFIG['multi_metrics'],
                'num_pareto_solutions': results['num_pareto_solutions'],
                'best_compromise': results.get('best_scores'),
                'pareto_front': pareto_data,
                'total_time': total_time
            }, f, indent=2, default=str)
        
        print_success(f"✓ Pareto front saved: {output_file}")
    
    # Visualize (if 2D)
    if OUTPUT_CONFIG['save_plots'] and len(METRICS_CONFIG['multi_metrics']) == 2:
        _plot_pareto_2d(results['pareto_front'], METRICS_CONFIG['multi_metrics'], 
                       output_dir / 'pareto_front.png')
    
    logger.blank()
    return results


# ============================================================================
# EXPERIMENT 3: MODEL COMPARISON
# ============================================================================

def run_model_comparison(X, y, output_dir, logger):
    """Compare multiple models."""
    
    print_header("EXPERIMENT 3: Model Comparison")
    print_info(f"Models: {', '.join(MODEL_CONFIG['comparison_models'])}")
    print_info(f"Metric: {METRICS_CONFIG['single_metric']}")
    print()
    
    all_results = {}
    
    for model_type in MODEL_CONFIG['comparison_models']:
        print_section(f"Optimizing:  {model_type.upper()}")
        print()
        
        optimizer = UnifiedOptimizer(
            X, y,
            model_type=model_type,
            fitness_metrics=METRICS_CONFIG['single_metric'],
            ga_config=create_ga_config(),
            cv_folds=GA_CONFIG['cv_folds'],
            test_size=DATASET_CONFIG['test_size'],
            random_state=DATASET_CONFIG['random_state'],
            logger=logger
        )
        
        start_time = time.time()
        results = optimizer.optimize()
        total_time = time.time() - start_time
        
        all_results[model_type] = {
            'cv_score': results.get('best_score'),
            'test_accuracy': results.get('test_accuracy'),
            'test_f1_score': results.get('test_f1_score'),
            'optimization_time': results.get('optimization_time', total_time),
            'config': results.get('best_config')
        }
        
        logger.blank()
    
    # Save comparison
    if OUTPUT_CONFIG['save_json']: 
        output_file = output_dir / 'model_comparison.json'
        with open(output_file, 'w') as f:
            json.dump({
                'experiment': 'model_comparison',
                'dataset':  DATASET_CONFIG['name'],
                'metric': METRICS_CONFIG['single_metric'],
                'results': all_results
            }, f, indent=2, default=str)
        
        print_success(f"✓ Comparison saved: {output_file}")
    
    # Print summary
    _print_comparison_table(all_results)
    
    logger.blank()
    return all_results


# ============================================================================
# EXPERIMENT 4: METRIC COMPARISON
# ============================================================================

def run_metric_comparison(X, y, output_dir, logger):
    """Compare different fitness metrics."""
    
    print_header("EXPERIMENT 4: Metric Comparison")
    print_info(f"Model: {MODEL_CONFIG['primary_model']}")
    print_info(f"Metrics: {', '.join(METRICS_CONFIG['comparison_metrics'])}")
    print()
    
    all_results = {}
    
    for metric in METRICS_CONFIG['comparison_metrics']:
        print_section(f"Optimizing for: {metric.upper()}")
        print()
        
        optimizer = UnifiedOptimizer(
            X, y,
            model_type=MODEL_CONFIG['primary_model'],
            fitness_metrics=metric,
            ga_config=create_ga_config(),
            cv_folds=GA_CONFIG['cv_folds'],
            test_size=DATASET_CONFIG['test_size'],
            random_state=DATASET_CONFIG['random_state'],
            logger=logger
        )
        
        start_time = time.time()
        results = optimizer.optimize()
        total_time = time.time() - start_time
        
        all_results[metric] = {
            'cv_score':  results.get('best_score'),
            'test_accuracy': results.get('test_accuracy'),
            'test_f1_score': results.get('test_f1_score'),
            'optimization_time': results.get('optimization_time', total_time),
            'config': results.get('best_config')
        }
        
        logger.blank()
    
    # Save comparison
    if OUTPUT_CONFIG['save_json']:
        output_file = output_dir / 'metric_comparison.json'
        with open(output_file, 'w') as f:
            json.dump({
                'experiment': 'metric_comparison',
                'dataset': DATASET_CONFIG['name'],
                'model': MODEL_CONFIG['primary_model'],
                'results': all_results
            }, f, indent=2, default=str)
        
        print_success(f"✓ Comparison saved: {output_file}")
    
    # Print summary
    _print_comparison_table(all_results)
    
    logger.blank()
    return all_results


# ============================================================================
# EXPERIMENT 5: FULL ANALYSIS
# ============================================================================

def run_full_analysis(X, y, output_dir, logger):
    """Run all experiments."""
    
    print_header("EXPERIMENT 5: Full Analysis")
    print_warning("This will run all experiments - may take several hours!")
    print()
    
    all_results = {}
    
    # Run all experiments
    all_results['single_objective'] = run_single_objective(X, y, output_dir, logger)
    all_results['multi_objective'] = run_multi_objective(X, y, output_dir, logger)
    all_results['model_comparison'] = run_model_comparison(X, y, output_dir, logger)
    all_results['metric_comparison'] = run_metric_comparison(X, y, output_dir, logger)
    
    # Save comprehensive results
    if OUTPUT_CONFIG['save_json']:
        output_file = output_dir / 'full_analysis.json'
        with open(output_file, 'w') as f:
            json.dump({
                'experiment': 'full_analysis',
                'dataset': DATASET_CONFIG['name'],
                'timestamp': datetime.now().isoformat(),
                'results': all_results
            }, f, indent=2, default=str)
        
        print_success(f"✓ Full analysis saved: {output_file}")
    
    return all_results

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def _plot_pareto_2d(pareto_front, metric_names, output_path):
    """Plot 2D Pareto front."""
    import matplotlib.pyplot as plt
    
    x_values = [sol.fitness[0] for sol in pareto_front]
    y_values = [sol.fitness[1] for sol in pareto_front]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(x_values, y_values, c='blue', s=100, alpha=0.6, edgecolors='black', linewidths=2)
    plt.xlabel(metric_names[0].replace('_', ' ').title(), fontsize=12)
    plt.ylabel(metric_names[1].replace('_', ' ').title(), fontsize=12)
    plt.title(f'Pareto Front:  {metric_names[0]} vs {metric_names[1]}', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_success(f"✓ Pareto plot saved: {output_path}")


def _print_comparison_table(results):
    """Print comparison table."""
    print()
    print("="*90)
    print(f"{'Item':<20} {'CV Score':<15} {'Test Accuracy':<15} {'Test F1':<15} {'Time (s)':<15}")
    print("="*90)
    
    for name, data in results.items():
        print(f"{name:<20} {data['cv_score']:<15.4f} {data['test_accuracy']:<15.4f} "
              f"{data['test_f1_score']:<15.4f} {data['optimization_time']:<15.1f}")
    
    print("="*90)
    print()


def save_configuration(output_dir):
    """Save configuration used for this run."""
    config_data = {
        'dataset':  DATASET_CONFIG,
        'model': MODEL_CONFIG,
        'metrics': METRICS_CONFIG,
        'ga':  GA_CONFIG,
        'output':  OUTPUT_CONFIG,
        'experiments_run': RUN_EXPERIMENTS
    }
    
    config_file = output_dir / 'run_configuration.json'
    with open(config_file, 'w') as f:
        json.dump(config_data, f, indent=2)
    
    print_info(f"✓ Configuration saved: {config_file}")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    # Setup
    logger = get_logger(name="MAIN_EXECUTION", verbose=True)
    output_dir = setup_output_directory()
    
    print_header("GA HYPERPARAMETER OPTIMIZATION")
    print_info(f"Output directory: {output_dir}")
    print()
    
    # Save configuration
    save_configuration(output_dir)
    logger.blank()
    
    # Load dataset
    X, y, metadata = load_dataset(logger)
    
    # Run experiments
    total_start = time.time()
    
    if RUN_EXPERIMENTS['single_objective']:
        run_single_objective(X, y, output_dir, logger)
    
    if RUN_EXPERIMENTS['multi_objective']:
        run_multi_objective(X, y, output_dir, logger)
    
    if RUN_EXPERIMENTS['model_comparison']: 
        run_model_comparison(X, y, output_dir, logger)
    
    if RUN_EXPERIMENTS['metric_comparison']: 
        run_metric_comparison(X, y, output_dir, logger)
    
    if RUN_EXPERIMENTS['full_analysis']:
        run_full_analysis(X, y, output_dir, logger)
    
    total_time = time.time() - total_start
    
    # Summary
    print_header("EXECUTION COMPLETE")
    print_success(f"✓ Total runtime: {total_time/60:.1f} minutes")
    print_info(f"✓ Results saved to: {output_dir}")
    print()
    
    # Show what was run
    experiments_run = [name for name, enabled in RUN_EXPERIMENTS.items() if enabled]
    print_info(f"Experiments completed: {', '.join(experiments_run)}")
    print()


if __name__ == '__main__':
    main()
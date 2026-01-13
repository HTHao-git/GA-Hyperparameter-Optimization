# ============================================================================
# SECOM DATASET OPTIMIZATION EXAMPLES
# ============================================================================
# Demonstrates best practices for optimizing on imbalanced datasets
# ============================================================================

import numpy as np
from pathlib import Path

from preprocessing.data_loader import DatasetLoader
from ga.unified_optimizer import UnifiedOptimizer
from ga.genetic_algorithm import GAConfig
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning

import json


# ============================================================================
# CONFIGURATION 1: SINGLE OBJECTIVE - MACRO F1 (RECOMMENDED)
# ============================================================================

def config_1_macro_f1():
    """
    Single objective: Optimize for macro F1-score. 
    
    WHY: Macro F1 treats both classes equally (unweighted average).
    EXPECTED:  85-90% accuracy, 40-70% minority recall
    RUNTIME: ~10-15 minutes
    """
    
    logger = get_logger(name="SECOM_F1_MACRO", verbose=True)
    
    print_header("SECOM OPTIMIZATION - Macro F1")
    print_info("Metric: F1-score (macro-averaged)")
    print_info("Why: Treats Pass and Fail classes equally")
    print()
    
    # Load SECOM
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset('secom')
    
    logger.blank()
    
    # GA Configuration
    ga_config = GAConfig(
        population_size=20,
        num_generations=15,
        crossover_rate=0.8,
        mutation_rate=0.25,  # Higher for exploration
        elitism_rate=0.10,
        
        # Adaptive mutation (helps with imbalanced data)
        adaptive_mutation=True,
        mutation_method='adaptive',
        mutation_strength='large',  # Large changes to find class_weight='balanced'
        adaptive_method='diversity_based',
        
        early_stopping=True,
        patience=8,
        verbose=1,
        random_state=42
    )
    
    # Create optimizer
    optimizer = UnifiedOptimizer(
        X, y,
        model_type='random_forest',  # Or 'xgboost', 'lightgbm', 'svm', 'neural_network'
        fitness_metrics='f1_macro',  # SINGLE OBJECTIVE → Uses GA
        ga_config=ga_config,
        cv_folds=5,
        logger=logger
    )
    
    # Optimize
    results = optimizer.optimize()
    
    # Save results
    output_dir = Path('outputs/secom_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config1_macro_f1_results.json', 'w') as f:
        json.dump({
            'config': results['best_config'],
            'cv_score_f1_macro': results['best_score'],
            'test_accuracy':  results['test_accuracy'],
            'test_f1':  results['test_f1_score'],
            'optimization_time': results['optimization_time']
        }, f, indent=2, default=str)
    
    logger.blank()
    print_success("✓ Configuration 1 complete!")
    print_info(f"  CV F1-Macro: {results['best_score']:.4f}")
    print_info(f"  Test Accuracy: {results['test_accuracy']:.4f}")
    print_info(f"  Results:  {output_dir / 'config1_macro_f1_results.json'}")
    
    return results


# ============================================================================
# CONFIGURATION 2: SINGLE OBJECTIVE - BALANCED ACCURACY
# ============================================================================

def config_2_balanced_accuracy():
    """
    Single objective: Optimize for balanced accuracy. 
    
    WHY: Balanced accuracy = average of per-class recall (0.5×Recall₀ + 0.5×Recall₁)
    EXPECTED: Similar to macro F1, slightly different configs
    RUNTIME: ~10-15 minutes
    """
    
    logger = get_logger(name="SECOM_BAL_ACC", verbose=True)
    
    print_header("SECOM OPTIMIZATION - Balanced Accuracy")
    print_info("Metric: Balanced Accuracy")
    print_info("Why:  Average of per-class recall (equal class weight)")
    print()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, _ = loader.load_dataset('secom')
    
    logger.blank()
    
    ga_config = GAConfig(
        population_size=20,
        num_generations=15,
        mutation_rate=0.25,
        adaptive_mutation=True,
        mutation_strength='large',
        patience=8,
        verbose=1,
        random_state=42
    )
    
    optimizer = UnifiedOptimizer(
        X, y,
        model_type='xgboost',  # XGBoost often good for imbalanced data
        fitness_metrics='balanced_accuracy',  # SINGLE OBJECTIVE
        ga_config=ga_config,
        logger=logger
    )
    
    results = optimizer.optimize()
    
    output_dir = Path('outputs/secom_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config2_balanced_acc_results.json', 'w') as f:
        json.dump({
            'config': results['best_config'],
            'cv_score_balanced_acc': results['best_score'],
            'test_accuracy': results['test_accuracy'],
            'optimization_time': results['optimization_time']
        }, f, indent=2, default=str)
    
    logger.blank()
    print_success("✓ Configuration 2 complete!")
    print_info(f"  CV Balanced Accuracy: {results['best_score']:.4f}")
    
    return results


# ============================================================================
# CONFIGURATION 3: MULTI-OBJECTIVE - ACCURACY vs MINORITY RECALL
# ============================================================================

def config_3_multi_objective_basic():
    """
    Multi-objective: Optimize for accuracy AND minority class recall.
    
    WHY: Explore trade-off between overall accuracy and detecting failures. 
    EXPECTED:  Pareto front with 10-20 solutions
    RUNTIME: ~20-30 minutes
    """
    
    logger = get_logger(name="SECOM_MULTI_BASIC", verbose=True)
    
    print_header("SECOM OPTIMIZATION - Multi-Objective (Basic)")
    print_info("Metrics: Accuracy + Minority Recall")
    print_info("Why: Explore trade-off between overall performance and failure detection")
    print()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, _ = loader.load_dataset('secom')
    
    logger.blank()
    
    ga_config = GAConfig(
        population_size=30,  # Larger for multi-objective
        num_generations=20,  # More generations
        mutation_rate=0.25,
        adaptive_mutation=True,
        mutation_strength='large',
        early_stopping=False,  # Run all generations for NSGA-II
        verbose=1,
        random_state=42
    )
    
    optimizer = UnifiedOptimizer(
        X, y,
        model_type='lightgbm',  # Fast model for multi-objective
        fitness_metrics=['accuracy', 'recall_minority'],  # MULTI-OBJECTIVE → NSGA-II
        ga_config=ga_config,
        logger=logger
    )
    
    results = optimizer.optimize()
    
    # Save Pareto front
    output_dir = Path('outputs/secom_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pareto_data = []
    for solution in results['pareto_front']:
        pareto_data.append({
            'config': solution.chromosome,
            'accuracy': solution.fitness[0],
            'recall_minority': solution.fitness[1]
        })
    
    with open(output_dir / 'config3_pareto_front.json', 'w') as f:
        json.dump(pareto_data, f, indent=2, default=str)
    
    logger.blank()
    print_success("✓ Configuration 3 complete!")
    print_info(f"  Pareto solutions:  {results['num_pareto_solutions']}")
    print_info(f"  Best compromise:  {results['best_scores']}")
    
    # Visualize Pareto front
    _visualize_pareto_front(results['pareto_front'], output_dir / 'config3_pareto_plot.png')
    
    return results


# ============================================================================
# CONFIGURATION 4: MULTI-OBJECTIVE - COMPREHENSIVE (3 METRICS)
# ============================================================================

def config_4_multi_objective_comprehensive():
    """
    Multi-objective: Optimize for F1-macro, balanced accuracy, AND minority recall.
    
    WHY: Comprehensive optimization for thesis/research. 
    EXPECTED: 3D Pareto front
    RUNTIME: ~30-45 minutes
    """
    
    logger = get_logger(name="SECOM_MULTI_COMP", verbose=True)
    
    print_header("SECOM OPTIMIZATION - Multi-Objective (Comprehensive)")
    print_info("Metrics:  F1-Macro + Balanced Accuracy + Minority Recall")
    print_info("Why: Comprehensive trade-off analysis for research")
    print()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, _ = loader.load_dataset('secom')
    
    logger.blank()
    
    ga_config = GAConfig(
        population_size=40,  # Larger for 3 objectives
        num_generations=25,
        mutation_rate=0.25,
        adaptive_mutation=True,
        mutation_strength='large',
        early_stopping=False,
        verbose=1,
        random_state=42
    )
    
    optimizer = UnifiedOptimizer(
        X, y,
        model_type='random_forest',
        fitness_metrics=['f1_macro', 'balanced_accuracy', 'recall_minority'],  # 3 OBJECTIVES
        ga_config=ga_config,
        logger=logger
    )
    
    results = optimizer.optimize()
    
    # Save results
    output_dir = Path('outputs/secom_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pareto_data = []
    for solution in results['pareto_front']:
        pareto_data.append({
            'config': solution.chromosome,
            'f1_macro': solution.fitness[0],
            'balanced_accuracy':  solution.fitness[1],
            'recall_minority': solution.fitness[2]
        })
    
    with open(output_dir / 'config4_comprehensive_pareto.json', 'w') as f:
        json.dump(pareto_data, f, indent=2, default=str)
    
    logger.blank()
    print_success("✓ Configuration 4 complete!")
    print_info(f"  Pareto solutions: {results['num_pareto_solutions']}")
    print_info(f"  Best compromise: {results['best_scores']}")
    
    return results


# ============================================================================
# CONFIGURATION 5: COMPARISON - ACCURACY vs F1-MACRO (THESIS DEMO)
# ============================================================================

def config_5_comparison_accuracy_vs_f1():
    """
    Compare accuracy-based vs F1-macro-based optimization.
    
    WHY: Demonstrates the impact of fitness metric choice (for thesis).
    EXPECTED: Accuracy → 93% acc, 0% recall; F1 → 85% acc, 50% recall
    RUNTIME: ~20-30 minutes (2 runs)
    """
    
    logger = get_logger(name="SECOM_COMPARISON", verbose=True)
    
    print_header("SECOM OPTIMIZATION - Accuracy vs F1-Macro Comparison")
    print()
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, _ = loader.load_dataset('secom')
    
    logger.blank()
    
    ga_config = GAConfig(
        population_size=20,
        num_generations=15,
        mutation_rate=0.25,
        adaptive_mutation=True,
        mutation_strength='large',
        patience=8,
        verbose=1,
        random_state=42
    )
    
    # ========================================================================
    # RUN 1: Accuracy (Problematic)
    # ========================================================================
    
    print_section("RUN 1: Accuracy-based Optimization")
    print_warning("Expected: High accuracy, low minority recall")
    print()
    
    optimizer_acc = UnifiedOptimizer(
        X, y,
        model_type='random_forest',
        fitness_metrics='accuracy',  # Problematic for imbalanced data
        ga_config=ga_config,
        logger=logger
    )
    
    results_acc = optimizer_acc.optimize()
    
    logger.blank()
    
    # ========================================================================
    # RUN 2: F1-Macro (Correct)
    # ========================================================================
    
    print_section("RUN 2: F1-Macro-based Optimization")
    print_success("Expected:  Balanced performance across classes")
    print()
    
    optimizer_f1 = UnifiedOptimizer(
        X, y,
        model_type='random_forest',
        fitness_metrics='f1_macro',  # Better for imbalanced data
        ga_config=ga_config,
        logger=logger
    )
    
    results_f1 = optimizer_f1.optimize()
    
    # ========================================================================
    # Compare Results
    # ========================================================================
    
    logger.blank()
    print_section("COMPARISON RESULTS")
    print()
    
    print("="*80)
    print(f"{'Metric':<25} {'Accuracy-based':<25} {'F1-Macro-based':<25}")
    print("="*80)
    print(f"{'CV Score':<25} {results_acc['best_score']:<25.4f} {results_f1['best_score']:<25.4f}")
    print(f"{'Test Accuracy':<25} {results_acc['test_accuracy']: <25.4f} {results_f1['test_accuracy']: <25.4f}")
    print(f"{'Test F1':<25} {results_acc['test_f1_score']:<25.4f} {results_f1['test_f1_score']:<25.4f}")
    print("="*80)
    
    # Save comparison
    output_dir = Path('outputs/secom_examples')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'config5_comparison.json', 'w') as f:
        json.dump({
            'accuracy_based': {
                'config': results_acc['best_config'],
                'cv_score': results_acc['best_score'],
                'test_accuracy': results_acc['test_accuracy'],
                'test_f1': results_acc['test_f1_score']
            },
            'f1_macro_based': {
                'config': results_f1['best_config'],
                'cv_score': results_f1['best_score'],
                'test_accuracy': results_f1['test_accuracy'],
                'test_f1': results_f1['test_f1_score']
            }
        }, f, indent=2, default=str)
    
    logger.blank()
    print_success("✓ Comparison complete!")
    
    return results_acc, results_f1


# ============================================================================
# VISUALIZATION HELPER
# ============================================================================

def _visualize_pareto_front(pareto_front, output_path):
    """Visualize 2D Pareto front."""
    import matplotlib.pyplot as plt
    
    if len(pareto_front[0].fitness) != 2:
        print_info("  Skipping visualization (not 2D)")
        return
    
    objectives_1 = [sol.fitness[0] for sol in pareto_front]
    objectives_2 = [sol.fitness[1] for sol in pareto_front]
    
    plt.figure(figsize=(10, 6))
    plt.scatter(objectives_1, objectives_2, c='blue', s=100, alpha=0.6, edgecolors='black')
    plt.xlabel('Accuracy', fontsize=12)
    plt.ylabel('Minority Recall', fontsize=12)
    plt.title('Pareto Front: Accuracy vs Minority Recall', fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_info(f"  Pareto plot saved:  {output_path}")


# ============================================================================
# MAIN MENU
# ============================================================================

def main():
    """Interactive menu for running configurations."""
    
    print_header("SECOM OPTIMIZATION EXAMPLES")
    print()
    print("Available Configurations:")
    print()
    print("  1. Single Objective: Macro F1 (Recommended) [~15 min]")
    print("  2. Single Objective: Balanced Accuracy [~15 min]")
    print("  3. Multi-Objective: Accuracy + Minority Recall [~25 min]")
    print("  4. Multi-Objective: Comprehensive (3 metrics) [~40 min]")
    print("  5. Comparison: Accuracy vs F1-Macro (Thesis demo) [~30 min]")
    print("  6. Run all configurations [~2 hours]")
    print()
    
    choice = input("Select configuration (1-6): ").strip()
    
    if choice == '1':
        config_1_macro_f1()
    elif choice == '2':
        config_2_balanced_accuracy()
    elif choice == '3': 
        config_3_multi_objective_basic()
    elif choice == '4':
        config_4_multi_objective_comprehensive()
    elif choice == '5': 
        config_5_comparison_accuracy_vs_f1()
    elif choice == '6': 
        print_info("Running all configurations...")
        config_1_macro_f1()
        config_2_balanced_accuracy()
        config_3_multi_objective_basic()
        config_4_multi_objective_comprehensive()
        config_5_comparison_accuracy_vs_f1()
        print_success("✓ All configurations complete!")
    else:
        print_warning("Invalid choice.Running Configuration 1 (default).")
        config_1_macro_f1()


if __name__ == '__main__':
    main()
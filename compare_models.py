# ============================================================================
# COMPREHENSIVE MODEL COMPARISON
# ============================================================================
# Compare Random Forest vs XGBoost with complete analysis
#
# FEATURES:
#   - GA optimization for both models
#   - Enhanced metrics (accuracy, precision, recall, F1, ROC-AUC)
#   - Convergence visualizations
#   - Statistical significance testing
#   - HTML/Text report generation
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from pathlib import Path
import time
import json

from preprocessing.data_loader import DatasetLoader
from ga.ml_optimizer import MLOptimizer
from models.xgboost_optimizer import XGBoostOptimizer
from ga.genetic_algorithm import GAConfig
from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning
from utils.metrics import MetricsCalculator
from utils.visualization import GAVisualizer, ComparisonVisualizer
from utils.statistical_tests import StatisticalTester
from utils.report_generator import ReportGenerator

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': 'secom',
    
    # GA settings (same for both models)
    'ga': {
        'population_size':  15,
        'num_generations': 15,
        'crossover_rate': 0.8,
        'mutation_method': 'adaptive',
        'mutation_rate': 0.20,
        'elitism_rate': 0.15,
        'early_stopping':  True,
        'patience': 5,
        'diversity_threshold': 0.0,
        'cache_fitness': False,
        'verbose': 1
    },
    
    # Evaluation
    'cv_folds': 5,  # Use 5 folds for statistical testing
    'test_size': 0.2,
    'random_state': 42,
    
    # Output
    'output_dir': 'outputs/model_comparison'
}


# ============================================================================
# HELPER:  GET CV SCORES
# ============================================================================

def get_cv_scores(model, X, y, cv_folds=5, random_state=42):
    """Get individual CV fold scores for statistical testing."""
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=1)
    return scores.tolist()


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Comprehensive model comparison."""
    
    logger = get_logger(name="MODEL_COMPARISON", verbose=True)
    
    print_header("COMPREHENSIVE MODEL COMPARISON")
    print_header("Random Forest vs XGBoost with GA Optimization")
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
    
    # Store dataset info for report
    dataset_info = {
        'name': CONFIG['dataset'],
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'n_classes': len(np.unique(y)),
        'class_distribution': str(dict(zip(*np.unique(y, return_counts=True))))
    }
    
    logger.blank()
    
    # ========================================================================
    # STEP 2: Configure GA
    # ========================================================================
    
    print_section("STEP 2: Configure GA")
    
    ga_config = GAConfig(**CONFIG['ga'])
    
    print_info(f"Population size: {ga_config.population_size}")
    print_info(f"Generations: {ga_config.num_generations}")
    print_info(f"Mutation rate: {ga_config.mutation_rate}")
    print_info(f"Crossover rate:  {ga_config.crossover_rate}")
    
    logger.blank()
    
    # ========================================================================
    # STEP 3: Optimize Random Forest
    # ========================================================================
    
    print_section("STEP 3: Optimize Random Forest")
    print()
    
    rf_start = time.time()
    
    rf_optimizer = MLOptimizer(
        X, y,
        model_type='random_forest',
        ga_config=ga_config,
        cv_folds=CONFIG['cv_folds'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        logger=logger
    )
    
    rf_results = rf_optimizer.optimize()
    rf_time = time.time() - rf_start
    
    rf_optimizer.save_results(output_dir / 'random_forest_results.json')
    
    logger.blank()
    
    # ========================================================================
    # STEP 4: Optimize XGBoost
    # ========================================================================
    
    print_section("STEP 4: Optimize XGBoost")
    print()
    
    xgb_start = time.time()
    
    xgb_optimizer = XGBoostOptimizer(
        X, y,
        ga_config=ga_config,
        cv_folds=CONFIG['cv_folds'],
        test_size=CONFIG['test_size'],
        random_state=CONFIG['random_state'],
        logger=logger
    )
    
    xgb_results = xgb_optimizer.optimize()
    xgb_time = time.time() - xgb_start
    
    xgb_optimizer.save_results(output_dir / 'xgboost_results.json')
    
    logger.blank()
    
    # ========================================================================
    # STEP 5: Generate Visualizations
    # ========================================================================
    
    print_section("STEP 5: Generate Visualizations")
    print()
    
    plots = {}
    
    # Random Forest convergence
    print_info("Creating Random Forest convergence plot...")
    rf_viz = GAVisualizer(rf_results['history'], logger)
    rf_conv_path = output_dir / 'rf_convergence.png'
    rf_viz.plot_combined(rf_conv_path)
    plots['Random Forest Convergence'] = rf_conv_path
    
    # XGBoost convergence
    print_info("Creating XGBoost convergence plot...")
    xgb_viz = GAVisualizer(xgb_results['history'], logger)
    xgb_conv_path = output_dir / 'xgb_convergence.png'
    xgb_viz.plot_combined(xgb_conv_path)
    plots['XGBoost Convergence'] = xgb_conv_path
    
    # Comparison charts
    print_info("Creating comparison charts...")
    comparison_metrics = {
        'Random Forest': {
            'accuracy': rf_results['test_score'],
            'cv_score': rf_results['cv_score'],
            'optimization_time': rf_time
        },
        'XGBoost': {
            'accuracy':  xgb_results['test_accuracy'],
            'cv_score': xgb_results['cv_score'],
            'f1_score': xgb_results.get('test_f1_score', 0),
            'optimization_time':  xgb_time
        }
    }
    
    comp_viz = ComparisonVisualizer(comparison_metrics, logger)
    
    bars_path = output_dir / 'comparison_bars.png'
    comp_viz.plot_metric_comparison(
        metrics=['accuracy', 'cv_score'],
        output_path=bars_path
    )
    plots['Performance Comparison'] = bars_path
    
    radar_path = output_dir / 'comparison_radar.png'
    comp_viz.plot_radar_chart(
        metrics=['accuracy', 'cv_score'],
        output_path=radar_path
    )
    plots['Radar Chart'] = radar_path
    
    logger.blank()
    
    # ========================================================================
    # STEP 6: Statistical Significance Testing
    # ========================================================================
    
    print_section("STEP 6: Statistical Significance Testing")
    print()
    
    print_info("Note: Using CV scores from optimization for statistical testing")
    print()
    
    # Extract CV scores from history (best scores per generation)
    rf_cv_scores = [h['best_fitness'] for h in rf_results['history'][-5:]]  # Last 5 generations
    xgb_cv_scores = [h['best_fitness'] for h in xgb_results['history'][-5:]]
    
    # Pad to same length if needed
    max_len = max(len(rf_cv_scores), len(xgb_cv_scores))
    if len(rf_cv_scores) < max_len:
        rf_cv_scores = rf_cv_scores + [rf_cv_scores[-1]] * (max_len - len(rf_cv_scores))
    if len(xgb_cv_scores) < max_len:
        xgb_cv_scores = xgb_cv_scores + [xgb_cv_scores[-1]] * (max_len - len(xgb_cv_scores))
    
    model_cv_scores = {
        'Random Forest': rf_cv_scores,
        'XGBoost': xgb_cv_scores
    }
    
    tester = StatisticalTester(logger=logger, alpha=0.05)
    statistical_results = tester.compare_models(model_cv_scores, baseline_model='Random Forest')
    
    tester.print_comparison_results(statistical_results)
    
    # Save statistical results
    stats_path = output_dir / 'statistical_results.json'
    with open(stats_path, 'w') as f:
        json.dump(statistical_results, f, indent=2, default=str)
    
    logger.blank()
    
    # ========================================================================
    # STEP 7: Generate Reports
    # ========================================================================
    
    print_section("STEP 7: Generate Reports")
    print()
    
    # Prepare model results for report
    model_results_report = {
        'Random Forest':  {
            'cv_score':  rf_results['cv_score'],
            'test_score': rf_results['test_score'],
            'optimization_time': rf_time,
            'config': rf_results['config']
        },
        'XGBoost': {
            'cv_score': xgb_results['cv_score'],
            'test_accuracy': xgb_results['test_accuracy'],
            'test_f1_score': xgb_results.get('test_f1_score', 0),
            'optimization_time': xgb_time,
            'config': xgb_results['config']
        }
    }
    
    # Generate reports
    report_gen = ReportGenerator(output_dir, logger)
    
    html_path = report_gen.generate_html_report(
        experiment_name="GA Model Comparison - Random Forest vs XGBoost",
        dataset_info=dataset_info,
        model_results=model_results_report,
        statistical_results=statistical_results,
        plots=plots
    )
    
    text_path = report_gen.generate_text_report(
        experiment_name="GA Model Comparison - Random Forest vs XGBoost",
        dataset_info=dataset_info,
        model_results=model_results_report,
        statistical_results=statistical_results
    )
    
    logger.blank()
    
    # ========================================================================
    # STEP 8: Final Summary
    # ========================================================================

    print_section("STEP 8: Final Summary")
    print()

    from utils.timing_helper import format_time
    
    print_info("Performance Summary:")
    print()
    print(f"{'Model':<20} {'CV Score':<15} {'Test Score':<15} {'Time (s)':<15}")
    print("=" * 65)
    print(f"{'Random Forest':<20} {rf_results['cv_score']: <15.4f} {rf_results['test_score']:<15.4f} {rf_time:<15.1f}")
    print(f"{'XGBoost':<20} {xgb_results['cv_score']:<15.4f} {xgb_results['test_accuracy']:<15.4f} {xgb_time:<15.1f}")
    print()
    
    # Determine winner
    if xgb_results['test_accuracy'] > rf_results['test_score']:
        winner = 'XGBoost'
        diff = xgb_results['test_accuracy'] - rf_results['test_score']
    elif rf_results['test_score'] > xgb_results['test_accuracy']:
        winner = 'Random Forest'
        diff = rf_results['test_score'] - xgb_results['test_accuracy']
    else: 
        winner = 'Tie'
        diff = 0
    
    if winner != 'Tie':
        print_success(f"🏆 Winner: {winner} (+{diff:.4f} or +{diff*100:.2f}%)")
    else:
        print_info("🤝 Performance is identical")
    
    print()
    print_info("📁 Output Files:")
    print(f"  - HTML Report:   {html_path}")
    print(f"  - Text Report:  {text_path}")
    print(f"  - Statistical Results: {stats_path}")
    print(f"  - Visualizations: {len(plots)} plots in {output_dir}")
    
    logger.blank()
    print_header("COMPARISON COMPLETE")
    print()
    print_success(f"✓ All results saved to: {output_dir}")
    print_info(f"✓ Open the HTML report in your browser for full details!")


if __name__ == '__main__':
    main()
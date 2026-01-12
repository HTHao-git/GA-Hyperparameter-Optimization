# ============================================================================
# FITNESS METRIC COMPARISON:  ACCURACY vs F1-SCORE
# ============================================================================
# Demonstrates the impact of fitness metric choice on imbalanced data
#
# BEFORE:   Accuracy-based optimization → predicts all majority class
# AFTER:   F1-based optimization → balanced class detection
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from pathlib import Path
import time
import json

from preprocessing.data_loader import DatasetLoader
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler
from preprocessing.smote_handler import SMOTEHandler
from preprocessing.pca import PCA

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (
    accuracy_score, f1_score, balanced_accuracy_score,
    confusion_matrix, classification_report
)

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.types import Individual

from utils.logger import get_logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning
import matplotlib.pyplot as plt
import seaborn as sns


# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'dataset': 'secom',
    'test_size': 0.2,
    'cv_folds': 3,  # Reduced for speed
    'random_state': 42,
    
    # GA settings (same for both)
    'ga': {
        'population_size':  20,
        'num_generations': 40,
        'crossover_rate': 0.8,
        'mutation_rate': 0.25,
        'elitism_rate': 0.10,
        'early_stopping': True,
        'patience': 5,
        'diversity_threshold': 0.0,
        'cache_fitness': False,
        'verbose': 1
    },
    
    'output_dir': 'outputs/fitness_comparison'
}

# Random Forest hyperparameter template
RF_TEMPLATE = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 15, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 4, 8],
    'class_weight': ['balanced', 'balanced_subsample', None],
    'max_features': ['sqrt', 'log2'],
    'pca_variance':  [0.95, 0.99],
    'smote_strategy': ['none', 'smote', 'adasyn', 'random_over'],
    'scaler': ['standard']
}


# ============================================================================
# FITNESS FUNCTIONS
# ============================================================================

class AccuracyFitness: 
    """Fitness based on accuracy (BEFORE - problematic for imbalanced data)."""
    
    def __init__(self, X, y, cv_folds=3, random_state=42):
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.random_state = random_state
    
    def evaluate(self, config):
        """Evaluate using accuracy."""
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
                y_train_fold, y_val_fold = self. y[train_idx], self. y[val_idx]
                
                # Preprocess
                X_train_p, y_train_p = self._preprocess(X_train_fold, y_train_fold, config, fit=True)
                X_val_p, _ = self._preprocess(X_val_fold, y_val_fold, config, fit=False)
                
                # Train
                model = self._create_model(config)
                model.fit(X_train_p, y_train_p)
                
                # Evaluate with ACCURACY
                y_pred = model.predict(X_val_p)
                accuracy = accuracy_score(y_val_fold, y_pred)
                scores.append(accuracy)
            
            return float(np.mean(scores))
        
        except Exception as e: 
            return 0.0
    
    def _preprocess(self, X, y, config, fit=True):
        """Apply preprocessing pipeline."""
        X_processed = X.copy()
        y_processed = y.copy()
        
        # Handle missing values
        if np.isnan(X_processed).any():
            if fit:
                self.mv_handler = MissingValuesHandler(strategy='mean')
                X_processed = self.mv_handler.fit_transform(X_processed)
            else:
                X_processed = self.mv_handler.transform(X_processed)
        
        # Scale
        scaler_type = config.get('scaler', 'standard')
        if fit:
            self.scaler = StandardScaler()
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        # SMOTE (training only)
        if fit:
            smote_strategy = config.get('smote_strategy', 'smote')
            if smote_strategy != 'none':
                smote_handler = SMOTEHandler(strategy=smote_strategy, random_state=self.random_state)
                X_processed, y_processed = smote_handler.fit_resample(X_processed, y_processed)
        
        # PCA
        pca_variance = config.get('pca_variance', 0.95)
        if fit:
            self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_processed)
        else:
            X_processed = self.pca.transform(X_processed)
        
        return X_processed, y_processed
    
    def _create_model(self, config):
        """Create Random Forest model."""
        # Extract class_weight separately
        class_weight = config.get('class_weight', None)
        
        # Filter out preprocessing params
        params = {k: v for k, v in config.items() 
                 if k not in ['pca_variance', 'smote_strategy', 'scaler', 'class_weight']}
        
        params['random_state'] = self.random_state
        
        # Add class_weight if specified
        if class_weight is not None:
            params['class_weight'] = class_weight
        
        return RandomForestClassifier(**params)

class F1Fitness(AccuracyFitness):
    """Fitness based on MACRO F1-score (AFTER - better for imbalanced data)."""
    
    def evaluate(self, config):
        """Evaluate using macro F1-score."""
        try:
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=self.random_state)
            scores = []
            
            for train_idx, val_idx in cv.split(self. X, self.y):
                X_train_fold, X_val_fold = self.X[train_idx], self.X[val_idx]
                y_train_fold, y_val_fold = self.y[train_idx], self.y[val_idx]
                
                # Preprocess
                X_train_p, y_train_p = self._preprocess(X_train_fold, y_train_fold, config, fit=True)
                X_val_p, _ = self._preprocess(X_val_fold, y_val_fold, config, fit=False)
                
                # Train
                model = self._create_model(config)
                model.fit(X_train_p, y_train_p)
                
                # Evaluate with MACRO F1-SCORE (treats classes equally)
                y_pred = model.predict(X_val_p)
                f1 = f1_score(y_val_fold, y_pred, average='macro', zero_division=0)
                scores.append(f1)
            
            return float(np.mean(scores))
        
        except Exception as e:
            return 0.0


# ============================================================================
# EVALUATION HELPER
# ============================================================================

def evaluate_model(X_train, X_test, y_train, y_test, config, fitness_evaluator, model_name):
    """Train and evaluate a model on test set."""
    
    print_info(f"Training {model_name} with best config...")
    
    # Preprocess
    X_train_p, y_train_p = fitness_evaluator._preprocess(X_train, y_train, config, fit=True)
    X_test_p, _ = fitness_evaluator._preprocess(X_test, y_test, config, fit=False)
    
    # Train
    model = fitness_evaluator._create_model(config)
    model.fit(X_train_p, y_train_p)
    
    # Predict
    y_pred = model.predict(X_test_p)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    # Minority class detection
    minority_class = 1
    if len(cm) > 1 and cm.shape[0] > minority_class and cm.shape[1] > minority_class:
        minority_detected = cm[minority_class, minority_class]
        minority_total = np.sum(y_test == minority_class)
        minority_recall = minority_detected / minority_total if minority_total > 0 else 0
    else:
        minority_recall = 0
    
    results = {
        'accuracy': accuracy,
        'f1_score': f1,
        'balanced_accuracy': bal_acc,
        'confusion_matrix': cm,
        'minority_recall': minority_recall,
        'predictions': y_pred
    }
    
    return results


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_confusion_matrices(results_before, results_after, output_dir):
    """Plot confusion matrices side by side."""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Before (Accuracy-based)
    cm_before = results_before['confusion_matrix']
    sns.heatmap(cm_before, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    ax1.set_title('BEFORE: Accuracy-Based Optimization', fontsize=12, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    
    # After (F1-based)
    cm_after = results_after['confusion_matrix']
    sns.heatmap(cm_after, annot=True, fmt='d', cmap='Greens', ax=ax2,
                xticklabels=['Class 0', 'Class 1'],
                yticklabels=['Class 0', 'Class 1'])
    ax2.set_title('AFTER: F1-Based Optimization', fontsize=12, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    
    plt.tight_layout()
    
    save_path = output_dir / 'confusion_matrix_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_info(f"Confusion matrix comparison saved:  {save_path}")


def plot_metrics_comparison(results_before, results_after, output_dir):
    """Plot metrics bar chart."""
    
    metrics = ['accuracy', 'f1_score', 'balanced_accuracy', 'minority_recall']
    metric_labels = ['Accuracy', 'F1-Score', 'Balanced\nAccuracy', 'Minority\nRecall']
    
    before_values = [results_before[m] for m in metrics]
    after_values = [results_after[m] for m in metrics]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars1 = ax.bar(x - width/2, before_values, width, label='Before (Accuracy)', color='#FF6B6B', alpha=0.8)
    bars2 = ax.bar(x + width/2, after_values, width, label='After (F1-Score)', color='#4ECDC4', alpha=0.8)
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Performance Comparison: Accuracy vs F1-Based Optimization', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels)
    ax.legend(fontsize=11)
    ax.set_ylim([0, 1.05])
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]: 
        for bar in bars: 
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    save_path = output_dir / 'metrics_comparison.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print_info(f"Metrics comparison saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    """Run before/after comparison."""
    
    logger = get_logger(name="FITNESS_COMPARISON", verbose=True)
    
    print_header("FITNESS METRIC COMPARISON")
    print_header("Accuracy vs F1-Score on Imbalanced Data")
    print()
    
    output_dir = Path(CONFIG['output_dir'])
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    
    print_section("STEP 1: Load Dataset")
    
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset(CONFIG['dataset'])
    
    logger.blank()
    print_info(f"Dataset:  {CONFIG['dataset']}")
    print_info(f"  Samples: {X.shape[0]}")
    print_info(f"  Features: {X.shape[1]}")
    print_info(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=CONFIG['test_size'], random_state=CONFIG['random_state'], stratify=y
    )
    
    print_info(f"  Test set: {len(y_test)} samples")
    print_info(f"    Class 0: {np.sum(y_test == 0)} ({np.sum(y_test == 0)/len(y_test)*100:.1f}%)")
    print_info(f"    Class 1: {np.sum(y_test == 1)} ({np.sum(y_test == 1)/len(y_test)*100:.1f}%)")
    
    logger.blank()
    
    # ========================================================================
    # STEP 2: BEFORE - Accuracy-Based Optimization
    # ========================================================================
    
    print_section("STEP 2: BEFORE - Accuracy-Based Optimization")
    print()
    
    print_warning("Using ACCURACY as fitness metric (problematic for imbalanced data)")
    print()
    
    accuracy_fitness = AccuracyFitness(X_train, y_train, cv_folds=CONFIG['cv_folds'], random_state=CONFIG['random_state'])
    
    ga_config = GAConfig(**CONFIG['ga'])
    
    ga_before = GeneticAlgorithm(
        config=ga_config,
        fitness_function=accuracy_fitness.evaluate,
        chromosome_template=RF_TEMPLATE,
        logger=logger
    )
    
    start_time = time.time()
    best_before = ga_before.run()
    time_before = time.time() - start_time
    
    # Evaluate on test set
    results_before = evaluate_model(X_train, X_test, y_train, y_test, 
                                    best_before.chromosome, accuracy_fitness, 
                                    "Accuracy-Optimized RF")
    
    logger.blank()
    print_info("BEFORE Results:")
    print(f"  Accuracy:           {results_before['accuracy']:.4f}")
    print(f"  F1-Score:          {results_before['f1_score']:.4f}")
    print(f"  Balanced Accuracy: {results_before['balanced_accuracy']:.4f}")
    print(f"  Minority Recall:   {results_before['minority_recall']:.4f} ({results_before['minority_recall']*100:.1f}%)")
    print()
    
    logger.blank()
    
    # ========================================================================
    # STEP 3: AFTER - F1-Based Optimization
    # ========================================================================
    
    print_section("STEP 3: AFTER - F1-Score-Based Optimization")
    print()
    
    print_success("Using F1-SCORE as fitness metric (better for imbalanced data)")
    print()
    
    f1_fitness = F1Fitness(X_train, y_train, cv_folds=CONFIG['cv_folds'], random_state=CONFIG['random_state'])
    
    ga_after = GeneticAlgorithm(
        config=ga_config,
        fitness_function=f1_fitness.evaluate,
        chromosome_template=RF_TEMPLATE,
        logger=logger
    )
    
    start_time = time.time()
    best_after = ga_after.run()
    time_after = time.time() - start_time
    
    # Evaluate on test set
    results_after = evaluate_model(X_train, X_test, y_train, y_test,
                                   best_after.chromosome, f1_fitness,
                                   "F1-Optimized RF")
    
    logger.blank()
    print_info("AFTER Results:")
    print(f"  Accuracy:          {results_after['accuracy']:.4f}")
    print(f"  F1-Score:          {results_after['f1_score']:.4f}")
    print(f"  Balanced Accuracy: {results_after['balanced_accuracy']:.4f}")
    print(f"  Minority Recall:   {results_after['minority_recall']:.4f} ({results_after['minority_recall']*100:.1f}%)")
    print()
    
    logger.blank()
    
    # ========================================================================
    # STEP 4: Compare Results
    # ========================================================================
    
    print_section("STEP 4: Comparison Summary")
    print()
    
    print("=" * 80)
    print(f"{'Metric':<25} {'BEFORE (Accuracy)':<25} {'AFTER (F1)':<25} {'Change':<10}")
    print("=" * 80)
    
    metrics_to_compare = [
        ('Accuracy', 'accuracy'),
        ('F1-Score', 'f1_score'),
        ('Balanced Accuracy', 'balanced_accuracy'),
        ('Minority Recall', 'minority_recall')
    ]
    
    for label, key in metrics_to_compare: 
        before_val = results_before[key]
        after_val = results_after[key]
        change = after_val - before_val
        change_str = f"+{change:.4f}" if change >= 0 else f"{change:.4f}"
        
        print(f"{label:<25} {before_val:<25.4f} {after_val: <25.4f} {change_str:<10}")
    
    print("=" * 80)
    print()
    
    # Highlight key finding
    if results_before['minority_recall'] < 0.1 and results_after['minority_recall'] > 0.3:
        print_success("🎯 KEY FINDING: F1-based optimization dramatically improved minority class detection!")
        print_info(f"   Minority recall improved from {results_before['minority_recall']*100:.1f}% to {results_after['minority_recall']*100:.1f}%")
    
    print()
    
    # ========================================================================
    # STEP 5: Visualizations
    # ========================================================================
    
    print_section("STEP 5: Generate Visualizations")
    print()
    
    plot_confusion_matrices(results_before, results_after, output_dir)
    plot_metrics_comparison(results_before, results_after, output_dir)
    
    logger.blank()
    
    # ========================================================================
    # STEP 6: Save Results
    # ========================================================================
    
    print_section("STEP 6: Save Results")
    print()
    
    comparison_results = {
        'before': {
            'fitness_metric': 'accuracy',
            'config': {k: (int(v) if isinstance(v, np.integer) else 
                          float(v) if isinstance(v, np.floating) else v) 
                      for k, v in best_before.chromosome.items()},
            'cv_score': float(best_before.fitness),
            'test_accuracy': float(results_before['accuracy']),
            'test_f1':  float(results_before['f1_score']),
            'balanced_accuracy': float(results_before['balanced_accuracy']),
            'minority_recall': float(results_before['minority_recall']),
            'optimization_time': float(time_before)
        },
        'after':  {
            'fitness_metric':  'f1_score',
            'config': {k: (int(v) if isinstance(v, np.integer) else 
                          float(v) if isinstance(v, np.floating) else v) 
                      for k, v in best_after.chromosome.items()},
            'cv_score': float(best_after.fitness),
            'test_accuracy': float(results_after['accuracy']),
            'test_f1': float(results_after['f1_score']),
            'balanced_accuracy': float(results_after['balanced_accuracy']),
            'minority_recall':  float(results_after['minority_recall']),
            'optimization_time': float(time_after)
        },
        'improvement': {
            'accuracy': float(results_after['accuracy'] - results_before['accuracy']),
            'f1_score': float(results_after['f1_score'] - results_before['f1_score']),
            'balanced_accuracy':  float(results_after['balanced_accuracy'] - results_before['balanced_accuracy']),
            'minority_recall': float(results_after['minority_recall'] - results_before['minority_recall'])
        }
    }
    
    results_path = output_dir / 'comparison_results.json'
    with open(results_path, 'w') as f:
        json.dump(comparison_results, f, indent=2)
    
    print_info(f"Results saved to: {results_path}")
    
    logger.blank()
    print_header("COMPARISON COMPLETE")
    print()
    print_success(f"✓ All results saved to:  {output_dir}")
    print_info(f"✓ Check the visualizations and results.json for detailed analysis!")


if __name__ == '__main__':
    main()
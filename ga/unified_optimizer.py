# ============================================================================
# UNIFIED OPTIMIZER
# ============================================================================
# Intelligent optimizer that automatically selects: 
#   - Genetic Algorithm (GA) for single-objective
#   - NSGA-II for multi-objective optimization
#
# FEATURES:
#   - Automatic algorithm selection
#   - Configurable fitness metrics
#   - Unified API for both approaches
#
# Last updated: 2026-01-12
# ============================================================================

import numpy as np
from typing import Dict, Any, Optional, List, Union, Tuple
from pathlib import Path

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.multi_objective import NSGAII
from ga.types import Individual

from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, roc_auc_score
)

from utils.logger import Logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning


# ============================================================================
# AVAILABLE FITNESS METRICS
# ============================================================================

AVAILABLE_METRICS = {
    'accuracy': {
        'function': accuracy_score,
        'requires_proba': False,
        'description': 'Overall accuracy (TP+TN)/(TP+TN+FP+FN)'
    },
    'f1_macro': {
        'function': lambda y_true, y_pred:  f1_score(y_true, y_pred, average='macro', zero_division=0),
        'requires_proba': False,
        'description': 'Macro-averaged F1 (unweighted mean of per-class F1)'
    },
    'f1_weighted': {
        'function': lambda y_true, y_pred:  f1_score(y_true, y_pred, average='weighted', zero_division=0),
        'requires_proba': False,
        'description': 'Weighted F1 (weighted by class support)'
    },
    'f1_binary': {
        'function': lambda y_true, y_pred:  f1_score(y_true, y_pred, average='binary', zero_division=0),
        'requires_proba': False,
        'description': 'Binary F1 score (for 2-class problems)'
    },
    'precision_macro': {
        'function': lambda y_true, y_pred:  precision_score(y_true, y_pred, average='macro', zero_division=0),
        'requires_proba': False,
        'description': 'Macro-averaged precision'
    },
    'recall_macro': {
        'function': lambda y_true, y_pred: recall_score(y_true, y_pred, average='macro', zero_division=0),
        'requires_proba':  False,
        'description': 'Macro-averaged recall'
    },
    'recall_minority': {
        'function': lambda y_true, y_pred:  recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        'requires_proba': False,
        'description': 'Recall of minority class (class 1)'
    },
    'balanced_accuracy': {
        'function': balanced_accuracy_score,
        'requires_proba': False,
        'description': 'Average of per-class recall (good for imbalanced data)'
    },
    'roc_auc': {
        'function': None,  # Custom handling needed
        'requires_proba':  True,
        'description': 'Area under ROC curve (requires predict_proba)'
    }
}


# ============================================================================
# UNIFIED OPTIMIZER CLASS
# ============================================================================

class UnifiedOptimizer:
    """
    Intelligent optimizer that automatically selects GA or NSGA-II. 
    
    - **Single objective** (1 metric) → Uses Genetic Algorithm
    - **Multi-objective** (2+ metrics) → Uses NSGA-II
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: 'random_forest', 'xgboost', 'lightgbm', 'neural_network', 'svm'
        fitness_metrics:  List of metric names or single metric string
        ga_config: GA configuration (optional)
        cv_folds: Cross-validation folds
        test_size: Test set size
        random_state: Random seed
        logger: Logger instance
    
    Example:
        >>> # Single objective
        >>> optimizer = UnifiedOptimizer(X, y, fitness_metrics='f1_macro')
        >>> results = optimizer.optimize()  # Uses GA
        
        >>> # Multi-objective
        >>> optimizer = UnifiedOptimizer(X, y, fitness_metrics=['accuracy', 'recall_minority'])
        >>> results = optimizer.optimize()  # Uses NSGA-II
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_type: str = 'random_forest',
                 fitness_metrics: Union[str, List[str]] = 'accuracy',
                 ga_config:  Optional[GAConfig] = None,
                 cv_folds:  int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        
        self.X = X
        self.y = y
        self.model_type = model_type
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger
        
        # Convert single metric to list
        if isinstance(fitness_metrics, str):
            fitness_metrics = [fitness_metrics]
        
        self.fitness_metrics = fitness_metrics
        self.num_objectives = len(fitness_metrics)
        
        # Validate metrics
        for metric in fitness_metrics:
            if metric not in AVAILABLE_METRICS:
                raise ValueError(f"Unknown metric: {metric}. Available: {list(AVAILABLE_METRICS.keys())}")
        
        # GA configuration
        if ga_config is None:
            ga_config = GAConfig(
                population_size=20,
                num_generations=15,
                crossover_rate=0.8,
                mutation_rate=0.2,
                elitism_rate=0.1,
                early_stopping=True,
                patience=5,
                verbose=1,
                random_state=random_state
            )
        
        self.ga_config = ga_config
        
        # Determine algorithm
        if self.num_objectives == 1:
            self.algorithm = 'GA'
            if self.logger:
                self.logger.info(f"Single objective ({fitness_metrics[0]}) → Using Genetic Algorithm")
        else:
            self.algorithm = 'NSGA-II'
            if self.logger:
                self.logger.info(f"Multi-objective ({len(fitness_metrics)} metrics) → Using NSGA-II")
        
        # Load appropriate model optimizer
        self._load_model_optimizer()
    
    # ========================================================================
    # MODEL OPTIMIZER LOADING
    # ========================================================================
    
    def _load_model_optimizer(self):
        """Load the appropriate model optimizer."""
        
        if self.model_type == 'random_forest':
            from ga.ml_optimizer import MLOptimizer
            self.base_optimizer = MLOptimizer(
                self.X, self.y,
                model_type='random_forest',
                ga_config=self.ga_config,
                cv_folds=self.cv_folds,
                test_size=self.test_size,
                random_state=self.random_state,
                logger=self.logger
            )
        
        elif self.model_type == 'xgboost':
            from models.xgboost_optimizer import XGBoostOptimizer
            self.base_optimizer = XGBoostOptimizer(
                self.X, self.y,
                ga_config=self.ga_config,
                cv_folds=self.cv_folds,
                test_size=self.test_size,
                random_state=self.random_state,
                logger=self.logger
            )
        
        elif self.model_type == 'lightgbm': 
            from models.lightgbm_optimizer import LightGBMOptimizer
            self.base_optimizer = LightGBMOptimizer(
                self.X, self.y,
                ga_config=self.ga_config,
                cv_folds=self.cv_folds,
                test_size=self.test_size,
                random_state=self.random_state,
                logger=self.logger
            )
        
        elif self.model_type == 'neural_network':
            from models.neural_network_optimizer import NeuralNetworkOptimizer
            self.base_optimizer = NeuralNetworkOptimizer(
                self.X, self.y,
                ga_config=self.ga_config,
                cv_folds=self.cv_folds,
                test_size=self.test_size,
                random_state=self.random_state,
                logger=self.logger
            )
        
        elif self.model_type == 'svm':
            from models.svm_optimizer import SVMOptimizer
            self.base_optimizer = SVMOptimizer(
                self.X, self.y,
                ga_config=self.ga_config,
                cv_folds=self.cv_folds,
                test_size=self.test_size,
                random_state=self.random_state,
                logger=self.logger
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Get chromosome template
        self.chromosome_template = self.base_optimizer.chromosome_template
    
    # ========================================================================
    # FITNESS EVALUATION
    # ========================================================================
    
    def _evaluate_single_objective(self, config: Dict[str, Any]) -> float:
        """
        Evaluate configuration for single objective.
        
        Returns:
            Fitness score (scalar)
        """
        metric_name = self.fitness_metrics[0]
        metric_info = AVAILABLE_METRICS[metric_name]
        
        try:
            from sklearn.model_selection import StratifiedKFold
            
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            scores = []
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train_fold = self.X[train_idx]
                y_train_fold = self.y[train_idx]
                X_val_fold = self.X[val_idx]
                y_val_fold = self.y[val_idx]
                
                # Preprocess
                X_train_p, y_train_p = self.base_optimizer._preprocess(
                    X_train_fold, y_train_fold, config, fit=True
                )
                X_val_p, _ = self.base_optimizer._preprocess(
                    X_val_fold, y_val_fold, config, fit=False
                )
                
                # Train
                model = self.base_optimizer._create_model(config)
                
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model.fit(X_train_p, y_train_p)
                
                # Predict
                y_pred = model.predict(X_val_p)
                
                # Calculate metric
                if metric_name == 'roc_auc':
                    try:
                        if hasattr(model, 'predict_proba'):
                            y_proba = model.predict_proba(X_val_p)[:, 1]
                            score = roc_auc_score(y_val_fold, y_proba)
                        else:
                            score = 0.0
                    except:
                        score = 0.0
                else:
                    score = metric_info['function'](y_val_fold, y_pred)
                
                scores.append(score)
            
            return float(np.mean(scores))
        
        except Exception as e: 
            if self.logger and self.ga_config.verbose >= 2:
                self.logger.warning(f"Configuration failed: {e}")
            return 0.0
    
    def _evaluate_multi_objective(self, config: Dict[str, Any]) -> List[float]:
        """
        Evaluate configuration for multiple objectives.
        
        Returns:
            List of fitness scores (one per objective)
        """
        try:
            from sklearn.model_selection import StratifiedKFold
            
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            # Store scores for each metric
            all_scores = {metric: [] for metric in self.fitness_metrics}
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                X_train_fold = self.X[train_idx]
                y_train_fold = self.y[train_idx]
                X_val_fold = self.X[val_idx]
                y_val_fold = self.y[val_idx]
                
                # Preprocess
                X_train_p, y_train_p = self.base_optimizer._preprocess(
                    X_train_fold, y_train_fold, config, fit=True
                )
                X_val_p, _ = self.base_optimizer._preprocess(
                    X_val_fold, y_val_fold, config, fit=False
                )
                
                # Train
                model = self.base_optimizer._create_model(config)
                
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore')
                    model.fit(X_train_p, y_train_p)
                
                # Predict
                y_pred = model.predict(X_val_p)
                
                # Calculate all metrics
                for metric_name in self.fitness_metrics:
                    metric_info = AVAILABLE_METRICS[metric_name]
                    
                    if metric_name == 'roc_auc': 
                        try:
                            if hasattr(model, 'predict_proba'):
                                y_proba = model.predict_proba(X_val_p)[:, 1]
                                score = roc_auc_score(y_val_fold, y_proba)
                            else:
                                score = 0.0
                        except:
                            score = 0.0
                    else:
                        score = metric_info['function'](y_val_fold, y_pred)
                    
                    all_scores[metric_name].append(score)
            
            # Return mean of each metric
            return [float(np.mean(all_scores[metric])) for metric in self.fitness_metrics]
        
        except Exception as e: 
            if self.logger and self.ga_config.verbose >= 2:
                self.logger.warning(f"Configuration failed: {e}")
            return [0.0] * self.num_objectives
    
    # ========================================================================
    # OPTIMIZATION
    # ========================================================================
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run optimization (automatically uses GA or NSGA-II).
        
        Returns:
            Results dictionary with best config(s) and scores
        """
        if self.logger:
            print_header(f"{self.model_type.upper()} OPTIMIZATION")
            print_info(f"Algorithm: {self.algorithm}")
            print_info(f"Objectives: {', '.join(self.fitness_metrics)}")
            print()
        
        import time
        optimization_start = time.time()
        
        if self.algorithm == 'GA': 
            results = self._optimize_single_objective()
        else:
            results = self._optimize_multi_objective()
        
        optimization_time = time.time() - optimization_start
        results['optimization_time'] = optimization_time
        results['algorithm'] = self.algorithm
        results['fitness_metrics'] = self.fitness_metrics
        
        return results
    
    def _optimize_single_objective(self) -> Dict[str, Any]:
        """Optimize using standard GA."""
        
        ga = GeneticAlgorithm(
            config=self.ga_config,
            fitness_function=self._evaluate_single_objective,
            chromosome_template=self.chromosome_template,
            logger=self.logger
        )
        
        best_individual = ga.run()
        
        # Final evaluation
        if self.logger:
            print()
            print_section("Final Evaluation")
        
        # Handle different return formats
        result = self.base_optimizer._final_evaluation(best_individual.chromosome)
        
        if isinstance(result, tuple):
            final_score, final_metrics = result
        else: 
            # Old format:  just returns score
            final_score = result
            final_metrics = {
                'accuracy': final_score,
                'f1_score': 0.0,
                'evaluation_time': 0.0
            }
        
        if self.logger:
            print_success(f"✓ Optimization complete!")
            print_info(f"  Best CV score ({self.fitness_metrics[0]}): {best_individual.fitness:.4f}")
            print_info(f"  Test accuracy: {final_metrics['accuracy']:.4f}")
            print_info(f"  Test F1-score: {final_metrics['f1_score']:.4f}")
        
        return {
            'best_config': best_individual.chromosome,
            'best_score': float(best_individual.fitness),
            'test_accuracy': float(final_metrics['accuracy']),
            'test_f1_score': float(final_metrics['f1_score']),
            'evaluation_time': float(final_metrics.get('evaluation_time', 0.0)),
            'history': ga.history
        }
    
    def _optimize_multi_objective(self) -> Dict[str, Any]:
        """Optimize using NSGA-II."""
        
        # All objectives are maximized
        objectives = ['maximize'] * self.num_objectives
        
        nsga2 = NSGAII(
            population_size=self.ga_config.population_size,
            num_generations=self.ga_config.num_generations,
            objectives=objectives,
            crossover_rate=self.ga_config.crossover_rate,
            mutation_rate=self.ga_config.mutation_rate,
            random_state=self.random_state,
            logger=self.logger
        )
        
        pareto_front = nsga2.optimize(
            self._evaluate_multi_objective,
            self.chromosome_template
        )
        
        # Select best compromise solution (middle of Pareto front)
        if pareto_front:
            middle_idx = len(pareto_front) // 2
            best_solution = pareto_front[middle_idx]
        else:
            best_solution = None
        
        if self.logger:
            print()
            print_section("Multi-Objective Results")
            print_success(f"✓ Found {len(pareto_front)} Pareto-optimal solutions!")
            
            if best_solution:
                print_info("  Best compromise solution (middle of Pareto front):")
                for metric, score in zip(self.fitness_metrics, best_solution.fitness):
                    print_info(f"    {metric}:  {score:.4f}")
        
        return {
            'pareto_front': pareto_front,
            'best_compromise': best_solution.chromosome if best_solution else None,
            'best_scores': {
                metric: float(score)
                for metric, score in zip(self.fitness_metrics, best_solution.fitness)
            } if best_solution else {},
            'num_pareto_solutions': len(pareto_front)
        }
    
    # ========================================================================
    # UTILITY
    # ========================================================================
    
    @staticmethod
    def list_available_metrics() -> Dict[str, str]:
        """List all available fitness metrics."""
        return {
            name: info['description']
            for name, info in AVAILABLE_METRICS.items()
        }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from preprocessing.data_loader import DatasetLoader
    from utils.logger import get_logger
    
    logger = get_logger(name="UNIFIED_TEST", verbose=True)
    
    # Load dataset
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, _ = loader.load_dataset('secom')
    
    print_header("UNIFIED OPTIMIZER TEST")
    print()
    
    # ========================================================================
    # TEST 1: Single Objective (GA)
    # ========================================================================
    
    print_section("TEST 1: Single Objective (Macro F1)")
    print()
    
    optimizer_single = UnifiedOptimizer(
        X, y,
        model_type='random_forest',
        fitness_metrics='f1_macro',  # Single → GA
        ga_config=GAConfig(population_size=10, num_generations=5),
        logger=logger
    )
    
    results_single = optimizer_single.optimize()
    
    print()
    print(f"Best config: {results_single['best_config']}")
    print(f"Best score: {results_single['best_score']:.4f}")
    
    logger.blank()
    
    # ========================================================================
    # TEST 2: Multi-Objective (NSGA-II)
    # ========================================================================
    
    print_section("TEST 2: Multi-Objective (Accuracy + Minority Recall)")
    print()
    
    optimizer_multi = UnifiedOptimizer(
        X, y,
        model_type='random_forest',
        fitness_metrics=['accuracy', 'recall_minority'],  # Multi → NSGA-II
        ga_config=GAConfig(population_size=10, num_generations=5),
        logger=logger
    )
    
    results_multi = optimizer_multi.optimize()
    
    print()
    print(f"Pareto front size: {results_multi['num_pareto_solutions']}")
    print(f"Best compromise scores: {results_multi['best_scores']}")
    
    logger.blank()
    print_success("✓ Unified Optimizer test complete!")
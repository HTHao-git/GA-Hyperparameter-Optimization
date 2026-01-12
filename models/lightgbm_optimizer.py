# ============================================================================
# LIGHTGBM HYPERPARAMETER OPTIMIZER
# ============================================================================
# GA-based optimization for LightGBM models
#
# FEATURES:
#   - Comprehensive LightGBM hyperparameter space
#   - Faster than XGBoost (leaf-wise growth)
#   - Integration with GA framework
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

from ga.genetic_algorithm import GeneticAlgorithm, GAConfig
from ga.types import Individual
from preprocessing.missing_values import MissingValuesHandler
from preprocessing.scaling import StandardScaler, MinMaxScaler, RobustScaler
from preprocessing.smote_handler import SMOTEHandler
from preprocessing.pca import PCA

from utils.logger import Logger
from utils.colors import print_header, print_section, print_success, print_info, print_warning

# LightGBM
try:
    import lightgbm as lgb
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print_warning("LightGBM not available. Install with: pip install lightgbm")


# ============================================================================
# LIGHTGBM HYPERPARAMETER TEMPLATE
# ============================================================================

LIGHTGBM_HYPERPARAMETER_TEMPLATE = {
    # Boosting parameters
    'n_estimators': [50, 100, 200, 300, 500],
    'max_depth': [3, 5, 7, 9, 11, -1],  # -1 = no limit
    'learning_rate': (0.01, 0.3),
    'num_leaves': [15, 31, 63, 127],  # Max leaves per tree
    
    # Sampling
    'subsample': (0.6, 1.0),
    'colsample_bytree': (0.6, 1.0),
    'subsample_freq': [0, 1, 5],  # Frequency of bagging
    
    # Regularization
    'reg_alpha': (0.0, 1.0),  # L1
    'reg_lambda': (0.0, 2.0),  # L2
    'min_child_samples': [5, 10, 20, 30],
    'min_child_weight': (0.001, 0.1),
    
    # Class balancing
    'class_weight': ['balanced', None],
    
    # Preprocessing
    'pca_variance':  [0.90, 0.95, 0.99],
    'smote_strategy': ['none', 'smote', 'adasyn'],
    'scaler': ['standard', 'minmax', 'robust']
}


# ============================================================================
# LIGHTGBM OPTIMIZER CLASS
# ============================================================================

class LightGBMOptimizer: 
    """
    LightGBM hyperparameter optimizer using Genetic Algorithm. 
    
    Args:
        X: Feature matrix
        y: Labels
        ga_config: GA configuration (optional)
        cv_folds: Number of cross-validation folds
        test_size: Test set size for final evaluation
        random_state: Random seed
        logger: Logger instance
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 ga_config: Optional[GAConfig] = None,
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        
        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        
        self.X = X
        self.y = y
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger
        
        # GA configuration
        if ga_config is None:
            ga_config = GAConfig(
                population_size=15,
                num_generations=15,
                crossover_rate=0.8,
                mutation_rate=0.20,
                elitism_rate=0.15,
                early_stopping=True,
                patience=5,
                diversity_threshold=0.0,
                cache_fitness=False,
                verbose=1,
                random_state=random_state
            )
        
        self.ga_config = ga_config
        self.chromosome_template = LIGHTGBM_HYPERPARAMETER_TEMPLATE
        
        # Results
        self.best_config = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        if self.logger:
            self.logger.info(f"LightGBMOptimizer initialized")
            self.logger.info(f"  Dataset: {X.shape}")
            self.logger.info(f"  CV folds: {cv_folds}")
    
    # ========================================================================
    # OPTIMIZATION
    # ========================================================================
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run GA optimization. 
        
        Returns:
            Best configuration found
        """
        if self.logger:
            print_header("LIGHTGBM HYPERPARAMETER OPTIMIZATION")
            print()
        
        # Create GA instance
        ga = GeneticAlgorithm(
            config=self.ga_config,
            fitness_function=self._evaluate_configuration,
            chromosome_template=self.chromosome_template,
            logger=self.logger
        )
        
        import time
        optimization_start = time.time()

        # Run optimization
        best_individual = ga.run()

        # Calculate optimization time        
        optimization_time = time.time() - optimization_start

        # Store results
        self.best_config = best_individual.chromosome
        self.best_score = best_individual.fitness
        self.optimization_history = ga.history
        
        # Final evaluation
        if self.logger:
            print()
            print_section("Final Evaluation")
        
        final_score, final_metrics = self._final_evaluation(self.best_config)
        
        if self.logger:
            print_success(f"✓ Optimization complete!")
            print_info(f"  Best CV score: {self.best_score:.4f}")
            print_info(f"  Test accuracy: {final_metrics['accuracy']:.4f}")
            print_info(f"  Test F1-score: {final_metrics['f1_score']:.4f}")

        self.optimization_time = optimization_time
        self.evaluation_time = final_metrics. get('evaluation_time', 0.0)
        self.total_time = optimization_time + self.evaluation_time

        return {
            'config': self.best_config,
            'cv_score': float(self.best_score),
            'test_accuracy': float(final_metrics['accuracy']),
            'test_f1_score': float(final_metrics['f1_score']),
            'optimization_time': float(optimization_time),
            'evaluation_time': float(final_metrics['evaluation_time']),
            'total_time': float(optimization_time + final_metrics['evaluation_time']),
            'history': [
                {
                    'generation':  int(stats.generation),
                    'best_fitness': float(stats.best_fitness),
                    'mean_fitness': float(stats.mean_fitness),
                    'time':  float(stats.time) if hasattr(stats, 'time') else 0.0
                }
                for stats in self.optimization_history
            ]
        }
    
    # ========================================================================
    # FITNESS EVALUATION
    # ========================================================================
    
    def _evaluate_configuration(self, config: Dict[str, Any]) -> float:
        """
        Evaluate a configuration using cross-validation.
        
        Args:
            config:  Hyperparameter configuration
            
        Returns:
            Fitness score (CV accuracy)
        """
        try:
            # Manual CV to prevent data leakage
            cv = StratifiedKFold(
                n_splits=self.cv_folds,
                shuffle=True,
                random_state=self.random_state
            )
            
            scores = []
            
            for train_idx, val_idx in cv.split(self.X, self.y):
                # Split data
                X_train_fold = self.X[train_idx]
                y_train_fold = self.y[train_idx]
                X_val_fold = self.X[val_idx]
                y_val_fold = self.y[val_idx]
                
                # Preprocess
                X_train_processed, y_train_processed = self._preprocess(
                    X_train_fold, y_train_fold, config, fit=True
                )
                X_val_processed, _ = self._preprocess(
                    X_val_fold, y_val_fold, config, fit=False
                )
                
                # Create and train model
                model = self._create_model(config)
                model.fit(
                    X_train_processed, y_train_processed,
                    eval_set=[(X_val_processed, y_val_fold)],
                    callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
                )
                
                # Predict and evaluate
                y_pred = model.predict(X_val_processed)
                accuracy = accuracy_score(y_val_fold, y_pred)
                scores.append(accuracy)
            
            # Return mean accuracy
            return float(np.mean(scores))
        
        except Exception as e: 
            # Invalid configuration
            if self.logger and self.ga_config.verbose >= 2:
                self.logger.warning(f"Configuration failed: {e}")
            return 0.0
    
    def _final_evaluation(self, config: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Final evaluation on held-out test set.
        
        Args:
            config: Best configuration
            
        Returns:
            (test_score, metrics_dict)
        """

        import time
        eval_start = time.time()

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self.y
        )
        
        # Preprocess
        X_train_processed, y_train_processed = self._preprocess(X_train, y_train, config, fit=True)
        X_test_processed, _ = self._preprocess(X_test, y_test, config, fit=False)
        
        # Train model
        model = self._create_model(config)
        model.fit(
            X_train_processed, y_train_processed,
            eval_set=[(X_test_processed, y_test)],
            callbacks=[lgb.early_stopping(stopping_rounds=10, verbose=False)]
        )
        
        # Predict
        y_pred = model.predict(X_test_processed)
        
        # Evaluate
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        eval_time = time.time() - eval_start

        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'evaluation_time': eval_time
        }
        
        return accuracy, metrics
    
    # ========================================================================
    # PREPROCESSING
    # ========================================================================
    
    def _preprocess(self,
                    X: np.ndarray,
                    y: np.ndarray,
                    config: Dict[str, Any],
                    fit: bool = True) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Apply preprocessing pipeline.
        
        Args:
            X: Features
            y: Labels
            config: Configuration with preprocessing params
            fit: Whether to fit transformers
            
        Returns: 
            (X_processed, y_processed) tuple
        """
        X_processed = X.copy()
        y_processed = y.copy()
        
        # 1. Handle missing values
        if np.isnan(X_processed).any():
            if fit:
                self.mv_handler = MissingValuesHandler(strategy='mean')
                X_processed = self.mv_handler.fit_transform(X_processed)
            else:
                X_processed = self.mv_handler.transform(X_processed)
        
        # 2. Scale features
        scaler_type = config.get('scaler', 'standard')
        
        if fit:
            if scaler_type == 'standard':
                self.scaler = StandardScaler()
            elif scaler_type == 'minmax':
                self.scaler = MinMaxScaler()
            elif scaler_type == 'robust':
                self.scaler = RobustScaler()
            
            X_processed = self.scaler.fit_transform(X_processed)
        else:
            X_processed = self.scaler.transform(X_processed)
        
        # 3. Balance classes (only for training)
        if fit:
            smote_strategy = config.get('smote_strategy', 'none')
            
            if smote_strategy != 'none':
                smote_handler = SMOTEHandler(
                    strategy=smote_strategy,
                    random_state=self.random_state
                )
                X_processed, y_processed = smote_handler.fit_resample(X_processed, y_processed)
        
        # 4. PCA
        pca_variance = config.get('pca_variance', 0.95)
        
        if fit:
            self.pca = PCA(n_components=pca_variance, random_state=self.random_state)
            X_processed = self.pca.fit_transform(X_processed)
        else:
            X_processed = self.pca.transform(X_processed)
        
        return X_processed, y_processed
    
    # ========================================================================
    # MODEL CREATION
    # ========================================================================
    
    def _create_model(self, config: Dict[str, Any]):
        """
        Create LightGBM model from configuration.
        
        Args:
            config: Model hyperparameters
            
        Returns:
            LightGBM classifier instance
        """
        # Extract model-specific params
        model_params = {
            k: v for k, v in config.items()
            if k not in ['pca_variance', 'smote_strategy', 'scaler']
        }
        
        # Add fixed parameters
        model_params['random_state'] = self.random_state
        model_params['verbose'] = -1  # Suppress warnings
        model_params['force_col_wise'] = True  # Faster for wide datasets
        
        # Create model
        return lgb.LGBMClassifier(**model_params)
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_results(self, filepath: Path):
        """Save optimization results."""
        
        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types recursively."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, bool):  # Regular Python bool
                return obj
            elif obj is None or isinstance(obj, (str, int, float)):
                return obj
            else:
                # Try to convert to string as fallback
                return str(obj)
        
        results = {
            'model_type': 'neural_network',
            'best_config': self.best_config,
            'architecture': self._decode_architecture(self.best_config),
            'best_cv_score': self.best_score,
            'timing': {
                'optimization_time': getattr(self, 'optimization_time', 0.0),
                'evaluation_time': getattr(self, 'evaluation_time', 0.0),
                'total_time': getattr(self, 'total_time', 0.0)
            },
            'ga_config': {
                'population_size': self.ga_config.population_size,
                'num_generations': self.ga_config.num_generations,
                'crossover_rate': self.ga_config.crossover_rate,
                'mutation_rate': self. ga_config.mutation_rate
            },
            'history': [
                {
                    'generation': stats.generation,
                    'best_fitness': stats.best_fitness,
                    'mean_fitness': stats.mean_fitness,
                    'diversity': stats.diversity
                }
                for stats in self.optimization_history
            ]
        }

        results = convert_to_serializable(results)
        
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.logger:
            self.logger.info(f"Results saved to:  {filepath}")

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from preprocessing.data_loader import DatasetLoader
    from utils.logger import get_logger
    
    logger = get_logger(name="LIGHTGBM_TEST", verbose=True)
    
    print_header("LIGHTGBM OPTIMIZER TEST")
    print()
    
    # Load dataset
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset('secom')
    
    logger.blank()
    
    # Create optimizer (small population for quick test)
    ga_config = GAConfig(
        population_size=10,
        num_generations=5,
        cache_fitness=False,
        verbose=1
    )
    
    optimizer = LightGBMOptimizer(
        X, y,
        ga_config=ga_config,
        logger=logger
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    optimizer.save_results(Path('outputs/lightgbm_test/optimization_results.json'))
    
    logger.blank()
    print_success("✓ LightGBM Optimizer test complete!")
# ============================================================================
# ML HYPERPARAMETER OPTIMIZER
# ============================================================================
# Genetic Algorithm for ML model hyperparameter optimization
#
# FEATURES:
#   - Optimize preprocessing + model hyperparameters
#   - Cross-validation fitness evaluation
#   - Multi-objective optimization (accuracy + speed)
#   - Integration with preprocessing pipeline
#
# USAGE: 
#   from ga.ml_optimizer import MLOptimizer
#   
#   optimizer = MLOptimizer(X, y, model_type='random_forest')
#   best_config = optimizer.optimize()
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import time
from typing import Dict, Any, Tuple
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

# ML models
try:
    from sklearn.model_selection import cross_val_score, StratifiedKFold
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neighbors import KNeighborsClassifier
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print_warning("scikit-learn not available")


# ============================================================================
# HYPERPARAMETER TEMPLATES
# ============================================================================

HYPERPARAMETER_TEMPLATES = {
    'random_forest': {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [5, 10, 15, 20, 25, None],
        'min_samples_split': [2, 5, 10, 20],
        'min_samples_leaf': [1, 2, 4, 8],
        'max_features': ['sqrt', 'log2', None],
        
        # Preprocessing
        'pca_variance':  [0.90, 0.95, 0.99],
        'smote_strategy': ['smote', 'random_over'],
        'scaler': ['standard', 'minmax', 'robust']
    },
    
    'svm': {
        'C': (0.1, 100.0),  # Continuous range
        'kernel': ['rbf', 'linear', 'poly'],
        'gamma': ['scale', 'auto'],
        
        # Preprocessing
        'pca_variance': [0.90, 0.95, 0.99],
        'smote_strategy': ['smote', 'random_over'],
        'scaler': ['standard', 'minmax', 'robust']
    },
    
    'knn': {
        'n_neighbors': [3, 5, 7, 9, 11, 15, 21],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        
        # Preprocessing
        'pca_variance': [0.90, 0.95, 0.99],
        'smote_strategy':  ['smote', 'random_over'],
        'scaler': ['standard', 'minmax', 'robust']
    }
}


# ============================================================================
# ML OPTIMIZER CLASS
# ============================================================================

class MLOptimizer:
    """
    Genetic Algorithm optimizer for ML hyperparameters.
    
    Args:
        X: Feature matrix
        y: Labels
        model_type: Type of model ('random_forest', 'svm', 'knn')
        ga_config: GA configuration (optional)
        cv_folds: Number of cross-validation folds
        test_size: Test set size for final evaluation
        random_state: Random seed
        logger: Logger instance
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 model_type: str = 'random_forest',
                 ga_config: Optional[GAConfig] = None,
                 fitness_metric: str = 'accuracy',
                 cv_folds: int = 5,
                 test_size: float = 0.2,
                 random_state: int = 42,
                 logger: Optional[Logger] = None):
        
        if not SKLEARN_AVAILABLE: 
            raise ImportError("scikit-learn is required for MLOptimizer")
        
        self.X = X
        self.y = y
        self.model_type = model_type
        self.fitness_metric = fitness_metric
        self.cv_folds = cv_folds
        self.test_size = test_size
        self.random_state = random_state
        self.logger = logger
        
        # GA configuration
        if ga_config is None:
            ga_config = GAConfig(
                population_size=20,
                num_generations=30,
                crossover_rate=0.8,
                mutation_rate=0.15,
                elitism_rate=0.1,
                early_stopping=True,
                patience=5,
                verbose=1,
                random_state=random_state
            )
        
        self.ga_config = ga_config
        
        # Get hyperparameter template
        if model_type not in HYPERPARAMETER_TEMPLATES:
            raise ValueError(f"Unknown model type:  {model_type}")
        
        self.chromosome_template = HYPERPARAMETER_TEMPLATES[model_type]
        
        # Results
        self.best_config = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        if self.logger:
            self.logger.info(f"MLOptimizer initialized for {model_type}")
            self.logger.info(f"  Dataset:  {X.shape}")
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
            print_header(f"HYPERPARAMETER OPTIMIZATION - {self.model_type.upper()}")
            print()
        
        # Create GA instance
        ga = GeneticAlgorithm(
            config=self.ga_config,
            fitness_function=self._evaluate_configuration,
            chromosome_template=self.chromosome_template,
            logger=self.logger
        )
        
        # Run optimization
        best_individual = ga.run()
        
        # Store results
        self.best_config = best_individual.chromosome
        self.best_score = best_individual.fitness
        self.optimization_history = ga.history
        
        # Final evaluation
        if self.logger:
            print()
            print_section("Final Evaluation")
        
        final_score = self._final_evaluation(self.best_config)
        
        if self.logger:
            print_success(f"✓ Optimization complete!")
            print_info(f"  Best CV score: {self.best_score:.4f}")
            print_info(f"  Test score: {final_score:.4f}")
        
        return {
            'config': self.best_config,
            'cv_score': float(self.best_score),
            'test_score': float(final_score),
            'history': [
                {
                    'generation':  int(stats.generation),
                    'best_fitness': float(stats.best_fitness),
                    'mean_fitness': float(stats.mean_fitness)
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
            from sklearn.metrics import accuracy_score
            
            # Manual CV to prevent data leakage
            cv = StratifiedKFold(
                n_splits=self. cv_folds,
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
                
                # Preprocess (fit on train, transform on val)
                X_train_processed, y_train_processed = self._preprocess(
                    X_train_fold, y_train_fold, config, fit=True
                )
                X_val_processed, _ = self._preprocess(
                    X_val_fold, y_val_fold, config, fit=False
                )
                
                # Create and train model
                model = self._create_model(config)
                model.fit(X_train_processed, y_train_processed)
                
                # Predict and evaluate
                y_pred = model.predict(X_val_processed)
                accuracy = accuracy_score(y_val_fold, y_pred)
                scores.append(accuracy)
            
            # Return mean accuracy
            return float(np.mean(scores))
        
        except Exception as e:
            # Invalid configuration
            if self.logger and self.ga_config.verbose >= 2:
                self.logger. warning(f"Configuration failed: {e}")
            return 0.0
    
    def _final_evaluation(self, config:  Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """
        Final evaluation on held-out test set.
        
        Returns:
            (test_score, metrics_dict)
        """
        import time
        eval_start = time.time()
        
        from sklearn.model_selection import train_test_split
        from sklearn.metrics import accuracy_score, f1_score
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=self. y
        )
        
        # Preprocess
        X_train_processed, y_train_processed = self._preprocess(X_train, y_train, config, fit=True)
        X_test_processed, _ = self._preprocess(X_test, y_test, config, fit=False)
        
        # Train model
        model = self._create_model(config)
        model.fit(X_train_processed, y_train_processed)
        
        # Predict
        y_pred = model. predict(X_test_processed)
        
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
                    config:  Dict[str, Any],
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
        
        # 1. Handle missing values (if any)
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
            smote_strategy = config.get('smote_strategy', 'smote')
            
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
        Create model from configuration.
        
        Args:
            config: Model hyperparameters
            
        Returns:
            Scikit-learn model instance
        """
        # Extract model-specific params (exclude preprocessing params)
        model_params = {
            k: v for k, v in config.items()
            if k not in ['pca_variance', 'smote_strategy', 'scaler']
        }
        
        # Add random state if applicable
        if self.model_type in ['random_forest', 'svm']: 
            model_params['random_state'] = self.random_state
        
        # Create model
        if self.model_type == 'random_forest':
            return RandomForestClassifier(**model_params)
        
        elif self.model_type == 'svm':
            return SVC(**model_params)
        
        elif self.model_type == 'knn': 
            return KNeighborsClassifier(**model_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    # ========================================================================
    # PERSISTENCE
    # ========================================================================
    
    def save_results(self, filepath:  Path):
        """Save optimization results."""
        
        # Helper to convert NumPy types
        def convert_to_serializable(obj):
            """Convert NumPy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np. floating, np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif obj is None or isinstance(obj, (str, bool)):
                return obj
            else:
                return obj
        
        results = {
            'model_type': self.model_type,
            'best_config':  convert_to_serializable(self.best_config),
            'best_cv_score': float(self.best_score),
            'ga_config':  {
                'population_size':  int(self.ga_config.population_size),
                'num_generations': int(self.ga_config.num_generations),
                'crossover_rate': float(self.ga_config.crossover_rate),
                'mutation_rate': float(self.ga_config.mutation_rate)
            },
            'history': [
                {
                    'generation':  int(stats.generation),
                    'best_fitness': float(stats.best_fitness),
                    'mean_fitness': float(stats.mean_fitness),
                    'diversity': float(stats. diversity)
                }
                for stats in self.optimization_history
            ]
        }
        
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
    
    logger = get_logger(name="ML_OPTIMIZER_TEST", verbose=True)
    
    print_header("ML OPTIMIZER TEST")
    print()
    
    # Load dataset
    loader = DatasetLoader(logger=logger, interactive=False)
    X, y, metadata = loader.load_dataset('secom')
    
    logger.blank()
    
    # Create optimizer with better settings
    ga_config = GAConfig(
        population_size=20,
        num_generations=20,
        crossover_rate=0.8,
        mutation_rate=0.15,
        elitism_rate=0.1,
        early_stopping=True,
        patience=8,
        diversity_threshold=0.001,
        verbose=1
    )
    
    optimizer = MLOptimizer(
        X, y,
        model_type='random_forest',
        ga_config=ga_config,
        cv_folds=3,
        logger=logger
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    optimizer.save_results(Path('outputs/ga_test/optimization_results.json'))
    
    logger.blank()
    print_success("✓ ML Optimizer test complete!")
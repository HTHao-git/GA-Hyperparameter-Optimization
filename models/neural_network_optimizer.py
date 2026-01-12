# ============================================================================
# NEURAL NETWORK HYPERPARAMETER OPTIMIZER
# ============================================================================
# GA-based optimization for Neural Networks (MLP)
#
# FEATURES:
#   - Architecture search (layers, neurons)
#   - Learning rate, optimizer, regularization
#   - Integration with GA framework
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import time
from typing import Dict, Any, Optional, Tuple, List
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

# Neural Network
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print_warning("scikit-learn not available for neural networks")


# ============================================================================
# NEURAL NETWORK HYPERPARAMETER TEMPLATE
# ============================================================================

NEURAL_NETWORK_HYPERPARAMETER_TEMPLATE = {
    # Architecture (encoded as layer sizes)
    'hidden_layer_1': [16, 32, 64, 128, 256],
    'hidden_layer_2': [0, 16, 32, 64, 128],  # 0 = no second layer
    'hidden_layer_3': [0, 16, 32, 64],       # 0 = no third layer
    
    # Learning parameters
    'learning_rate_init': (0.0001, 0.01),
    'alpha': (0.0001, 0.01),  # L2 regularization
    'batch_size': [32, 64, 128, 256, 'auto'],
    
    # Optimization
    'solver': ['adam', 'sgd'],
    'activation': ['relu', 'tanh', 'logistic'],
    'max_iter': [200, 300, 500],
    
    # Early stopping
    'early_stopping':  [True, False],
    'validation_fraction': [0.1, 0.15, 0.2],
    
    # Class balancing
    'class_weight':  ['balanced', None],
    
    # Preprocessing
    'pca_variance': [0.90, 0.95, 0.99],
    'smote_strategy': ['none', 'smote', 'adasyn'],
    'scaler': ['standard', 'minmax']  # Neural nets prefer scaling
}


# ============================================================================
# NEURAL NETWORK OPTIMIZER CLASS
# ============================================================================

class NeuralNetworkOptimizer:
    """
    Neural Network (MLP) hyperparameter optimizer using Genetic Algorithm. 
    
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
        
        if not SKLEARN_AVAILABLE: 
            raise ImportError("scikit-learn is required for neural networks")
        
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
        self.chromosome_template = NEURAL_NETWORK_HYPERPARAMETER_TEMPLATE
        
        # Results
        self.best_config = None
        self.best_score = -np.inf
        self.optimization_history = []
        
        if self.logger:
            self.logger.info(f"NeuralNetworkOptimizer initialized")
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
            print_header("NEURAL NETWORK HYPERPARAMETER OPTIMIZATION")
            print()
        
        # Create GA instance
        ga = GeneticAlgorithm(
            config=self.ga_config,
            fitness_function=self._evaluate_configuration,
            chromosome_template=self.chromosome_template,
            logger=self.logger
        )
        
        # Track optimization time
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
            
            # Show architecture
            architecture = self._decode_architecture(self.best_config)
            print_info(f"  Architecture: {architecture}")

        self.optimization_time = optimization_time
        self.evaluation_time = final_metrics. get('evaluation_time', 0.0)
        self.total_time = optimization_time + self.evaluation_time
        
        return {
            'config': self.best_config,
            'architecture': self._decode_architecture(self.best_config),
            'cv_score': float(self.best_score),
            'test_accuracy': float(final_metrics['accuracy']),
            'test_f1_score': float(final_metrics['f1_score']),
            'optimization_time': float(optimization_time),
            'evaluation_time': float(final_metrics['evaluation_time']),
            'total_time': float(optimization_time + final_metrics['evaluation_time']),
            'history': [
                {
                    'generation':   int(stats.generation),
                    'best_fitness': float(stats.best_fitness),
                    'mean_fitness': float(stats.mean_fitness),
                    'time':  float(stats.time) if hasattr(stats, 'time') else 0.0
                }
                for stats in self. optimization_history
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
                
                # Suppress convergence warnings
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings('ignore', category=Warning)
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
	
	#Add time tracker
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
        
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=Warning)
            model.fit(X_train_processed, y_train_processed)
        
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
        
        # 2. Scale features (IMPORTANT for neural networks!)
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
    
    def _decode_architecture(self, config: Dict[str, Any]) -> List[int]:
        """
        Decode architecture from config.
        
        Args:
            config: Configuration with layer sizes
            
        Returns: 
            List of layer sizes (excluding 0s)
        """
        layers = []
        for i in [1, 2, 3]: 
            layer_size = config.get(f'hidden_layer_{i}', 0)
            if layer_size > 0:
                layers.append(layer_size)
        
        return layers if layers else [100]  # Default if all 0
    
    def _create_model(self, config: Dict[str, Any]):
        """
        Create Neural Network model from configuration.
        
        Args:
            config: Model hyperparameters
            
        Returns:
            MLPClassifier instance
        """
        # Decode architecture
        hidden_layer_sizes = tuple(self._decode_architecture(config))
        
        # Extract parameters
        model_params = {
            'hidden_layer_sizes': hidden_layer_sizes,
            'learning_rate_init': config.get('learning_rate_init', 0.001),
            'alpha': config.get('alpha', 0.0001),
            'batch_size': config.get('batch_size', 'auto'),
            'solver': config.get('solver', 'adam'),
            'activation':  config.get('activation', 'relu'),
            'max_iter': config.get('max_iter', 200),
            'early_stopping': config.get('early_stopping', True),
            'validation_fraction': config.get('validation_fraction', 0.1),
        }
        
        # Class weight
        class_weight = config.get('class_weight', None)
        if class_weight is not None:
            # MLPClassifier doesn't directly support class_weight
            # We'll handle this through SMOTE instead
            pass
        
        # Add fixed parameters
        model_params['random_state'] = self.random_state
        model_params['verbose'] = False
        
        # Create model
        return MLPClassifier(**model_params)
    
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
    
    logger = get_logger(name="NEURAL_NET_TEST", verbose=True)
    
    print_header("NEURAL NETWORK OPTIMIZER TEST")
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
    
    optimizer = NeuralNetworkOptimizer(
        X, y,
        ga_config=ga_config,
        logger=logger
    )
    
    # Run optimization
    results = optimizer.optimize()
    
    # Save results
    optimizer.save_results(Path('outputs/neural_net_test/optimization_results.json'))
    
    logger.blank()
    print_success("✓ Neural Network Optimizer test complete!")
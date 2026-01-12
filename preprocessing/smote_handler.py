# ============================================================================
# SMOTE HANDLER - Synthetic Minority Over-sampling Technique
# ============================================================================
# Handle imbalanced datasets using various sampling strategies
#
# FEATURES:
#   - SMOTE (Synthetic Minority Over-sampling)
#   - ADASYN (Adaptive Synthetic Sampling)
#   - Random over/under sampling
#   - Combined strategies
#   - Configurable sampling ratios
#
# USAGE:
#   from preprocessing.smote_handler import SMOTEHandler
#   
#   handler = SMOTEHandler(strategy='smote')
#   X_balanced, y_balanced = handler.fit_resample(X, y)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Optional, Dict, Any, Tuple
from collections import Counter

from utils.logger import Logger
from utils.colors import print_section, print_info, print_warning, print_success


# ============================================================================
# SMOTE HANDLER CLASS
# ============================================================================

class SMOTEHandler: 
    """
    Handle imbalanced datasets using various sampling strategies.
    
    Args:
        strategy:  Sampling strategy ('smote', 'adasyn', 'random_over', 'random_under', 'combined')
        sampling_strategy: Desired ratio or dict of class ratios (default: 'auto' = balance all classes)
        k_neighbors: Number of nearest neighbors for SMOTE/ADASYN
        random_state: Random seed for reproducibility
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 strategy: str = 'smote',
                 sampling_strategy: str = 'auto',
                 k_neighbors: int = 5,
                 random_state:  int = 42,
                 logger: Optional[Logger] = None):
        
        self.strategy = strategy
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.logger = logger
        
        # Valid strategies
        self.valid_strategies = ['smote', 'adasyn', 'random_over', 'random_under', 'combined', 'none']
        
        if strategy not in self.valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'.Valid:  {self.valid_strategies}")
        
        # Set random seed
        np.random.seed(random_state)
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def analyze_imbalance(self, y: np.ndarray) -> Dict[str, Any]:
        """
        Analyze class imbalance. 
        
        Args:
            y: Labels
            
        Returns:
            Dictionary with imbalance statistics
        """
        class_counts = Counter(y)
        total = len(y)
        
        classes = sorted(class_counts.keys())
        counts = [class_counts[c] for c in classes]
        percentages = [(count / total) * 100 for count in counts]
        
        # Find majority and minority
        majority_class = max(class_counts, key=class_counts.get)
        minority_class = min(class_counts, key=class_counts.get)
        
        imbalance_ratio = class_counts[majority_class] / class_counts[minority_class]
        
        stats = {
            'total_samples': int(total),
            'num_classes': int(len(classes)),
            'class_counts': {int(c): int(count) for c, count in class_counts.items()},
            'class_percentages': {int(c): float(pct) for c, pct in zip(classes, percentages)},
            'majority_class': int(majority_class),
            'minority_class': int(minority_class),
            'imbalance_ratio': float(imbalance_ratio)
        }
        
        return stats
    
    def report_imbalance(self, y: np.ndarray):
        """
        Print class imbalance report.
        
        Args:
            y: Labels
        """
        stats = self.analyze_imbalance(y)
        
        print_section("CLASS IMBALANCE REPORT")
        print()
        
        print_info(f"Total samples: {stats['total_samples']}")
        print_info(f"Number of classes: {stats['num_classes']}")
        print()
        
        print_info("Class distribution:")
        for cls in sorted(stats['class_counts'].keys()):
            count = stats['class_counts'][cls]
            pct = stats['class_percentages'][cls]
            print(f"  Class {cls}: {count: 5} samples ({pct: 5.1f}%)")
        
        print()
        
        if stats['imbalance_ratio'] > 1.5:
            print_warning(f"Dataset is imbalanced!")
            print_info(f"  Majority class: {stats['majority_class']} ({stats['class_counts'][stats['majority_class']]} samples)")
            print_info(f"  Minority class: {stats['minority_class']} ({stats['class_counts'][stats['minority_class']]} samples)")
            print_info(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}: 1")
        else:
            print_success("Dataset is relatively balanced")
    
    # ========================================================================
    # SAMPLING STRATEGIES
    # ========================================================================
    
    def fit_resample(self, X:  np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Resample dataset to handle imbalance.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            (X_resampled, y_resampled) tuple
        """
        if self.strategy == 'none': 
            return X, y
        
        if self.logger:
            self.logger.info(f"Applying {self.strategy} resampling...")
        
        # Analyze imbalance
        stats_before = self.analyze_imbalance(y)
        
        if self.logger:
            self.logger.info(f"  Before: {stats_before['class_counts']}")
        
        # Apply strategy
        if self.strategy == 'smote':
            X_resampled, y_resampled = self._smote(X, y)
        
        elif self.strategy == 'adasyn':
            X_resampled, y_resampled = self._adasyn(X, y)
        
        elif self.strategy == 'random_over':
            X_resampled, y_resampled = self._random_oversample(X, y)
        
        elif self.strategy == 'random_under':
            X_resampled, y_resampled = self._random_undersample(X, y)
        
        elif self.strategy == 'combined':
            # SMOTE + undersampling
            X_temp, y_temp = self._smote(X, y)
            X_resampled, y_resampled = self._random_undersample(X_temp, y_temp)
        
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
        
        # Report results
        stats_after = self.analyze_imbalance(y_resampled)
        
        if self.logger:
            self.logger.info(f"  After:   {stats_after['class_counts']}")
            self.logger.success(f"Resampling complete:  {len(y)} → {len(y_resampled)} samples")
        
        return X_resampled, y_resampled
    
    # ========================================================================
    # SMOTE IMPLEMENTATION
    # ========================================================================
    
    def _smote(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        """
        SMOTE:  Synthetic Minority Over-sampling Technique.
        
        Creates synthetic samples by interpolating between existing minority samples.
        """
        # Get class counts
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        # For each minority class
        for minority_class in minority_classes:
            # Get minority samples
            minority_mask = y == minority_class
            X_minority = X[minority_mask]
            
            # Calculate how many synthetic samples to generate
            if self.sampling_strategy == 'auto':
                # Balance to majority class
                n_samples_needed = class_counts[majority_class] - class_counts[minority_class]
            else:
                # Custom ratio
                n_samples_needed = int(class_counts[minority_class] * self.sampling_strategy)
            
            if n_samples_needed <= 0:
                continue
            
            # Generate synthetic samples
            synthetic_samples = self._generate_synthetic_samples(
                X_minority, 
                n_samples_needed, 
                self.k_neighbors
            )
            
            # Add to dataset
            X_resampled = np.vstack([X_resampled, synthetic_samples])
            y_resampled = np.hstack([y_resampled, np.full(len(synthetic_samples), minority_class)])
        
        return X_resampled, y_resampled
    
    def _generate_synthetic_samples(self, 
                                    X: np.ndarray, 
                                    n_samples:  int, 
                                    k_neighbors: int) -> np.ndarray:
        """
        Generate synthetic samples using SMOTE algorithm.
        
        Args:
            X:  Minority class samples
            n_samples: Number of synthetic samples to generate
            k_neighbors: Number of nearest neighbors
            
        Returns: 
            Synthetic samples
        """
        synthetic = []
        n_minority = len(X)
        
        # Adjust k_neighbors if there aren't enough samples
        k = min(k_neighbors, n_minority - 1)
        
        if k <= 0:
            # Not enough samples for SMOTE, just duplicate
            return X[np.random.choice(n_minority, n_samples, replace=True)]
        
        for _ in range(n_samples):
            # Randomly select a sample
            idx = np.random.randint(0, n_minority)
            sample = X[idx]
            
            # Find k nearest neighbors
            distances = np.sum((X - sample) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]  # Exclude the sample itself
            
            # Randomly select one neighbor
            neighbor_idx = np.random.choice(nearest_indices)
            neighbor = X[neighbor_idx]
            
            # Generate synthetic sample (interpolate)
            gap = np.random.random()
            synthetic_sample = sample + gap * (neighbor - sample)
            
            synthetic.append(synthetic_sample)
        
        return np.array(synthetic)
    
    # ========================================================================
    # ADASYN IMPLEMENTATION
    # ========================================================================
    
    def _adasyn(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        ADASYN: Adaptive Synthetic Sampling.
        
        Generates more synthetic samples for harder-to-learn minority examples.
        """
        # Get class counts
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for minority_class in minority_classes: 
            minority_mask = y == minority_class
            X_minority = X[minority_mask]
            
            # Calculate needed samples
            if self.sampling_strategy == 'auto':
                n_samples_needed = class_counts[majority_class] - class_counts[minority_class]
            else:
                n_samples_needed = int(class_counts[minority_class] * self.sampling_strategy)
            
            if n_samples_needed <= 0:
                continue
            
            # Calculate density distribution (ratio of majority neighbors)
            density = self._calculate_density(X_minority, X, y, majority_class)
            
            # Normalize to get probability distribution
            if density.sum() == 0:
                # All samples have same density, use uniform
                probabilities = np.ones(len(X_minority)) / len(X_minority)
            else:
                probabilities = density / density.sum()
            
            # Generate samples based on density
            synthetic_samples = []
            
            for _ in range(n_samples_needed):
                # Select sample based on probability (harder examples more likely)
                idx = np.random.choice(len(X_minority), p=probabilities)
                sample = X_minority[idx]
                
                # Find k nearest neighbors in minority class
                distances = np.sum((X_minority - sample) ** 2, axis=1)
                k = min(self.k_neighbors, len(X_minority) - 1)
                nearest_indices = np.argsort(distances)[1:k+1]
                
                # Generate synthetic sample
                neighbor_idx = np.random.choice(nearest_indices)
                neighbor = X_minority[neighbor_idx]
                
                gap = np.random.random()
                synthetic_sample = sample + gap * (neighbor - sample)
                
                synthetic_samples.append(synthetic_sample)
            
            # Add to dataset
            X_resampled = np.vstack([X_resampled, np.array(synthetic_samples)])
            y_resampled = np.hstack([y_resampled, np.full(len(synthetic_samples), minority_class)])
        
        return X_resampled, y_resampled
    
    def _calculate_density(self, 
                          X_minority: np.ndarray, 
                          X_all: np.ndarray, 
                          y_all: np.ndarray,
                          majority_class: int) -> np.ndarray:
        """
        Calculate density (ratio of majority neighbors) for each minority sample.
        
        Higher density = more majority neighbors = harder to learn.
        """
        density = np.zeros(len(X_minority))
        
        k = min(self.k_neighbors, len(X_all) - 1)
        
        for i, sample in enumerate(X_minority):
            # Find k nearest neighbors in full dataset
            distances = np.sum((X_all - sample) ** 2, axis=1)
            nearest_indices = np.argsort(distances)[1:k+1]
            
            # Count how many are from majority class
            nearest_labels = y_all[nearest_indices]
            majority_count = np.sum(nearest_labels == majority_class)
            
            density[i] = majority_count / k
        
        return density
    
    # ========================================================================
    # RANDOM SAMPLING
    # ========================================================================
    
    def _random_oversample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Random oversampling:  duplicate minority samples.
        """
        class_counts = Counter(y)
        majority_class = max(class_counts, key=class_counts.get)
        minority_classes = [c for c in class_counts.keys() if c != majority_class]
        
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        for minority_class in minority_classes:
            minority_mask = y == minority_class
            X_minority = X[minority_mask]
            
            # Calculate needed samples
            if self.sampling_strategy == 'auto': 
                n_samples_needed = class_counts[majority_class] - class_counts[minority_class]
            else:
                n_samples_needed = int(class_counts[minority_class] * self.sampling_strategy)
            
            if n_samples_needed <= 0:
                continue
            
            # Randomly duplicate samples
            indices = np.random.choice(len(X_minority), n_samples_needed, replace=True)
            X_duplicated = X_minority[indices]
            
            # Add to dataset
            X_resampled = np.vstack([X_resampled, X_duplicated])
            y_resampled = np.hstack([y_resampled, np.full(len(X_duplicated), minority_class)])
        
        return X_resampled, y_resampled
    
    def _random_undersample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]: 
        """
        Random undersampling: remove majority samples.
        """
        class_counts = Counter(y)
        minority_class = min(class_counts, key=class_counts.get)
        minority_count = class_counts[minority_class]
        
        # Keep all samples from minority classes
        X_resampled = []
        y_resampled = []
        
        for cls in class_counts.keys():
            class_mask = y == cls
            X_class = X[class_mask]
            
            if cls == minority_class or self.sampling_strategy != 'auto':
                # Keep all minority samples
                X_resampled.append(X_class)
                y_resampled.append(np.full(len(X_class), cls))
            else:
                # Undersample majority to match minority
                n_samples = minority_count
                indices = np.random.choice(len(X_class), n_samples, replace=False)
                X_resampled.append(X_class[indices])
                y_resampled.append(np.full(n_samples, cls))
        
        X_resampled = np.vstack(X_resampled)
        y_resampled = np.hstack(y_resampled)
        
        return X_resampled, y_resampled


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def resample_dataset(X: np.ndarray,
                    y: np.ndarray,
                    strategy: str = 'smote',
                    logger: Optional[Logger] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convenience function to resample imbalanced dataset.
    
    Args:
        X: Features
        y: Labels
        strategy:  Sampling strategy
        logger: Logger instance
        
    Returns:
        (X_resampled, y_resampled) tuple
    """
    handler = SMOTEHandler(strategy=strategy, logger=logger)
    return handler.fit_resample(X, y)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header
    
    logger = get_logger(name="SMOTE_TEST", verbose=True)
    
    print_header("SMOTE HANDLER TEST")
    
    # Create imbalanced dataset
    np.random.seed(42)
    
    # Majority class: 1000 samples
    X_majority = np.random.randn(1000, 10)
    y_majority = np.zeros(1000)
    
    # Minority class: 100 samples
    X_minority = np.random.randn(100, 10) + 2  # Shifted distribution
    y_minority = np.ones(100)
    
    X = np.vstack([X_majority, X_minority])
    y = np.hstack([y_majority, y_minority])
    
    # Shuffle
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    # Create handler
    handler = SMOTEHandler(strategy='smote', logger=logger)
    
    # Report imbalance
    handler.report_imbalance(y)
    
    # Resample
    logger.blank()
    logger.section("Resampling with SMOTE")
    
    X_resampled, y_resampled = handler.fit_resample(X, y)
    
    # Report after
    logger.blank()
    handler.report_imbalance(y_resampled)
    
    logger.blank()
    print_success("✓ SMOTE handler working correctly!")
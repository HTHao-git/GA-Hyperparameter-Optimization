# ============================================================================
# FEATURE SCALING
# ============================================================================
# Scale features for machine learning models
#
# FEATURES:
#   - Multiple scaling strategies
#   - Fit/transform pattern
#   - Handle edge cases (zero variance, outliers)
#   - Compatible with scikit-learn API
#
# USAGE:
#   from preprocessing.scaling import StandardScaler
#   
#   scaler = StandardScaler()
#   X_scaled = scaler.fit_transform(X_train)
#   X_test_scaled = scaler.transform(X_test)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

from utils.logger import Logger


# ============================================================================
# BASE SCALER CLASS
# ============================================================================

class BaseScaler(ABC):
    """Base class for all scalers."""
    
    def __init__(self, logger: Optional[Logger] = None):
        self.logger = logger
        self.is_fitted_ = False
    
    @abstractmethod
    def fit(self, X: np.ndarray) -> 'BaseScaler':
        """Fit scaler to data."""
        pass
    
    @abstractmethod
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data using fitted scaler."""
        pass
    
    def fit_transform(self, X:  np.ndarray) -> np.ndarray:
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """Transform scaled data back to original scale."""
        raise NotImplementedError("Inverse transform not implemented for this scaler")


# ============================================================================
# STANDARD SCALER (Z-score normalization)
# ============================================================================

class StandardScaler(BaseScaler):
    """
    Standardize features by removing mean and scaling to unit variance.
    
    Formula: z = (x - μ) / σ
    
    Args:
        with_mean:  If True, center data before scaling
        with_std: If True, scale data to unit variance
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 with_mean: bool = True,
                 with_std: bool = True,
                 logger: Optional[Logger] = None):
        
        super().__init__(logger)
        self.with_mean = with_mean
        self.with_std = with_std
        
        self.mean_ = None
        self.std_ = None
    
    def fit(self, X: np.ndarray) -> 'StandardScaler':
        """
        Compute mean and std for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self
        """
        if self.logger:
            self.logger.info("Fitting StandardScaler...")
        
        if self.with_mean:
            self.mean_ = np.mean(X, axis=0)
        else:
            self.mean_ = np.zeros(X.shape[1])
        
        if self.with_std:
            self.std_ = np.std(X, axis=0)
            
            # Handle zero variance features
            self.std_[self.std_ == 0] = 1.0
            
            if self.logger:
                zero_var_count = np.sum(np.std(X, axis=0) == 0)
                if zero_var_count > 0:
                    self.logger.warning(f"  {zero_var_count} features have zero variance")
        else:
            self.std_ = np.ones(X.shape[1])
        
        self.is_fitted_ = True
        
        if self.logger:
            self.logger.success("StandardScaler fitted")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize data. 
        
        Args:
            X: Data to transform
            
        Returns:
            Scaled data
        """
        if not self.is_fitted_: 
            raise RuntimeError("Scaler must be fitted before transform")
        
        X_scaled = (X - self.mean_) / self.std_
        
        return X_scaled
    
    def inverse_transform(self, X:  np.ndarray) -> np.ndarray:
        """
        Transform scaled data back to original scale.
        
        Args:
            X:  Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        return X * self.std_ + self.mean_


# ============================================================================
# MIN-MAX SCALER
# ============================================================================

class MinMaxScaler(BaseScaler):
    """
    Scale features to a given range (default [0, 1]).
    
    Formula: x_scaled = (x - min) / (max - min) * (max_range - min_range) + min_range
    
    Args: 
        feature_range:  Desired range (min, max)
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 feature_range: tuple = (0, 1),
                 logger: Optional[Logger] = None):
        
        super().__init__(logger)
        self.feature_range = feature_range
        
        self.min_ = None
        self.max_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'MinMaxScaler':
        """
        Compute min and max for later scaling. 
        
        Args:
            X: Training data
            
        Returns:
            self
        """
        if self.logger:
            self.logger.info(f"Fitting MinMaxScaler (range:  {self.feature_range})...")
        
        self.min_ = np.min(X, axis=0)
        self.max_ = np.max(X, axis=0)
        
        # Compute scale
        data_range = self.max_ - self.min_
        
        # Handle constant features (min == max)
        data_range[data_range == 0] = 1.0
        
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / data_range
        
        if self.logger:
            constant_count = np.sum((self.max_ - self.min_) == 0)
            if constant_count > 0:
                self.logger.warning(f"  {constant_count} constant features detected")
        
        self.is_fitted_ = True
        
        if self.logger:
            self.logger.success("MinMaxScaler fitted")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data to feature range.
        
        Args:
            X: Data to transform
            
        Returns:
            Scaled data
        """
        if not self.is_fitted_: 
            raise RuntimeError("Scaler must be fitted before transform")
        
        feature_min, feature_max = self.feature_range
        
        X_scaled = (X - self.min_) * self.scale_ + feature_min
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform scaled data back to original scale.
        
        Args:
            X: Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted_: 
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        feature_min, _ = self.feature_range
        
        X_original = (X - feature_min) / self.scale_ + self.min_
        
        return X_original


# ============================================================================
# ROBUST SCALER
# ============================================================================

class RobustScaler(BaseScaler):
    """
    Scale features using statistics robust to outliers.
    
    Uses median and interquartile range instead of mean and std.
    
    Formula: x_scaled = (x - median) / IQR
    
    Args:
        with_centering: If True, center data using median
        with_scaling: If True, scale data using IQR
        quantile_range: Quantile range for IQR (default:  25-75 percentile)
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 with_centering: bool = True,
                 with_scaling: bool = True,
                 quantile_range: tuple = (25.0, 75.0),
                 logger: Optional[Logger] = None):
        
        super().__init__(logger)
        self.with_centering = with_centering
        self.with_scaling = with_scaling
        self.quantile_range = quantile_range
        
        self.center_ = None
        self.scale_ = None
    
    def fit(self, X: np.ndarray) -> 'RobustScaler':
        """
        Compute median and IQR for later scaling.
        
        Args:
            X: Training data
            
        Returns:
            self
        """
        if self.logger:
            self.logger.info("Fitting RobustScaler...")
        
        if self.with_centering:
            self.center_ = np.median(X, axis=0)
        else:
            self.center_ = np.zeros(X.shape[1])
        
        if self.with_scaling:
            q_min, q_max = self.quantile_range
            
            quantiles = np.percentile(X, [q_min, q_max], axis=0)
            self.scale_ = quantiles[1] - quantiles[0]
            
            # Handle zero IQR
            self.scale_[self.scale_ == 0] = 1.0
            
            if self.logger:
                zero_iqr_count = np.sum((quantiles[1] - quantiles[0]) == 0)
                if zero_iqr_count > 0:
                    self.logger.warning(f"  {zero_iqr_count} features have zero IQR")
        else:
            self.scale_ = np.ones(X.shape[1])
        
        self.is_fitted_ = True
        
        if self.logger:
            self.logger.success("RobustScaler fitted")
        
        return self
    
    def transform(self, X:  np.ndarray) -> np.ndarray:
        """
        Scale data using median and IQR.
        
        Args:
            X: Data to transform
            
        Returns: 
            Scaled data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform")
        
        X_scaled = (X - self.center_) / self.scale_
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform scaled data back to original scale.
        
        Args:
            X: Scaled data
            
        Returns: 
            Original scale data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        return X * self.scale_ + self.center_


# ============================================================================
# MAX ABS SCALER
# ============================================================================

class MaxAbsScaler(BaseScaler):
    """
    Scale features by their maximum absolute value.
    
    Scales to range [-1, 1]. Preserves sparsity (doesn't center).
    
    Formula: x_scaled = x / max(|x|)
    
    Args:
        logger: Logger instance (optional)
    """
    
    def __init__(self, logger: Optional[Logger] = None):
        super().__init__(logger)
        self.max_abs_ = None
    
    def fit(self, X: np.ndarray) -> 'MaxAbsScaler': 
        """
        Compute max absolute value for later scaling.
        
        Args:
            X:  Training data
            
        Returns: 
            self
        """
        if self.logger:
            self.logger.info("Fitting MaxAbsScaler...")
        
        self.max_abs_ = np.max(np.abs(X), axis=0)
        
        # Handle zero max abs
        self.max_abs_[self.max_abs_ == 0] = 1.0
        
        self.is_fitted_ = True
        
        if self.logger:
            self.logger.success("MaxAbsScaler fitted")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale data by max absolute value.
        
        Args:
            X: Data to transform
            
        Returns: 
            Scaled data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before transform")
        
        X_scaled = X / self.max_abs_
        
        return X_scaled
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform scaled data back to original scale.
        
        Args:
            X:  Scaled data
            
        Returns:
            Original scale data
        """
        if not self.is_fitted_:
            raise RuntimeError("Scaler must be fitted before inverse transform")
        
        return X * self.max_abs_


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def get_scaler(scaler_type: str = 'standard', **kwargs):
    """
    Get scaler by name. 
    
    Args:
        scaler_type: Type of scaler ('standard', 'minmax', 'robust', 'maxabs')
        **kwargs: Additional arguments for scaler
        
    Returns: 
        Scaler instance
    """
    scalers = {
        'standard':  StandardScaler,
        'minmax': MinMaxScaler,
        'robust': RobustScaler,
        'maxabs':  MaxAbsScaler
    }
    
    if scaler_type not in scalers:
        raise ValueError(f"Unknown scaler type: {scaler_type}.Valid:  {list(scalers.keys())}")
    
    return scalers[scaler_type](**kwargs)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success
    
    logger = get_logger(name="SCALING_TEST", verbose=True)
    
    print_header("FEATURE SCALING TEST")
    
    # Create sample data with different scales
    np.random.seed(42)
    X = np.random.randn(100, 3)
    X[:, 0] = X[:, 0] * 1000 + 5000  # Large scale, shifted
    X[:, 1] = X[:, 1] * 0.01  # Small scale
    X[:, 2] = X[:, 2] + 10  # Shifted
    
    print_section("Original Data Statistics")
    print(f"Feature 0: mean={X[:, 0].mean():.2f}, std={X[:, 0].std():.2f}, min={X[:, 0].min():.2f}, max={X[:, 0].max():.2f}")
    print(f"Feature 1: mean={X[:, 1].mean():.4f}, std={X[:, 1].std():.4f}, min={X[:, 1].min():.4f}, max={X[:, 1].max():.4f}")
    print(f"Feature 2: mean={X[:, 2].mean():.2f}, std={X[:, 2].std():.2f}, min={X[:, 2].min():.2f}, max={X[:, 2].max():.2f}")
    
    logger.blank()
    
    # Test each scaler
    scalers = [
        ('StandardScaler', StandardScaler(logger=logger)),
        ('MinMaxScaler', MinMaxScaler(logger=logger)),
        ('RobustScaler', RobustScaler(logger=logger)),
        ('MaxAbsScaler', MaxAbsScaler(logger=logger))
    ]
    
    for name, scaler in scalers:
        print_section(name)
        
        X_scaled = scaler.fit_transform(X)
        
        print(f"Feature 0: mean={X_scaled[:, 0].mean():.4f}, std={X_scaled[: , 0].std():.4f}, min={X_scaled[: , 0].min():.4f}, max={X_scaled[: , 0].max():.4f}")
        print(f"Feature 1: mean={X_scaled[:, 1].mean():.4f}, std={X_scaled[:, 1].std():.4f}, min={X_scaled[:, 1].min():.4f}, max={X_scaled[:, 1].max():.4f}")
        print(f"Feature 2: mean={X_scaled[:, 2].mean():.4f}, std={X_scaled[:, 2].std():.4f}, min={X_scaled[:, 2].min():.4f}, max={X_scaled[:, 2].max():.4f}")
        
        # Test inverse transform
        X_reconstructed = scaler.inverse_transform(X_scaled)
        reconstruction_error = np.max(np.abs(X - X_reconstructed))
        
        if reconstruction_error < 1e-10:
            print_success(f"✓ Inverse transform accurate (error: {reconstruction_error:.2e})")
        else:
            logger.warning(f"⚠ Inverse transform error: {reconstruction_error:.2e}")
        
        logger.blank()
    
    print_header("TEST COMPLETE")
    print_success("All scalers working correctly!")
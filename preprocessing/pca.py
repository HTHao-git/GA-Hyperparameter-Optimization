# ============================================================================
# PCA - PRINCIPAL COMPONENT ANALYSIS
# ============================================================================
# Dimensionality reduction using PCA
#
# FEATURES:
#   - Automatic component selection by variance threshold
#   - Variance explained analysis
#   - Scree plot data generation
#   - Inverse transform support
#   - Handle zero-variance features
#
# USAGE:
#   from preprocessing.pca import PCA
#   
#   pca = PCA(n_components=0.95)  # Keep 95% variance
#   X_reduced = pca.fit_transform(X)
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Optional, Union, Dict, Any, Tuple

from utils.logger import Logger
from utils.colors import print_section, print_info, print_warning, print_success


# ============================================================================
# PCA CLASS
# ============================================================================

class PCA:
    """
    Principal Component Analysis for dimensionality reduction. 
    
    Args:
        n_components:  Number of components to keep
            - int: Keep exactly n components
            - float (0-1): Keep enough components to explain this much variance
            - None: Keep all components
        whiten: If True, multiply components by sqrt(n_samples) for unit variance
        random_state: Random seed for reproducibility
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 n_components: Optional[Union[int, float]] = None,
                 whiten:  bool = False,
                 random_state:  int = 42,
                 logger: Optional[Logger] = None):
        
        self.n_components = n_components
        self.whiten = whiten
        self.random_state = random_state
        self.logger = logger
        
        # Fitted attributes
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None
        self.mean_ = None
        self.n_components_ = None
        self.n_features_ = None
        self.n_samples_ = None
        
        self.is_fitted_ = False
        
        np.random.seed(random_state)
    
    # ========================================================================
    # FITTING
    # ========================================================================
    
    def fit(self, X: np.ndarray) -> 'PCA':
        """
        Fit PCA on the data.
        
        Args:
            X: Training data (n_samples, n_features)
            
        Returns:
            self
        """
        if self.logger:
            self.logger.info(f"Fitting PCA on data:  {X.shape}")
        
        self.n_samples_, self.n_features_ = X.shape
        
        # Center the data
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Compute covariance matrix
        # Using SVD instead of eigenvalue decomposition (more stable)
        # X = U * S * V^T
        # Covariance = (X^T * X) / (n-1) = V * S^2 * V^T / (n-1)
        
        if self.logger:
            self.logger.debug("Computing SVD...")
        
        # SVD
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # V^T rows are the principal components
        components = Vt
        
        # Explained variance
        explained_variance = (S ** 2) / (self.n_samples_ - 1)
        
        # Total variance
        total_variance = explained_variance.sum()
        
        # Explained variance ratio
        explained_variance_ratio = explained_variance / total_variance
        
        # Determine number of components to keep
        if self.n_components is None:
            # Keep all components
            n_components_to_keep = self.n_features_
        
        elif isinstance(self.n_components, int):
            # Keep exact number
            n_components_to_keep = min(self.n_components, self.n_features_)
        
        elif isinstance(self.n_components, float):
            # Keep enough to explain this much variance
            cumsum = np.cumsum(explained_variance_ratio)
            n_components_to_keep = np.searchsorted(cumsum, self.n_components) + 1
            n_components_to_keep = min(n_components_to_keep, self.n_features_)
            
            if self.logger:
                variance_explained = cumsum[n_components_to_keep - 1] * 100
                self.logger.info(f"  Keeping {n_components_to_keep} components to explain {variance_explained:.2f}% variance")
        
        else: 
            raise ValueError(f"Invalid n_components:  {self.n_components}")
        
        # Store results
        self.n_components_ = n_components_to_keep
        self.components_ = components[: n_components_to_keep]
        self.explained_variance_ = explained_variance[:n_components_to_keep]
        self.explained_variance_ratio_ = explained_variance_ratio[:n_components_to_keep]
        self.singular_values_ = S[:n_components_to_keep]
        
        self.is_fitted_ = True
        
        if self.logger:
            total_var_explained = self.explained_variance_ratio_.sum() * 100
            self.logger.success(f"PCA fitted:  {self.n_features_} → {self.n_components_} features ({total_var_explained:.2f}% variance)")
        
        return self
    
    # ========================================================================
    # TRANSFORM
    # ========================================================================
    
    def transform(self, X:  np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.
        
        Args:
            X: Data to transform (n_samples, n_features)
            
        Returns:
            Transformed data (n_samples, n_components)
        """
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted before transform")
        
        # Center the data
        X_centered = X - self.mean_
        
        # Project onto principal components
        X_transformed = np.dot(X_centered, self.components_.T)
        
        # Whiten if requested
        if self.whiten:
            X_transformed /= np.sqrt(self.explained_variance_)
        
        return X_transformed
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit PCA and transform data in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns:
            Transformed data
        """
        return self.fit(X).transform(X)
    
    # ========================================================================
    # INVERSE TRANSFORM
    # ========================================================================
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to original space.
        
        Args:
            X:  Transformed data (n_samples, n_components)
            
        Returns:
            Data in original space (n_samples, n_features)
        """
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted before inverse transform")
        
        # Reverse whitening if applied
        if self.whiten:
            X = X * np.sqrt(self.explained_variance_)
        
        # Project back to original space
        X_original = np.dot(X, self.components_)
        
        # Add back the mean
        X_original += self.mean_
        
        return X_original
    
    # ========================================================================
    # ANALYSIS
    # ========================================================================
    
    def get_variance_explained(self) -> np.ndarray:
        """
        Get cumulative variance explained by components.
        
        Returns:
            Cumulative variance ratio
        """
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted first")
        
        return np.cumsum(self.explained_variance_ratio_)
    
    def get_scree_data(self) -> Dict[str, np.ndarray]:
        """
        Get data for scree plot. 
        
        Returns:
            Dictionary with component numbers, variance ratios, and cumulative variance
        """
        if not self.is_fitted_: 
            raise RuntimeError("PCA must be fitted first")
        
        return {
            'components': np.arange(1, len(self.explained_variance_ratio_) + 1),
            'variance_ratio': self.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(self.explained_variance_ratio_)
        }
    
    def report_variance(self):
        """Print variance explained report."""
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted first")
        
        print_section("PCA VARIANCE REPORT")
        print()
        
        print_info(f"Original features: {self.n_features_}")
        print_info(f"Components kept: {self.n_components_}")
        print_info(f"Reduction: {(1 - self.n_components_ / self.n_features_) * 100:.1f}%")
        print()
        
        cumsum = np.cumsum(self.explained_variance_ratio_)
        
        print_info("Variance explained by top components:")
        
        # Show first 10 components
        n_to_show = min(10, self.n_components_)
        
        for i in range(n_to_show):
            var_ratio = self.explained_variance_ratio_[i] * 100
            cumvar = cumsum[i] * 100
            print(f"  PC{i+1:2}:  {var_ratio: 6.2f}% (cumulative: {cumvar:6.2f}%)")
        
        if self.n_components_ > n_to_show:
            print(f"  ... ({self.n_components_ - n_to_show} more components)")
        
        print()
        
        total_var = cumsum[-1] * 100
        print_success(f"Total variance explained: {total_var:.2f}%")
    
    def get_component_importance(self, feature_names: Optional[list] = None, 
                                 n_components: int = 3,
                                 n_features:  int = 5) -> Dict[int, list]:
        """
        Get most important features for each principal component.
        
        Args:
            feature_names: Optional feature names
            n_components: Number of components to analyze
            n_features: Number of top features per component
            
        Returns: 
            Dictionary mapping component index to list of (feature, weight) tuples
        """
        if not self.is_fitted_:
            raise RuntimeError("PCA must be fitted first")
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(self.n_features_)]
        
        importance = {}
        
        n_comp = min(n_components, self.n_components_)
        
        for i in range(n_comp):
            # Get absolute weights for this component
            weights = np.abs(self.components_[i])
            
            # Get top features
            top_indices = np.argsort(-weights)[:n_features]
            
            top_features = [
                (feature_names[idx], float(self.components_[i, idx]))
                for idx in top_indices
            ]
            
            importance[i] = top_features
        
        return importance
    
    def report_component_importance(self, feature_names: Optional[list] = None,
                                   n_components: int = 3,
                                   n_features: int = 5):
        """Print component importance report."""
        
        importance = self.get_component_importance(feature_names, n_components, n_features)
        
        print_section("PRINCIPAL COMPONENT IMPORTANCE")
        print()
        
        for comp_idx, features in importance.items():
            var_explained = self.explained_variance_ratio_[comp_idx] * 100
            print_info(f"PC{comp_idx + 1} (explains {var_explained:.2f}% variance):")
            
            for feat_name, weight in features: 
                sign = '+' if weight > 0 else '-'
                print(f"  {sign} {feat_name[:40]:40} ({abs(weight):.4f})")
            
            print()


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def apply_pca(X: np.ndarray,
             n_components: Union[int, float] = 0.95,
             logger: Optional[Logger] = None) -> Tuple[np.ndarray, PCA]:
    """
    Convenience function to apply PCA. 
    
    Args:
        X: Data to transform
        n_components: Number of components or variance threshold
        logger: Logger instance
        
    Returns:
        (X_transformed, pca_model) tuple
    """
    pca = PCA(n_components=n_components, logger=logger)
    X_transformed = pca.fit_transform(X)
    
    return X_transformed, pca


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header
    
    logger = get_logger(name="PCA_TEST", verbose=True)
    
    print_header("PCA TEST")
    
    # Create sample data with correlated features
    np.random.seed(42)
    n_samples = 200
    
    # Generate data with intrinsic dimensionality of 3
    latent = np.random.randn(n_samples, 3)
    
    # Create 50 features as linear combinations of latent factors
    mixing_matrix = np.random.randn(3, 50)
    X = np.dot(latent, mixing_matrix)
    
    # Add some noise
    X += np.random.randn(n_samples, 50) * 0.1
    
    logger.blank()
    logger.info(f"Original data shape: {X.shape}")
    
    # Apply PCA with 95% variance threshold
    logger.blank()
    print_section("Apply PCA (95% variance threshold)")
    
    pca = PCA(n_components=0.95, logger=logger)
    X_reduced = pca.fit_transform(X)
    
    logger.blank()
    logger.info(f"Reduced data shape: {X_reduced.shape}")
    
    # Show variance report
    logger.blank()
    pca.report_variance()
    
    # Test reconstruction
    logger.blank()
    print_section("Test Reconstruction")
    
    X_reconstructed = pca.inverse_transform(X_reduced)
    reconstruction_error = np.mean((X - X_reconstructed) ** 2)
    
    logger.info(f"Reconstruction MSE: {reconstruction_error:.6f}")
    
    if reconstruction_error < 0.1:
        print_success("✓ Good reconstruction quality")
    else:
        print_warning(f"⚠ High reconstruction error: {reconstruction_error:.6f}")
    
    logger.blank()
    print_header("TEST COMPLETE")
    print_success("PCA working correctly!")
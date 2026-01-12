# ============================================================================
# MISSING VALUES HANDLER
# ============================================================================
# Detect and handle missing values in datasets
#
# FEATURES:
#   - Detection of NaN, None, inf values
#   - Multiple imputation strategies
#   - Missing value analysis and reporting
#   - Column/row dropping with thresholds
#
# USAGE:
#   from preprocessing.missing_values import MissingValuesHandler
#   
#   handler = MissingValuesHandler()
#   X_clean = handler.fit_transform(X, strategy='mean')
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, Tuple, List
from pathlib import Path

from utils.logger import Logger
from utils.colors import print_section, print_info, print_warning, print_success


# ============================================================================
# MISSING VALUES HANDLER CLASS
# ============================================================================

class MissingValuesHandler: 
    """
    Handle missing values in datasets.
    
    Args:
        strategy: Imputation strategy ('mean', 'median', 'mode', 'knn', 'drop', 'ffill', 'bfill')
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 strategy: str = 'mean',
                 logger:  Optional[Logger] = None):
        
        self.strategy = strategy
        self.logger = logger
        
        # Store fitted values for transform
        self.fill_values_ = None
        self.columns_to_drop_ = []
        self.is_fitted_ = False
        
        # Valid strategies
        self.valid_strategies = ['mean', 'median', 'mode', 'knn', 'drop', 'ffill', 'bfill', 'constant']
        
        if strategy not in self.valid_strategies:
            raise ValueError(f"Invalid strategy '{strategy}'.Valid:  {self.valid_strategies}")
    
    # ========================================================================
    # DETECTION
    # ========================================================================
    
    def detect_missing(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Detect missing values and return statistics.
        
        Args:
            X: Data array
            
        Returns: 
            Dictionary with missing value statistics
        """
        # Count NaN values
        nan_count = np.isnan(X).sum()
        
        # Count infinite values
        inf_count = np.isinf(X).sum()
        
        # Total missing
        total_missing = nan_count + inf_count
        
        # Percentage
        total_values = X.size
        missing_percentage = (total_missing / total_values) * 100
        
        # Per-column statistics
        nan_per_col = np.isnan(X).sum(axis=0)
        inf_per_col = np.isinf(X).sum(axis=0)
        missing_per_col = nan_per_col + inf_per_col
        
        # Per-row statistics
        nan_per_row = np.isnan(X).sum(axis=1)
        inf_per_row = np.isinf(X).sum(axis=1)
        missing_per_row = nan_per_row + inf_per_row
        
        stats = {
            'total_values': int(total_values),
            'nan_count': int(nan_count),
            'inf_count': int(inf_count),
            'total_missing': int(total_missing),
            'missing_percentage': float(missing_percentage),
            'missing_per_column': missing_per_col,
            'missing_per_row': missing_per_row,
            'columns_with_missing': int(np.sum(missing_per_col > 0)),
            'rows_with_missing': int(np.sum(missing_per_row > 0))
        }
        
        return stats
    
    def report_missing(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        Print detailed missing value report.
        
        Args:
            X: Data array
            feature_names: Optional feature names
        """
        stats = self.detect_missing(X)
        
        print_section("MISSING VALUES REPORT")
        print()
        
        print_info(f"Dataset shape: {X.shape}")
        print_info(f"Total values:   {stats['total_values']}")
        print()
        
        if stats['total_missing'] == 0:
            print_success("✓ No missing values detected!")
            return
        
        print_warning(f"Missing values:  {stats['total_missing']} ({stats['missing_percentage']:.2f}%)")
        print_info(f"  NaN values:   {stats['nan_count']}")
        print_info(f"  Inf values:  {stats['inf_count']}")
        print()
        
        print_info(f"Columns with missing values:  {stats['columns_with_missing']}/{X.shape[1]}")
        print_info(f"Rows with missing values:     {stats['rows_with_missing']}/{X.shape[0]}")
        print()
        
        # Show top columns with most missing values
        missing_per_col = stats['missing_per_column']
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(len(missing_per_col))]
        
        # Get indices of columns with missing values
        cols_with_missing = np.where(missing_per_col > 0)[0]
        
        if len(cols_with_missing) > 0:
            print_info("Top 10 columns with missing values:")
            
            # Sort by missing count (descending)
            sorted_indices = cols_with_missing[np.argsort(-missing_per_col[cols_with_missing])]
            
            for i, col_idx in enumerate(sorted_indices[: 10]):
                count = missing_per_col[col_idx]
                percentage = (count / X.shape[0]) * 100
                name = feature_names[col_idx] if col_idx < len(feature_names) else f"Feature_{col_idx}"
                print(f"  {i+1}.{name:30} - {count:5} ({percentage: 5.1f}%)")
    
    # ========================================================================
    # IMPUTATION
    # ========================================================================
    
    def fit(self, X: np.ndarray) -> 'MissingValuesHandler': 
        """
        Fit the imputation strategy on the data.
        
        Args:
            X: Training data
            
        Returns: 
            self
        """
        if self.logger:
            self.logger.info(f"Fitting missing values handler (strategy: {self.strategy})")
        
        # Replace inf with nan for consistent handling
        X_clean = X.copy()
        X_clean[np.isinf(X_clean)] = np.nan
        
        if self.strategy == 'mean': 
            # Calculate mean for each column (ignoring NaN)
            self.fill_values_ = np.nanmean(X_clean, axis=0)
        
        elif self.strategy == 'median':
            # Calculate median for each column (ignoring NaN)
            self.fill_values_ = np.nanmedian(X_clean, axis=0)
        
        elif self.strategy == 'mode':
            # Calculate mode for each column
            self.fill_values_ = []
            for col_idx in range(X_clean.shape[1]):
                col = X_clean[:, col_idx]
                col_no_nan = col[~np.isnan(col)]
                
                if len(col_no_nan) > 0:
                    # Get most common value
                    values, counts = np.unique(col_no_nan, return_counts=True)
                    mode_value = values[np.argmax(counts)]
                    self.fill_values_.append(mode_value)
                else:
                    self.fill_values_.append(0.0)
            
            self.fill_values_ = np.array(self.fill_values_)
        
        elif self.strategy == 'knn':
            # KNN imputation requires sklearn
            try:
                from sklearn.impute import KNNImputer
                
                self.imputer_ = KNNImputer(n_neighbors=5)
                self.imputer_.fit(X_clean)
                
            except ImportError:
                if self.logger:
                    self.logger.warning("sklearn not available, falling back to mean imputation")
                self.strategy = 'mean'
                self.fill_values_ = np.nanmean(X_clean, axis=0)
        
        elif self.strategy == 'drop':
            # Identify columns with >50% missing values to drop
            missing_per_col = np.isnan(X_clean).sum(axis=0)
            threshold = X_clean.shape[0] * 0.5
            self.columns_to_drop_ = np.where(missing_per_col > threshold)[0]
            
            if self.logger and len(self.columns_to_drop_) > 0:
                self.logger.info(f"  Will drop {len(self.columns_to_drop_)} columns with >50% missing")
        
        elif self.strategy in ['ffill', 'bfill', 'constant']:
            # No fitting needed for these strategies
            pass
        
        self.is_fitted_ = True
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data by imputing missing values.
        
        Args:
            X: Data to transform
            
        Returns: 
            Transformed data
        """
        if not self.is_fitted_:
            raise RuntimeError("Handler must be fitted before transform. Call fit() first.")
        
        X_clean = X.copy()
        
        # Replace inf with nan
        X_clean[np.isinf(X_clean)] = np.nan
        
        if self.strategy in ['mean', 'median', 'mode']: 
            # Replace NaN with fill values
            for col_idx in range(X_clean.shape[1]):
                mask = np.isnan(X_clean[:, col_idx])
                X_clean[mask, col_idx] = self.fill_values_[col_idx]
        
        elif self.strategy == 'knn':
            # Use fitted KNN imputer
            X_clean = self.imputer_.transform(X_clean)
        
        elif self.strategy == 'drop':
            # Drop columns with too many missing values
            if len(self.columns_to_drop_) > 0:
                X_clean = np.delete(X_clean, self.columns_to_drop_, axis=1)
            
            # Drop rows with any remaining missing values
            mask = ~np.isnan(X_clean).any(axis=1)
            X_clean = X_clean[mask]
        
        elif self.strategy == 'ffill':
            # Forward fill (use pandas for convenience)
            df = pd.DataFrame(X_clean)
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)  # Backfill any remaining
            X_clean = df.values
        
        elif self.strategy == 'bfill': 
            # Backward fill
            df = pd.DataFrame(X_clean)
            df.fillna(method='bfill', inplace=True)
            df.fillna(method='ffill', inplace=True)  # Forward fill any remaining
            X_clean = df.values
        
        elif self.strategy == 'constant': 
            # Fill with constant (0)
            X_clean[np.isnan(X_clean)] = 0.0
        
        if self.logger:
            remaining = np.isnan(X_clean).sum()
            if remaining > 0:
                self.logger.warning(f"  {remaining} missing values remain after imputation")
            else:
                self.logger.success(f"  All missing values imputed")
        
        return X_clean
    
    def fit_transform(self, X:  np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            X: Data to fit and transform
            
        Returns: 
            Transformed data
        """
        return self.fit(X).transform(X)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def handle_missing_values(X: np.ndarray,
                         strategy: str = 'mean',
                         logger: Optional[Logger] = None) -> np.ndarray:
    """
    Convenience function to handle missing values.
    
    Args:
        X: Data array
        strategy:  Imputation strategy
        logger: Logger instance
        
    Returns:
        Data with missing values handled
    """
    handler = MissingValuesHandler(strategy=strategy, logger=logger)
    return handler.fit_transform(X)


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header
    
    logger = get_logger(name="MISSING_VALUES_TEST", verbose=True)
    
    print_header("MISSING VALUES HANDLER TEST")
    
    # Create sample data with missing values
    np.random.seed(42)
    X = np.random.randn(100, 10)
    
    # Introduce missing values (20%)
    mask = np.random.random(X.shape) < 0.2
    X[mask] = np.nan
    
    # Create handler
    handler = MissingValuesHandler(strategy='mean', logger=logger)
    
    # Report missing values
    handler.report_missing(X)
    
    # Handle missing values
    logger.blank()
    logger.section("Imputing Missing Values")
    
    X_clean = handler.fit_transform(X)
    
    # Verify
    logger.blank()
    stats_before = handler.detect_missing(X)
    stats_after = handler.detect_missing(X_clean)
    
    print_info("Before imputation:")
    print(f"  Missing values: {stats_before['total_missing']}")
    print()
    print_info("After imputation:")
    print(f"  Missing values: {stats_after['total_missing']}")
    
    if stats_after['total_missing'] == 0:
        logger.blank()
        print_success("✓ All missing values handled successfully!")
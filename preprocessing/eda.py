# ============================================================================
# EXPLORATORY DATA ANALYSIS (EDA)
# ============================================================================
# Comprehensive data analysis and visualization tools
#
# FEATURES:
#   - Statistical summaries
#   - Distribution analysis
#   - Correlation analysis
#   - Outlier detection
#   - Class balance visualization
#   - Missing value patterns
#   - PCA variance plots
#
# USAGE:
#   from preprocessing.eda import EDA
#   
#   eda = EDA(X, y)
#   eda.generate_report()
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
import pandas as pd
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import warnings

from utils.logger import Logger
from utils.colors import print_section, print_info, print_warning, print_success

# Optional visualization imports
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    warnings.warn("matplotlib/seaborn not available. Visualizations will be skipped.")


# ============================================================================
# EDA CLASS
# ============================================================================

class EDA:
    """
    Exploratory Data Analysis toolkit. 
    
    Args:
        X: Feature matrix
        y: Labels (optional)
        feature_names: Feature names (optional)
        class_names: Class names (optional)
        logger: Logger instance (optional)
    """
    
    def __init__(self,
                 X: np.ndarray,
                 y: Optional[np.ndarray] = None,
                 feature_names: Optional[List[str]] = None,
                 class_names: Optional[List[str]] = None,
                 logger:  Optional[Logger] = None):
        
        self.X = X
        self.y = y
        self.logger = logger
        
        # Generate feature names if not provided
        if feature_names is None:
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        # Generate class names if not provided
        if class_names is None and y is not None:
            unique_classes = np.unique(y)
            self.class_names = [f"Class_{c}" for c in unique_classes]
        else:
            self.class_names = class_names
        
        # Statistics cache
        self._stats = None
        self._correlation = None
    
    # ========================================================================
    # STATISTICAL SUMMARY
    # ========================================================================
    
    def get_statistics(self) -> pd.DataFrame:
        """
        Get statistical summary of features.
        
        Returns:
            DataFrame with statistics
        """
        if self._stats is not None:
            return self._stats
        
        stats = {
            'mean': np.mean(self.X, axis=0),
            'std':  np.std(self.X, axis=0),
            'min': np.min(self.X, axis=0),
            'max': np.max(self.X, axis=0),
            'median': np.median(self.X, axis=0),
            'q25': np.percentile(self.X, 25, axis=0),
            'q75':  np.percentile(self.X, 75, axis=0),
            'missing':  np.isnan(self.X).sum(axis=0),
            'zeros': (self.X == 0).sum(axis=0),
            'unique': np.array([len(np.unique(self.X[: , i])) for i in range(self.X.shape[1])])
        }
        
        self._stats = pd.DataFrame(stats, index=self.feature_names)
        
        return self._stats
    
    def report_statistics(self, n_features: int = 10):
        """
        Print statistical summary. 
        
        Args:
            n_features:  Number of features to show details for
        """
        print_section("STATISTICAL SUMMARY")
        print()
        
        print_info(f"Dataset shape: {self.X.shape}")
        print_info(f"Number of features: {self.X.shape[1]}")
        print_info(f"Number of samples: {self.X.shape[0]}")
        
        if self.y is not None:
            print_info(f"Number of classes:  {len(np.unique(self.y))}")
        
        print()
        
        stats = self.get_statistics()
        
        # Overall statistics
        print_info("Overall statistics:")
        print(f"  Total missing values: {stats['missing']. sum()}")
        print(f"  Features with missing: {(stats['missing'] > 0).sum()}")
        print(f"  Constant features: {(stats['std'] == 0).sum()}")
        
        print()
        
        # Show details for top features (by variance)
        variance = stats['std'] ** 2
        
        # Filter out NaN variances
        valid_variance_mask = ~np.isnan(variance)
        valid_indices = np.where(valid_variance_mask)[0]
        
        if len(valid_indices) == 0:
            print_warning("All features have NaN variance")
            return
        
        # Sort valid variances
        valid_variances = variance[valid_indices]
        sorted_valid_indices = valid_indices[np.argsort(-valid_variances)]
        top_indices = sorted_valid_indices[: min(n_features, len(sorted_valid_indices))]
        
        print_info(f"Top {len(top_indices)} features by variance:")
        print()
        
        print(f"{'Feature':<30} {'Mean': >10} {'Std':>10} {'Min':>10} {'Max':>10}")
        print("-" * 70)
        
        for idx in top_indices:
            name = self.feature_names[idx][: 28]
            mean = stats. loc[self.feature_names[idx], 'mean']
            std = stats.loc[self.feature_names[idx], 'std']
            min_val = stats.loc[self.feature_names[idx], 'min']
            max_val = stats.loc[self.feature_names[idx], 'max']
            
            print(f"{name:<30} {mean:>10.2f} {std:>10.2f} {min_val: >10.2f} {max_val:>10.2f}")
    
    # ========================================================================
    # CORRELATION ANALYSIS
    # ========================================================================
    
    def get_correlation_matrix(self) -> np.ndarray:
        """
        Compute correlation matrix.
        
        Returns:
            Correlation matrix
        """
        if self._correlation is not None:
            return self._correlation
        
        # Handle missing values
        X_clean = self.X.copy()
        
        # Replace NaN with column mean (or 0 if all NaN)
        for i in range(X_clean.shape[1]):
            col = X_clean[:, i]
            if np.isnan(col).any():
                col_mean = np.nanmean(col)
                
                # If all NaN, use 0
                if np.isnan(col_mean):
                    col_mean = 0.0
                
                col[np.isnan(col)] = col_mean
                X_clean[:, i] = col
        
        # Compute correlation (suppress warnings for constant columns)
        with np.errstate(invalid='ignore', divide='ignore'):
            self._correlation = np.corrcoef(X_clean.T)
        
        # Replace NaN correlations with 0
        self._correlation = np.nan_to_num(self._correlation, nan=0.0)
        
        return self._correlation
    
    def find_highly_correlated_features(self, threshold: float = 0.9) -> List[Tuple[str, str, float]]:
        """
        Find pairs of highly correlated features.
        
        Args:
            threshold: Correlation threshold
            
        Returns: 
            List of (feature1, feature2, correlation) tuples
        """
        corr_matrix = self.get_correlation_matrix()
        
        highly_correlated = []
        
        n_features = len(self.feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr = abs(corr_matrix[i, j])
                
                if corr > threshold: 
                    highly_correlated.append((
                        self.feature_names[i],
                        self.feature_names[j],
                        corr_matrix[i, j]
                    ))
        
        # Sort by correlation (descending)
        highly_correlated.sort(key=lambda x: abs(x[2]), reverse=True)
        
        return highly_correlated
    
    def report_correlations(self, threshold: float = 0.9):
        """
        Print correlation analysis.  
        
        Args: 
            threshold: Correlation threshold
        """
        print_section("CORRELATION ANALYSIS")
        print()
        
        highly_corr = self.find_highly_correlated_features(threshold)
        
        if len(highly_corr) == 0:
            print_success(f"✓ No feature pairs with correlation > {threshold}")
        else:
            print_warning(f"Found {len(highly_corr)} highly correlated feature pairs (> {threshold}):")
            print()
            
            # Show top 10
            for i, (feat1, feat2, corr) in enumerate(highly_corr[: 10]):
                print(f"  {i+1}. {feat1[: 25]:<25} <-> {feat2[:25]:<25} (r = {corr:.3f})")
            
            if len(highly_corr) > 10:
                print(f"  ... and {len(highly_corr) - 10} more pairs")
    
    # ========================================================================
    # CLASS BALANCE ANALYSIS
    # ========================================================================
    
    def analyze_class_balance(self) -> Dict[str, Any]:
        """
        Analyze class distribution. 
        
        Returns:
            Dictionary with class statistics
        """
        if self.y is None:
            return {}
        
        unique, counts = np.unique(self.y, return_counts=True)
        
        total = len(self.y)
        percentages = (counts / total) * 100
        
        majority_class = unique[np.argmax(counts)]
        minority_class = unique[np.argmin(counts)]
        
        imbalance_ratio = counts.max() / counts.min()
        
        return {
            'classes': unique,
            'counts':  counts,
            'percentages':  percentages,
            'total': total,
            'majority_class': majority_class,
            'minority_class': minority_class,
            'imbalance_ratio': imbalance_ratio
        }
    
    def report_class_balance(self):
        """Print class balance report."""
        if self.y is None:
            print_warning("No labels provided")
            return
        
        print_section("CLASS BALANCE ANALYSIS")
        print()
        
        stats = self.analyze_class_balance()
        
        print_info(f"Total samples: {stats['total']}")
        print_info(f"Number of classes: {len(stats['classes'])}")
        print()
        
        print_info("Class distribution:")
        for cls, count, pct in zip(stats['classes'], stats['counts'], stats['percentages']):
            class_name = self.class_names[int(cls)] if self.class_names else f"Class {cls}"
            print(f"  {class_name: <20} {count:>6} samples ({pct:>5.1f}%)")
        
        print()
        
        if stats['imbalance_ratio'] > 1.5:
            print_warning(f"Dataset is imbalanced!")
            print_info(f"  Imbalance ratio: {stats['imbalance_ratio']:.1f}: 1")
            print_info(f"  Consider using SMOTE or class weights")
        else:
            print_success("Dataset is relatively balanced")
    
    # ========================================================================
    # OUTLIER DETECTION
    # ========================================================================
    
    def detect_outliers(self, method: str = 'iqr', threshold: float = 3.0) -> Dict[str, Any]:
        """
        Detect outliers in features.
        
        Args:
            method: Detection method ('iqr' or 'zscore')
            threshold: Threshold for detection
            
        Returns:
            Dictionary with outlier information
        """
        outliers_per_feature = []
        outliers_per_sample = np.zeros(self.X.shape[0])
        
        for i in range(self.X.shape[1]):
            col = self.X[:, i]
            
            # Skip if all NaN
            if np.all(np.isnan(col)):
                outliers_per_feature.append(0)
                continue
            
            if method == 'iqr': 
                # IQR method
                q25 = np.nanpercentile(col, 25)
                q75 = np.nanpercentile(col, 75)
                iqr = q75 - q25
                
                lower_bound = q25 - threshold * iqr
                upper_bound = q75 + threshold * iqr
                
                outliers = (col < lower_bound) | (col > upper_bound)
            
            elif method == 'zscore':
                # Z-score method
                mean = np.nanmean(col)
                std = np.nanstd(col)
                
                if std == 0:
                    outliers = np.zeros(len(col), dtype=bool)
                else: 
                    z_scores = np.abs((col - mean) / std)
                    outliers = z_scores > threshold
            
            else:
                raise ValueError(f"Unknown method: {method}")
            
            # Count outliers
            n_outliers = np.sum(outliers & ~np.isnan(col))
            outliers_per_feature.append(n_outliers)
            
            # Track per sample
            outliers_per_sample += outliers
        
        return {
            'method': method,
            'threshold': threshold,
            'outliers_per_feature': np.array(outliers_per_feature),
            'outliers_per_sample': outliers_per_sample,
            'total_outliers': sum(outliers_per_feature)
        }
    
    def report_outliers(self, method: str = 'iqr', threshold: float = 3.0):
        """Print outlier report."""
        print_section("OUTLIER DETECTION")
        print()
        
        outlier_info = self.detect_outliers(method, threshold)
        
        print_info(f"Method: {method.upper()}")
        print_info(f"Threshold: {threshold}")
        print()
        
        total_values = self.X.size
        outlier_pct = (outlier_info['total_outliers'] / total_values) * 100
        
        print_info(f"Total outlier values: {outlier_info['total_outliers']} ({outlier_pct:.2f}%)")
        
        # Features with most outliers
        top_indices = np.argsort(-outlier_info['outliers_per_feature'])[:10]
        
        print()
        print_info("Features with most outliers:")
        
        for idx in top_indices: 
            if outlier_info['outliers_per_feature'][idx] > 0:
                name = self.feature_names[idx]
                count = outlier_info['outliers_per_feature'][idx]
                pct = (count / self.X.shape[0]) * 100
                print(f"  {name[: 35]:<35} {count:>5} ({pct:>5.1f}%)")
    
    # ========================================================================
    # MISSING VALUE ANALYSIS
    # ========================================================================
    
    def analyze_missing_values(self) -> Dict[str, Any]:
        """
        Analyze missing value patterns.
        
        Returns:
            Dictionary with missing value statistics
        """
        missing_mask = np.isnan(self.X)
        
        missing_per_feature = missing_mask.sum(axis=0)
        missing_per_sample = missing_mask.sum(axis=1)
        
        total_missing = missing_mask.sum()
        total_values = self.X.size
        
        return {
            'total_missing': int(total_missing),
            'total_values': int(total_values),
            'missing_percentage': float(total_missing / total_values * 100),
            'missing_per_feature': missing_per_feature,
            'missing_per_sample': missing_per_sample,
            'features_with_missing': int((missing_per_feature > 0).sum()),
            'samples_with_missing': int((missing_per_sample > 0).sum())
        }
    
    def report_missing_values(self):
        """Print missing value report."""
        print_section("MISSING VALUES ANALYSIS")
        print()
        
        mv_info = self.analyze_missing_values()
        
        if mv_info['total_missing'] == 0:
            print_success("✓ No missing values detected")
            return
        
        print_warning(f"Missing values:  {mv_info['total_missing']} ({mv_info['missing_percentage']:.2f}%)")
        print_info(f"Features with missing:  {mv_info['features_with_missing']}/{self.X.shape[1]}")
        print_info(f"Samples with missing: {mv_info['samples_with_missing']}/{self.X.shape[0]}")
        
        print()
        
        # Features with most missing values
        top_indices = np.argsort(-mv_info['missing_per_feature'])[:10]
        
        print_info("Features with most missing values:")
        
        for idx in top_indices:
            if mv_info['missing_per_feature'][idx] > 0:
                name = self.feature_names[idx]
                count = mv_info['missing_per_feature'][idx]
                pct = (count / self.X.shape[0]) * 100
                print(f"  {name[:35]:<35} {count:>5} ({pct: >5.1f}%)")
    
    # ========================================================================
    # COMPREHENSIVE REPORT
    # ========================================================================
    
    def generate_report(self, output_dir: Optional[Path] = None):
        """
        Generate comprehensive EDA report.
        
        Args:
            output_dir: Directory to save visualizations (optional)
        """
        from utils.colors import print_header
        
        print_header("EXPLORATORY DATA ANALYSIS REPORT")
        print()
        
        # Statistical summary
        self.report_statistics()
        
        if self.logger:
            self.logger.blank()
        else:
            print()
        
        # Missing values
        self.report_missing_values()
        
        if self.logger:
            self.logger.blank()
        else:
            print()
        
        # Class balance
        if self.y is not None:
            self.report_class_balance()
            
            if self.logger:
                self.logger.blank()
            else:
                print()
        
        # Correlations
        self.report_correlations(threshold=0.9)
        
        if self.logger:
            self.logger.blank()
        else:
            print()
        
        # Outliers
        self.report_outliers(method='iqr', threshold=3.0)
        
        if self.logger:
            self.logger.blank()
        else:
            print()
        
        # Generate visualizations if requested
        if output_dir is not None and VISUALIZATION_AVAILABLE:
            print_section("GENERATING VISUALIZATIONS")
            print()
            
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            self.plot_all(output_dir)
            
            print_success(f"✓ Visualizations saved to:  {output_dir}")
        
        elif output_dir is not None and not VISUALIZATION_AVAILABLE:
            print_warning("Visualization libraries not available.Skipping plots.")
    
    # ========================================================================
    # VISUALIZATIONS
    # ========================================================================
    
    def plot_all(self, output_dir: Path):
        """
        Generate all visualizations.
        
        Args:
            output_dir: Directory to save plots
        """
        if not VISUALIZATION_AVAILABLE: 
            if self.logger:
                self.logger.warning("Visualization libraries not available")
            return
        
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        sns.set_style("whitegrid")
        
        # 1. Class distribution
        if self.y is not None:
            self._plot_class_distribution(output_dir / "class_distribution.png")
        
        # 2. Correlation heatmap (sample if too many features)
        self._plot_correlation_heatmap(output_dir / "correlation_heatmap.png")
        
        # 3. Missing values heatmap
        if np.isnan(self.X).any():
            self._plot_missing_values(output_dir / "missing_values.png")
        
        # 4. Feature distributions (top 12 by variance)
        self._plot_feature_distributions(output_dir / "feature_distributions.png")
        
        # 5. Outlier summary
        self._plot_outlier_summary(output_dir / "outliers.png")
        
        plt.close('all')
    
    def _plot_class_distribution(self, filepath:  Path):
        """Plot class distribution."""
        stats = self.analyze_class_balance()
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        classes = [self.class_names[int(c)] if self.class_names else f"Class {c}" 
                  for c in stats['classes']]
        
        bars = ax.bar(classes, stats['counts'], color='steelblue', alpha=0.7)
        
        # Add percentage labels
        for bar, pct in zip(bars, stats['percentages']):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{pct:.1f}%',
                   ha='center', va='bottom')
        
        ax.set_xlabel('Class', fontsize=12)
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Class Distribution', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_correlation_heatmap(self, filepath: Path, max_features: int = 50):
        """Plot correlation heatmap."""
        corr_matrix = self.get_correlation_matrix()
        
        # Sample features if too many
        if corr_matrix.shape[0] > max_features:
            # Select features with highest variance
            variances = np.var(self.X, axis=0)
            top_indices = np.argsort(-variances)[:max_features]
            
            corr_matrix = corr_matrix[top_indices][: , top_indices]
            feature_labels = [self.feature_names[i] for i in top_indices]
        else:
            feature_labels = self.feature_names
        
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(corr_matrix, 
                   cmap='coolwarm', 
                   center=0,
                   vmin=-1, 
                   vmax=1,
                   square=True,
                   linewidths=0.5,
                   cbar_kws={"shrink": 0.8},
                   xticklabels=False,
                   yticklabels=False,
                   ax=ax)
        
        ax.set_title(f'Feature Correlation Heatmap (top {len(feature_labels)} features)', 
                    fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_missing_values(self, filepath: Path):
        """Plot missing value patterns."""
        missing_mask = np.isnan(self.X)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Missing per feature
        missing_per_feature = missing_mask.sum(axis=0)
        missing_pct = (missing_per_feature / self.X.shape[0]) * 100
        
        # Show only features with missing values
        has_missing = missing_pct > 0
        
        if has_missing.sum() > 0:
            indices = np.where(has_missing)[0]
            
            axes[0].barh(range(len(indices)), missing_pct[indices], color='coral', alpha=0.7)
            axes[0].set_yticks(range(len(indices)))
            axes[0].set_yticklabels([self.feature_names[i][:20] for i in indices], fontsize=8)
            axes[0].set_xlabel('Missing (%)', fontsize=10)
            axes[0].set_title('Missing Values per Feature', fontsize=12, fontweight='bold')
            axes[0].invert_yaxis()
        
        # Missing per sample
        missing_per_sample = missing_mask.sum(axis=1)
        
        axes[1].hist(missing_per_sample, bins=30, color='coral', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Number of Missing Features', fontsize=10)
        axes[1].set_ylabel('Number of Samples', fontsize=10)
        axes[1].set_title('Missing Values per Sample', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_feature_distributions(self, filepath: Path, n_features: int = 12):
        """Plot feature distributions."""
        # Select top features by variance
        variances = np.nanvar(self.X, axis=0)
        top_indices = np.argsort(-variances)[:n_features]
        
        # Create subplots
        n_rows = 3
        n_cols = 4
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 10))
        axes = axes.flatten()
        
        for i, idx in enumerate(top_indices):
            data = self.X[:, idx]
            data = data[~np.isnan(data)]  # Remove NaN
            
            axes[i].hist(data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
            axes[i].set_title(self.feature_names[idx][: 25], fontsize=10)
            axes[i].set_ylabel('Frequency', fontsize=8)
        
        plt.suptitle('Feature Distributions (Top 12 by Variance)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _plot_outlier_summary(self, filepath: Path):
        """Plot outlier summary."""
        outlier_info = self.detect_outliers(method='iqr', threshold=3.0)
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Outliers per feature
        top_indices = np.argsort(-outlier_info['outliers_per_feature'])[:20]
        
        outlier_counts = outlier_info['outliers_per_feature'][top_indices]
        outlier_pct = (outlier_counts / self.X.shape[0]) * 100
        
        axes[0].barh(range(len(top_indices)), outlier_pct, color='tomato', alpha=0.7)
        axes[0].set_yticks(range(len(top_indices)))
        axes[0].set_yticklabels([self.feature_names[i][:20] for i in top_indices], fontsize=8)
        axes[0].set_xlabel('Outliers (%)', fontsize=10)
        axes[0].set_title('Top 20 Features by Outlier Count', fontsize=12, fontweight='bold')
        axes[0].invert_yaxis()
        
        # Outliers per sample
        axes[1].hist(outlier_info['outliers_per_sample'], bins=30, color='tomato', alpha=0.7, edgecolor='black')
        axes[1].set_xlabel('Number of Outlier Features', fontsize=10)
        axes[1].set_ylabel('Number of Samples', fontsize=10)
        axes[1].set_title('Outliers per Sample', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header
    
    logger = get_logger(name="EDA_TEST", verbose=True)
    
    print_header("EDA TEST")
    
    # Create sample data
    np.random.seed(42)
    
    X = np.random.randn(500, 20)
    y = np.random.randint(0, 3, 500)
    
    # Add some missing values
    missing_mask = np.random.random(X.shape) < 0.1
    X[missing_mask] = np.nan
    
    # Add some outliers
    outlier_mask = np.random.random(X.shape) < 0.05
    X[outlier_mask] = np.random.randn(outlier_mask.sum()) * 10
    
    # Create EDA instance
    eda = EDA(X, y, logger=logger)
    
    # Generate report
    eda.generate_report()
    
    logger.blank()
    print_header("TEST COMPLETE")
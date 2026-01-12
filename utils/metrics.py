# ============================================================================
# COMPREHENSIVE METRICS MODULE
# ============================================================================
# Calculate and visualize model performance metrics
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from pathlib import Path

try:
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        confusion_matrix, classification_report, roc_auc_score, roc_curve,
        precision_recall_curve, average_precision_score
    )
    import matplotlib.pyplot as plt
    import seaborn as sns
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from utils.logger import Logger
from utils.colors import print_info, print_success


# ============================================================================
# METRICS CALCULATOR
# ============================================================================

class MetricsCalculator: 
    """
    Calculate comprehensive classification metrics. 
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities (optional, for ROC/AUC)
        labels: Class labels
        logger: Logger instance
    """
    
    def __init__(self,
                 y_true: np.ndarray,
                 y_pred: np.ndarray,
                 y_prob: Optional[np.ndarray] = None,
                 labels: Optional[List[str]] = None,
                 logger: Optional[Logger] = None):
        
        if not SKLEARN_AVAILABLE: 
            raise ImportError("scikit-learn required for metrics")
        
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.labels = labels
        self.logger = logger
        
        # Calculate all metrics
        self.metrics = self._calculate_all_metrics()
    
    def _calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all classification metrics."""
        
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(self.y_true, self.y_pred)
        metrics['precision'] = precision_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['recall'] = recall_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        metrics['f1_score'] = f1_score(self.y_true, self.y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(self.y_true, self.y_pred, average=None, zero_division=0)
        metrics['recall_per_class'] = recall_score(self.y_true, self.y_pred, average=None, zero_division=0)
        metrics['f1_per_class'] = f1_score(self.y_true, self.y_pred, average=None, zero_division=0)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(self.y_true, self.y_pred)
        
        # ROC-AUC (if probabilities available)
        if self.y_prob is not None: 
            try:
                if len(np.unique(self.y_true)) == 2:
                    # Binary classification
                    if self.y_prob.ndim == 2:
                        y_prob_positive = self.y_prob[:, 1]
                    else:
                        y_prob_positive = self.y_prob
                    
                    metrics['roc_auc'] = roc_auc_score(self.y_true, y_prob_positive)
                    metrics['fpr'], metrics['tpr'], metrics['roc_thresholds'] = roc_curve(self.y_true, y_prob_positive)
                    metrics['average_precision'] = average_precision_score(self.y_true, y_prob_positive)
                else:
                    # Multi-class
                    metrics['roc_auc'] = roc_auc_score(self.y_true, self.y_prob, average='weighted', multi_class='ovr')
            except Exception as e:
                if self.logger:
                    self.logger.warning(f"Could not calculate ROC-AUC: {e}")
        
        # Classification report
        metrics['classification_report'] = classification_report(
            self.y_true, self.y_pred,
            target_names=self.labels if self.labels else None,
            output_dict=True,
            zero_division=0
        )
        
        return metrics
    
    def get_summary(self) -> Dict[str, float]:
        """Get summary metrics as simple dict."""
        return {
            'accuracy': float(self.metrics['accuracy']),
            'precision': float(self.metrics['precision']),
            'recall': float(self.metrics['recall']),
            'f1_score': float(self.metrics['f1_score']),
            'roc_auc': float(self.metrics.get('roc_auc', 0.0))
        }
    
    def print_summary(self):
        """Print metrics summary."""
        print_info("Performance Metrics:")
        print(f"  Accuracy:   {self.metrics['accuracy']:.4f}")
        print(f"  Precision: {self.metrics['precision']:.4f}")
        print(f"  Recall:    {self.metrics['recall']:.4f}")
        print(f"  F1-Score:  {self.metrics['f1_score']:.4f}")
        
        if 'roc_auc' in self.metrics:
            print(f"  ROC-AUC:   {self.metrics['roc_auc']:.4f}")
    
    def plot_confusion_matrix(self, output_path: Optional[Path] = None, figsize=(8, 6)):
        """Plot confusion matrix."""
        plt.figure(figsize=figsize)
        
        cm = self.metrics['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.labels if self.labels else 'auto',
                   yticklabels=self.labels if self.labels else 'auto')
        
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Confusion matrix saved to: {output_path}")
        
        plt.close()
    
    def plot_roc_curve(self, output_path: Optional[Path] = None, figsize=(8, 6)):
        """Plot ROC curve (binary classification only)."""
        if 'fpr' not in self.metrics or 'tpr' not in self.metrics:
            if self.logger:
                self.logger.warning("ROC curve not available (need probabilities for binary classification)")
            return
        
        plt.figure(figsize=figsize)
        
        plt.plot(self.metrics['fpr'], self.metrics['tpr'], 
                label=f"ROC Curve (AUC = {self.metrics['roc_auc']:.3f})", linewidth=2)
        plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"ROC curve saved to: {output_path}")
        
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_success
    
    logger = get_logger(name="METRICS_TEST", verbose=True)
    
    print_header("METRICS CALCULATOR TEST")
    print()
    
    # Generate fake predictions
    np.random.seed(42)
    n_samples = 100
    
    y_true = np.random.randint(0, 2, n_samples)
    y_pred = y_true. copy()
    y_pred[np.random.choice(n_samples, 10, replace=False)] = 1 - y_pred[np.random.choice(n_samples, 10, replace=False)]  # Add some errors
    y_prob = np.random.random((n_samples, 2))
    y_prob = y_prob / y_prob.sum(axis=1, keepdims=True)  # Normalize
    
    # Calculate metrics
    calc = MetricsCalculator(y_true, y_pred, y_prob, labels=['Class 0', 'Class 1'], logger=logger)
    
    # Print summary
    calc.print_summary()
    print()
    
    # Plot confusion matrix
    output_dir = Path('outputs/metrics_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    calc.plot_confusion_matrix(output_dir / 'confusion_matrix.png')
    calc.plot_roc_curve(output_dir / 'roc_curve.png')
    
    print_success("✓ Metrics test complete!")
    print_info(f"  Plots saved to: {output_dir}")
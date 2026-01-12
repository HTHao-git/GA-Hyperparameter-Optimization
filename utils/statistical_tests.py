# ============================================================================
# STATISTICAL SIGNIFICANCE TESTING
# ============================================================================
# Determine if differences between models are statistically significant
#
# Last updated: 2026-01-02
# ============================================================================

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from scipy import stats

from utils.logger import Logger
from utils.colors import print_info, print_warning, print_success


# ============================================================================
# STATISTICAL TESTER
# ============================================================================

class StatisticalTester: 
    """
    Perform statistical significance tests on model comparisons.
    
    Args:
        logger: Logger instance
        alpha: Significance level (default 0.05)
    """
    
    def __init__(self, logger: Optional[Logger] = None, alpha:  float = 0.05):
        self.logger = logger
        self.alpha = alpha
    
    # ========================================================================
    # PAIRED T-TEST
    # ========================================================================
    
    def paired_t_test(self, 
                      scores1: List[float], 
                      scores2: List[float],
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Perform paired t-test to compare two models. 
        
        Used when you have paired samples (e.g., same CV folds for both models).
        
        H0: Mean difference = 0 (no difference)
        H1: Mean difference ≠ 0 (significant difference)
        
        Args: 
            scores1: Scores from model 1
            scores2: Scores from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score arrays must have same length for paired test")
        
        # Perform paired t-test
        statistic, p_value = stats.ttest_rel(scores1, scores2)
        
        # Interpret results
        is_significant = p_value < self.alpha
        
        mean_diff = np.mean(scores1) - np.mean(scores2)
        
        if is_significant:
            if mean_diff > 0:
                conclusion = f"{model1_name} is significantly better than {model2_name}"
            else:
                conclusion = f"{model2_name} is significantly better than {model1_name}"
        else:
            conclusion = f"No significant difference between {model1_name} and {model2_name}"
        
        results = {
            'test':  'Paired t-test',
            'statistic': float(statistic),
            'p_value': float(p_value),
            'alpha': self.alpha,
            'is_significant': is_significant,
            'mean_diff': float(mean_diff),
            'model1_mean': float(np.mean(scores1)),
            'model2_mean': float(np.mean(scores2)),
            'conclusion':  conclusion
        }
        
        return results
    
    # ========================================================================
    # WILCOXON SIGNED-RANK TEST
    # ========================================================================
    
    def wilcoxon_test(self,
                      scores1: List[float],
                      scores2: List[float],
                      model1_name: str = "Model 1",
                      model2_name: str = "Model 2") -> Dict[str, Any]:
        """
        Perform Wilcoxon signed-rank test (non-parametric alternative to paired t-test).
        
        More robust when data is not normally distributed.
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
            model1_name: Name of model 1
            model2_name: Name of model 2
            
        Returns:
            Dictionary with test results
        """
        if len(scores1) != len(scores2):
            raise ValueError("Score arrays must have same length")
        
        # Check if all differences are zero
        differences = np.array(scores1) - np.array(scores2)
        if np.all(differences == 0):
            # All values identical - no difference
            results = {
                'test':  'Wilcoxon signed-rank test',
                'statistic': 0.0,
                'p_value': 1.0,  # No difference
                'alpha': self.alpha,
                'is_significant': False,
                'median_diff': 0.0,
                'model1_median': float(np.median(scores1)),
                'model2_median': float(np.median(scores2)),
                'conclusion': f"No difference between {model1_name} and {model2_name} (identical scores)"
            }
            return results
        
        try:
            # Perform Wilcoxon test
            statistic, p_value = stats.wilcoxon(scores1, scores2, zero_method='zsplit')
        except Exception as e:
            # Fallback if test fails
            if self.logger:
                self.logger.warning(f"Wilcoxon test failed: {e}. Using median comparison.")
            
            median_diff = np.median(scores1) - np.median(scores2)
            results = {
                'test': 'Wilcoxon signed-rank test',
                'statistic': 0.0,
                'p_value': 1.0,  # Conservative
                'alpha': self.alpha,
                'is_significant': False,
                'median_diff':  float(median_diff),
                'model1_median': float(np.median(scores1)),
                'model2_median': float(np.median(scores2)),
                'conclusion': f"Test failed - scores may be too similar"
            }
            return results
        
        # Interpret results
        is_significant = p_value < self.alpha
        
        median_diff = np.median(scores1) - np.median(scores2)
        
        if is_significant:
            if median_diff > 0:
                conclusion = f"{model1_name} is significantly better than {model2_name}"
            else:
                conclusion = f"{model2_name} is significantly better than {model1_name}"
        else:  
            conclusion = f"No significant difference between {model1_name} and {model2_name}"
        
        results = {
            'test': 'Wilcoxon signed-rank test',
            'statistic': float(statistic),
            'p_value':   float(p_value),
            'alpha': self.alpha,
            'is_significant': is_significant,
            'median_diff': float(median_diff),
            'model1_median':  float(np.median(scores1)),
            'model2_median': float(np.median(scores2)),
            'conclusion':   conclusion
        }
        
        return results
    
    # ========================================================================
    # EFFECT SIZE (COHEN'S D)
    # ========================================================================
    
    def cohens_d(self, scores1: List[float], scores2: List[float]) -> float:
        """
        Calculate Cohen's d effect size. 
        
        Interpretation:
        - Small:  d = 0.2
        - Medium: d = 0.5
        - Large: d = 0.8
        
        Args:
            scores1: Scores from model 1
            scores2: Scores from model 2
            
        Returns:
            Cohen's d value
        """
        mean1 = np.mean(scores1)
        mean2 = np.mean(scores2)
        
        std1 = np.std(scores1, ddof=1)
        std2 = np.std(scores2, ddof=1)
        
        n1 = len(scores1)
        n2 = len(scores2)
        
        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        
        # Cohen's d
        d = (mean1 - mean2) / pooled_std
        
        return float(d)
    
    def interpret_cohens_d(self, d: float) -> str:
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        
        if d_abs < 0.2:
            return "negligible"
        elif d_abs < 0.5:
            return "small"
        elif d_abs < 0.8:
            return "medium"
        else:
            return "large"
    
    # ========================================================================
    # COMPREHENSIVE COMPARISON
    # ========================================================================
    
    def compare_models(self,
                       model_scores: Dict[str, List[float]],
                       baseline_model: Optional[str] = None) -> Dict[str, Any]: 
        """
        Comprehensive comparison of multiple models.
        
        Args:
            model_scores: Dictionary of {model_name: [cv_scores]}
            baseline_model: Name of baseline model (if None, uses first model)
            
        Returns:
            Dictionary with all comparison results
        """
        model_names = list(model_scores.keys())
        
        if baseline_model is None:
            baseline_model = model_names[0]
        
        if baseline_model not in model_names: 
            raise ValueError(f"Baseline model '{baseline_model}' not found")
        
        baseline_scores = model_scores[baseline_model]
        
        results = {
            'baseline':  baseline_model,
            'comparisons': {}
        }
        
        # Compare each model to baseline
        for model_name in model_names:
            if model_name == baseline_model:
                continue
            
            model_scores_list = model_scores[model_name]
            
            # Paired t-test
            t_test = self.paired_t_test(
                model_scores_list, baseline_scores,
                model_name, baseline_model
            )
            
            # Wilcoxon test
            wilcoxon = self.wilcoxon_test(
                model_scores_list, baseline_scores,
                model_name, baseline_model
            )
            
            # Effect size
            d = self.cohens_d(model_scores_list, baseline_scores)
            effect_size_interpretation = self.interpret_cohens_d(d)
            
            results['comparisons'][model_name] = {
                't_test': t_test,
                'wilcoxon': wilcoxon,
                'cohens_d': d,
                'effect_size': effect_size_interpretation
            }
        
        return results
    
    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    
    def print_comparison_results(self, results: Dict[str, Any]):
        """Print comparison results in readable format."""
        
        print_info(f"Baseline model: {results['baseline']}")
        print()
        
        for model_name, comparison in results['comparisons'].items():
            print_info(f"Comparing {model_name} vs {results['baseline']}:")
            print()
            
            # T-test
            t_test = comparison['t_test']
            print(f"  Paired t-test:")
            print(f"    p-value: {t_test['p_value']:.4f}")
            print(f"    Significant: {'Yes' if t_test['is_significant'] else 'No'} (α={t_test['alpha']})")
            print(f"    Conclusion: {t_test['conclusion']}")
            print()
            
            # Wilcoxon
            wilcoxon = comparison['wilcoxon']
            print(f"  Wilcoxon test:")
            print(f"    p-value: {wilcoxon['p_value']:.4f}")
            print(f"    Significant: {'Yes' if wilcoxon['is_significant'] else 'No'}")
            print()
            
            # Effect size
            print(f"  Effect size:")
            print(f"    Cohen's d: {comparison['cohens_d']:.3f}")
            print(f"    Interpretation: {comparison['effect_size']}")
            print()
            print("-" * 70)
            print()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_section, print_success
    
    logger = get_logger(name="STATS_TEST", verbose=True)
    
    print_header("STATISTICAL TESTING")
    print()
    
    # Simulate CV scores for 3 models (5-fold CV)
    np.random.seed(42)
    
    model_scores = {
        'Random Forest': [0.95, 0.94, 0.96, 0.95, 0.94],
        'XGBoost':  [0.96, 0.95, 0.97, 0.96, 0.95],
        'SVM': [0.92, 0.91, 0.93, 0.92, 0.91]
    }
    
    # Create tester
    tester = StatisticalTester(logger=logger, alpha=0.05)
    
    # Compare all models
    print_section("Model Comparison (vs Random Forest)")
    print()
    
    results = tester.compare_models(model_scores, baseline_model='Random Forest')
    tester.print_comparison_results(results)
    
    print_success("✓ Statistical testing complete!")
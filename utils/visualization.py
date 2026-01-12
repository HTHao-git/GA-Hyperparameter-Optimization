# ============================================================================
# VISUALIZATION MODULE
# ============================================================================
# Create plots for GA optimization results
#
# Last updated:  2026-01-02
# ============================================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Any, Optional
from pathlib import Path

from utils.logger import Logger
from utils.colors import print_info


# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 100


# ============================================================================
# GA VISUALIZATION
# ============================================================================

class GAVisualizer:
    """
    Visualize GA optimization results. 
    
    Args:
        history: GA optimization history
        logger: Logger instance
    """
    
    def __init__(self, history: List[Dict[str, Any]], logger: Optional[Logger] = None):
        self.history = history
        self.logger = logger
    
    def plot_convergence(self, output_path: Optional[Path] = None, figsize=(10, 6)):
        """Plot fitness convergence over generations."""
        
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        mean_fitness = [h['mean_fitness'] for h in self.history]
        
        plt.figure(figsize=figsize)
        
        plt.plot(generations, best_fitness, 'b-o', label='Best Fitness', linewidth=2, markersize=6)
        plt.plot(generations, mean_fitness, 'g--s', label='Mean Fitness', linewidth=2, markersize=5, alpha=0.7)
        
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Fitness', fontsize=12)
        plt.title('GA Convergence Curve', fontsize=14, fontweight='bold')
        plt.legend(fontsize=11)
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Convergence plot saved to: {output_path}")
        
        plt.close()
    
    def plot_diversity(self, output_path: Optional[Path] = None, figsize=(10, 6)):
        """Plot diversity over generations."""
        
        if 'diversity' not in self.history[0]:
            if self.logger:
                self.logger.warning("Diversity data not available in history")
            return
        
        generations = [h['generation'] for h in self.history]
        diversity = [h['diversity'] for h in self.history]
        
        plt.figure(figsize=figsize)
        
        plt.plot(generations, diversity, 'r-o', linewidth=2, markersize=6)
        plt.xlabel('Generation', fontsize=12)
        plt.ylabel('Diversity', fontsize=12)
        plt.title('Population Diversity Over Time', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Diversity plot saved to: {output_path}")
        
        plt.close()
    
    def plot_combined(self, output_path: Optional[Path] = None, figsize=(14, 5)):
        """Plot convergence and diversity side-by-side."""
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        generations = [h['generation'] for h in self.history]
        best_fitness = [h['best_fitness'] for h in self.history]
        mean_fitness = [h['mean_fitness'] for h in self.history]
        
        # Convergence
        ax1.plot(generations, best_fitness, 'b-o', label='Best', linewidth=2, markersize=6)
        ax1.plot(generations, mean_fitness, 'g--s', label='Mean', linewidth=2, markersize=5, alpha=0.7)
        ax1.set_xlabel('Generation', fontsize=11)
        ax1.set_ylabel('Fitness', fontsize=11)
        ax1.set_title('Convergence', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Diversity
        if 'diversity' in self.history[0]:
            diversity = [h['diversity'] for h in self.history]
            ax2.plot(generations, diversity, 'r-o', linewidth=2, markersize=6)
            ax2.set_xlabel('Generation', fontsize=11)
            ax2.set_ylabel('Diversity', fontsize=11)
            ax2.set_title('Diversity', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Diversity data\nnot available', 
                    ha='center', va='center', transform=ax2.transAxes, fontsize=12)
            ax2.set_xticks([])
            ax2.set_yticks([])
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Combined plot saved to: {output_path}")
        
        plt.close()


# ============================================================================
# MODEL COMPARISON VISUALIZATION
# ============================================================================

class ComparisonVisualizer:
    """
    Visualize model comparison results.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        logger: Logger instance
    """
    
    def __init__(self, results: Dict[str, Dict[str, float]], logger: Optional[Logger] = None):
        self.results = results
        self.logger = logger
    
    def plot_metric_comparison(self, 
                               metrics:  List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                               output_path: Optional[Path] = None,
                               figsize=(10, 6)):
        """Plot bar chart comparing metrics across models."""
        
        models = list(self.results.keys())
        n_metrics = len(metrics)
        n_models = len(models)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        colors = sns.color_palette("husl", n_models)
        
        for i, model in enumerate(models):
            values = [self.results[model].get(m, 0) for m in metrics]
            offset = (i - n_models/2 + 0.5) * width
            ax.bar(x + offset, values, width, label=model, color=colors[i], alpha=0.8)
        
        ax.set_xlabel('Metrics', fontsize=12)
        ax.set_ylabel('Score', fontsize=12)
        ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics])
        ax.legend(fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Comparison plot saved to: {output_path}")
        
        plt.close()
    
    def plot_radar_chart(self,
                        metrics: List[str] = ['accuracy', 'precision', 'recall', 'f1_score'],
                        output_path:  Optional[Path] = None,
                        figsize=(8, 8)):
        """Plot radar chart comparing models."""
        
        models = list(self.results.keys())
        n_metrics = len(metrics)
        
        angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
        angles += angles[: 1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        colors = sns.color_palette("husl", len(models))
        
        for i, model in enumerate(models):
            values = [self.results[model].get(m, 0) for m in metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors[i])
            ax.fill(angles, values, alpha=0.15, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([m.replace('_', ' ').title() for m in metrics], fontsize=10)
        ax.set_ylim(0, 1)
        ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
        ax.set_yticklabels(['0.2', '0.4', '0.6', '0.8', '1.0'], fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=11)
        
        plt.title('Model Performance Radar Chart', fontsize=14, fontweight='bold', pad=20)
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            if self.logger:
                self.logger.info(f"Radar chart saved to: {output_path}")
        
        plt.close()


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils.colors import print_header, print_success
    
    logger = get_logger(name="VIZ_TEST", verbose=True)
    
    print_header("VISUALIZATION TEST")
    print()
    
    output_dir = Path('outputs/viz_test')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Test GA visualization
    history = [
        {'generation': i, 'best_fitness': 0.5 + 0.04*i + np.random.random()*0.05, 
         'mean_fitness': 0.3 + 0.03*i + np.random.random()*0.05,
         'diversity': 0.3 - 0.015*i + np.random.random()*0.02}
        for i in range(20)
    ]
    
    ga_viz = GAVisualizer(history, logger)
    ga_viz.plot_convergence(output_dir / 'convergence.png')
    ga_viz.plot_diversity(output_dir / 'diversity.png')
    ga_viz.plot_combined(output_dir / 'combined.png')
    
    # Test comparison visualization
    results = {
        'Random Forest': {'accuracy': 0.95, 'precision': 0.93, 'recall': 0.94, 'f1_score':  0.935},
        'XGBoost': {'accuracy': 0.96, 'precision': 0.95, 'recall': 0.93, 'f1_score':  0.94},
        'SVM': {'accuracy': 0.92, 'precision': 0.91, 'recall': 0.90, 'f1_score': 0.905}
    }
    
    comp_viz = ComparisonVisualizer(results, logger)
    comp_viz.plot_metric_comparison(output_path=output_dir / 'comparison_bars.png')
    comp_viz.plot_radar_chart(output_path=output_dir / 'comparison_radar.png')
    
    print_success("✓ Visualization test complete!")
    print_info(f"  Plots saved to: {output_dir}")
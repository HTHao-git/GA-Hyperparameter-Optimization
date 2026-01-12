# ============================================================================
# REPORT GENERATOR
# ============================================================================
# Generate comprehensive HTML/text reports for experiments
#
# Last updated: 2026-01-02
# ============================================================================

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime

from utils.logger import Logger
from utils.colors import print_info, print_success


# ============================================================================
# REPORT GENERATOR
# ============================================================================

class ReportGenerator:
    """
    Generate comprehensive experiment reports. 
    
    Args:
        output_dir: Directory to save reports
        logger: Logger instance
    """
    
    def __init__(self, output_dir: Path, logger: Optional[Logger] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger = logger
    
    # ========================================================================
    # HTML REPORT
    # ========================================================================
    
    def generate_html_report(self,
                            experiment_name: str,
                            dataset_info: Dict[str, Any],
                            model_results: Dict[str, Dict[str, Any]],
                            statistical_results: Optional[Dict[str, Any]] = None,
                            plots:  Optional[Dict[str, Path]] = None) -> Path:
        """
        Generate comprehensive HTML report.
        
        Args:
            experiment_name: Name of experiment
            dataset_info: Dataset information
            model_results: Results for each model
            statistical_results:  Statistical test results
            plots: Dictionary of {plot_name: plot_path}
            
        Returns:
            Path to generated HTML file
        """
        
        html_content = f"""
<! DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{experiment_name} - Experiment Report</title>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .header {{
            background:  linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding:  30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        h1 {{
            margin: 0;
            font-size: 2.5em;
        }}
        .timestamp {{
            opacity: 0.9;
            margin-top: 10px;
        }}
        .section {{
            background: white;
            padding: 25px;
            margin-bottom:  20px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h2 {{
            color: #667eea;
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #667eea;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric-value {{
            font-weight: bold;
            color: #2d3748;
        }}
        .winner {{
            background-color: #48bb78;
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.9em;
        }}
        .plot {{
            margin: 20px 0;
            text-align: center;
        }}
        .plot img {{
            max-width: 100%;
            border-radius: 8px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        .config-item {{
            background: #f7fafc;
            padding: 8px 12px;
            margin: 5px 0;
            border-left: 3px solid #667eea;
            border-radius: 4px;
        }}
        .significant {{
            color: #48bb78;
            font-weight: bold;
        }}
        .not-significant {{
            color: #a0aec0;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>{experiment_name}</h1>
        <div class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>
    </div>
    
    <div class="section">
        <h2>📊 Dataset Information</h2>
        <table>
            <tr><th>Property</th><th>Value</th></tr>
            <tr><td>Dataset</td><td>{dataset_info.get('name', 'N/A')}</td></tr>
            <tr><td>Samples</td><td>{dataset_info.get('n_samples', 'N/A')}</td></tr>
            <tr><td>Features</td><td>{dataset_info.get('n_features', 'N/A')}</td></tr>
            <tr><td>Classes</td><td>{dataset_info.get('n_classes', 'N/A')}</td></tr>
            <tr><td>Class Distribution</td><td>{dataset_info.get('class_distribution', 'N/A')}</td></tr>
        </table>
    </div>
    
    <div class="section">
        <h2>🏆 Model Performance Comparison</h2>
        {self._generate_performance_table(model_results)}
    </div>
"""
        
        # Add statistical tests if available
        if statistical_results: 
            html_content += f"""
    <div class="section">
        <h2>📈 Statistical Significance</h2>
        {self._generate_statistical_table(statistical_results)}
    </div>
"""
        
        # Add plots if available
        if plots:
            html_content += """
    <div class="section">
        <h2>📉 Visualizations</h2>
"""
            for plot_name, plot_path in plots.items():
                # Make path relative
                rel_path = plot_path.relative_to(self.output_dir) if plot_path.is_absolute() else plot_path
                html_content += f"""
        <div class="plot">
            <h3>{plot_name}</h3>
            <img src="{rel_path}" alt="{plot_name}">
        </div>
"""
            html_content += "    </div>\n"
        
        # Add model configurations
        html_content += """
    <div class="section">
        <h2>⚙️ Best Configurations</h2>
"""
        for model_name, results in model_results.items():
            if 'config' in results:
                html_content += f"""
        <h3>{model_name}</h3>
        {self._generate_config_display(results['config'])}
"""
        
        html_content += """
    </div>
</body>
</html>
"""
        
        # Save HTML file
        html_path = self.output_dir / f"{experiment_name.replace(' ', '_')}_report.html"
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        if self.logger:
            self.logger.info(f"HTML report saved to: {html_path}")
        
        return html_path
    
    def _generate_performance_table(self, model_results: Dict[str, Dict[str, Any]]) -> str:
        """Generate HTML table for performance metrics."""
        
        # Get all metrics
        all_metrics = set()
        for results in model_results.values():
            all_metrics.update(results.keys())
        
        metrics = sorted([m for m in all_metrics if m not in ['config', 'history', 'optimization_time']])
        
        html = "<table>\n<tr><th>Model</th>"
        for metric in metrics: 
            html += f"<th>{metric.replace('_', ' ').title()}</th>"
        html += "</tr>\n"
        
        for model_name, results in model_results.items():
            html += f"<tr><td><strong>{model_name}</strong></td>"
            for metric in metrics:
                value = results.get(metric, 'N/A')
                if isinstance(value, (int, float)):
                    html += f"<td class='metric-value'>{value:.4f}</td>"
                else: 
                    html += f"<td>{value}</td>"
            html += "</tr>\n"
        
        html += "</table>"
        return html
    
    def _generate_statistical_table(self, statistical_results: Dict[str, Any]) -> str:
        """Generate HTML table for statistical tests."""
        
        baseline = statistical_results.get('baseline', 'Baseline')
        comparisons = statistical_results.get('comparisons', {})
        
        html = f"<p><strong>Baseline Model:</strong> {baseline}</p>\n"
        html += "<table>\n<tr><th>Model</th><th>Test</th><th>p-value</th><th>Significant? </th><th>Effect Size</th></tr>\n"
        
        for model_name, comp in comparisons.items():
            t_test = comp.get('t_test', {})
            p_value = t_test.get('p_value', 0)
            is_sig = t_test.get('is_significant', False)
            effect = comp.get('effect_size', 'N/A')
            
            sig_class = 'significant' if is_sig else 'not-significant'
            sig_text = 'Yes ✓' if is_sig else 'No'
            
            html += f"<tr><td><strong>{model_name}</strong></td>"
            html += f"<td>Paired t-test</td>"
            html += f"<td>{p_value:.4f}</td>"
            html += f"<td class='{sig_class}'>{sig_text}</td>"
            html += f"<td>{effect}</td></tr>\n"
        
        html += "</table>"
        return html
    
    def _generate_config_display(self, config: Dict[str, Any]) -> str:
        """Generate HTML for configuration display."""
        html = "<div>"
        for key, value in config.items():
            if not key.startswith('_'):  # Skip internal parameters
                html += f"<div class='config-item'><strong>{key}:</strong> {value}</div>\n"
        html += "</div>"
        return html
    
    # ========================================================================
    # TEXT REPORT
    # ========================================================================
    
    def generate_text_report(self,
                            experiment_name: str,
                            dataset_info:  Dict[str, Any],
                            model_results: Dict[str, Dict[str, Any]],
                            statistical_results: Optional[Dict[str, Any]] = None) -> Path:
        """Generate simple text report."""
        
        lines = []
        lines.append("=" * 80)
        lines.append(f"{experiment_name.upper()}")
        lines.append("=" * 80)
        lines.append(f"Generated:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("")
        
        # Dataset info
        lines.append("DATASET INFORMATION")
        lines.append("-" * 80)
        for key, value in dataset_info.items():
            lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Model results
        lines.append("MODEL PERFORMANCE")
        lines.append("-" * 80)
        for model_name, results in model_results.items():
            lines.append(f"\n{model_name}:")
            for key, value in results.items():
                if key not in ['config', 'history']: 
                    if isinstance(value, (int, float)):
                        lines.append(f"  {key}: {value:.4f}")
                    else:
                        lines.append(f"  {key}: {value}")
        lines.append("")
        
        # Statistical results
        if statistical_results:
            lines.append("STATISTICAL SIGNIFICANCE")
            lines.append("-" * 80)
            baseline = statistical_results.get('baseline', 'Baseline')
            lines.append(f"Baseline: {baseline}\n")
            
            for model_name, comp in statistical_results.get('comparisons', {}).items():
                lines.append(f"{model_name} vs {baseline}:")
                t_test = comp.get('t_test', {})
                lines.append(f"  p-value: {t_test.get('p_value', 0):.4f}")
                lines.append(f"  Significant: {'Yes' if t_test.get('is_significant', False) else 'No'}")
                lines.append(f"  Effect size: {comp.get('effect_size', 'N/A')}")
                lines.append("")
        
        lines.append("=" * 80)
        
        # Save text file
        text_path = self.output_dir / f"{experiment_name.replace(' ', '_')}_report.txt"
        with open(text_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        if self.logger:
            self.logger.info(f"Text report saved to: {text_path}")
        
        return text_path


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    from utils.logger import get_logger
    from utils. colors import print_header, print_success
    
    logger = get_logger(name="REPORT_TEST", verbose=True)
    
    print_header("REPORT GENERATOR TEST")
    print()
    
    output_dir = Path('outputs/report_test')
    
    # Sample data
    dataset_info = {
        'name': 'SECOM',
        'n_samples': 1567,
        'n_features':  590,
        'n_classes':  2,
        'class_distribution': '{0: 1463, 1: 104}'
    }
    
    model_results = {
        'Random Forest': {
            'cv_score': 0.9500,
            'test_score': 0.9450,
            'f1_score': 0.9400,
            'optimization_time': 150.5
        },
        'XGBoost': {
            'cv_score': 0.9600,
            'test_accuracy': 0.9550,
            'test_f1_score': 0.9500,
            'optimization_time': 120.3
        }
    }
    
    # Generate reports
    generator = ReportGenerator(output_dir, logger)
    
    html_path = generator.generate_html_report(
        experiment_name="Model Comparison Test",
        dataset_info=dataset_info,
        model_results=model_results
    )
    
    text_path = generator.generate_text_report(
        experiment_name="Model Comparison Test",
        dataset_info=dataset_info,
        model_results=model_results
    )
    
    print()
    print_success("✓ Report generation complete!")
    print_info(f"  HTML:  {html_path}")
    print_info(f"  Text:  {text_path}")
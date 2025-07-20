# Utilities and Logging Configuration
# This file contains both plotting utilities and logging configuration

# =============================================================================
# File: src/utils/plotting.py
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from typing import Dict, List, Optional, Any
import logging

logger = logging.getLogger(__name__)

class DemandForecastingVisualizer:
    """
    Visualization utilities for M5 demand forecasting
    Designed for intermittent demand patterns (68.2% zeros)
    """
    
    def __init__(self, style: str = "seaborn-v0_8", figsize: tuple = (12, 8)):
        plt.style.use(style)
        self.figsize = figsize
        self.colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
    
    def plot_model_comparison(self, results: Dict[str, Dict], save_path: str = None) -> plt.Figure:
        """Plot comparison of multiple models"""
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # Extract metrics
        models = list(results.keys())
        rmse_values = [results[m]['metrics']['rmse'] for m in models]
        mae_values = [results[m]['metrics']['mae'] for m in models]
        zero_f1_values = [results[m]['metrics']['zero_f1'] for m in models]
        service_levels = [results[m].get('business_impact', {}).get('average_service_level', 0) for m in models]
        
        # RMSE comparison
        bars1 = axes[0, 0].bar(models, rmse_values, color=self.colors[:len(models)])
        axes[0, 0].set_title('Root Mean Squared Error')
        axes[0, 0].set_ylabel('RMSE')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars1, rmse_values):
            axes[0, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # MAE comparison
        bars2 = axes[0, 1].bar(models, mae_values, color=self.colors[:len(models)])
        axes[0, 1].set_title('Mean Absolute Error')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars2, mae_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Zero prediction F1 score
        bars3 = axes[1, 0].bar(models, zero_f1_values, color=self.colors[:len(models)])
        axes[1, 0].set_title('Zero Prediction F1 Score')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].tick_params(axis='x', rotation=45)
        
        for bar, value in zip(bars3, zero_f1_values):
            axes[1, 0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Service level (if available)
        if any(sl > 0 for sl in service_levels):
            bars4 = axes[1, 1].bar(models, service_levels, color=self.colors[:len(models)])
            axes[1, 1].set_title('Service Level')
            axes[1, 1].set_ylabel('Service Level')
            axes[1, 1].tick_params(axis='x', rotation=45)
            axes[1, 1].set_ylim(0, 1)
            
            for bar, value in zip(bars4, service_levels):
                axes[1, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                               f'{value:.3f}', ha='center', va='bottom')
        else:
            axes[1, 1].text(0.5, 0.5, 'Service Level\nData Not Available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_feature_importance(self, feature_importance: np.ndarray, 
                               feature_names: List[str], top_n: int = 20,
                               title: str = "Feature Importance", save_path: str = None) -> plt.Figure:
        """Plot feature importance"""
        
        # Sort features by importance
        sorted_idx = np.argsort(feature_importance)[-top_n:]
        sorted_importance = feature_importance[sorted_idx]
        sorted_names = [feature_names[i] for i in sorted_idx]
        
        fig, ax = plt.subplots(figsize=(10, max(8, top_n * 0.4)))
        
        bars = ax.barh(range(len(sorted_names)), sorted_importance, color='steelblue')
        ax.set_yticks(range(len(sorted_names)))
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance')
        ax.set_title(title)
        ax.grid(axis='x', alpha=0.3)
        
        # Add value labels
        for i, (bar, value) in enumerate(zip(bars, sorted_importance)):
            ax.text(value + max(sorted_importance) * 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', va='center', ha='left')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_prediction_analysis(self, y_true: np.ndarray, y_pred: np.ndarray,
                                model_name: str = "Model", save_path: str = None) -> plt.Figure:
        """Plot detailed prediction analysis for intermittent demand"""
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - Prediction Analysis', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted (log scale for better visibility)
        non_zero_mask = (y_true > 0) | (y_pred > 0)
        if non_zero_mask.sum() > 0:
            axes[0, 0].scatter(y_true[non_zero_mask], y_pred[non_zero_mask], alpha=0.6, s=10)
            max_val = max(y_true.max(), y_pred.max())
            axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
            axes[0, 0].set_xlabel('Actual (Non-zero only)')
            axes[0, 0].set_ylabel('Predicted')
            axes[0, 0].set_title('Actual vs Predicted (Non-zero)')
            axes[0, 0].set_xscale('log')
            axes[0, 0].set_yscale('log')
        
        # 2. Zero prediction confusion matrix
        zero_true = (y_true == 0)
        zero_pred = (y_pred < 0.5)  # Threshold for zero prediction
        
        confusion_data = np.array([
            [(~zero_true & ~zero_pred).sum(), (~zero_true & zero_pred).sum()],
            [(zero_true & ~zero_pred).sum(), (zero_true & zero_pred).sum()]
        ])
        
        im = axes[0, 1].imshow(confusion_data, cmap='Blues')
        axes[0, 1].set_xticks([0, 1])
        axes[0, 1].set_yticks([0, 1])
        axes[0, 1].set_xticklabels(['Pred: Non-zero', 'Pred: Zero'])
        axes[0, 1].set_yticklabels(['True: Non-zero', 'True: Zero'])
        axes[0, 1].set_title('Zero Prediction Analysis')
        
        for i in range(2):
            for j in range(2):
                axes[0, 1].text(j, i, confusion_data[i, j], ha="center", va="center", 
                               color="white" if confusion_data[i, j] > confusion_data.max()/2 else "black")
        
        # 3. Error distribution
        errors = y_pred - y_true
        axes[0, 2].hist(errors, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 2].axvline(x=0, color='r', linestyle='--')
        axes[0, 2].set_xlabel('Prediction Error')
        axes[0, 2].set_ylabel('Frequency')
        axes[0, 2].set_title('Error Distribution')
        
        # 4. Demand intensity comparison
        true_nonzero = y_true[y_true > 0]
        pred_nonzero = y_pred[y_pred > 0]
        
        if len(true_nonzero) > 0 and len(pred_nonzero) > 0:
            axes[1, 0].hist(true_nonzero, bins=30, alpha=0.7, label='Actual', density=True)
            axes[1, 0].hist(pred_nonzero, bins=30, alpha=0.7, label='Predicted', density=True)
            axes[1, 0].set_xlabel('Non-zero Demand Values')
            axes[1, 0].set_ylabel('Density')
            axes[1, 0].set_title('Non-zero Demand Distribution')
            axes[1, 0].legend()
            axes[1, 0].set_yscale('log')
        
        # 5. Residuals vs Predicted
        axes[1, 1].scatter(y_pred, errors, alpha=0.5, s=10)
        axes[1, 1].axhline(y=0, color='r', linestyle='--')
        axes[1, 1].set_xlabel('Predicted Values')
        axes[1, 1].set_ylabel('Residuals')
        axes[1, 1].set_title('Residuals vs Predicted')
        
        # 6. Performance by demand level
        demand_bins = np.quantile(y_true[y_true > 0], [0, 0.25, 0.5, 0.75, 0.9, 1.0]) if (y_true > 0).sum() > 0 else [0, 1]
        bin_labels = [f'{demand_bins[i]:.1f}-{demand_bins[i+1]:.1f}' for i in range(len(demand_bins)-1)]
        bin_rmse = []
        
        for i in range(len(demand_bins)-1):
            mask = (y_true >= demand_bins[i]) & (y_true < demand_bins[i+1])
            if mask.sum() > 0:
                rmse = np.sqrt(np.mean((y_true[mask] - y_pred[mask])**2))
                bin_rmse.append(rmse)
            else:
                bin_rmse.append(0)
        
        if bin_rmse:
            axes[1, 2].bar(range(len(bin_labels)), bin_rmse, alpha=0.7)
            axes[1, 2].set_xticks(range(len(bin_labels)))
            axes[1, 2].set_xticklabels(bin_labels, rotation=45)
            axes[1, 2].set_xlabel('Demand Range')
            axes[1, 2].set_ylabel('RMSE')
            axes[1, 2].set_title('RMSE by Demand Level')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def create_interactive_dashboard(self, results: Dict[str, Dict], 
                                   save_path: str = None) -> go.Figure:
        """Create interactive dashboard with Plotly"""
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Model Performance', 'Zero Prediction Accuracy', 
                          'Error Analysis', 'Business Impact'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        models = list(results.keys())
        
        # Model performance metrics
        rmse_values = [results[m]['metrics']['rmse'] for m in models]
        mae_values = [results[m]['metrics']['mae'] for m in models]
        
        fig.add_trace(
            go.Bar(x=models, y=rmse_values, name='RMSE', marker_color='lightblue'),
            row=1, col=1
        )
        
        # Zero prediction metrics
        zero_f1_values = [results[m]['metrics']['zero_f1'] for m in models]
        fig.add_trace(
            go.Bar(x=models, y=zero_f1_values, name='Zero F1', marker_color='lightgreen'),
            row=1, col=2
        )
        
        # Add more interactive elements as needed
        fig.update_layout(
            title_text="M5 Demand Forecasting - Model Performance Dashboard",
            showlegend=False,
            height=800
        )
        
        if save_path:
            fig.write_html(save_path)
        
        return fig


class ReportGenerator:
    """Generate HTML reports for model training results"""
    
    def __init__(self):
        self.template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>M5 Demand Forecasting - Model Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                .header { background-color: #f0f8ff; padding: 20px; border-radius: 10px; }
                .metric { background-color: #f9f9f9; padding: 15px; margin: 10px 0; border-radius: 5px; }
                .model-section { border-left: 4px solid #4CAF50; padding-left: 20px; margin: 20px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                .best-model { background-color: #e8f5e8; }
            </style>
        </head>
        <body>
            <div class="header">
                <h1>🎯 M5 Demand Forecasting Model Report</h1>
                <p>Generated on: {timestamp}</p>
            </div>
            
            <h2>📊 Executive Summary</h2>
            <div class="metric">
                <strong>Best Model:</strong> {best_model}<br>
                <strong>Best RMSE:</strong> {best_rmse:.4f}<br>
                <strong>Zero Prediction Accuracy:</strong> {best_zero_f1:.3f}<br>
                <strong>Models Trained:</strong> {num_models}
            </div>
            
            <h2>📈 Model Performance Comparison</h2>
            {comparison_table}
            
            <h2>🔍 Detailed Results</h2>
            {detailed_results}
            
            <h2>💼 Business Impact</h2>
            <div class="metric">
                <p>Intermittent demand patterns (68.2% zeros) successfully handled with specialized metrics and model architectures.</p>
                <p>Recommended deployment: <strong>{best_model}</strong> for production use.</p>
            </div>
        </body>
        </html>
        """
    
    def generate_report(self, results: Dict[str, Dict], save_path: str) -> str:
        """Generate HTML report"""
        
        from datetime import datetime
        
        # Find best model
        best_model = min(results.keys(), 
                        key=lambda x: results[x]['metrics']['rmse'])
        best_rmse = results[best_model]['metrics']['rmse']
        best_zero_f1 = results[best_model]['metrics']['zero_f1']
        
        # Create comparison table
        comparison_data = []
        for model, result in results.items():
            metrics = result['metrics']
            comparison_data.append([
                model,
                f"{metrics['rmse']:.4f}",
                f"{metrics['mae']:.4f}",
                f"{metrics['zero_f1']:.3f}",
                f"{result.get('zero_rate_predicted', 0):.1%}"
            ])
        
        # Sort by RMSE
        comparison_data.sort(key=lambda x: float(x[1]))
        
        # Create HTML table
        table_html = "<table><tr><th>Model</th><th>RMSE</th><th>MAE</th><th>Zero F1</th><th>Zero Rate</th></tr>"
        for i, row in enumerate(comparison_data):
            row_class = ' class="best-model"' if i == 0 else ''
            table_html += f"<tr{row_class}>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>"
        table_html += "</table>"
        
        # Create detailed results
        detailed_html = ""
        for model, result in results.items():
            metrics = result['metrics']
            detailed_html += f"""
            <div class="model-section">
                <h3>🤖 {model.upper()}</h3>
                <div class="metric">
                    <strong>Core Metrics:</strong><br>
                    RMSE: {metrics['rmse']:.4f} | MAE: {metrics['mae']:.4f} | R²: {metrics.get('r2', 'N/A')}<br>
                    <strong>Intermittent Demand Metrics:</strong><br>
                    Zero F1: {metrics['zero_f1']:.3f} | Non-zero RMSE: {metrics.get('nonzero_rmse', 'N/A'):.4f}<br>
                    Zero Rate (Predicted): {result.get('zero_rate_predicted', 0):.1%}
                </div>
            </div>
            """
        
        # Fill template
        report_html = self.template.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            best_model=best_model.upper(),
            best_rmse=best_rmse,
            best_zero_f1=best_zero_f1,
            num_models=len(results),
            comparison_table=table_html,
            detailed_results=detailed_html
        )
        
        # Save report
        with open(save_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Report generated: {save_path}")
        return save_path


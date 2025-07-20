# Model Evaluation System for M5 Demand Forecasting
# File: src/models/evaluation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Comprehensive model evaluation for M5 demand forecasting.
    Handles intermittent demand patterns (68.2% zeros from your EDA).
    """
    
    def __init__(self):
        self.evaluation_history = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str = "model") -> Dict[str, Any]:
        """
        Comprehensive model evaluation with M5-specific metrics
        """
        logger.info(f"Evaluating {model_name}...")
        
        # Ensure arrays are numpy arrays
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        # Basic regression metrics
        metrics = self._calculate_basic_metrics(y_true, y_pred)
        
        # M5-specific metrics
        m5_metrics = self._calculate_m5_metrics(y_true, y_pred)
        metrics.update(m5_metrics)
        
        # Intermittent demand metrics (for 68.2% zeros)
        intermittent_metrics = self._calculate_intermittent_metrics(y_true, y_pred)
        metrics.update(intermittent_metrics)
        
        # Error analysis
        error_analysis = self._analyze_errors(y_true, y_pred)
        
        # Prediction distribution analysis
        distribution_analysis = self._analyze_distributions(y_true, y_pred)
        
        results = {
            'model_name': model_name,
            'metrics': metrics,
            'error_analysis': error_analysis,
            'distribution_analysis': distribution_analysis,
            'n_samples': len(y_true),
            'zero_rate_actual': (y_true == 0).mean(),
            'zero_rate_predicted': (y_pred == 0).mean()
        }
        
        # Store in history
        self.evaluation_history[model_name] = results
        
        logger.info(f"{model_name} evaluation completed. RMSE: {metrics['rmse']:.4f}")
        
        return results
    
    def _calculate_basic_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate basic regression metrics"""
        
        metrics = {}
        
        # Root Mean Squared Error
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        
        # Mean Absolute Error
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        
        # Mean Squared Error
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        
        # R-squared
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # Mean Absolute Percentage Error (handle zeros carefully)
        non_zero_mask = y_true != 0
        if non_zero_mask.sum() > 0:
            metrics['mape'] = mean_absolute_percentage_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )
        else:
            metrics['mape'] = np.inf
        
        # Mean Error (bias)
        metrics['mean_error'] = np.mean(y_pred - y_true)
        
        # Standard deviation of errors
        metrics['error_std'] = np.std(y_pred - y_true)
        
        return metrics
    
    def _calculate_m5_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calculate M5 competition specific metrics"""
        
        metrics = {}
        
        # Weighted Root Mean Squared Scaled Error (similar to M5 WRMSSE)
        # Simplified version - full M5 WRMSSE requires hierarchical structure
        naive_forecast = np.roll(y_true, 1)  # Previous day as naive forecast
        naive_forecast[0] = 0  # Handle first element
        
        mse_model = mean_squared_error(y_true, y_pred)
        mse_naive = mean_squared_error(y_true, naive_forecast)
        
        if mse_naive > 0:
            metrics['rmsse'] = np.sqrt(mse_model / mse_naive)
        else:
            metrics['rmsse'] = np.inf
        
        # Scale-dependent metrics
        metrics['rmse_scaled'] = metrics['rmsse']  # Alias for compatibility
        
        return metrics
    
    def _calculate_intermittent_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate metrics specific to intermittent demand (68.2% zeros)
        """
        
        metrics = {}
        
        # Zero prediction accuracy
        true_zeros = (y_true == 0)
        pred_zeros = (y_pred == 0)
        
        metrics['zero_precision'] = (true_zeros & pred_zeros).sum() / pred_zeros.sum() if pred_zeros.sum() > 0 else 0
        metrics['zero_recall'] = (true_zeros & pred_zeros).sum() / true_zeros.sum() if true_zeros.sum() > 0 else 0
        
        # F1 score for zero prediction
        if metrics['zero_precision'] + metrics['zero_recall'] > 0:
            metrics['zero_f1'] = 2 * metrics['zero_precision'] * metrics['zero_recall'] / (
                metrics['zero_precision'] + metrics['zero_recall']
            )
        else:
            metrics['zero_f1'] = 0
        
        # Non-zero demand metrics
        non_zero_mask = y_true > 0
        if non_zero_mask.sum() > 0:
            metrics['nonzero_rmse'] = np.sqrt(mean_squared_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            ))
            metrics['nonzero_mae'] = mean_absolute_error(
                y_true[non_zero_mask], y_pred[non_zero_mask]
            )
        else:
            metrics['nonzero_rmse'] = 0
            metrics['nonzero_mae'] = 0
        
        # Intermittency pattern metrics
        metrics['intermittency_rate_true'] = true_zeros.mean()
        metrics['intermittency_rate_pred'] = pred_zeros.mean()
        metrics['intermittency_bias'] = metrics['intermittency_rate_pred'] - metrics['intermittency_rate_true']
        
        # Demand intensity (average demand when non-zero)
        if non_zero_mask.sum() > 0:
            metrics['demand_intensity_true'] = y_true[y_true > 0].mean()
            pred_non_zero = y_pred > 0
            if pred_non_zero.sum() > 0:
                metrics['demand_intensity_pred'] = y_pred[pred_non_zero].mean()
            else:
                metrics['demand_intensity_pred'] = 0
        else:
            metrics['demand_intensity_true'] = 0
            metrics['demand_intensity_pred'] = 0
        
        return metrics
    
    def _analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction errors in detail"""
        
        errors = y_pred - y_true
        abs_errors = np.abs(errors)
        
        analysis = {
            'error_percentiles': {
                'p5': np.percentile(errors, 5),
                'p25': np.percentile(errors, 25),
                'p50': np.percentile(errors, 50),
                'p75': np.percentile(errors, 75),
                'p95': np.percentile(errors, 95)
            },
            'abs_error_percentiles': {
                'p5': np.percentile(abs_errors, 5),
                'p25': np.percentile(abs_errors, 25),
                'p50': np.percentile(abs_errors, 50),
                'p75': np.percentile(abs_errors, 75),
                'p95': np.percentile(abs_errors, 95)
            },
            'largest_errors': {
                'underestimation': {
                    'max_error': errors.min(),
                    'instances': (errors < -2).sum()
                },
                'overestimation': {
                    'max_error': errors.max(),
                    'instances': (errors > 2).sum()
                }
            }
        }
        
        return analysis
    
    def _analyze_distributions(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze prediction vs actual distributions"""
        
        analysis = {
            'actual_stats': {
                'mean': np.mean(y_true),
                'std': np.std(y_true),
                'min': np.min(y_true),
                'max': np.max(y_true),
                'zeros': (y_true == 0).sum(),
                'positive': (y_true > 0).sum()
            },
            'predicted_stats': {
                'mean': np.mean(y_pred),
                'std': np.std(y_pred),
                'min': np.min(y_pred),
                'max': np.max(y_pred),
                'zeros': (y_pred == 0).sum(),
                'positive': (y_pred > 0).sum()
            }
        }
        
        # Distribution comparison
        # KS test for distribution similarity
        try:
            ks_stat, ks_pvalue = stats.ks_2samp(y_true, y_pred)
            analysis['ks_test'] = {
                'statistic': ks_stat,
                'p_value': ks_pvalue,
                'distributions_similar': ks_pvalue > 0.05
            }
        except:
            analysis['ks_test'] = None
        
        return analysis
    
    def evaluate_by_category(self, y_true: np.ndarray, y_pred: np.ndarray, 
                           categories: np.ndarray, category_col: str = 'category') -> Dict[str, Dict]:
        """Evaluate model performance by category (FOODS, HOUSEHOLD, HOBBIES)"""
        
        logger.info("Evaluating by category...")
        
        results = {}
        unique_categories = np.unique(categories)
        
        for category in unique_categories:
            mask = categories == category
            if mask.sum() == 0:
                continue
                
            cat_true = y_true[mask]
            cat_pred = y_pred[mask]
            
            # Calculate metrics for this category
            cat_metrics = self._calculate_basic_metrics(cat_true, cat_pred)
            cat_intermittent = self._calculate_intermittent_metrics(cat_true, cat_pred)
            
            results[category] = {
                'metrics': {**cat_metrics, **cat_intermittent},
                'n_samples': len(cat_true),
                'zero_rate': (cat_true == 0).mean()
            }
        
        return results
    
    def compare_models(self, model_results: Dict[str, Dict]) -> pd.DataFrame:
        """Compare multiple models' performance"""
        
        logger.info("Comparing model performance...")
        
        comparison_data = []
        
        for model_name, results in model_results.items():
            metrics = results.get('metrics', {})
            
            comparison_data.append({
                'Model': model_name,
                'RMSE': metrics.get('rmse', np.nan),
                'MAE': metrics.get('mae', np.nan),
                'R²': metrics.get('r2', np.nan),
                'MAPE': metrics.get('mape', np.nan),
                'Zero F1': metrics.get('zero_f1', np.nan),
                'Non-zero RMSE': metrics.get('nonzero_rmse', np.nan),
                'Zero Rate (Pred)': results.get('zero_rate_predicted', np.nan),
                'Zero Rate (True)': results.get('zero_rate_actual', np.nan)
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.round(4)
        
        return comparison_df
    
    def create_evaluation_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               model_name: str = "Model", save_path: str = None) -> plt.Figure:
        """Create comprehensive evaluation plots"""
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{model_name} - Model Evaluation', fontsize=16, fontweight='bold')
        
        # 1. Actual vs Predicted scatter plot
        axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=1)
        max_val = max(y_true.max(), y_pred.max())
        axes[0, 0].plot([0, max_val], [0, max_val], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual')
        axes[0, 0].set_ylabel('Predicted')
        axes[0, 0].set_title('Actual vs Predicted')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals plot
        residuals = y_pred - y_true
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=1)
        axes[0, 1].axhline(y=0, color='r', linestyle='--')
        axes[0, 1].set_xlabel('Predicted')
        axes[0, 1].set_ylabel('Residuals')
        axes[0, 1].set_title('Residual Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Distribution comparison
        axes[0, 2].hist(y_true, bins=50, alpha=0.7, label='Actual', density=True)
        axes[0, 2].hist(y_pred, bins=50, alpha=0.7, label='Predicted', density=True)
        axes[0, 2].set_xlabel('Value')
        axes[0, 2].set_ylabel('Density')
        axes[0, 2].set_title('Distribution Comparison')
        axes[0, 2].legend()
        axes[0, 2].set_yscale('log')
        
        # 4. Error distribution
        axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(x=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Prediction Error')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Error Distribution')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 5. Zero prediction analysis
        zero_true = (y_true == 0)
        zero_pred = (y_pred == 0)
        
        confusion_matrix = np.array([
            [(~zero_true & ~zero_pred).sum(), (~zero_true & zero_pred).sum()],
            [(zero_true & ~zero_pred).sum(), (zero_true & zero_pred).sum()]
        ])
        
        im = axes[1, 1].imshow(confusion_matrix, cmap='Blues')
        axes[1, 1].set_xticks([0, 1])
        axes[1, 1].set_yticks([0, 1])
        axes[1, 1].set_xticklabels(['Non-zero Pred', 'Zero Pred'])
        axes[1, 1].set_yticklabels(['Non-zero True', 'Zero True'])
        axes[1, 1].set_title('Zero Prediction Confusion Matrix')
        
        # Add text annotations
        for i in range(2):
            for j in range(2):
                axes[1, 1].text(j, i, confusion_matrix[i, j], 
                               ha="center", va="center", color="white" if confusion_matrix[i, j] > confusion_matrix.max()/2 else "black")
        
        # 6. Performance by value range
        bins = np.linspace(0, y_true.max(), 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_errors = []
        
        for i in range(len(bins) - 1):
            mask = (y_true >= bins[i]) & (y_true < bins[i+1])
            if mask.sum() > 0:
                bin_error = np.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
                bin_errors.append(bin_error)
            else:
                bin_errors.append(0)
        
        axes[1, 2].bar(bin_centers, bin_errors, width=bin_centers[1]-bin_centers[0], alpha=0.7)
        axes[1, 2].set_xlabel('Actual Value Range')
        axes[1, 2].set_ylabel('RMSE')
        axes[1, 2].set_title('RMSE by Value Range')
        axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Evaluation plots saved to {save_path}")
        
        return fig


class BusinessImpactCalculator:
    """
    Calculate business impact metrics for demand forecasting
    Based on inventory optimization and service level analysis
    """
    
    def __init__(self, holding_cost_rate: float = 0.25, stockout_cost_multiplier: float = 3.0):
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
    
    def calculate_inventory_impact(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 safety_stock_factor: float = 1.25) -> Dict[str, float]:
        """
        Calculate inventory-related business impact
        """
        
        # Calculate optimal inventory levels
        optimal_inventory = y_true * safety_stock_factor
        predicted_inventory = y_pred * safety_stock_factor
        
        # Holding costs (excess inventory)
        excess_inventory = np.maximum(predicted_inventory - y_true, 0)
        holding_costs = excess_inventory * self.holding_cost_rate
        
        # Stockout costs (undersupply)
        stockout_quantity = np.maximum(y_true - predicted_inventory, 0)
        stockout_costs = stockout_quantity * self.stockout_cost_multiplier
        
        # Total costs
        total_costs = holding_costs + stockout_costs
        optimal_costs = np.maximum(optimal_inventory - y_true, 0) * self.holding_cost_rate
        
        # Service level (% of demand met)
        service_level = np.minimum(predicted_inventory / np.maximum(y_true, 1e-8), 1.0)
        avg_service_level = np.mean(service_level[y_true > 0]) if (y_true > 0).sum() > 0 else 1.0
        
        return {
            'total_holding_costs': holding_costs.sum(),
            'total_stockout_costs': stockout_costs.sum(),
            'total_costs': total_costs.sum(),
            'optimal_costs': optimal_costs.sum(),
            'cost_efficiency': optimal_costs.sum() / total_costs.sum() if total_costs.sum() > 0 else 1.0,
            'average_service_level': avg_service_level,
            'stockout_incidents': (stockout_quantity > 0).sum(),
            'excess_inventory_incidents': (excess_inventory > 0).sum()
        }
    
    def calculate_forecast_value(self, y_true: np.ndarray, y_pred: np.ndarray, 
                               baseline_pred: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate the business value of improved forecasting
        """
        
        # Use naive forecast as baseline if not provided
        if baseline_pred is None:
            baseline_pred = np.roll(y_true, 1)  # Previous period as baseline
            baseline_pred[0] = 0
        
        # Calculate business impact for both predictions
        model_impact = self.calculate_inventory_impact(y_true, y_pred)
        baseline_impact = self.calculate_inventory_impact(y_true, baseline_pred)
        
        # Calculate improvements
        cost_reduction = baseline_impact['total_costs'] - model_impact['total_costs']
        service_improvement = model_impact['average_service_level'] - baseline_impact['average_service_level']
        
        return {
            'cost_reduction': cost_reduction,
            'cost_reduction_pct': cost_reduction / baseline_impact['total_costs'] * 100 if baseline_impact['total_costs'] > 0 else 0,
            'service_level_improvement': service_improvement,
            'roi_estimate': cost_reduction,  # Simplified ROI
            'baseline_costs': baseline_impact['total_costs'],
            'model_costs': model_impact['total_costs']
        }


if __name__ == "__main__":
    # Test the evaluation system
    print("Testing Model Evaluation System...")
    
    # Create sample data with intermittent demand pattern
    np.random.seed(42)
    n_samples = 1000
    
    # True values with 70% zeros (similar to M5)
    y_true = np.random.choice([0, 1, 2, 3, 4, 5], size=n_samples, p=[0.7, 0.15, 0.08, 0.04, 0.02, 0.01])
    
    # Predictions with some noise
    y_pred = y_true + np.random.normal(0, 0.5, n_samples)
    y_pred = np.maximum(y_pred, 0)  # Ensure non-negative
    
    # Test evaluator
    evaluator = ModelEvaluator()
    results = evaluator.evaluate_model(y_true, y_pred, "test_model")
    
    print(f"Test Results:")
    print(f"RMSE: {results['metrics']['rmse']:.4f}")
    print(f"Zero F1: {results['metrics']['zero_f1']:.4f}")
    print(f"Intermittency Rate (True): {results['zero_rate_actual']:.1%}")
    print(f"Intermittency Rate (Pred): {results['zero_rate_predicted']:.1%}")
    
    # Test business impact
    impact_calc = BusinessImpactCalculator()
    business_impact = impact_calc.calculate_inventory_impact(y_true, y_pred)
    
    print(f"\nBusiness Impact:")
    print(f"Service Level: {business_impact['average_service_level']:.1%}")
    print(f"Cost Efficiency: {business_impact['cost_efficiency']:.3f}")
    
    print("\nEvaluation system test completed successfully!")
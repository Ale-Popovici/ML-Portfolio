# Model Evaluation and Metrics for Demand Forecasting
# Path: src/models/evaluation.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    mean_absolute_percentage_error
)
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DemandForecastingMetrics:
    """Comprehensive metrics for demand forecasting evaluation"""
    
    @staticmethod
    def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Root Mean Squared Error"""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Error"""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Mean Absolute Percentage Error"""
        # Handle zero values
        mask = y_true != 0
        if not mask.any():
            return np.inf
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Symmetric Mean Absolute Percentage Error"""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        if not mask.any():
            return 0.0
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    @staticmethod
    def wmape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Weighted Mean Absolute Percentage Error"""
        return np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100
    
    @staticmethod
    def wrmsse(y_true: np.ndarray, y_pred: np.ndarray, y_train: np.ndarray = None) -> float:
        """Weighted Root Mean Squared Scaled Error (M5 competition metric)"""
        
        # Calculate naive forecast error (seasonal naive with period=7)
        if y_train is not None and len(y_train) >= 7:
            naive_forecast = np.roll(y_train, 7)[-len(y_true):]
            naive_mse = mean_squared_error(y_true, naive_forecast)
        else:
            # Fallback to simple naive forecast
            naive_forecast = y_true[:-1] if len(y_true) > 1 else [y_true.mean()]
            naive_forecast = np.append(naive_forecast, y_true[-1])
            naive_mse = mean_squared_error(y_true, naive_forecast)
        
        if naive_mse == 0:
            return 0.0
        
        # Calculate RMSSE
        mse = mean_squared_error(y_true, y_pred)
        rmsse = np.sqrt(mse / naive_mse)
        
        # Weight by sales volume (simple weighting)
        weight = np.sum(y_true) / np.sum(y_true)  # This would be more complex in practice
        
        return rmsse * weight
    
    @staticmethod
    def bias(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Forecast bias"""
        return np.mean(y_pred - y_true)
    
    @staticmethod
    def coverage_probability(y_true: np.ndarray, y_pred_lower: np.ndarray, 
                           y_pred_upper: np.ndarray) -> float:
        """Coverage probability for prediction intervals"""
        coverage = np.mean((y_true >= y_pred_lower) & (y_true <= y_pred_upper))
        return coverage * 100
    
    @staticmethod
    def inventory_cost_metric(y_true: np.ndarray, y_pred: np.ndarray, 
                             holding_cost: float = 0.1, stockout_cost: float = 1.0) -> float:
        """Business metric: inventory cost optimization"""
        
        # Excess inventory cost
        excess = np.maximum(0, y_pred - y_true)
        holding_costs = excess * holding_cost
        
        # Stockout cost
        shortage = np.maximum(0, y_true - y_pred)
        stockout_costs = shortage * stockout_cost
        
        total_cost = np.sum(holding_costs + stockout_costs)
        return total_cost


class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self, metrics_config: Dict = None):
        self.metrics_config = metrics_config or {
            'rmse': True,
            'mae': True,
            'mape': True,
            'smape': True,
            'wmape': True,
            'bias': True,
            'r2': True
        }
        
        self.metrics = DemandForecastingMetrics()
        self.evaluation_results = {}
    
    def evaluate_model(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      model_name: str, y_train: np.ndarray = None) -> Dict[str, float]:
        """Evaluate a single model"""
        
        results = {}
        
        if self.metrics_config.get('rmse', True):
            results['RMSE'] = self.metrics.rmse(y_true, y_pred)
        
        if self.metrics_config.get('mae', True):
            results['MAE'] = self.metrics.mae(y_true, y_pred)
        
        if self.metrics_config.get('mape', True):
            results['MAPE'] = self.metrics.mape(y_true, y_pred)
        
        if self.metrics_config.get('smape', True):
            results['SMAPE'] = self.metrics.smape(y_true, y_pred)
        
        if self.metrics_config.get('wmape', True):
            results['WMAPE'] = self.metrics.wmape(y_true, y_pred)
        
        if self.metrics_config.get('bias', True):
            results['Bias'] = self.metrics.bias(y_true, y_pred)
        
        if self.metrics_config.get('r2', True):
            results['R²'] = r2_score(y_true, y_pred)
        
        if y_train is not None:
            results['WRMSSE'] = self.metrics.wrmsse(y_true, y_pred, y_train)
        
        # Business metrics
        results['Inventory_Cost'] = self.metrics.inventory_cost_metric(y_true, y_pred)
        
        self.evaluation_results[model_name] = results
        
        logger.info(f"Evaluated {model_name}: RMSE={results['RMSE']:.4f}, MAPE={results['MAPE']:.2f}%")
        
        return results
    
    def compare_models(self) -> pd.DataFrame:
        """Compare all evaluated models"""
        
        if not self.evaluation_results:
            logger.warning("No models have been evaluated yet")
            return pd.DataFrame()
        
        comparison_df = pd.DataFrame(self.evaluation_results).T
        comparison_df = comparison_df.round(4)
        
        # Rank models (lower is better for most metrics, higher for R²)
        for metric in comparison_df.columns:
            if metric == 'R²':
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=False)
            else:
                comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=True)
        
        # Calculate overall rank (average of all metric ranks)
        rank_cols = [col for col in comparison_df.columns if col.endswith('_rank')]
        comparison_df['Overall_Rank'] = comparison_df[rank_cols].mean(axis=1)
        
        return comparison_df.sort_values('Overall_Rank')
    
    def create_evaluation_report(self, output_dir: str = "outputs/reports") -> str:
        """Create comprehensive evaluation report"""
        
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        report_path = os.path.join(output_dir, "model_evaluation_report.html")
        
        # Generate HTML report
        html_content = self._generate_html_report()
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        logger.info(f"Evaluation report saved to {report_path}")
        return report_path
    
    def _generate_html_report(self) -> str:
        """Generate HTML evaluation report"""
        
        comparison_df = self.compare_models()
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Model Evaluation Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
                .metric-table {{ margin: 20px 0; }}
                .best-model {{ background-color: #d4edda; }}
                .summary {{ background-color: #f8f9fa; padding: 20px; margin: 20px 0; }}
            </style>
        </head>
        <body>
            <h1>Demand Forecasting Model Evaluation Report</h1>
            
            <div class="summary">
                <h2>Executive Summary</h2>
                <p><strong>Best Model:</strong> {comparison_df.index[0]} (Rank: {comparison_df.iloc[0]['Overall_Rank']:.2f})</p>
                <p><strong>Best RMSE:</strong> {comparison_df['RMSE'].min():.4f}</p>
                <p><strong>Best MAPE:</strong> {comparison_df['MAPE'].min():.2f}%</p>
                <p><strong>Number of Models Evaluated:</strong> {len(comparison_df)}</p>
            </div>
            
            <h2>Model Comparison</h2>
            <div class="metric-table">
                {comparison_df.to_html(table_id="comparison_table", classes="table")}
            </div>
            
            <h2>Recommendations</h2>
            <ul>
                <li>Primary recommendation: Use <strong>{comparison_df.index[0]}</strong> for production deployment</li>
                <li>Consider ensemble combining top 3 models for improved robustness</li>
                <li>Monitor model performance continuously for concept drift</li>
                <li>Retrain models monthly or when performance degrades by 5%</li>
            </ul>
        </body>
        </html>
        """
        
        return html


class BusinessImpactEvaluator:
    """Evaluate business impact of forecasting models"""
    
    def __init__(self, holding_cost_rate: float = 0.1, stockout_cost_multiplier: float = 5.0):
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_multiplier = stockout_cost_multiplier
    
    def calculate_inventory_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  item_costs: np.ndarray = None) -> Dict[str, float]:
        """Calculate inventory-related business metrics"""
        
        if item_costs is None:
            item_costs = np.ones_like(y_true)  # Assume unit cost of 1
        
        # Safety stock calculation (simplified)
        forecast_error = np.abs(y_true - y_pred)
        safety_stock = np.percentile(forecast_error, 95)  # 95th percentile service level
        
        # Inventory costs
        excess_inventory = np.maximum(0, y_pred - y_true)
        holding_costs = excess_inventory * item_costs * self.holding_cost_rate
        
        # Stockout costs
        stockouts = np.maximum(0, y_true - y_pred)
        stockout_costs = stockouts * item_costs * self.stockout_cost_multiplier
        
        # Service level
        service_level = np.mean(y_pred >= y_true) * 100
        
        # Inventory turnover
        avg_inventory = (y_pred + safety_stock).mean()
        inventory_turnover = y_true.sum() / avg_inventory if avg_inventory > 0 else 0
        
        return {
            'total_holding_cost': holding_costs.sum(),
            'total_stockout_cost': stockout_costs.sum(),
            'total_inventory_cost': holding_costs.sum() + stockout_costs.sum(),
            'service_level_pct': service_level,
            'safety_stock_units': safety_stock,
            'inventory_turnover': inventory_turnover,
            'avg_inventory_level': avg_inventory
        }
    
    def calculate_forecast_value(self, baseline_y_pred: np.ndarray, improved_y_pred: np.ndarray, 
                               y_true: np.ndarray, annual_revenue: float = 1000000) -> Dict[str, float]:
        """Calculate business value of improved forecasting"""
        
        # Calculate costs for baseline and improved forecasts
        baseline_metrics = self.calculate_inventory_metrics(y_true, baseline_y_pred)
        improved_metrics = self.calculate_inventory_metrics(y_true, improved_y_pred)
        
        # Cost savings
        cost_savings = baseline_metrics['total_inventory_cost'] - improved_metrics['total_inventory_cost']
        cost_savings_pct = (cost_savings / baseline_metrics['total_inventory_cost']) * 100
        
        # Service level improvement
        service_level_improvement = improved_metrics['service_level_pct'] - baseline_metrics['service_level_pct']
        
        # Revenue impact (simplified)
        revenue_impact = service_level_improvement * annual_revenue * 0.01  # 1% service level = 1% revenue
        
        # ROI calculation
        implementation_cost = 50000  # Assumed ML implementation cost
        annual_savings = cost_savings * (365 / len(y_true))  # Annualized
        roi = ((annual_savings + revenue_impact - implementation_cost) / implementation_cost) * 100
        
        return {
            'cost_savings': cost_savings,
            'cost_savings_pct': cost_savings_pct,
            'service_level_improvement': service_level_improvement,
            'revenue_impact': revenue_impact,
            'annual_savings': annual_savings,
            'roi_pct': roi,
            'payback_months': implementation_cost / (annual_savings / 12) if annual_savings > 0 else np.inf
        }


class CrossValidationEvaluator:
    """Time series cross-validation evaluation"""
    
    def __init__(self, n_splits: int = 3, test_size: int = 28):
        self.n_splits = n_splits
        self.test_size = test_size
        
    def time_series_cv_score(self, model, X: pd.DataFrame, y: pd.Series, 
                           date_col: str = 'date') -> Dict[str, List[float]]:
        """Perform time series cross-validation"""
        
        from sklearn.model_selection import TimeSeriesSplit
        
        # Sort by date
        data_df = X.copy()
        data_df[date_col] = pd.to_datetime(data_df[date_col])
        data_df['target'] = y
        data_df = data_df.sort_values(date_col)
        
        # Create time series splits
        tscv = TimeSeriesSplit(n_splits=self.n_splits, test_size=self.test_size)
        
        cv_scores = {
            'rmse': [],
            'mae': [],
            'mape': []
        }
        
        X_sorted = data_df.drop(['target', date_col], axis=1)
        y_sorted = data_df['target']
        
        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_sorted)):
            logger.info(f"Evaluating fold {fold + 1}/{self.n_splits}")
            
            X_train, X_test = X_sorted.iloc[train_idx], X_sorted.iloc[test_idx]
            y_train, y_test = y_sorted.iloc[train_idx], y_sorted.iloc[test_idx]
            
            # Train model
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            metrics = DemandForecastingMetrics()
            cv_scores['rmse'].append(metrics.rmse(y_test.values, y_pred))
            cv_scores['mae'].append(metrics.mae(y_test.values, y_pred))
            cv_scores['mape'].append(metrics.mape(y_test.values, y_pred))
        
        # Calculate summary statistics
        cv_summary = {}
        for metric, scores in cv_scores.items():
            cv_summary[f'{metric}_mean'] = np.mean(scores)
            cv_summary[f'{metric}_std'] = np.std(scores)
            cv_summary[f'{metric}_scores'] = scores
        
        logger.info(f"CV Results - RMSE: {cv_summary['rmse_mean']:.4f} ± {cv_summary['rmse_std']:.4f}")
        
        return cv_summary


def create_residual_plots(y_true: np.ndarray, y_pred: np.ndarray, 
                         model_name: str, save_path: str = None):
    """Create residual analysis plots"""
    
    residuals = y_true - y_pred
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Residual Analysis - {model_name}', fontsize=16)
    
    # Residuals vs Fitted
    axes[0, 0].scatter(y_pred, residuals, alpha=0.6)
    axes[0, 0].axhline(y=0, color='r', linestyle='--')
    axes[0, 0].set_xlabel('Fitted Values')
    axes[0, 0].set_ylabel('Residuals')
    axes[0, 0].set_title('Residuals vs Fitted')
    
    # QQ plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[0, 1])
    axes[0, 1].set_title('Q-Q Plot')
    
    # Histogram of residuals
    axes[1, 0].hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    axes[1, 0].set_xlabel('Residuals')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Histogram of Residuals')
    
    # Actual vs Predicted
    axes[1, 1].scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    axes[1, 1].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    axes[1, 1].set_xlabel('Actual Values')
    axes[1, 1].set_ylabel('Predicted Values')
    axes[1, 1].set_title('Actual vs Predicted')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Residual plots saved to {save_path}")
    
    plt.show()


def main():
    """Test evaluation module"""
    logger.info("Evaluation module loaded successfully")
    
    # Test metrics
    y_true = np.array([10, 15, 12, 8, 20])
    y_pred = np.array([12, 14, 10, 9, 18])
    
    metrics = DemandForecastingMetrics()
    rmse = metrics.rmse(y_true, y_pred)
    mape = metrics.mape(y_true, y_pred)
    
    logger.info(f"Test RMSE: {rmse:.4f}, MAPE: {mape:.2f}%")

if __name__ == "__main__":
    main()
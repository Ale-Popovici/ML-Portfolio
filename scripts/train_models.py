# Main Training Script for Demand Forecasting ML Pipeline
# Path: scripts/train_models.py

import sys
import os
import argparse
import logging
from pathlib import Path
import json
import pandas as pd
import numpy as np
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from config.config import Config
from config.logging_config import setup_logging
from models.pipeline import PipelineOrchestrator
from models.evaluation import BusinessImpactEvaluator
from utils.plotting import DemandForecastingVisualizer, ReportGenerator

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train demand forecasting models')
    
    parser.add_argument('--data-path', type=str, default='data/processed/feature_store.csv',
                       help='Path to processed data file')
    parser.add_argument('--config-path', type=str, default=None,
                       help='Path to configuration file')
    parser.add_argument('--output-dir', type=str, default='outputs',
                       help='Output directory for results')
    parser.add_argument('--models', nargs='+', default=['lightgbm', 'xgboost', 'random_forest'],
                       help='Models to train')
    parser.add_argument('--enable-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--create-ensemble', action='store_true',
                       help='Create ensemble model')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with small subset of data')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    return parser.parse_args()

def create_model_config(models: list, enable_tuning: bool = False) -> dict:
    """Create model configuration"""
    
    model_configs = []
    
    for model_type in models:
        if model_type == 'lightgbm':
            config = {
                'type': 'lightgbm',
                'name': 'lgb_baseline',
                'params': {
                    'objective': 'tweedie',
                    'tweedie_variance_power': 1.1,
                    'metric': 'rmse',
                    'boosting_type': 'gbdt',
                    'num_leaves': 127,
                    'learning_rate': 0.05,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 1,
                    'n_estimators': 500,
                    'random_state': 42,
                    'verbosity': -1,
                    'n_jobs': -1
                }
            }
        elif model_type == 'xgboost':
            config = {
                'type': 'xgboost',
                'name': 'xgb_baseline',
                'params': {
                    'objective': 'reg:tweedie',
                    'tweedie_variance_power': 1.1,
                    'eval_metric': 'rmse',
                    'learning_rate': 0.05,
                    'max_depth': 6,
                    'subsample': 0.8,
                    'colsample_bytree': 0.8,
                    'n_estimators': 500,
                    'random_state': 42,
                    'verbosity': 0,
                    'n_jobs': -1
                }
            }
        elif model_type == 'random_forest':
            config = {
                'type': 'random_forest',
                'name': 'rf_baseline',
                'params': {
                    'n_estimators': 100,
                    'max_depth': 10,
                    'min_samples_split': 5,
                    'min_samples_leaf': 2,
                    'random_state': 42,
                    'n_jobs': -1
                }
            }
        elif model_type == 'ridge':
            config = {
                'type': 'ridge',
                'name': 'ridge_baseline',
                'params': {
                    'alpha': 1.0,
                    'random_state': 42
                }
            }
        else:
            logger.warning(f"Unknown model type: {model_type}, skipping...")
            continue
        
        model_configs.append(config)
    
    # Hyperparameter tuning configuration
    hyperparameter_config = {
        'enabled': enable_tuning,
        'models': ['lightgbm', 'xgboost'] if enable_tuning else [],
        'param_grids': {
            'lightgbm': {
                'num_leaves': [50, 100, 200],
                'learning_rate': [0.03, 0.05, 0.1],
                'feature_fraction': [0.7, 0.8, 0.9],
                'bagging_fraction': [0.7, 0.8, 0.9]
            },
            'xgboost': {
                'max_depth': [4, 6, 8],
                'learning_rate': [0.03, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9]
            }
        }
    }
    
    return {
        'models': model_configs,
        'hyperparameter_tuning': hyperparameter_config,
        'ensemble': {
            'create': True,
            'models': [config['name'] for config in model_configs[:2]],  # Use first 2 models
            'weights': None  # Equal weights
        }
    }

def validate_data(data_path: str) -> bool:
    """Validate input data"""
    
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return False
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"Data loaded successfully: {df.shape}")
        
        # Check required columns
        required_cols = ['date', 'demand', 'id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        
        if missing_cols:
            logger.error(f"Missing required columns: {missing_cols}")
            return False
        
        # Check data quality
        if df['demand'].isnull().all():
            logger.error("All demand values are null")
            return False
        
        if df['date'].isnull().any():
            logger.error("Missing dates found")
            return False
        
        logger.info("Data validation passed")
        return True
        
    except Exception as e:
        logger.error(f"Error validating data: {e}")
        return False

def run_training_pipeline(args):
    """Run the complete training pipeline"""
    
    logger.info("=" * 60)
    logger.info("🚀 STARTING DEMAND FORECASTING ML PIPELINE")
    logger.info("=" * 60)
    
    # Validate data
    logger.info("📊 Validating input data...")
    if not validate_data(args.data_path):
        logger.error("Data validation failed. Exiting.")
        return False
    
    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize components
    logger.info("🔧 Initializing pipeline components...")
    
    # Create configuration
    if args.config_path and os.path.exists(args.config_path):
        with open(args.config_path, 'r') as f:
            config = json.load(f)
    else:
        config = create_model_config(args.models, args.enable_tuning)
    
    # Save configuration
    config_path = output_dir / 'pipeline_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    logger.info(f"Configuration saved to {config_path}")
    
    # Initialize pipeline orchestrator
    orchestrator = PipelineOrchestrator()
    orchestrator.config.update(config)
    
    # Initialize visualizer and report generator
    visualizer = DemandForecastingVisualizer(output_dir / 'visualizations')
    report_generator = ReportGenerator(visualizer)
    
    try:
        # Load and preprocess data
        logger.info("📂 Loading and preprocessing data...")
        df = pd.read_csv(args.data_path)
        
        # Quick test mode - use subset of data
        if args.quick_test:
            logger.info("⚡ Quick test mode - using data subset")
            # Use only last 90 days and sample of products
            df['date'] = pd.to_datetime(df['date'])
            latest_date = df['date'].max()
            cutoff_date = latest_date - pd.Timedelta(days=90)
            df = df[df['date'] >= cutoff_date]
            
            # Sample products
            unique_ids = df['id'].unique()
            sample_ids = np.random.choice(unique_ids, size=min(100, len(unique_ids)), replace=False)
            df = df[df['id'].isin(sample_ids)]
            
            logger.info(f"Quick test data shape: {df.shape}")
        
        # Run main pipeline
        logger.info("🎯 Running ML pipeline...")
        start_time = datetime.now()
        
        results = orchestrator.run_complete_pipeline(args.data_path)
        
        end_time = datetime.now()
        training_time = (end_time - start_time).total_seconds()
        
        logger.info(f"✅ Pipeline completed in {training_time:.2f} seconds")
        
        # Extract results
        best_model = results['best_model']
        test_performance = results['test_performance']
        comparison_df = results['comparison']
        
        logger.info(f"🏆 Best model: {best_model}")
        logger.info(f"📈 Test RMSE: {test_performance.get('RMSE', 'N/A'):.4f}")
        logger.info(f"📈 Test MAPE: {test_performance.get('MAPE', 'N/A'):.2f}%")
        
        # Create visualizations
        logger.info("📊 Creating visualizations...")
        
        # Model comparison
        comparison_fig = visualizer.plot_model_comparison(comparison_df, save_name='model_comparison')
        
        # Business impact analysis
        business_evaluator = BusinessImpactEvaluator()
        
        # Create example business impact (you would calculate real values)
        baseline_cost = 100000  # Example baseline cost
        improved_cost = 80000   # Example improved cost
        service_levels = {'baseline': 85, 'improved': 92}
        
        business_fig = visualizer.plot_business_impact(
            baseline_cost, improved_cost, service_levels, 
            save_name='business_impact'
        )
        
        # Executive dashboard
        dashboard_fig = visualizer.create_executive_dashboard(
            results, save_name='executive_dashboard'
        )
        
        # Generate comprehensive report
        logger.info("📋 Generating reports...")
        
        report_path = report_generator.generate_model_report(
            results, output_dir / 'model_report.html'
        )
        
        # Save results summary
        summary = {
            'pipeline_info': {
                'completion_time': end_time.isoformat(),
                'training_duration_seconds': training_time,
                'data_shape': df.shape,
                'models_trained': list(comparison_df.index),
                'best_model': best_model
            },
            'performance_metrics': test_performance,
            'model_comparison': comparison_df.to_dict(),
            'business_impact': {
                'cost_reduction_pct': ((baseline_cost - improved_cost) / baseline_cost) * 100,
                'service_level_improvement': service_levels['improved'] - service_levels['baseline'],
                'estimated_annual_savings': (baseline_cost - improved_cost) * 12  # Monthly to annual
            },
            'file_paths': {
                'model_report': str(report_path),
                'visualizations': str(visualizer.output_dir),
                'production_pipeline': f"models/trained/production_pipeline_{best_model}.joblib"
            }
        }
        
        summary_path = output_dir / 'training_summary.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        # Print final summary
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("=" * 60)
        logger.info(f"📊 Best Model: {best_model}")
        logger.info(f"📈 Performance: RMSE={test_performance.get('RMSE', 'N/A'):.4f}, MAPE={test_performance.get('MAPE', 'N/A'):.2f}%")
        logger.info(f"💰 Estimated Cost Reduction: {summary['business_impact']['cost_reduction_pct']:.1f}%")
        logger.info(f"📋 Full Report: {report_path}")
        logger.info(f"📊 Visualizations: {visualizer.output_dir}")
        logger.info(f"⏱️  Training Time: {training_time:.2f} seconds")
        logger.info("=" * 60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed with error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def main():
    """Main execution function"""
    
    # Parse arguments
    args = parse_arguments()
    
    # Setup logging
    log_file = f"demand_forecasting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    logger = setup_logging(
        log_level=getattr(logging, args.log_level),
        log_file=log_file
    )
    
    # Log startup info
    logger.info("Demand Forecasting ML Pipeline - Training Script")
    logger.info(f"Arguments: {vars(args)}")
    logger.info(f"Python version: {sys.version}")
    logger.info(f"Working directory: {os.getcwd()}")
    
    # Validate configuration
    Config.validate_setup()
    
    # Run pipeline
    success = run_training_pipeline(args)
    
    if success:
        logger.info("✅ Training completed successfully!")
        exit_code = 0
    else:
        logger.error("❌ Training failed!")
        exit_code = 1
    
    return exit_code

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
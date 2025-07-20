# Main Training Script for M5 Demand Forecasting
# File: scripts/train_models.py

import sys
import os
import argparse
import logging
import json
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Dict, List, Tuple, Optional, Any
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import our modules
from config.config import get_config, initialize_config
from features.feature_engineering import AdvancedFeatureEngineer
from models.ml_models import ModelTrainer, ModelFactory, HyperparameterTuner, EnsembleModel

# Setup logging
def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration"""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('training.log')
        ]
    )

logger = logging.getLogger(__name__)

# Simple evaluator for now
class SimpleModelEvaluator:
    """Simple model evaluator for basic metrics"""
    
    def evaluate_model(self, y_true, y_pred, model_name="model"):
        """Basic model evaluation"""
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Zero prediction metrics
        zero_true = (y_true == 0)
        zero_pred = (y_pred < 0.5)
        zero_precision = (zero_true & zero_pred).sum() / zero_pred.sum() if zero_pred.sum() > 0 else 0
        zero_recall = (zero_true & zero_pred).sum() / zero_true.sum() if zero_true.sum() > 0 else 0
        zero_f1 = 2 * zero_precision * zero_recall / (zero_precision + zero_recall) if (zero_precision + zero_recall) > 0 else 0
        
        return {
            'model_name': model_name,
            'metrics': {
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'zero_f1': zero_f1
            },
            'zero_rate_actual': zero_true.mean(),
            'zero_rate_predicted': (y_pred == 0).mean()
        }
    
    def evaluate_by_category(self, y_true, y_pred, categories):
        """Simple category evaluation"""
        results = {}
        for cat in np.unique(categories):
            mask = categories == cat
            if mask.sum() > 0:
                cat_result = self.evaluate_model(y_true[mask], y_pred[mask], f"category_{cat}")
                results[cat] = cat_result
        return results

class M5TrainingPipeline:
    """
    Complete training pipeline for M5 demand forecasting
    Handles your intermittent demand patterns (68.2% zeros)
    """
    
    def __init__(self, config_path: str = None):
        self.config = initialize_config(config_path)
        self.feature_engineer = AdvancedFeatureEngineer(self.config.get_full_config_dict())
        self.trainer = ModelTrainer(self.config.get_full_config_dict())
        self.evaluator = SimpleModelEvaluator()
        
        # Create directories
        self.config.create_directories()
        
        # Training data containers
        self.raw_data = {}
        self.processed_data = {}
        self.features_data = {}
        self.results = {}
    
    def load_data(self) -> Dict[str, pd.DataFrame]:
        """Load M5 competition data"""
        logger.info("Loading M5 competition data...")
        
        try:
            # Load main datasets - check multiple possible locations
            possible_paths = [
                # Standard location
                (self.config.data.raw_data_dir / self.config.data.sales_file,
                 self.config.data.raw_data_dir / self.config.data.calendar_file,
                 self.config.data.raw_data_dir / self.config.data.prices_file),
                # Alternative location in src folder
                (Path("src/data/raw") / self.config.data.sales_file,
                 Path("src/data/raw") / self.config.data.calendar_file,
                 Path("src/data/raw") / self.config.data.prices_file),
                # Current directory data folder
                (Path("data/raw") / self.config.data.sales_file,
                 Path("data/raw") / self.config.data.calendar_file,
                 Path("data/raw") / self.config.data.prices_file)
            ]
            
            sales_path = calendar_path = prices_path = None
            
            for paths in possible_paths:
                if all(p.exists() for p in paths):
                    sales_path, calendar_path, prices_path = paths
                    logger.info(f"Found data files in: {sales_path.parent}")
                    break
            
            if not all([sales_path, calendar_path, prices_path]):
                logger.error("M5 data files not found in any expected location:")
                for i, paths in enumerate(possible_paths):
                    logger.error(f"  Location {i+1}: {paths[0].parent}")
                    for p in paths:
                        logger.error(f"    {p.name}: {'EXISTS' if p.exists() else 'MISSING'}")
                raise FileNotFoundError("M5 data files not found. Please ensure data is in data/raw/ or src/data/raw/")
            
            # Load sales data
            logger.info("Loading sales data...")
            sales_df = pd.read_csv(sales_path)
            
            # Convert to long format (like your EDA)
            logger.info("Converting sales data to long format...")
            id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
            sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
            
            sales_long = pd.melt(
                sales_df,
                id_vars=id_cols,
                value_vars=sales_cols,
                var_name='d',
                value_name='demand'
            )
            sales_long['day_num'] = sales_long['d'].str.extract(r'(\d+)').astype(int)
            
            # Load calendar and prices
            logger.info("Loading calendar and prices data...")
            calendar_df = pd.read_csv(calendar_path)
            prices_df = pd.read_csv(prices_path)
            
            self.raw_data = {
                'sales': sales_long,
                'calendar': calendar_df,
                'prices': prices_df
            }
            
            logger.info(f"Data loaded successfully!")
            logger.info(f"Sales shape: {sales_long.shape}")
            logger.info(f"Calendar shape: {calendar_df.shape}")
            logger.info(f"Prices shape: {prices_df.shape}")
            
            return self.raw_data
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def create_features(self, quick_test: bool = False) -> pd.DataFrame:
        """Create features using the advanced feature engineering"""
        logger.info("Creating features...")
        
        sales_df = self.raw_data['sales']
        calendar_df = self.raw_data['calendar']
        prices_df = self.raw_data['prices']
        
        # Quick test mode - use subset of data
        if quick_test:
            logger.info("Quick test mode - using subset of data")
            # Use last 100 days and sample of items
            max_day = sales_df['day_num'].max()
            sales_df = sales_df[sales_df['day_num'] >= max_day - 100]
            
            # Sample items
            sample_items = sales_df['item_id'].unique()[:100]
            sales_df = sales_df[sales_df['item_id'].isin(sample_items)]
        
        # Create features
        features_df = self.feature_engineer.create_all_features(
            sales_df, calendar_df, prices_df, target_col='demand'
        )
        
        logger.info(f"Features created. Shape: {features_df.shape}")
        
        # Save features
        features_path = self.config.data.features_data_dir / "feature_store.csv"
        features_df.to_csv(features_path, index=False)
        logger.info(f"Features saved to {features_path}")
        
        self.features_data = features_df
        return features_df
    
    def split_data(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Split data for time series validation (M5 standard)"""
        logger.info("Splitting data for time series validation...")
        
        # Sort by date to ensure proper time series split
        features_df = features_df.sort_values(['item_id', 'store_id', 'date'])
        
        # M5 standard: use last 28 days for test, previous 28 for validation
        max_date = features_df['date'].max()
        test_start = max_date - pd.Timedelta(days=self.config.model.test_size - 1)
        val_start = test_start - pd.Timedelta(days=self.config.model.validation_size)
        
        # Create splits
        train_df = features_df[features_df['date'] < val_start].copy()
        val_df = features_df[
            (features_df['date'] >= val_start) & 
            (features_df['date'] < test_start)
        ].copy()
        test_df = features_df[features_df['date'] >= test_start].copy()
        
        logger.info(f"Data split completed:")
        logger.info(f"  Train: {train_df.shape} ({train_df['date'].min()} to {train_df['date'].max()})")
        logger.info(f"  Val:   {val_df.shape} ({val_df['date'].min()} to {val_df['date'].max()})")
        logger.info(f"  Test:  {test_df.shape} ({test_df['date'].min()} to {test_df['date'].max()})")
        
        return train_df, val_df, test_df
    
    def prepare_model_data(self, train_df: pd.DataFrame, val_df: pd.DataFrame, 
                          test_df: pd.DataFrame) -> Tuple[pd.DataFrame, ...]:
        """Prepare data for modeling"""
        logger.info("Preparing data for modeling...")
        
        # Use feature engineer to prepare features
        X_train, y_train = self.feature_engineer.prepare_features_for_modeling(
            train_df, self.config.model.target_col
        )
        X_val, y_val = self.feature_engineer.prepare_features_for_modeling(
            val_df, self.config.model.target_col
        )
        X_test, y_test = self.feature_engineer.prepare_features_for_modeling(
            test_df, self.config.model.target_col
        )
        
        logger.info(f"Model data prepared:")
        logger.info(f"  Features: {X_train.shape[1]}")
        logger.info(f"  Train samples: {X_train.shape[0]}")
        logger.info(f"  Val samples: {X_val.shape[0]}")
        logger.info(f"  Test samples: {X_test.shape[0]}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame, y_val: pd.Series,
                    models_to_train: List[str], enable_tuning: bool = False) -> Dict:
        """Train the specified models"""
        logger.info(f"Training models: {models_to_train}")
        
        trained_models = {}
        
        for model_type in models_to_train:
            logger.info(f"Training {model_type}...")
            
            # Get base parameters
            base_params = self.config.model_params.get(model_type, {})
            
            # Hyperparameter tuning if enabled
            if enable_tuning and model_type in ['lightgbm', 'xgboost']:
                logger.info(f"Tuning hyperparameters for {model_type}...")
                tuner = HyperparameterTuner(model_type, n_trials=50)
                best_params = tuner.tune(X_train, y_train)
                
                # Merge with base parameters
                base_params.update(best_params)
                logger.info(f"Best parameters: {best_params}")
            
            # Train model
            model = self.trainer.train_single_model(
                model_type, X_train, y_train, X_val, y_val, 
                params=base_params, model_name=model_type
            )
            
            trained_models[model_type] = model
        
        return trained_models
    
    def create_ensemble(self, trained_models: Dict, 
                       X_val: pd.DataFrame, y_val: pd.Series) -> EnsembleModel:
        """Create ensemble model with optimized weights"""
        logger.info("Creating ensemble model...")
        
        # Use config weights or optimize based on validation performance
        if hasattr(self.config.model, 'ensemble_weights'):
            weights = self.config.model.ensemble_weights
        else:
            # Simple equal weighting
            weights = {name: 1.0/len(trained_models) for name in trained_models.keys()}
        
        ensemble = self.trainer.create_ensemble(
            list(trained_models.keys()), weights
        )
        
        return ensemble
    
    def evaluate_models(self, models: Dict, X_test: pd.DataFrame, y_test: pd.Series,
                       test_df: pd.DataFrame) -> Dict:
        """Evaluate all models"""
        logger.info("Evaluating models...")
        
        results = {}
        
        for name, model in models.items():
            logger.info(f"Evaluating {name}...")
            
            # Make predictions
            predictions = model.predict(X_test)
            
            # Evaluate
            model_results = self.evaluator.evaluate_model(
                y_test, predictions, model_name=name
            )
            
            # Add business metrics (using test_df for additional context)
            if 'cat_id' in test_df.columns:
                category_results = self.evaluator.evaluate_by_category(
                    y_test, predictions, test_df['cat_id']
                )
                model_results['category_performance'] = category_results
            
            results[name] = model_results
        
        return results
    
    def save_results(self, results: Dict, trained_models: Dict):
        """Save training results and models"""
        logger.info("Saving results...")
        
        # Save models
        self.trainer.save_models(str(self.config.model.trained_models_dir))
        
        # Save results
        import json
        results_path = self.config.output.outputs_dir / "training_results.json"
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        logger.info(f"Results saved to {results_path}")
        
        return results_path
    
    def run_complete_pipeline(self, models_to_train: List[str] = None,
                            enable_tuning: bool = False, 
                            create_ensemble: bool = True,
                            quick_test: bool = False) -> Dict:
        """Run the complete training pipeline"""
        
        logger.info("Starting M5 Demand Forecasting Training Pipeline")
        logger.info("=" * 60)
        
        try:
            # Default models based on your EDA insights
            if models_to_train is None:
                models_to_train = ['lightgbm', 'xgboost', 'random_forest']
            
            # Step 1: Load data
            logger.info("Step 1: Loading data...")
            self.load_data()
            
            # Step 2: Create features
            logger.info("Step 2: Creating features...")
            features_df = self.create_features(quick_test=quick_test)
            
            # Step 3: Split data
            logger.info("Step 3: Splitting data...")
            train_df, val_df, test_df = self.split_data(features_df)
            
            # Step 4: Prepare model data
            logger.info("Step 4: Preparing model data...")
            X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_model_data(
                train_df, val_df, test_df
            )
            
            # Step 5: Train models
            logger.info("Step 5: Training models...")
            trained_models = self.train_models(
                X_train, y_train, X_val, y_val, models_to_train, enable_tuning
            )
            
            # Step 6: Create ensemble
            ensemble_model = None
            if create_ensemble and len(trained_models) > 1:
                logger.info("Step 6: Creating ensemble...")
                ensemble_model = self.create_ensemble(trained_models, X_val, y_val)
                trained_models['ensemble'] = ensemble_model
            
            # Step 7: Evaluate models
            logger.info("Step 7: Evaluating models...")
            results = self.evaluate_models(trained_models, X_test, y_test, test_df)
            
            # Step 8: Save results
            logger.info("Step 8: Saving results...")
            self.save_results(results, trained_models)
            
            # Summary
            logger.info("Training pipeline completed successfully!")
            logger.info("=" * 60)
            
            # Print summary
            self.print_summary(results)
            
            return {
                'models': trained_models,
                'results': results,
                'feature_engineer': self.feature_engineer
            }
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            raise
    
    def print_summary(self, results: Dict):
        """Print training summary"""
        print("\nTRAINING SUMMARY")
        print("=" * 40)
        
        for model_name, model_results in results.items():
            metrics = model_results.get('metrics', {})
            rmse = metrics.get('rmse', 'N/A')
            mae = metrics.get('mae', 'N/A')
            
            print(f"{model_name.upper()}")
            print(f"   RMSE: {rmse}")
            print(f"   MAE:  {mae}")
        
        # Find best model
        best_model = min(results.keys(), 
                        key=lambda x: results[x].get('metrics', {}).get('rmse', float('inf')))
        best_rmse = results[best_model]['metrics']['rmse']
        
        print(f"\nBEST MODEL: {best_model.upper()} (RMSE: {best_rmse:.4f})")
        print("=" * 40)


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Train M5 Demand Forecasting Models')
    
    parser.add_argument('--config', type=str, help='Path to configuration file')
    parser.add_argument('--models', nargs='+', default=['lightgbm', 'xgboost', 'random_forest'],
                       help='Models to train')
    parser.add_argument('--enable-tuning', action='store_true',
                       help='Enable hyperparameter tuning')
    parser.add_argument('--create-ensemble', action='store_true', default=True,
                       help='Create ensemble model')
    parser.add_argument('--quick-test', action='store_true',
                       help='Quick test with subset of data')
    parser.add_argument('--log-level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       help='Logging level')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    try:
        # Initialize pipeline
        pipeline = M5TrainingPipeline(args.config)
        
        # Run pipeline
        results = pipeline.run_complete_pipeline(
            models_to_train=args.models,
            enable_tuning=args.enable_tuning,
            create_ensemble=args.create_ensemble,
            quick_test=args.quick_test
        )
        
        print("\nTraining completed successfully!")
        print("Check outputs/ directory for results and models/trained/ for saved models")
        
        return 0
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
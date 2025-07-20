# ML Pipeline for Demand Forecasting
# Path: src/models/pipeline.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
import joblib
import os
from pathlib import Path

# Scikit-learn
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

# Our modules
from src.features.engineering import AdvancedFeatureEngineer, TimeSeriesSplitter
from src.models.ml_models import ModelFactory, BaseModel
from src.models.evaluation import ModelEvaluator, CrossValidationEvaluator

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class DemandForecastingTransformer(BaseEstimator, TransformerMixin):
    """Custom transformer for demand forecasting preprocessing"""
    
    def __init__(self, feature_engineer: AdvancedFeatureEngineer = None, 
                 target_col: str = 'demand', date_col: str = 'date'):
        self.feature_engineer = feature_engineer or AdvancedFeatureEngineer()
        self.target_col = target_col
        self.date_col = date_col
        self.feature_names_out_ = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series = None):
        """Fit the transformer"""
        logger.info("Fitting DemandForecastingTransformer...")
        
        # Create a copy with target for feature engineering
        if y is not None:
            X_with_target = X.copy()
            X_with_target[self.target_col] = y
        else:
            X_with_target = X.copy()
        
        # Fit feature engineer
        X_transformed = self.feature_engineer.create_all_features(X_with_target, self.target_col)
        
        # Prepare features for modeling
        X_features, feature_names = self.feature_engineer.prepare_features_for_modeling(
            X_transformed, self.target_col
        )
        
        self.feature_names_out_ = feature_names
        
        logger.info(f"Fitted transformer with {len(feature_names)} features")
        return self
    
    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Transform the data"""
        if self.feature_names_out_ is None:
            raise ValueError("Transformer must be fitted before transform")
        
        # Create features
        X_transformed = self.feature_engineer.create_all_features(X, self.target_col)
        
        # Prepare features for modeling
        X_features, _ = self.feature_engineer.prepare_features_for_modeling(
            X_transformed, self.target_col
        )
        
        # Ensure same columns as fitted
        missing_cols = set(self.feature_names_out_) - set(X_features.columns)
        extra_cols = set(X_features.columns) - set(self.feature_names_out_)
        
        # Add missing columns with zeros
        for col in missing_cols:
            X_features[col] = 0
        
        # Remove extra columns
        X_features = X_features[self.feature_names_out_]
        
        return X_features
    
    def get_feature_names_out(self, input_features=None):
        """Get output feature names"""
        return self.feature_names_out_


class ModelWrapper(BaseEstimator, RegressorMixin):
    """Wrapper to make our custom models compatible with sklearn Pipeline"""
    
    def __init__(self, model: BaseModel):
        self.model = model
        
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs):
        """Fit the wrapped model"""
        self.model.fit(X, y, **kwargs)
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """Get parameters"""
        return self.model.params
    
    def set_params(self, **params):
        """Set parameters"""
        self.model.params.update(params)
        return self


class DemandForecastingPipeline:
    """Complete ML pipeline for demand forecasting"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Components
        self.feature_engineer = None
        self.preprocessing_pipeline = None
        self.model_pipeline = None
        self.fitted_models = {}
        self.best_model = None
        self.evaluator = ModelEvaluator()
        
        # Configuration
        self.target_col = self.config.get('target_col', 'demand')
        self.date_col = self.config.get('date_col', 'date')
        self.test_size = self.config.get('test_size', 28)
        self.validation_size = self.config.get('validation_size', 28)
        
    def create_preprocessing_pipeline(self) -> Pipeline:
        """Create preprocessing pipeline"""
        logger.info("Creating preprocessing pipeline...")
        
        # Feature engineering transformer
        self.feature_engineer = AdvancedFeatureEngineer()
        feature_transformer = DemandForecastingTransformer(
            self.feature_engineer, self.target_col, self.date_col
        )
        
        # Scaling (optional)
        if self.config.get('scale_features', False):
            scaler = RobustScaler()  # More robust to outliers than StandardScaler
        else:
            scaler = 'passthrough'
        
        # Create pipeline
        self.preprocessing_pipeline = Pipeline([
            ('feature_engineering', feature_transformer),
            ('scaling', scaler)
        ])
        
        logger.info("Preprocessing pipeline created")
        return self.preprocessing_pipeline
    
    def create_model_pipeline(self, model_config: Dict) -> Pipeline:
        """Create complete model pipeline"""
        logger.info(f"Creating model pipeline for {model_config['type']}...")
        
        # Create preprocessing pipeline if not exists
        if self.preprocessing_pipeline is None:
            self.create_preprocessing_pipeline()
        
        # Create model
        model = ModelFactory.create_model(
            model_config['type'], 
            model_config.get('params', {})
        )
        
        # Wrap model for sklearn compatibility
        wrapped_model = ModelWrapper(model)
        
        # Create complete pipeline
        pipeline = Pipeline([
            ('preprocessing', self.preprocessing_pipeline),
            ('model', wrapped_model)
        ])
        
        self.model_pipeline = pipeline
        logger.info("Model pipeline created")
        return pipeline
    
    def train_single_model(self, model_config: Dict, X_train: pd.DataFrame, 
                          y_train: pd.Series, X_val: pd.DataFrame = None, 
                          y_val: pd.Series = None) -> Pipeline:
        """Train a single model"""
        logger.info(f"Training {model_config['type']} model...")
        
        # Create pipeline
        pipeline = self.create_model_pipeline(model_config)
        
        # Prepare validation data for early stopping (if supported)
        fit_params = {}
        if X_val is not None and y_val is not None:
            # Transform validation data for early stopping
            X_val_transformed = pipeline[:-1].fit_transform(X_train)  # Fit on train, transform val
            X_val_transformed = pipeline[:-1].transform(X_val)
            
            # Set validation data for models that support it
            if model_config['type'] in ['lightgbm', 'xgboost']:
                fit_params = {
                    'model__X_val': X_val_transformed,
                    'model__y_val': y_val
                }
        
        # Train pipeline
        pipeline.fit(X_train, y_train, **fit_params)
        
        # Store fitted model
        model_name = model_config.get('name', model_config['type'])
        self.fitted_models[model_name] = pipeline
        
        logger.info(f"{model_name} training completed")
        return pipeline
    
    def train_multiple_models(self, model_configs: List[Dict], 
                             X_train: pd.DataFrame, y_train: pd.Series,
                             X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Pipeline]:
        """Train multiple models"""
        logger.info(f"Training {len(model_configs)} models...")
        
        trained_models = {}
        
        for model_config in model_configs:
            try:
                model_name = model_config.get('name', model_config['type'])
                
                # Train model
                pipeline = self.train_single_model(
                    model_config, X_train, y_train, X_val, y_val
                )
                
                trained_models[model_name] = pipeline
                
                # Evaluate on validation set if available
                if X_val is not None and y_val is not None:
                    y_pred = pipeline.predict(X_val)
                    self.evaluator.evaluate_model(
                        y_val.values, y_pred, model_name, y_train.values
                    )
                
            except Exception as e:
                logger.error(f"Failed to train {model_config['type']}: {e}")
                continue
        
        self.fitted_models.update(trained_models)
        logger.info(f"Successfully trained {len(trained_models)} models")
        
        return trained_models
    
    def hyperparameter_tuning(self, model_config: Dict, param_grid: Dict,
                             X_train: pd.DataFrame, y_train: pd.Series,
                             cv_strategy: str = 'time_series', n_splits: int = 3) -> Pipeline:
        """Perform hyperparameter tuning"""
        logger.info(f"Tuning hyperparameters for {model_config['type']}...")
        
        # Create base pipeline
        base_pipeline = self.create_model_pipeline(model_config)
        
        # Prepare cross-validation
        if cv_strategy == 'time_series':
            from sklearn.model_selection import TimeSeriesSplit
            cv = TimeSeriesSplit(n_splits=n_splits, test_size=self.test_size)
        else:
            cv = n_splits
        
        # Create parameter grid with correct pipeline naming
        pipeline_param_grid = {}
        for param, values in param_grid.items():
            pipeline_param_grid[f'model__{param}'] = values
        
        # Perform grid search
        grid_search = GridSearchCV(
            base_pipeline,
            pipeline_param_grid,
            cv=cv,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        grid_search.fit(X_train, y_train)
        
        # Store best model
        model_name = f"{model_config.get('name', model_config['type'])}_tuned"
        self.fitted_models[model_name] = grid_search.best_estimator_
        
        logger.info(f"Best parameters: {grid_search.best_params_}")
        logger.info(f"Best CV score: {-grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def create_ensemble_pipeline(self, model_names: List[str], 
                                weights: List[float] = None) -> Pipeline:
        """Create ensemble pipeline"""
        logger.info(f"Creating ensemble from {len(model_names)} models...")
        
        if not all(name in self.fitted_models for name in model_names):
            missing_models = [name for name in model_names if name not in self.fitted_models]
            raise ValueError(f"Models not found: {missing_models}")
        
        # Create ensemble model config
        ensemble_config = {
            'type': 'ensemble',
            'models': [self.fitted_models[name] for name in model_names],
            'weights': weights
        }
        
        # For now, create a simple ensemble predictor
        class EnsemblePipeline:
            def __init__(self, models, weights=None):
                self.models = models
                self.weights = weights or [1.0] * len(models)
                self.weights = np.array(self.weights) / np.sum(self.weights)
            
            def predict(self, X):
                predictions = []
                for model in self.models:
                    pred = model.predict(X)
                    predictions.append(pred)
                
                predictions = np.array(predictions)
                return np.average(predictions, axis=0, weights=self.weights)
            
            def fit(self, X, y):
                # Models are already fitted
                return self
        
        ensemble = EnsemblePipeline(
            [self.fitted_models[name] for name in model_names],
            weights
        )
        
        ensemble_name = f"ensemble_{'_'.join(model_names)}"
        self.fitted_models[ensemble_name] = ensemble
        
        logger.info(f"Ensemble pipeline created: {ensemble_name}")
        return ensemble
    
    def select_best_model(self, metric: str = 'RMSE') -> Tuple[str, Pipeline]:
        """Select best model based on evaluation metric"""
        
        comparison_df = self.evaluator.compare_models()
        
        if comparison_df.empty:
            raise ValueError("No models have been evaluated")
        
        # Select best model (lowest for most metrics, highest for R²)
        if metric == 'R²':
            best_model_name = comparison_df[metric].idxmax()
        else:
            best_model_name = comparison_df[metric].idxmin()
        
        self.best_model = self.fitted_models[best_model_name]
        
        logger.info(f"Best model selected: {best_model_name} ({metric}: {comparison_df.loc[best_model_name, metric]:.4f})")
        
        return best_model_name, self.best_model
    
    def save_pipeline(self, pipeline: Pipeline, filepath: str):
        """Save pipeline to disk"""
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        joblib.dump(pipeline, filepath)
        logger.info(f"Pipeline saved to {filepath}")
    
    def load_pipeline(self, filepath: str) -> Pipeline:
        """Load pipeline from disk"""
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Pipeline file not found: {filepath}")
        
        pipeline = joblib.load(filepath)
        logger.info(f"Pipeline loaded from {filepath}")
        
        return pipeline
    
    def create_production_pipeline(self, best_model_name: str = None) -> Pipeline:
        """Create production-ready pipeline"""
        
        if best_model_name is None:
            if self.best_model is None:
                best_model_name, _ = self.select_best_model()
            else:
                # Find best model name
                for name, model in self.fitted_models.items():
                    if model == self.best_model:
                        best_model_name = name
                        break
        
        production_pipeline = self.fitted_models[best_model_name]
        
        # Save production pipeline
        production_path = f"models/trained/production_pipeline_{best_model_name}.joblib"
        self.save_pipeline(production_pipeline, production_path)
        
        # Save metadata
        metadata = {
            'model_name': best_model_name,
            'creation_date': pd.Timestamp.now().isoformat(),
            'performance_metrics': self.evaluator.evaluation_results.get(best_model_name, {}),
            'features': getattr(production_pipeline.named_steps['preprocessing'].named_steps['feature_engineering'], 'feature_names_out_', [])
        }
        
        metadata_path = f"models/trained/production_metadata_{best_model_name}.json"
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Production pipeline created and saved: {production_path}")
        
        return production_pipeline


class PipelineOrchestrator:
    """Orchestrate the complete ML pipeline workflow"""
    
    def __init__(self, config_path: str = None):
        self.config = self._load_config(config_path)
        self.pipeline = DemandForecastingPipeline(self.config)
        self.splitter = TimeSeriesSplitter(
            test_size=self.config.get('test_size', 28),
            validation_size=self.config.get('validation_size', 28)
        )
        
    def _load_config(self, config_path: str) -> Dict:
        """Load configuration"""
        default_config = {
            'models': [
                {
                    'type': 'lightgbm',
                    'name': 'lgb_default',
                    'params': {}
                },
                {
                    'type': 'xgboost', 
                    'name': 'xgb_default',
                    'params': {}
                },
                {
                    'type': 'random_forest',
                    'name': 'rf_default',
                    'params': {}
                }
            ],
            'ensemble': {
                'create': True,
                'models': ['lgb_default', 'xgb_default'],
                'weights': [0.6, 0.4]
            },
            'hyperparameter_tuning': {
                'enabled': True,
                'models': ['lightgbm'],
                'param_grids': {
                    'lightgbm': {
                        'num_leaves': [50, 100, 200],
                        'learning_rate': [0.01, 0.03, 0.1],
                        'feature_fraction': [0.8, 0.9]
                    }
                }
            }
        }
        
        if config_path and os.path.exists(config_path):
            import json
            with open(config_path, 'r') as f:
                loaded_config = json.load(f)
            default_config.update(loaded_config)
        
        return default_config
    
    def run_complete_pipeline(self, data_path: str) -> Dict[str, Any]:
        """Run the complete ML pipeline"""
        logger.info("Starting complete ML pipeline...")
        
        # Load data
        logger.info(f"Loading data from {data_path}")
        df = pd.read_csv(data_path)
        
        # Split data
        train_df, val_df, test_df = self.splitter.split_by_time(df)
        
        # Prepare features and targets
        X_train = train_df.drop([self.config.get('target_col', 'demand')], axis=1)
        y_train = train_df[self.config.get('target_col', 'demand')]
        X_val = val_df.drop([self.config.get('target_col', 'demand')], axis=1)
        y_val = val_df[self.config.get('target_col', 'demand')]
        X_test = test_df.drop([self.config.get('target_col', 'demand')], axis=1)
        y_test = test_df[self.config.get('target_col', 'demand')]
        
        # Train models
        logger.info("Training baseline models...")
        trained_models = self.pipeline.train_multiple_models(
            self.config['models'], X_train, y_train, X_val, y_val
        )
        
        # Hyperparameter tuning
        if self.config.get('hyperparameter_tuning', {}).get('enabled', False):
            logger.info("Performing hyperparameter tuning...")
            for model_type in self.config['hyperparameter_tuning']['models']:
                if model_type in self.config['hyperparameter_tuning']['param_grids']:
                    model_config = {'type': model_type, 'name': f'{model_type}_baseline'}
                    param_grid = self.config['hyperparameter_tuning']['param_grids'][model_type]
                    
                    tuned_model = self.pipeline.hyperparameter_tuning(
                        model_config, param_grid, X_train, y_train
                    )
        
        # Create ensemble
        if self.config.get('ensemble', {}).get('create', False):
            logger.info("Creating ensemble model...")
            ensemble_models = self.config['ensemble']['models']
            ensemble_weights = self.config['ensemble'].get('weights')
            
            available_models = [m for m in ensemble_models if m in self.pipeline.fitted_models]
            if len(available_models) >= 2:
                ensemble = self.pipeline.create_ensemble_pipeline(available_models, ensemble_weights)
                
                # Evaluate ensemble
                y_pred_ensemble = ensemble.predict(X_val)
                self.pipeline.evaluator.evaluate_model(
                    y_val.values, y_pred_ensemble, 'ensemble', y_train.values
                )
        
        # Select best model
        best_model_name, best_model = self.pipeline.select_best_model()
        
        # Final evaluation on test set
        logger.info("Final evaluation on test set...")
        y_pred_test = best_model.predict(X_test)
        test_results = self.pipeline.evaluator.evaluate_model(
            y_test.values, y_pred_test, f'{best_model_name}_test', y_train.values
        )
        
        # Create production pipeline
        production_pipeline = self.pipeline.create_production_pipeline(best_model_name)
        
        # Generate reports
        report_path = self.pipeline.evaluator.create_evaluation_report()
        
        results = {
            'best_model': best_model_name,
            'test_performance': test_results,
            'all_results': self.pipeline.evaluator.evaluation_results,
            'comparison': self.pipeline.evaluator.compare_models(),
            'production_pipeline': production_pipeline,
            'report_path': report_path
        }
        
        logger.info("Complete ML pipeline finished successfully!")
        
        return results


def main():
    """Test pipeline module"""
    logger.info("Pipeline module loaded successfully")

if __name__ == "__main__":
    main()
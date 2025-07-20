# ML Models and Training Pipeline for M5 Demand Forecasting
# File: src/models/ml_models.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
import logging
from abc import ABC, abstractmethod
from pathlib import Path
import joblib
import json
from datetime import datetime
import warnings

# ML Libraries
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
import optuna

# Statistical Models
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    warnings.warn("Prophet not available. Install with: pip install prophet")

logger = logging.getLogger(__name__)

class BaseModel(ABC):
    """Abstract base class for all forecasting models"""
    
    def __init__(self, name: str, params: Dict = None):
        self.name = name
        self.params = params or {}
        self.model = None
        self.is_fitted = False
        self.feature_importance_ = None
        self.training_history = {}
        
    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Fit the model"""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        pass
    
    def save(self, path: str):
        """Save the model"""
        model_data = {
            'name': self.name,
            'params': self.params,
            'model': self.model,
            'is_fitted': self.is_fitted,
            'feature_importance_': self.feature_importance_,
            'training_history': self.training_history
        }
        joblib.dump(model_data, path)
        logger.info(f"Model saved to {path}")
    
    @classmethod
    def load(cls, path: str):
        """Load a saved model"""
        model_data = joblib.load(path)
        instance = cls(model_data['name'], model_data['params'])
        instance.model = model_data['model']
        instance.is_fitted = model_data['is_fitted']
        instance.feature_importance_ = model_data.get('feature_importance_')
        instance.training_history = model_data.get('training_history', {})
        return instance


class LightGBMModel(BaseModel):
    """
    LightGBM model optimized for intermittent demand forecasting.
    Handles the 68.2% zero sales pattern from your EDA.
    """
    
    def __init__(self, name: str = "lightgbm", params: Dict = None):
        default_params = {
            'objective': 'poisson',  # Better for count data with many zeros
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 100,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 1000,
            'min_child_samples': 20,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Fit LightGBM with early stopping and validation"""
        
        logger.info(f"Training {self.name} with shape {X.shape}")
        
        # Prepare datasets
        train_data = lgb.Dataset(X, label=y)
        valid_sets = [train_data]
        valid_names = ['train']
        
        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
            valid_sets.append(val_data)
            valid_names.append('valid')
        
        # Callbacks for early stopping
        callbacks = [
            lgb.early_stopping(stopping_rounds=100, verbose=False),
            lgb.log_evaluation(period=0)  # Silent training
        ]
        
        # Train model
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importance()
        
        # Store training history
        self.training_history = {
            'train_rmse': self.model.best_score['train']['rmse'],
            'valid_rmse': self.model.best_score.get('valid', {}).get('rmse'),
            'best_iteration': self.model.best_iteration,
            'num_features': X.shape[1]
        }
        
        logger.info(f"LightGBM training completed. Best iteration: {self.model.best_iteration}")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        # Ensure non-negative predictions for demand
        return np.maximum(predictions, 0)


class XGBoostModel(BaseModel):
    """
    XGBoost model for demand forecasting with zero-inflation handling
    """
    
    def __init__(self, name: str = "xgboost", params: Dict = None):
        default_params = {
            'objective': 'reg:squarederror',  # Changed from count:poisson to avoid issues
            'eval_metric': 'rmse',
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.9,
            'random_state': 42,
            'n_estimators': 500,  # Reduced for stability
            'min_child_weight': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 0.1,
            'verbosity': 0,
            'tree_method': 'hist'  # Faster training
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Fit XGBoost with early stopping"""
        
        logger.info(f"Training {self.name} with shape {X.shape}")
        
        # Prepare parameters
        fit_params = {}
        if X_val is not None and y_val is not None:
            fit_params['eval_set'] = [(X, y), (X_val, y_val)]
            fit_params['verbose'] = False
        
        # Train model
        self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, **fit_params)
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        # Store training history
        self.training_history = {
            'best_iteration': getattr(self.model, 'best_iteration', self.params['n_estimators']),
            'num_features': X.shape[1]
        }
        
        logger.info(f"XGBoost training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)


class RandomForestModel(BaseModel):
    """
    Random Forest model optimized for intermittent demand
    """
    
    def __init__(self, name: str = "random_forest", params: Dict = None):
        default_params = {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'random_state': 42,
            'n_jobs': -1,
            'max_features': 'sqrt'
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Fit Random Forest"""
        
        logger.info(f"Training {self.name} with shape {X.shape}")
        
        self.model = RandomForestRegressor(**self.params)
        self.model.fit(X, y)
        
        self.is_fitted = True
        self.feature_importance_ = self.model.feature_importances_
        
        self.training_history = {
            'num_features': X.shape[1],
            'oob_score': getattr(self.model, 'oob_score_', None)
        }
        
        logger.info(f"Random Forest training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        predictions = self.model.predict(X)
        return np.maximum(predictions, 0)


class ProphetModel(BaseModel):
    """
    Prophet model for time series forecasting with seasonality
    """
    
    def __init__(self, name: str = "prophet", params: Dict = None):
        if not PROPHET_AVAILABLE:
            raise ImportError("Prophet not available. Install with: pip install prophet")
        
        default_params = {
            'seasonality_mode': 'multiplicative',
            'daily_seasonality': True,
            'weekly_seasonality': True,
            'yearly_seasonality': True,
            'interval_width': 0.95,
            'changepoint_prior_scale': 0.05,
            'seasonality_prior_scale': 10.0
        }
        
        if params:
            default_params.update(params)
            
        super().__init__(name, default_params)
    
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None, **kwargs):
        """Fit Prophet model"""
        
        logger.info(f"Training {self.name} with shape {X.shape}")
        
        # Prophet expects specific format
        if 'date' not in X.columns:
            raise ValueError("Prophet requires 'date' column in features")
        
        # Prepare data for Prophet
        prophet_df = pd.DataFrame({
            'ds': X['date'],
            'y': y
        })
        
        self.model = Prophet(**self.params)
        self.model.fit(prophet_df)
        
        self.is_fitted = True
        
        logger.info("Prophet training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Prepare data for Prophet prediction
        future_df = pd.DataFrame({'ds': X['date']})
        forecast = self.model.predict(future_df)
        
        predictions = forecast['yhat'].values
        return np.maximum(predictions, 0)


class EnsembleModel:
    """
    Ensemble model that combines multiple base models
    Optimized for your M5 intermittent demand patterns
    """
    
    def __init__(self, models: Dict[str, BaseModel], weights: Dict[str, float] = None):
        self.models = models
        self.weights = weights or {name: 1.0/len(models) for name in models.keys()}
        self.is_fitted = False
        
        # Normalize weights
        total_weight = sum(self.weights.values())
        self.weights = {k: v/total_weight for k, v in self.weights.items()}
        
    def fit(self, X: pd.DataFrame, y: pd.Series, 
            X_val: pd.DataFrame = None, y_val: pd.Series = None):
        """Fit all models in the ensemble"""
        
        logger.info(f"Training ensemble with {len(self.models)} models")
        
        for name, model in self.models.items():
            logger.info(f"Training {name}...")
            model.fit(X, y, X_val, y_val)
        
        self.is_fitted = True
        logger.info("Ensemble training completed")
        
        return self
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make weighted ensemble predictions"""
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before prediction")
        
        predictions = np.zeros(len(X))
        
        for name, model in self.models.items():
            model_pred = model.predict(X)
            predictions += self.weights[name] * model_pred
        
        return predictions
    
    def get_model_predictions(self, X: pd.DataFrame) -> Dict[str, np.ndarray]:
        """Get predictions from individual models"""
        return {name: model.predict(X) for name, model in self.models.items()}


class ModelFactory:
    """Factory class to create models"""
    
    @staticmethod
    def create_model(model_type: str, params: Dict = None, name: str = None) -> BaseModel:
        """Create a model instance"""
        
        model_name = name or model_type
        
        if model_type.lower() in ['lightgbm', 'lgb']:
            return LightGBMModel(model_name, params)
        elif model_type.lower() in ['xgboost', 'xgb']:
            return XGBoostModel(model_name, params)
        elif model_type.lower() in ['randomforest', 'rf', 'random_forest']:
            return RandomForestModel(model_name, params)
        elif model_type.lower() == 'prophet':
            return ProphetModel(model_name, params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    @staticmethod
    def create_ensemble(model_configs: List[Dict], weights: Dict[str, float] = None) -> EnsembleModel:
        """Create an ensemble model"""
        
        models = {}
        for config in model_configs:
            model = ModelFactory.create_model(
                config['type'], 
                config.get('params'),
                config.get('name', config['type'])
            )
            models[model.name] = model
        
        return EnsembleModel(models, weights)


class HyperparameterTuner:
    """
    Hyperparameter tuning using Optuna
    Optimized for intermittent demand patterns
    """
    
    def __init__(self, model_type: str, cv_splits: int = 3, n_trials: int = 100):
        self.model_type = model_type
        self.cv_splits = cv_splits
        self.n_trials = n_trials
        self.best_params = None
        self.study = None
    
    def _objective_lightgbm(self, trial, X, y):
        """Objective function for LightGBM tuning"""
        params = {
            'objective': 'poisson',
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': trial.suggest_int('num_leaves', 50, 300),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
            'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
            'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
            'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbose': -1,
            'random_state': 42,
            'n_estimators': 500
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = LightGBMModel(params=params)
            model.fit(X_train, y_train, X_val, y_val)
            
            pred = model.predict(X_val)
            score = mean_squared_error(y_val, pred, squared=False)
            scores.append(score)
        
        return np.mean(scores)
    
    def _objective_xgboost(self, trial, X, y):
        """Objective function for XGBoost tuning"""
        params = {
            'objective': 'count:poisson',
            'eval_metric': 'rmse',
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'verbosity': 0,
            'random_state': 42,
            'n_estimators': 500
        }
        
        # Time series cross-validation
        tscv = TimeSeriesSplit(n_splits=self.cv_splits)
        scores = []
        
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = XGBoostModel(params=params)
            model.fit(X_train, y_train, X_val, y_val)
            
            pred = model.predict(X_val)
            score = mean_squared_error(y_val, pred, squared=False)
            scores.append(score)
        
        return np.mean(scores)
    
    def tune(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Perform hyperparameter tuning"""
        
        logger.info(f"Starting hyperparameter tuning for {self.model_type}")
        
        if self.model_type.lower() in ['lightgbm', 'lgb']:
            objective_func = lambda trial: self._objective_lightgbm(trial, X, y)
        elif self.model_type.lower() in ['xgboost', 'xgb']:
            objective_func = lambda trial: self._objective_xgboost(trial, X, y)
        else:
            raise ValueError(f"Tuning not implemented for {self.model_type}")
        
        # Create study
        self.study = optuna.create_study(direction='minimize')
        self.study.optimize(objective_func, n_trials=self.n_trials, show_progress_bar=True)
        
        self.best_params = self.study.best_params
        
        logger.info(f"Tuning completed. Best score: {self.study.best_value:.4f}")
        logger.info(f"Best parameters: {self.best_params}")
        
        return self.best_params


class ModelTrainer:
    """
    Main training orchestrator for M5 demand forecasting
    Handles your intermittent demand patterns (68.2% zeros)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.trained_models = {}
        self.ensemble_model = None
        self.training_history = {}
    
    def train_single_model(self, model_type: str, X_train: pd.DataFrame, y_train: pd.Series,
                          X_val: pd.DataFrame = None, y_val: pd.Series = None,
                          params: Dict = None, model_name: str = None) -> BaseModel:
        """Train a single model"""
        
        model_name = model_name or model_type
        logger.info(f"Training {model_name}...")
        
        # Create model
        model = ModelFactory.create_model(model_type, params, model_name)
        
        # Train
        start_time = datetime.now()
        model.fit(X_train, y_train, X_val, y_val)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Store model and history
        self.trained_models[model_name] = model
        self.training_history[model_name] = {
            'training_time': training_time,
            'model_params': model.params,
            'training_history': model.training_history
        }
        
        logger.info(f"{model_name} training completed in {training_time:.2f} seconds")
        
        return model
    
    def train_multiple_models(self, model_configs: List[Dict], 
                            X_train: pd.DataFrame, y_train: pd.Series,
                            X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, BaseModel]:
        """Train multiple models"""
        
        logger.info(f"Training {len(model_configs)} models...")
        
        for config in model_configs:
            self.train_single_model(
                config['type'],
                X_train, y_train, X_val, y_val,
                config.get('params'),
                config.get('name')
            )
        
        return self.trained_models
    
    def create_ensemble(self, model_names: List[str] = None, 
                       weights: Dict[str, float] = None) -> EnsembleModel:
        """Create ensemble from trained models"""
        
        if not self.trained_models:
            raise ValueError("No trained models available for ensemble")
        
        # Use all models if none specified
        if model_names is None:
            model_names = list(self.trained_models.keys())
        
        # Select models for ensemble
        ensemble_models = {name: self.trained_models[name] for name in model_names}
        
        # Create ensemble
        self.ensemble_model = EnsembleModel(ensemble_models, weights)
        self.ensemble_model.is_fitted = True  # Models already fitted
        
        logger.info(f"Ensemble created with models: {model_names}")
        
        return self.ensemble_model
    
    def tune_hyperparameters(self, model_type: str, X: pd.DataFrame, y: pd.Series,
                           n_trials: int = 100) -> Dict:
        """Tune hyperparameters for a model"""
        
        tuner = HyperparameterTuner(model_type, n_trials=n_trials)
        best_params = tuner.tune(X, y)
        
        return best_params
    
    def save_models(self, output_dir: str):
        """Save all trained models"""
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual models
        for name, model in self.trained_models.items():
            model_path = output_path / f"{name}.joblib"
            model.save(str(model_path))
        
        # Save ensemble if available
        if self.ensemble_model:
            ensemble_path = output_path / "ensemble.joblib"
            joblib.dump(self.ensemble_model, ensemble_path)
        
        # Save training history
        history_path = output_path / "training_history.json"
        with open(history_path, 'w') as f:
            json.dump(self.training_history, f, indent=2, default=str)
        
        logger.info(f"Models saved to {output_dir}")


if __name__ == "__main__":
    # Example usage and testing
    print("Testing Model Training Pipeline...")
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    n_features = 20
    
    X = pd.DataFrame(np.random.randn(n_samples, n_features), 
                    columns=[f'feature_{i}' for i in range(n_features)])
    
    # Create intermittent demand pattern (like M5 with 68% zeros)
    y = np.random.poisson(0.5, n_samples)  # Creates many zeros
    y = pd.Series(y)
    
    print(f"Sample data: {X.shape}, Zero rate: {(y == 0).mean():.1%}")
    
    # Split data
    split_idx = int(0.8 * len(X))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]
    
    # Test individual models
    print("\nTesting individual models...")
    
    # LightGBM
    lgb_model = LightGBMModel()
    lgb_model.fit(X_train, y_train, X_val, y_val)
    lgb_pred = lgb_model.predict(X_val)
    print(f"LightGBM RMSE: {mean_squared_error(y_val, lgb_pred, squared=False):.4f}")
    
    # XGBoost
    xgb_model = XGBoostModel()
    xgb_model.fit(X_train, y_train, X_val, y_val)
    xgb_pred = xgb_model.predict(X_val)
    print(f"XGBoost RMSE: {mean_squared_error(y_val, xgb_pred, squared=False):.4f}")
    
    # Random Forest
    rf_model = RandomForestModel()
    rf_model.fit(X_train, y_train, X_val, y_val)
    rf_pred = rf_model.predict(X_val)
    print(f"Random Forest RMSE: {mean_squared_error(y_val, rf_pred, squared=False):.4f}")
    
    # Test ensemble
    print("\nTesting ensemble...")
    models = {
        'lightgbm': lgb_model,
        'xgboost': xgb_model,
        'random_forest': rf_model
    }
    
    ensemble = EnsembleModel(models, weights={'lightgbm': 0.5, 'xgboost': 0.3, 'random_forest': 0.2})
    ensemble.is_fitted = True
    ensemble_pred = ensemble.predict(X_val)
    print(f"Ensemble RMSE: {mean_squared_error(y_val, ensemble_pred, squared=False):.4f}")
    
    print("\nModel testing completed successfully!")
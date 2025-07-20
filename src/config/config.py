# Configuration settings
# File: src/config/config.py

import os
from pathlib import Path

class Config:
    # Data paths
    DATA_DIR = Path("data")
    RAW_DATA_DIR = DATA_DIR / "raw"
    PROCESSED_DATA_DIR = DATA_DIR / "processed"
    EXTERNAL_DATA_DIR = DATA_DIR / "external"
    FEATURES_DATA_DIR = DATA_DIR / "features"
    
    # Model paths
    MODELS_DIR = Path("models")
    TRAINED_MODELS_DIR = MODELS_DIR / "trained"
    MODEL_CONFIGS_DIR = MODELS_DIR / "configs"
    MODEL_REGISTRY_DIR = MODELS_DIR / "registry"
    
    # Output paths
    OUTPUTS_DIR = Path("outputs")
    REPORTS_DIR = OUTPUTS_DIR / "reports"
    VISUALIZATIONS_DIR = OUTPUTS_DIR / "visualizations"
    
    # API Keys (from environment variables)
    WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
    FRED_API_KEY = os.getenv('FRED_API_KEY')
    KAGGLE_USERNAME = os.getenv('KAGGLE_USERNAME')
    KAGGLE_KEY = os.getenv('KAGGLE_KEY')
    
    # Model parameters
    TEST_SIZE = 28  # Days for testing (M5 competition standard)
    VALIDATION_SIZE = 28  # Days for validation
    
    # Feature engineering parameters
    LAG_PERIODS = [1, 7, 14, 21, 28]
    ROLLING_WINDOWS = [7, 14, 28, 56]
    
    # Model configurations
    MODELS_CONFIG = {
        'lgb': {
            'objective': 'tweedie',
            'tweedie_variance_power': 1.1,
            'metric': 'rmse',
            'boosting_type': 'gbdt',
            'num_leaves': 255,
            'learning_rate': 0.03,
            'feature_fraction': 0.8,
            'bagging_fraction': 0.8,
            'bagging_freq': 1,
            'n_estimators': 1000,
            'random_state': 42,
            'verbosity': -1
        },
        'xgb': {
            'objective': 'reg:tweedie',
            'tweedie_variance_power': 1.1,
            'eval_metric': 'rmse',
            'learning_rate': 0.03,
            'max_depth': 8,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'n_estimators': 1000,
            'random_state': 42,
            'verbosity': 0
        }
    }
    
    # Monitoring thresholds
    DRIFT_THRESHOLD = 0.1
    PERFORMANCE_THRESHOLD = 0.05
    
    @classmethod
    def validate_setup(cls):
        """Validate that required directories and API keys exist"""
        required_dirs = [cls.RAW_DATA_DIR, cls.PROCESSED_DATA_DIR, cls.EXTERNAL_DATA_DIR]
        for dir_path in required_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        warnings = []
        if not cls.WEATHER_API_KEY:
            warnings.append("WEATHER_API_KEY not set - weather data integration will be skipped")
        if not cls.FRED_API_KEY:
            warnings.append("FRED_API_KEY not set - economic data integration will be skipped")
        if not cls.KAGGLE_USERNAME or not cls.KAGGLE_KEY:
            warnings.append("Kaggle credentials not set - manual data download required")
        
        return warnings

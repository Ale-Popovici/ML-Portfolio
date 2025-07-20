# Configuration Management System
# File: src/config/config.py

import os
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass

@dataclass
class DataConfig:
    """Data-related configuration"""
    # Base paths
    data_dir: Path = Path("data")
    raw_data_dir: Path = Path("data/raw")
    processed_data_dir: Path = Path("data/processed")
    external_data_dir: Path = Path("data/external")
    features_data_dir: Path = Path("data/features")
    
    # M5 data files
    sales_file: str = "sales_train_validation.csv"
    calendar_file: str = "calendar.csv"
    prices_file: str = "sell_prices.csv"
    
    # External data
    weather_file: str = "weather_data.csv"
    economic_file: str = "economic_data.csv"
    
    # Data validation thresholds
    max_missing_pct: float = 0.05
    min_sales_value: float = 0
    max_sales_value: float = 100000


@dataclass
class ModelConfig:
    """Model-related configuration"""
    # Base paths
    models_dir: Path = Path("models")
    trained_models_dir: Path = Path("models/trained")
    model_configs_dir: Path = Path("models/configs")
    model_registry_dir: Path = Path("models/registry")
    
    # Model parameters
    test_size: int = 28  # Days for testing (M5 standard)
    validation_size: int = 28  # Days for validation
    target_col: str = "demand"
    date_col: str = "date"
    
    # Cross-validation
    cv_folds: int = 3
    gap_days: int = 28  # Gap between train and validation
    
    # Early stopping
    early_stopping_rounds: int = 100
    
    # Model ensemble
    ensemble_weights: Dict[str, float] = None
    
    def __post_init__(self):
        if self.ensemble_weights is None:
            self.ensemble_weights = {
                'lightgbm': 0.4,
                'xgboost': 0.3,
                'random_forest': 0.2,
                'prophet': 0.1
            }


@dataclass
class FeatureConfig:
    """Feature engineering configuration"""
    # Lag features
    lag_periods: List[int] = None
    
    # Rolling window features
    rolling_windows: List[int] = None
    
    # Moving averages
    ma_windows: List[int] = None
    
    # External features
    include_weather: bool = True
    include_economic: bool = True
    include_events: bool = True
    
    # Feature selection
    max_features: int = 1000
    feature_importance_threshold: float = 0.001
    correlation_threshold: float = 0.95
    
    def __post_init__(self):
        if self.lag_periods is None:
            self.lag_periods = [1, 7, 14, 21, 28, 35, 42]
        
        if self.rolling_windows is None:
            self.rolling_windows = [7, 14, 28, 56, 84]
            
        if self.ma_windows is None:
            self.ma_windows = [7, 14, 28]


@dataclass
class TrainingConfig:
    """Training configuration"""
    # Model types to train
    models_to_train: List[str] = None
    
    # Hyperparameter tuning
    enable_tuning: bool = False
    tuning_trials: int = 100
    tuning_timeout: int = 7200  # seconds
    
    # Ensemble
    create_ensemble: bool = True
    ensemble_method: str = "weighted_average"  # or "stacking"
    
    # Resources
    n_jobs: int = -1
    random_state: int = 42
    
    # Quick test mode
    quick_test: bool = False
    quick_test_samples: int = 10000
    
    def __post_init__(self):
        if self.models_to_train is None:
            self.models_to_train = ['lightgbm', 'xgboost', 'random_forest']


@dataclass
class OutputConfig:
    """Output and reporting configuration"""
    # Output paths
    outputs_dir: Path = Path("outputs")
    reports_dir: Path = Path("outputs/reports")
    visualizations_dir: Path = Path("outputs/visualizations")
    model_cards_dir: Path = Path("outputs/model_cards")
    
    # Report settings
    generate_html_report: bool = True
    generate_model_cards: bool = True
    create_visualizations: bool = True
    
    # Visualization settings
    plot_style: str = "seaborn-v0_8"
    figure_size: tuple = (12, 8)
    dpi: int = 300


class Config:
    """Main configuration class that orchestrates all configs"""
    
    def __init__(self, config_path: Optional[str] = None):
        # Initialize sub-configs
        self.data = DataConfig()
        self.model = ModelConfig()
        self.features = FeatureConfig()
        self.training = TrainingConfig()
        self.output = OutputConfig()
        
        # API Keys
        self.weather_api_key = os.getenv('WEATHER_API_KEY')
        self.fred_api_key = os.getenv('FRED_API_KEY')
        self.kaggle_username = os.getenv('KAGGLE_USERNAME')
        self.kaggle_key = os.getenv('KAGGLE_KEY')
        
        # Model-specific configurations
        self.model_params = self._get_model_params()
        
        # Load custom config if provided
        if config_path:
            self.load_config(config_path)
    
    def _get_model_params(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameters for each model type"""
        return {
            'lightgbm': {
                'objective': 'regression',
                'metric': 'rmse',
                'boosting_type': 'gbdt',
                'num_leaves': 100,
                'learning_rate': 0.05,
                'feature_fraction': 0.9,
                'bagging_fraction': 0.8,
                'bagging_freq': 5,
                'verbose': -1,
                'random_state': self.training.random_state,
                'n_estimators': 1000
            },
            'xgboost': {
                'objective': 'reg:squarederror',
                'eval_metric': 'rmse',
                'max_depth': 6,
                'learning_rate': 0.05,
                'subsample': 0.8,
                'colsample_bytree': 0.9,
                'random_state': self.training.random_state,
                'n_estimators': 1000,
                'verbosity': 0
            },
            'random_forest': {
                'n_estimators': 200,
                'max_depth': 15,
                'min_samples_split': 5,
                'min_samples_leaf': 2,
                'random_state': self.training.random_state,
                'n_jobs': self.training.n_jobs
            },
            'prophet': {
                'seasonality_mode': 'multiplicative',
                'daily_seasonality': True,
                'weekly_seasonality': True,
                'yearly_seasonality': True,
                'interval_width': 0.95
            }
        }
    
    def load_config(self, config_path: str):
        """Load configuration from JSON file"""
        import json
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Update configurations
        for section, values in config_dict.items():
            if hasattr(self, section):
                section_config = getattr(self, section)
                for key, value in values.items():
                    if hasattr(section_config, key):
                        setattr(section_config, key, value)
    
    def save_config(self, config_path: str):
        """Save current configuration to JSON file"""
        import json
        from dataclasses import asdict
        
        config_dict = {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'features': asdict(self.features),
            'training': asdict(self.training),
            'output': asdict(self.output)
        }
        
        # Convert Path objects to strings for JSON serialization
        def convert_paths(obj):
            if isinstance(obj, dict):
                return {k: convert_paths(v) for k, v in obj.items()}
            elif isinstance(obj, Path):
                return str(obj)
            elif isinstance(obj, list):
                return [convert_paths(item) for item in obj]
            else:
                return obj
        
        config_dict = convert_paths(config_dict)
        
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        with open(config_path, 'w') as f:
            json.dump(config_dict, f, indent=2)
    
    def create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.data.data_dir,
            self.data.raw_data_dir,
            self.data.processed_data_dir,
            self.data.external_data_dir,
            self.data.features_data_dir,
            self.model.models_dir,
            self.model.trained_models_dir,
            self.model.model_configs_dir,
            self.model.model_registry_dir,
            self.output.outputs_dir,
            self.output.reports_dir,
            self.output.visualizations_dir,
            self.output.model_cards_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def validate_config(self) -> List[str]:
        """Validate configuration settings"""
        issues = []
        
        # Check required files exist
        required_files = [
            self.data.raw_data_dir / self.data.sales_file,
            self.data.raw_data_dir / self.data.calendar_file,
            self.data.raw_data_dir / self.data.prices_file
        ]
        
        for file_path in required_files:
            if not file_path.exists():
                issues.append(f"Required data file missing: {file_path}")
        
        # Check model parameters
        if self.model.test_size <= 0:
            issues.append("Test size must be positive")
        
        if self.model.validation_size <= 0:
            issues.append("Validation size must be positive")
        
        # Check feature configuration
        if not self.features.lag_periods:
            issues.append("At least one lag period must be specified")
        
        return issues
    
    def get_full_config_dict(self) -> Dict[str, Any]:
        """Get full configuration as dictionary"""
        from dataclasses import asdict
        
        return {
            'data': asdict(self.data),
            'model': asdict(self.model),
            'features': asdict(self.features),
            'training': asdict(self.training),
            'output': asdict(self.output),
            'model_params': self.model_params
        }


# Global config instance
config = Config()

# Convenience function to get config
def get_config() -> Config:
    """Get the global configuration instance"""
    return config

# Function to initialize config with custom path
def initialize_config(config_path: Optional[str] = None) -> Config:
    """Initialize configuration with optional custom config file"""
    global config
    config = Config(config_path)
    config.create_directories()
    return config
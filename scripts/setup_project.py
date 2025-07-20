# Project Setup Script
# File: scripts/setup_project.py

import os
import subprocess
import sys
from pathlib import Path
import requests
import zipfile
import io

def create_project_structure():
    """Create the complete project directory structure"""
    print("📁 Creating project structure...")
    
    directories = [
        'data/raw',
        'data/processed', 
        'data/external',
        'data/features',
        'src/config',
        'src/data',
        'src/features',
        'src/models',
        'src/serving',
        'src/monitoring',
        'src/utils',
        'notebooks',
        'tests/test_data',
        'tests/test_features',
        'tests/test_models',
        'tests/test_serving',
        'models/trained',
        'models/configs',
        'models/registry',
        'outputs/reports',
        'outputs/visualizations',
        'outputs/model_cards',
        'scripts',
        'docs/model_cards',
        'docs/technical_specs',
        'docs/business_impact'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Create .gitkeep files for empty directories
        gitkeep = Path(directory) / '.gitkeep'
        if not any(Path(directory).iterdir()) and not gitkeep.exists():
            gitkeep.touch()
    
    print("✅ Project structure created!")

def create_config_files():
    """Create configuration files"""
    print("⚙️ Creating configuration files...")
    
    # Config.py
    config_content = '''# Configuration settings
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
'''
    
    with open('src/config/config.py', 'w') as f:
        f.write(config_content)
    
    # Logging config
    logging_config = '''# Logging configuration
# File: src/config/logging_config.py

import logging
import sys
from pathlib import Path

def setup_logging(log_level=logging.INFO, log_file=None):
    """Setup logging configuration"""
    
    # Create logs directory
    logs_dir = Path("logs")
    logs_dir.mkdir(exist_ok=True)
    
    # Configure logging format
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Setup root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.FileHandler(logs_dir / log_file)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
'''
    
    with open('src/config/logging_config.py', 'w') as f:
        f.write(logging_config)
    
    print("✅ Configuration files created!")

def setup_environment():
    """Setup Python environment with required packages"""
    print("🐍 Setting up Python environment...")
    
    requirements = '''# Core data science libraries
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
scikit-learn>=1.1.0

# Visualization
matplotlib>=3.5.0
seaborn>=0.11.0
plotly>=5.10.0
altair>=4.2.0

# Time series and forecasting
statsmodels>=0.13.0
prophet>=1.1.0
sktime>=0.13.0

# Machine learning
lightgbm>=3.3.0
xgboost>=1.6.0
catboost>=1.1.0
optuna>=3.0.0

# Deep learning (optional)
tensorflow>=2.9.0

# Model interpretation
shap>=0.41.0
lime>=0.2.0
interpret>=0.3.0

# MLOps and production
mlflow>=1.28.0
fastapi>=0.85.0
pydantic>=1.10.0
great-expectations>=0.15.0

# Data quality and validation
pandera>=0.12.0

# API and web scraping
requests>=2.28.0
beautifulsoup4>=4.11.0

# Progress bars and utilities
tqdm>=4.64.0
joblib>=1.2.0

# Testing
pytest>=7.1.0
pytest-cov>=3.0.0
hypothesis>=6.54.0

# Code quality
black>=22.6.0
flake8>=5.0.0
isort>=5.10.0

# Documentation
jupyter>=1.0.0
jupyterlab>=3.4.0
nbconvert>=6.5.0

# Environment management
python-dotenv>=0.20.0

# AWS/Cloud (optional)
boto3>=1.24.0
'''
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements)
    
    # Install packages
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Python packages installed!")
    except subprocess.CalledProcessError:
        print("⚠️ Some packages failed to install. Please run 'pip install -r requirements.txt' manually.")

def download_m5_data():
    """Download M5 competition data"""
    print("📥 Downloading M5 competition data...")
    
    # Check if Kaggle API is configured
    try:
        import kaggle
        
        # Download M5 data
        kaggle.api.competition_download_files('m5-forecasting-accuracy', path='data/raw')
        
        # Extract ZIP file
        with zipfile.ZipFile('data/raw/m5-forecasting-accuracy.zip', 'r') as zip_ref:
            zip_ref.extractall('data/raw')
        
        # Remove ZIP file
        os.remove('data/raw/m5-forecasting-accuracy.zip')
        
        print("✅ M5 data downloaded and extracted!")
        return True
        
    except ImportError:
        print("⚠️ Kaggle API not installed. Installing...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'kaggle'])
        return download_m5_data()
        
    except Exception as e:
        print(f"❌ Failed to download M5 data: {e}")
        print("Please manually download the data from:")
        print("https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        return False

def create_initial_notebooks():
    """Create initial Jupyter notebooks"""
    print("📓 Creating initial notebooks...")
    
    notebooks = {
        '01_data_exploration.ipynb': '''# Data Exploration and EDA
This notebook contains initial exploration of the M5 dataset and external data sources.

## Objectives
1. Load and examine M5 competition data
2. Analyze sales patterns and trends
3. Explore external data relationships
4. Identify data quality issues
''',
        '02_external_data_analysis.ipynb': '''# External Data Analysis
Analysis of weather and economic data integration with sales patterns.

## Objectives
1. Weather data correlation analysis
2. Economic indicators impact assessment
3. Feature correlation analysis
4. External signal validation
''',
        '03_feature_engineering.ipynb': '''# Feature Engineering
Advanced feature engineering for demand forecasting.

## Objectives
1. Temporal feature creation
2. External signal integration
3. Business logic features
4. Feature selection and validation
''',
        '04_baseline_models.ipynb': '''# Baseline Models
Implementation and evaluation of baseline forecasting models.

## Objectives
1. Naive forecasting baselines
2. Statistical methods (ARIMA, Prophet)
3. Simple ML models
4. Baseline performance evaluation
''',
        '05_advanced_models.ipynb': '''# Advanced ML Models
Implementation of advanced machine learning models.

## Objectives
1. Gradient boosting models (XGBoost, LightGBM)
2. Deep learning approaches (LSTM)
3. Hyperparameter optimization
4. Cross-validation strategies
''',
        '06_ensemble_methods.ipynb': '''# Ensemble Methods
Model ensemble and stacking approaches.

## Objectives
1. Weighted ensemble methods
2. Stacking and blending
3. Dynamic ensemble selection
4. Uncertainty quantification
''',
        '07_business_impact_analysis.ipynb': '''# Business Impact Analysis
Translation of model performance to business value.

## Objectives
1. Inventory optimization simulation
2. Service level analysis
3. Cost-benefit analysis
4. ROI calculations
''',
        '08_model_interpretation.ipynb': '''# Model Interpretation and Monitoring
Model interpretation and production monitoring setup.

## Objectives
1. SHAP and LIME analysis
2. Feature importance analysis
3. Model bias assessment
4. Monitoring dashboard creation
'''
    }
    
    for notebook_name, content in notebooks.items():
        notebook_path = Path('notebooks') / notebook_name
        
        # Create basic notebook structure
        notebook_content = {
            "cells": [
                {
                    "cell_type": "markdown",
                    "metadata": {},
                    "source": [content]
                },
                {
                    "cell_type": "code",
                    "execution_count": None,
                    "metadata": {},
                    "outputs": [],
                    "source": [
                        "# Import libraries\n",
                        "import pandas as pd\n",
                        "import numpy as np\n",
                        "import matplotlib.pyplot as plt\n",
                        "import seaborn as sns\n",
                        "import sys\n",
                        "sys.path.append('../src')\n",
                        "\n",
                        "# Configure plotting\n",
                        "plt.style.use('seaborn-v0_8')\n",
                        "sns.set_palette('husl')\n",
                        "%matplotlib inline"
                    ]
                }
            ],
            "metadata": {
                "kernelspec": {
                    "display_name": "Python 3",
                    "language": "python",
                    "name": "python3"
                },
                "language_info": {
                    "name": "python",
                    "version": "3.9.0"
                }
            },
            "nbformat": 4,
            "nbformat_minor": 4
        }
        
        import json
        with open(notebook_path, 'w') as f:
            json.dump(notebook_content, f, indent=2)
    
    print("✅ Initial notebooks created!")

def create_docker_files():
    """Create Docker configuration"""
    print("🐳 Creating Docker configuration...")
    
    dockerfile = '''FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port for API
EXPOSE 8000

# Default command
CMD ["python", "-m", "src.serving.api"]
'''
    
    with open('Dockerfile', 'w') as f:
        f.write(dockerfile)
    
    docker_compose = '''version: '3.8'

services:
  demand-forecasting:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - WEATHER_API_KEY=${WEATHER_API_KEY}
      - FRED_API_KEY=${FRED_API_KEY}
    
  jupyter:
    build: .
    ports:
      - "8888:8888"
    volumes:
      - .:/app
    command: jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root
    environment:
      - WEATHER_API_KEY=${WEATHER_API_KEY}
      - FRED_API_KEY=${FRED_API_KEY}
'''
    
    with open('docker-compose.yml', 'w') as f:
        f.write(docker_compose)
    
    print("✅ Docker files created!")

def create_github_workflows():
    """Create GitHub Actions workflows"""
    print("🔧 Creating GitHub Actions workflows...")
    
    Path('.github/workflows').mkdir(parents=True, exist_ok=True)
    
    ci_workflow = '''name: CI

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  test:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        python -m pip install --upgrade pip
        pip install -r requirements.txt
    
    - name: Run tests
      run: |
        pytest tests/ --cov=src --cov-report=xml
    
    - name: Code quality checks
      run: |
        black --check src/
        flake8 src/
        isort --check-only src/
'''
    
    with open('.github/workflows/ci.yml', 'w') as f:
        f.write(ci_workflow)
    
    print("✅ GitHub workflows created!")

def main():
    """Main setup function"""
    print("🚀 Setting up Demand Forecasting ML Project...")
    print("=" * 50)
    
    # Create project structure
    create_project_structure()
    
    # Create configuration files
    create_config_files()
    
    # Setup Python environment
    setup_environment()
    
    # Download M5 data
    download_m5_data()
    
    # Create initial notebooks
    create_initial_notebooks()
    
    # Create Docker files
    create_docker_files()
    
    # Create GitHub workflows
    create_github_workflows()
    
    print("\n" + "=" * 50)
    print("🎉 Project setup complete!")
    print("\n📋 Next steps:")
    print("1. Set up API keys in environment variables:")
    print("   - WEATHER_API_KEY (Visual Crossing Weather API)")
    print("   - FRED_API_KEY (Federal Reserve Economic Data)")
    print("   - KAGGLE_USERNAME and KAGGLE_KEY (for data download)")
    print("\n2. Run the ETL pipeline:")
    print("   python -m src.data.etl_pipeline")
    print("\n3. Start with data exploration:")
    print("   jupyter lab")
    print("   Open notebooks/01_data_exploration.ipynb")
    print("\n4. Optional: Set up version control:")
    print("   git init")
    print("   git add .")
    print("   git commit -m 'Initial project setup'")

if __name__ == "__main__":
    main()
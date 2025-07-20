# ETL Pipeline for Multi-Modal Demand Forecasting
# File: src/data/etl_pipeline.py

import pandas as pd
import numpy as np
import requests
import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import logging
from pathlib import Path
import zipfile
import io

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class M5DataExtractor:
    """Extract M5 Competition Data from Kaggle"""
    
    def __init__(self, data_dir: str = "data/raw"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # M5 Dataset URLs (these would be from Kaggle API or direct download)
        self.m5_files = {
            'sales_train_validation': 'sales_train_validation.csv',
            'calendar': 'calendar.csv',
            'sell_prices': 'sell_prices.csv',
            'sample_submission': 'sample_submission.csv'
        }
    
    def download_m5_data(self) -> bool:
        """Download M5 competition data"""
        try:
            # In real implementation, use Kaggle API:
            # kaggle competitions download -c m5-forecasting-accuracy
            logger.info("Downloading M5 competition data...")
            
            # For now, assume data is already downloaded
            # Check if files exist
            for file_name in self.m5_files.values():
                file_path = self.data_dir / file_name
                if not file_path.exists():
                    logger.error(f"Missing file: {file_path}")
                    return False
            
            logger.info("M5 data successfully verified")
            return True
            
        except Exception as e:
            logger.error(f"Error downloading M5 data: {e}")
            return False
    
    def extract_sales_data(self) -> pd.DataFrame:
        """Extract and initial processing of sales data"""
        logger.info("Extracting sales data...")
        
        # Load sales data
        sales_df = pd.read_csv(self.data_dir / 'sales_train_validation.csv')
        
        # Melt the dataframe to long format
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        sales_melted = pd.melt(
            sales_df, 
            id_vars=id_cols,
            var_name='d', 
            value_name='demand'
        )
        
        # Extract day number
        sales_melted['d_num'] = sales_melted['d'].str.extract('(\d+)').astype(int)
        
        logger.info(f"Sales data shape: {sales_melted.shape}")
        return sales_melted
    
    def extract_calendar_data(self) -> pd.DataFrame:
        """Extract calendar and events data"""
        logger.info("Extracting calendar data...")
        
        calendar_df = pd.read_csv(self.data_dir / 'calendar.csv')
        
        # Parse date
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        # Extract additional date features
        calendar_df['year'] = calendar_df['date'].dt.year
        calendar_df['month'] = calendar_df['date'].dt.month
        calendar_df['day'] = calendar_df['date'].dt.day
        calendar_df['dayofweek'] = calendar_df['date'].dt.dayofweek
        calendar_df['dayofyear'] = calendar_df['date'].dt.dayofyear
        calendar_df['week'] = calendar_df['date'].dt.isocalendar().week
        calendar_df['quarter'] = calendar_df['date'].dt.quarter
        
        # Boolean features for events
        event_cols = ['event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        for col in event_cols:
            calendar_df[f'{col}_exists'] = calendar_df[col].notna()
        
        logger.info(f"Calendar data shape: {calendar_df.shape}")
        return calendar_df
    
    def extract_prices_data(self) -> pd.DataFrame:
        """Extract pricing data"""
        logger.info("Extracting pricing data...")
        
        prices_df = pd.read_csv(self.data_dir / 'sell_prices.csv')
        
        logger.info(f"Prices data shape: {prices_df.shape}")
        return prices_df


class WeatherDataExtractor:
    """Extract weather data from Visual Crossing Weather API"""
    
    def __init__(self, api_key: str, data_dir: str = "data/external"):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline"
        
        # Store locations (approximate coordinates for M5 states)
        self.store_locations = {
            'CA': {'lat': 34.0522, 'lon': -118.2437, 'name': 'Los Angeles, CA'},
            'TX': {'lat': 32.7767, 'lon': -96.7970, 'name': 'Dallas, TX'},
            'WI': {'lat': 43.0389, 'lon': -87.9065, 'name': 'Milwaukee, WI'}
        }
    
    def get_historical_weather(self, state: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get historical weather data for a state"""
        logger.info(f"Fetching weather data for {state} from {start_date} to {end_date}")
        
        location = self.store_locations[state]
        
        # Construct API URL
        url = f"{self.base_url}/{location['lat']},{location['lon']}/{start_date}/{end_date}"
        
        params = {
            'key': self.api_key,
            'include': 'days',
            'elements': 'datetime,tempmax,tempmin,temp,humidity,precip,windspeed,cloudcover,conditions'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            
            data = response.json()
            
            # Convert to DataFrame
            weather_df = pd.DataFrame(data['days'])
            weather_df['state_id'] = state
            weather_df['date'] = pd.to_datetime(weather_df['datetime'])
            
            # Feature engineering
            weather_df['temp_range'] = weather_df['tempmax'] - weather_df['tempmin']
            weather_df['is_rainy'] = weather_df['precip'] > 0
            weather_df['temp_category'] = pd.cut(
                weather_df['temp'], 
                bins=[-np.inf, 32, 50, 70, 85, np.inf],
                labels=['freezing', 'cold', 'mild', 'warm', 'hot']
            )
            
            logger.info(f"Retrieved {len(weather_df)} weather records for {state}")
            return weather_df
            
        except Exception as e:
            logger.error(f"Error fetching weather data for {state}: {e}")
            return pd.DataFrame()
    
    def extract_all_weather_data(self, start_date: str = "2011-01-29", end_date: str = "2016-06-19") -> pd.DataFrame:
        """Extract weather data for all states"""
        all_weather = []
        
        for state in self.store_locations.keys():
            weather_df = self.get_historical_weather(state, start_date, end_date)
            if not weather_df.empty:
                all_weather.append(weather_df)
        
        if all_weather:
            combined_weather = pd.concat(all_weather, ignore_index=True)
            
            # Save to file
            weather_file = self.data_dir / 'weather_data.csv'
            combined_weather.to_csv(weather_file, index=False)
            logger.info(f"Weather data saved to {weather_file}")
            
            return combined_weather
        else:
            logger.warning("No weather data retrieved")
            return pd.DataFrame()


class EconomicDataExtractor:
    """Extract economic indicators from FRED API"""
    
    def __init__(self, api_key: str, data_dir: str = "data/external"):
        self.api_key = api_key
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.base_url = "https://api.stlouisfed.org/fred/series/observations"
        
        # Economic indicators to track
        self.indicators = {
            'unemployment': 'UNRATE',
            'consumer_confidence': 'UMCSENT',
            'cpi': 'CPIAUCSL',
            'gdp': 'GDP',
            'gas_prices': 'GASREGW'
        }
    
    def get_economic_indicator(self, series_id: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Get economic indicator data"""
        logger.info(f"Fetching economic data for {series_id}")
        
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'observation_start': start_date,
            'observation_end': end_date,
            'frequency': 'm'  # Monthly data
        }
        
        try:
            response = requests.get(self.base_url, params=params)
            response.raise_for_status()
            
            data = response.json()
            observations = data['observations']
            
            # Convert to DataFrame
            econ_df = pd.DataFrame(observations)
            econ_df['date'] = pd.to_datetime(econ_df['date'])
            econ_df['value'] = pd.to_numeric(econ_df['value'], errors='coerce')
            econ_df = econ_df[['date', 'value']].rename(columns={'value': series_id})
            
            return econ_df
            
        except Exception as e:
            logger.error(f"Error fetching economic data for {series_id}: {e}")
            return pd.DataFrame()
    
    def extract_all_economic_data(self, start_date: str = "2011-01-01", end_date: str = "2016-12-31") -> pd.DataFrame:
        """Extract all economic indicators"""
        all_indicators = []
        
        for name, series_id in self.indicators.items():
            indicator_df = self.get_economic_indicator(series_id, start_date, end_date)
            if not indicator_df.empty:
                all_indicators.append(indicator_df)
        
        if all_indicators:
            # Merge all indicators on date
            combined_econ = all_indicators[0]
            for df in all_indicators[1:]:
                combined_econ = pd.merge(combined_econ, df, on='date', how='outer')
            
            # Forward fill missing values
            combined_econ = combined_econ.sort_values('date').fillna(method='ffill')
            
            # Save to file
            econ_file = self.data_dir / 'economic_data.csv'
            combined_econ.to_csv(econ_file, index=False)
            logger.info(f"Economic data saved to {econ_file}")
            
            return combined_econ
        else:
            logger.warning("No economic data retrieved")
            return pd.DataFrame()


class DataTransformer:
    """Transform and integrate all data sources"""
    
    def __init__(self, output_dir: str = "data/processed"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def integrate_all_data(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame, 
                          prices_df: pd.DataFrame, weather_df: pd.DataFrame = None,
                          economic_df: pd.DataFrame = None) -> pd.DataFrame:
        """Integrate all data sources into a single dataset"""
        logger.info("Integrating all data sources...")
        
        # Start with sales data
        integrated_df = sales_df.copy()
        
        # Merge with calendar data
        integrated_df = pd.merge(integrated_df, calendar_df, on='d', how='left')
        
        # Merge with pricing data
        integrated_df = pd.merge(
            integrated_df, 
            prices_df, 
            on=['store_id', 'item_id', 'wm_yr_wk'], 
            how='left'
        )
        
        # Merge with weather data if available
        if weather_df is not None and not weather_df.empty:
            integrated_df = pd.merge(
                integrated_df,
                weather_df[['date', 'state_id', 'temp', 'humidity', 'precip', 'windspeed']],
                on=['date', 'state_id'],
                how='left'
            )
        
        # Merge with economic data if available
        if economic_df is not None and not economic_df.empty:
            # Create year-month for merging monthly economic data
            integrated_df['year_month'] = integrated_df['date'].dt.to_period('M')
            economic_df['year_month'] = economic_df['date'].dt.to_period('M')
            
            integrated_df = pd.merge(
                integrated_df,
                economic_df.drop('date', axis=1),
                on='year_month',
                how='left'
            )
            
            integrated_df = integrated_df.drop('year_month', axis=1)
        
        # Data quality checks
        logger.info(f"Integrated dataset shape: {integrated_df.shape}")
        logger.info(f"Missing values per column:\n{integrated_df.isnull().sum()}")
        
        return integrated_df
    
    def create_feature_store(self, integrated_df: pd.DataFrame) -> pd.DataFrame:
        """Create advanced features and save to feature store"""
        logger.info("Creating feature store...")
        
        # Sort by item and date for time series features
        feature_df = integrated_df.sort_values(['id', 'date']).copy()
        
        # Lag features (sales from previous periods)
        for lag in [1, 7, 14, 28]:
            feature_df[f'demand_lag_{lag}'] = feature_df.groupby('id')['demand'].shift(lag)
        
        # Rolling window features
        for window in [7, 14, 28]:
            feature_df[f'demand_rolling_mean_{window}'] = (
                feature_df.groupby('id')['demand']
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            feature_df[f'demand_rolling_std_{window}'] = (
                feature_df.groupby('id')['demand']
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
        
        # Price features
        if 'sell_price' in feature_df.columns:
            feature_df['price_change'] = (
                feature_df.groupby('id')['sell_price'].pct_change()
            )
            
            feature_df['price_relative_to_category'] = (
                feature_df['sell_price'] / 
                feature_df.groupby(['cat_id', 'date'])['sell_price'].transform('mean')
            )
        
        # Holiday and event features
        feature_df['is_holiday'] = (
            feature_df['event_name_1_exists'] | feature_df['event_name_2_exists']
        )
        
        # Seasonal features
        feature_df['sin_dayofyear'] = np.sin(2 * np.pi * feature_df['dayofyear'] / 365.25)
        feature_df['cos_dayofyear'] = np.cos(2 * np.pi * feature_df['dayofyear'] / 365.25)
        feature_df['sin_week'] = np.sin(2 * np.pi * feature_df['week'] / 52)
        feature_df['cos_week'] = np.cos(2 * np.pi * feature_df['week'] / 52)
        
        # Save feature store
        feature_file = self.output_dir / 'feature_store.csv'
        feature_df.to_csv(feature_file, index=False)
        logger.info(f"Feature store saved to {feature_file}")
        
        return feature_df


class ETLPipeline:
    """Main ETL Pipeline orchestrator"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.m5_extractor = M5DataExtractor(config.get('raw_data_dir', 'data/raw'))
        
        # Initialize external data extractors if API keys provided
        self.weather_extractor = None
        self.economic_extractor = None
        
        if config.get('weather_api_key'):
            self.weather_extractor = WeatherDataExtractor(
                config['weather_api_key'], 
                config.get('external_data_dir', 'data/external')
            )
        
        if config.get('fred_api_key'):
            self.economic_extractor = EconomicDataExtractor(
                config['fred_api_key'],
                config.get('external_data_dir', 'data/external')
            )
        
        self.transformer = DataTransformer(config.get('processed_data_dir', 'data/processed'))
    
    def run_full_pipeline(self) -> bool:
        """Run the complete ETL pipeline"""
        logger.info("Starting full ETL pipeline...")
        
        try:
            # Extract M5 data
            if not self.m5_extractor.download_m5_data():
                logger.error("Failed to download M5 data")
                return False
            
            sales_df = self.m5_extractor.extract_sales_data()
            calendar_df = self.m5_extractor.extract_calendar_data()
            prices_df = self.m5_extractor.extract_prices_data()
            
            # Extract external data
            weather_df = None
            economic_df = None
            
            if self.weather_extractor:
                weather_df = self.weather_extractor.extract_all_weather_data()
            
            if self.economic_extractor:
                economic_df = self.economic_extractor.extract_all_economic_data()
            
            # Transform and integrate
            integrated_df = self.transformer.integrate_all_data(
                sales_df, calendar_df, prices_df, weather_df, economic_df
            )
            
            # Create feature store
            feature_df = self.transformer.create_feature_store(integrated_df)
            
            logger.info("ETL pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"ETL pipeline failed: {e}")
            return False


# Configuration and execution
if __name__ == "__main__":
    # Configuration
    config = {
        'raw_data_dir': 'data/raw',
        'external_data_dir': 'data/external', 
        'processed_data_dir': 'data/processed',
        'weather_api_key': os.getenv('WEATHER_API_KEY'),  # Get from environment
        'fred_api_key': os.getenv('FRED_API_KEY')         # Get from environment
    }
    
    # Run pipeline
    pipeline = ETLPipeline(config)
    success = pipeline.run_full_pipeline()
    
    if success:
        print("✅ ETL Pipeline completed successfully!")
    else:
        print("❌ ETL Pipeline failed!")
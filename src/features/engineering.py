# Advanced Feature Engineering for Demand Forecasting
# Path: src/features/engineering.py

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import logging
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """Advanced feature engineering for demand forecasting"""
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.label_encoders = {}
        self.scalers = {}
        self.feature_names = []
        
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features"""
        logger.info("Creating temporal features...")
        
        df = df.copy()
        
        # Ensure date is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        # Basic date features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        df['is_month_start'] = df['date'].dt.is_month_start.astype(int)
        df['is_month_end'] = df['date'].dt.is_month_end.astype(int)
        df['is_quarter_start'] = df['date'].dt.is_quarter_start.astype(int)
        df['is_quarter_end'] = df['date'].dt.is_quarter_end.astype(int)
        df['is_year_start'] = df['date'].dt.is_year_start.astype(int)
        df['is_year_end'] = df['date'].dt.is_year_end.astype(int)
        
        # Cyclical encoding for seasonality
        df['sin_dayofyear'] = np.sin(2 * np.pi * df['dayofyear'] / 365.25)
        df['cos_dayofyear'] = np.cos(2 * np.pi * df['dayofyear'] / 365.25)
        df['sin_week'] = np.sin(2 * np.pi * df['week'] / 52)
        df['cos_week'] = np.cos(2 * np.pi * df['week'] / 52)
        df['sin_month'] = np.sin(2 * np.pi * df['month'] / 12)
        df['cos_month'] = np.cos(2 * np.pi * df['month'] / 12)
        df['sin_dayofweek'] = np.sin(2 * np.pi * df['dayofweek'] / 7)
        df['cos_dayofweek'] = np.cos(2 * np.pi * df['dayofweek'] / 7)
        
        logger.info(f"Created {len([c for c in df.columns if c.startswith(('sin_', 'cos_', 'is_', 'year', 'month', 'day', 'week', 'quarter'))])} temporal features")
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'demand', 
                           lags: List[int] = None) -> pd.DataFrame:
        """Create lag features for time series"""
        logger.info("Creating lag features...")
        
        if lags is None:
            lags = [1, 7, 14, 21, 28, 35, 42]
        
        df = df.copy()
        df = df.sort_values(['id', 'date'])
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df.groupby('id')[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str = 'demand',
                               windows: List[int] = None) -> pd.DataFrame:
        """Create rolling window statistical features"""
        logger.info("Creating rolling features...")
        
        if windows is None:
            windows = [7, 14, 28, 56]
        
        df = df.copy()
        df = df.sort_values(['id', 'date'])
        
        for window in windows:
            # Rolling statistics
            df[f'{target_col}_rolling_mean_{window}'] = (
                df.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .mean()
                .reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_std_{window}'] = (
                df.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .std()
                .reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_min_{window}'] = (
                df.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .min()
                .reset_index(0, drop=True)
            )
            
            df[f'{target_col}_rolling_max_{window}'] = (
                df.groupby('id')[target_col]
                .rolling(window=window, min_periods=1)
                .max()
                .reset_index(0, drop=True)
            )
            
            # Exponential weighted features
            df[f'{target_col}_ewm_{window}'] = (
                df.groupby('id')[target_col]
                .ewm(span=window, adjust=False)
                .mean()
                .reset_index(0, drop=True)
            )
        
        logger.info(f"Created {len(windows) * 5} rolling features")
        return df
    
    def create_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create price-related features"""
        logger.info("Creating price features...")
        
        df = df.copy()
        
        if 'sell_price' not in df.columns:
            logger.warning("No sell_price column found, skipping price features")
            return df
        
        # Price change features
        df = df.sort_values(['id', 'date'])
        df['price_change'] = df.groupby('id')['sell_price'].pct_change()
        df['price_change_abs'] = df['price_change'].abs()
        
        # Price relative to category/store averages
        df['price_relative_to_category'] = (
            df['sell_price'] / 
            df.groupby(['cat_id', 'date'])['sell_price'].transform('mean')
        )
        
        df['price_relative_to_store'] = (
            df['sell_price'] / 
            df.groupby(['store_id', 'date'])['sell_price'].transform('mean')
        )
        
        # Price statistics
        df['price_rolling_mean_28'] = (
            df.groupby('id')['sell_price']
            .rolling(window=28, min_periods=1)
            .mean()
            .reset_index(0, drop=True)
        )
        
        df['price_relative_to_avg'] = df['sell_price'] / df['price_rolling_mean_28']
        
        # Price bins
        df['price_bin'] = pd.qcut(df['sell_price'], q=5, labels=['very_low', 'low', 'medium', 'high', 'very_high'])
        
        logger.info("Created price features")
        return df
    
    def create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create event and holiday features"""
        logger.info("Creating event features...")
        
        df = df.copy()
        
        # Basic event features from M5 data
        if 'event_name_1' in df.columns:
            df['has_event_1'] = df['event_name_1'].notna().astype(int)
        if 'event_name_2' in df.columns:
            df['has_event_2'] = df['event_name_2'].notna().astype(int)
        
        df['has_any_event'] = df.get('has_event_1', 0) | df.get('has_event_2', 0)
        
        # SNAP benefits features (if available)
        snap_cols = [col for col in df.columns if col.startswith('snap_')]
        if snap_cols:
            df['snap_total'] = df[snap_cols].sum(axis=1)
            df['has_snap'] = (df['snap_total'] > 0).astype(int)
        
        # Major holidays (approximate dates)
        df['is_christmas_week'] = ((df['month'] == 12) & (df['day'] >= 22)).astype(int)
        df['is_thanksgiving_week'] = ((df['month'] == 11) & (df['day'] >= 22) & (df['day'] <= 28)).astype(int)
        df['is_new_year_week'] = ((df['month'] == 1) & (df['day'] <= 7)).astype(int)
        df['is_memorial_day'] = ((df['month'] == 5) & (df['dayofweek'] == 0) & (df['day'] >= 25)).astype(int)
        df['is_labor_day'] = ((df['month'] == 9) & (df['dayofweek'] == 0) & (df['day'] <= 7)).astype(int)
        
        logger.info("Created event features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features"""
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Category-Season interactions
        df['cat_month'] = df['cat_id'] + '_' + df['month'].astype(str)
        df['cat_quarter'] = df['cat_id'] + '_' + df['quarter'].astype(str)
        
        # Store-Season interactions
        df['store_month'] = df['store_id'] + '_' + df['month'].astype(str)
        df['store_quarter'] = df['store_id'] + '_' + df['quarter'].astype(str)
        
        # Price-Event interactions
        if 'sell_price' in df.columns and 'has_any_event' in df.columns:
            df['price_event_interaction'] = df['sell_price'] * df['has_any_event']
        
        # Weekend-Category interactions
        df['weekend_cat'] = df['is_weekend'].astype(str) + '_' + df['cat_id']
        
        logger.info("Created interaction features")
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, categorical_cols: List[str] = None) -> pd.DataFrame:
        """Encode categorical features using target encoding and label encoding"""
        logger.info("Encoding categorical features...")
        
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id', 
                              'cat_month', 'cat_quarter', 'store_month', 'store_quarter', 'weekend_cat']
        
        # Filter for existing columns
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
            else:
                # Handle unseen categories
                known_classes = set(self.label_encoders[col].classes_)
                df[f'{col}_temp'] = df[col].astype(str)
                df.loc[~df[f'{col}_temp'].isin(known_classes), f'{col}_temp'] = 'unknown'
                
                if 'unknown' not in known_classes:
                    # Add unknown class
                    self.label_encoders[col].classes_ = np.append(self.label_encoders[col].classes_, 'unknown')
                
                df[f'{col}_encoded'] = self.label_encoders[col].transform(df[f'{col}_temp'])
                df = df.drop(f'{col}_temp', axis=1)
        
        logger.info(f"Encoded {len(categorical_cols)} categorical features")
        return df
    
    def create_target_encoding_features(self, df: pd.DataFrame, target_col: str = 'demand',
                                      categorical_cols: List[str] = None) -> pd.DataFrame:
        """Create target encoding features (mean encoding)"""
        logger.info("Creating target encoding features...")
        
        df = df.copy()
        
        if categorical_cols is None:
            categorical_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        # Filter for existing columns
        categorical_cols = [col for col in categorical_cols if col in df.columns]
        
        for col in categorical_cols:
            # Calculate target mean for each category
            target_mean = df.groupby(col)[target_col].mean()
            df[f'{col}_target_mean'] = df[col].map(target_mean)
            
            # Calculate target std for each category
            target_std = df.groupby(col)[target_col].std()
            df[f'{col}_target_std'] = df[col].map(target_std).fillna(0)
            
            # Calculate target count for each category
            target_count = df.groupby(col)[target_col].count()
            df[f'{col}_target_count'] = df[col].map(target_count)
        
        logger.info(f"Created target encoding for {len(categorical_cols)} features")
        return df
    
    def create_external_signal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features from external signals (weather, economic)"""
        logger.info("Creating external signal features...")
        
        df = df.copy()
        
        # Weather features
        if 'temp' in df.columns:
            df['temp_squared'] = df['temp'] ** 2
            df['temp_binned'] = pd.cut(df['temp'], bins=5, labels=['very_cold', 'cold', 'mild', 'warm', 'hot'])
            
            # Temperature relative to seasonal average
            df['temp_seasonal_avg'] = df.groupby(['month'])['temp'].transform('mean')
            df['temp_relative_seasonal'] = df['temp'] - df['temp_seasonal_avg']
        
        if 'humidity' in df.columns:
            df['humidity_binned'] = pd.cut(df['humidity'], bins=3, labels=['low', 'medium', 'high'])
        
        if 'precip' in df.columns:
            df['has_precipitation'] = (df['precip'] > 0).astype(int)
            df['heavy_rain'] = (df['precip'] > df['precip'].quantile(0.75)).astype(int)
        
        # Economic features (if available)
        economic_cols = ['UNRATE', 'UMCSENT', 'CPIAUCSL', 'GASREGW']
        for col in economic_cols:
            if col in df.columns:
                # Create trend features
                df[f'{col}_trend'] = df.groupby('state_id')[col].pct_change(periods=12)  # Year-over-year change
                df[f'{col}_ma_3'] = df.groupby('state_id')[col].rolling(window=3).mean().reset_index(0, drop=True)
        
        logger.info("Created external signal features")
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, target_col: str = 'demand') -> Tuple[pd.DataFrame, List[str]]:
        """Prepare final feature set for modeling"""
        logger.info("Preparing features for modeling...")
        
        df = df.copy()
        
        # Remove non-feature columns
        exclude_cols = ['id', 'date', 'd', target_col, 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        # Include encoded versions but exclude original categorical
        feature_cols = []
        for col in df.columns:
            if col not in exclude_cols:
                if not any(col.endswith(suffix) for suffix in ['_temp', '_bin']) or col.endswith('_encoded'):
                    feature_cols.append(col)
        
        # Handle categorical features that weren't encoded
        for col in feature_cols.copy():
            if df[col].dtype == 'object':
                logger.warning(f"Removing object column {col} - should be encoded first")
                feature_cols.remove(col)
        
        # Create feature matrix
        X = df[feature_cols].copy()
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan)
        
        # Fill missing values with median for numeric features
        for col in X.columns:
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
        
        self.feature_names = feature_cols
        logger.info(f"Prepared {len(feature_cols)} features for modeling")
        
        return X, feature_cols
    
    def create_all_features(self, df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        """Create all features in the correct order"""
        logger.info("Creating all features...")
        
        # Step 1: Temporal features
        df = self.create_temporal_features(df)
        
        # Step 2: Lag features (must be before rolling features)
        df = self.create_lag_features(df, target_col)
        
        # Step 3: Rolling features
        df = self.create_rolling_features(df, target_col)
        
        # Step 4: Price features
        df = self.create_price_features(df)
        
        # Step 5: Event features
        df = self.create_event_features(df)
        
        # Step 6: External signal features
        df = self.create_external_signal_features(df)
        
        # Step 7: Interaction features
        df = self.create_interaction_features(df)
        
        # Step 8: Target encoding
        df = self.create_target_encoding_features(df, target_col)
        
        # Step 9: Categorical encoding
        df = self.encode_categorical_features(df)
        
        logger.info("All features created successfully")
        return df


class TimeSeriesSplitter:
    """Custom time series train/validation/test splitter"""
    
    def __init__(self, test_size: int = 28, validation_size: int = 28):
        self.test_size = test_size
        self.validation_size = validation_size
    
    def split_by_time(self, df: pd.DataFrame, date_col: str = 'date') -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test based on time"""
        logger.info("Splitting data by time...")
        
        df = df.sort_values(date_col)
        
        # Get unique dates
        unique_dates = sorted(df[date_col].unique())
        total_days = len(unique_dates)
        
        # Calculate split points
        test_start_idx = total_days - self.test_size
        val_start_idx = test_start_idx - self.validation_size
        
        test_start_date = unique_dates[test_start_idx]
        val_start_date = unique_dates[val_start_idx]
        
        # Split data
        train_df = df[df[date_col] < val_start_date].copy()
        val_df = df[(df[date_col] >= val_start_date) & (df[date_col] < test_start_date)].copy()
        test_df = df[df[date_col] >= test_start_date].copy()
        
        logger.info(f"Train: {len(train_df)} samples ({train_df[date_col].min()} to {train_df[date_col].max()})")
        logger.info(f"Validation: {len(val_df)} samples ({val_df[date_col].min()} to {val_df[date_col].max()})")
        logger.info(f"Test: {len(test_df)} samples ({test_df[date_col].min()} to {test_df[date_col].max()})")
        
        return train_df, val_df, test_df
    
    def create_time_series_cv(self, df: pd.DataFrame, n_splits: int = 3, date_col: str = 'date'):
        """Create time series cross-validation splits"""
        logger.info(f"Creating {n_splits} time series CV splits...")
        
        unique_dates = sorted(df[date_col].unique())
        total_days = len(unique_dates)
        
        # Calculate step size
        step_size = (total_days - self.test_size - self.validation_size) // n_splits
        
        splits = []
        for i in range(n_splits):
            val_end_idx = total_days - self.test_size - (i * step_size)
            val_start_idx = val_end_idx - self.validation_size
            
            val_start_date = unique_dates[val_start_idx]
            val_end_date = unique_dates[val_end_idx]
            
            train_mask = df[date_col] < val_start_date
            val_mask = (df[date_col] >= val_start_date) & (df[date_col] < val_end_date)
            
            train_idx = df[train_mask].index.tolist()
            val_idx = df[val_mask].index.tolist()
            
            splits.append((train_idx, val_idx))
        
        logger.info(f"Created {len(splits)} CV splits")
        return splits


def main():
    """Test feature engineering pipeline"""
    # This would be used for testing
    logger.info("Feature engineering module loaded successfully")

if __name__ == "__main__":
    main()
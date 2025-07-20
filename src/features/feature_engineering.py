# Advanced Feature Engineering for M5 Demand Forecasting
# File: src/features/feature_engineering.py

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import warnings
from datetime import datetime, timedelta
from sklearn.preprocessing import LabelEncoder
import logging

logger = logging.getLogger(__name__)

class AdvancedFeatureEngineer:
    """
    Comprehensive feature engineering for M5 demand forecasting.
    Based on EDA insights: handles intermittent demand (68.2% zeros), 
    category differences, and temporal patterns.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        self.lag_periods = self.config.get('lag_periods', [1, 7, 14, 21, 28, 35, 42])
        self.rolling_windows = self.config.get('rolling_windows', [7, 14, 28, 56])
        self.ma_windows = self.config.get('ma_windows', [7, 14, 28])
        
        # Encoders for categorical variables
        self.label_encoders = {}
        self.fitted = False
        
        # Feature names tracking
        self.feature_names = []
        
    def create_all_features(self, sales_df: pd.DataFrame, calendar_df: pd.DataFrame, 
                          prices_df: pd.DataFrame, target_col: str = 'demand') -> pd.DataFrame:
        """
        Create all features for the demand forecasting model.
        
        Args:
            sales_df: Sales data in long format (from your EDA)
            calendar_df: Calendar data with dates and events
            prices_df: Pricing data
            target_col: Target column name
            
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Creating comprehensive feature set...")
        
        # Start with sales data
        df = sales_df.copy()
        
        # Merge with calendar data
        df = self._merge_calendar_data(df, calendar_df)
        
        # Merge with pricing data
        df = self._merge_pricing_data(df, prices_df)
        
        # Create temporal features
        df = self._create_temporal_features(df)
        
        # Create lag features (crucial for intermittent demand)
        df = self._create_lag_features(df, target_col)
        
        # Create rolling window features
        df = self._create_rolling_features(df, target_col)
        
        # Create demand pattern features
        df = self._create_demand_pattern_features(df, target_col)
        
        # Create price elasticity features
        df = self._create_price_features(df, target_col)
        
        # Create categorical features
        df = self._create_categorical_features(df)
        
        # Create business logic features
        df = self._create_business_features(df, target_col)
        
        # Create interaction features
        df = self._create_interaction_features(df)
        
        # Create external event features
        df = self._create_event_features(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        self.fitted = True
        
        return df
    
    def _merge_calendar_data(self, df: pd.DataFrame, calendar_df: pd.DataFrame) -> pd.DataFrame:
        """Merge calendar data with sales data"""
        logger.info("Merging calendar data...")
        
        # Ensure date is datetime
        calendar_df = calendar_df.copy()
        calendar_df['date'] = pd.to_datetime(calendar_df['date'])
        
        # Merge on 'd' column
        df = df.merge(calendar_df, on='d', how='left')
        
        # Encode categorical columns from calendar data
        categorical_cols = ['weekday', 'event_name_1', 'event_type_1', 'event_name_2', 'event_type_2']
        
        for col in categorical_cols:
            if col in df.columns:
                # Fill NaN values first
                df[col] = df[col].fillna('None')
                
                # Encode as numeric
                if not self.fitted:
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col].astype(str))
                else:
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col].astype(str))
                    except ValueError:
                        df[f'{col}_encoded'] = 0
                
                # Drop original column to avoid object types
                df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _merge_pricing_data(self, df: pd.DataFrame, prices_df: pd.DataFrame) -> pd.DataFrame:
        """Merge pricing data with sales data"""
        logger.info("Merging pricing data...")
        
        # Merge on store_id, item_id, and wm_yr_wk
        df = df.merge(prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        
        # Forward fill missing prices (common in M5 data)
        df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].fillna(method='ffill')
        df['sell_price'] = df.groupby(['store_id', 'item_id'])['sell_price'].fillna(method='bfill')
        
        return df
    
    def _create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create comprehensive temporal features based on EDA patterns"""
        logger.info("Creating temporal features...")
        
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        
        # Basic temporal features
        df['year'] = df['date'].dt.year
        df['month'] = df['date'].dt.month
        df['day'] = df['date'].dt.day
        df['dayofweek'] = df['date'].dt.dayofweek
        df['dayofyear'] = df['date'].dt.dayofyear
        df['week'] = df['date'].dt.isocalendar().week
        df['quarter'] = df['date'].dt.quarter
        
        # Weekend indicator (EDA showed weekend patterns)
        df['is_weekend'] = df['dayofweek'].isin([5, 6]).astype(int)
        
        # Month-end and month-start effects
        df['is_month_start'] = (df['day'] <= 3).astype(int)
        df['is_month_end'] = (df['day'] >= 28).astype(int)
        
        # Cyclical encoding for temporal features
        for col, max_val in [('month', 12), ('dayofweek', 7), ('day', 31)]:
            df[f'{col}_sin'] = np.sin(2 * np.pi * df[col] / max_val)
            df[f'{col}_cos'] = np.cos(2 * np.pi * df[col] / max_val)
        
        # Days since start
        min_date = df['date'].min()
        df['days_since_start'] = (df['date'] - min_date).dt.days
        
        # Relative position in year
        df['relative_day_of_year'] = df['dayofyear'] / 365.25
        
        return df
    
    def _create_lag_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create lag features - crucial for intermittent demand (68.2% zeros)"""
        logger.info("Creating lag features...")
        
        df = df.copy()
        
        # Sort by identifiers and date
        df = df.sort_values(['store_id', 'item_id', 'date'])
        
        # Create lag features
        for lag in self.lag_periods:
            col_name = f'{target_col}_lag_{lag}'
            df[col_name] = df.groupby(['store_id', 'item_id'])[target_col].shift(lag)
            
            # Log of lag features (handle zeros)
            df[f'{col_name}_log'] = np.log1p(df[col_name])
            
            # Binary indicators for non-zero lags (important for intermittent demand)
            df[f'{col_name}_nonzero'] = (df[col_name] > 0).astype(int)
        
        # Recent activity indicators
        for window in [7, 14, 28]:
            df[f'active_days_last_{window}'] = df.groupby(['store_id', 'item_id'])[target_col]\
                .rolling(window=window, min_periods=1)\
                .apply(lambda x: (x > 0).sum())\
                .reset_index(level=[0,1], drop=True)
        
        return df
    
    def _create_rolling_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create rolling window features"""
        logger.info("Creating rolling window features...")
        
        df = df.copy()
        df = df.sort_values(['store_id', 'item_id', 'date'])
        
        for window in self.rolling_windows:
            # Rolling statistics
            rolling = df.groupby(['store_id', 'item_id'])[target_col].rolling(window=window, min_periods=1)
            
            df[f'{target_col}_rolling_mean_{window}'] = rolling.mean().reset_index(level=[0,1], drop=True)
            df[f'{target_col}_rolling_std_{window}'] = rolling.std().reset_index(level=[0,1], drop=True)
            df[f'{target_col}_rolling_max_{window}'] = rolling.max().reset_index(level=[0,1], drop=True)
            df[f'{target_col}_rolling_min_{window}'] = rolling.min().reset_index(level=[0,1], drop=True)
            
            # Coefficient of variation (std/mean) - important for intermittent demand
            df[f'{target_col}_rolling_cv_{window}'] = (
                df[f'{target_col}_rolling_std_{window}'] / 
                (df[f'{target_col}_rolling_mean_{window}'] + 1e-8)
            )
            
            # Trend features
            df[f'{target_col}_trend_{window}'] = (
                df[target_col] - df[f'{target_col}_rolling_mean_{window}']
            ) / (df[f'{target_col}_rolling_std_{window}'] + 1e-8)
            
        # Exponentially weighted moving averages
        for alpha in [0.1, 0.3, 0.5]:
            df[f'{target_col}_ewm_{alpha}'] = df.groupby(['store_id', 'item_id'])[target_col]\
                .ewm(alpha=alpha).mean().reset_index(level=[0,1], drop=True)
        
        return df
    
    def _create_demand_pattern_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create demand pattern features specific to M5 intermittent demand"""
        logger.info("Creating demand pattern features...")
        
        df = df.copy()
        
        # Days since last sale
        df['days_since_last_sale'] = 0
        for (store, item), group in df.groupby(['store_id', 'item_id']):
            sale_dates = group[group[target_col] > 0]['date']
            for idx, row in group.iterrows():
                recent_sales = sale_dates[sale_dates < row['date']]
                if len(recent_sales) > 0:
                    df.loc[idx, 'days_since_last_sale'] = (row['date'] - recent_sales.max()).days
                else:
                    df.loc[idx, 'days_since_last_sale'] = 999  # No previous sales
        
        # Intermittency indicators
        for window in [14, 28, 56]:
            # Intermittency ratio (zero sales / total sales)
            df[f'intermittency_ratio_{window}'] = df.groupby(['store_id', 'item_id'])[target_col]\
                .rolling(window=window, min_periods=1)\
                .apply(lambda x: (x == 0).mean())\
                .reset_index(level=[0,1], drop=True)
            
            # Average demand intensity (demand when non-zero)
            df[f'demand_intensity_{window}'] = df.groupby(['store_id', 'item_id'])[target_col]\
                .rolling(window=window, min_periods=1)\
                .apply(lambda x: x[x > 0].mean() if (x > 0).any() else 0)\
                .reset_index(level=[0,1], drop=True)
        
        # Demand velocity (change in recent demand)
        for lag1, lag2 in [(7, 14), (14, 28), (28, 56)]:
            df[f'demand_velocity_{lag1}_{lag2}'] = (
                df[f'{target_col}_rolling_mean_{lag1}'] - df[f'{target_col}_rolling_mean_{lag2}']
            )
        
        return df
    
    def _create_price_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create price elasticity and pricing features"""
        logger.info("Creating price features...")
        
        df = df.copy()
        
        if 'sell_price' not in df.columns:
            logger.warning("No price data available")
            return df
        
        # Price change indicators
        df['price_change'] = df.groupby(['store_id', 'item_id'])['sell_price'].diff()
        df['price_change_pct'] = df.groupby(['store_id', 'item_id'])['sell_price'].pct_change()
        
        # Price relative to item average
        df['price_vs_item_avg'] = df.groupby('item_id')['sell_price'].transform(
            lambda x: x / x.mean()
        )
        
        # Price relative to category average
        df['price_vs_cat_avg'] = df.groupby('cat_id')['sell_price'].transform(
            lambda x: x / x.mean()
        )
        
        # Price momentum
        for window in [7, 14, 28]:
            df[f'price_momentum_{window}'] = df.groupby(['store_id', 'item_id'])['sell_price']\
                .rolling(window=window).mean().reset_index(level=[0,1], drop=True)
        
        # Price elasticity indicators (when price changes, how does demand change)
        df['price_elasticity_signal'] = (
            df['price_change_pct'].fillna(0) * df[f'{target_col}_lag_7'].fillna(0)
        )
        
        # Discount indicators
        df['is_discount'] = (df['price_change'] < -0.01).astype(int)
        df['discount_magnitude'] = np.where(df['price_change'] < 0, 
                                          abs(df['price_change']), 0)
        
        return df
    
    def _create_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create and encode categorical features"""
        logger.info("Creating categorical features...")
        
        df = df.copy()
        
        # Categorical columns to encode
        cat_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        
        for col in cat_cols:
            if col in df.columns:
                if not self.fitted:
                    # Fit encoder
                    self.label_encoders[col] = LabelEncoder()
                    df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(
                        df[col].astype(str)
                    )
                else:
                    # Transform using fitted encoder
                    try:
                        df[f'{col}_encoded'] = self.label_encoders[col].transform(
                            df[col].astype(str)
                        )
                    except ValueError:
                        # Handle unseen categories
                        df[f'{col}_encoded'] = 0
        
        # Hierarchical features - encode these too
        df['state_cat_combined'] = df['state_id'].astype(str) + '_' + df['cat_id'].astype(str)
        df['store_cat_combined'] = df['store_id'].astype(str) + '_' + df['cat_id'].astype(str)
        df['store_dept_combined'] = df['store_id'].astype(str) + '_' + df['dept_id'].astype(str)
        
        # Encode the hierarchical features
        for col in ['state_cat_combined', 'store_cat_combined', 'store_dept_combined']:
            if not self.fitted:
                self.label_encoders[col] = LabelEncoder()
                df[f'{col}_encoded'] = self.label_encoders[col].fit_transform(df[col])
            else:
                try:
                    df[f'{col}_encoded'] = self.label_encoders[col].transform(df[col])
                except ValueError:
                    df[f'{col}_encoded'] = 0
            
            # Drop the original combined column (keep only encoded version)
            df.drop(col, axis=1, inplace=True)
        
        return df
    
    def _create_business_features(self, df: pd.DataFrame, target_col: str) -> pd.DataFrame:
        """Create business logic features based on domain knowledge"""
        logger.info("Creating business logic features...")
        
        df = df.copy()
        
        # Product lifecycle stage (based on sales history)
        item_first_sale = df[df[target_col] > 0].groupby('item_id')['date'].min()
        df['item_age_days'] = df.apply(
            lambda row: (row['date'] - item_first_sale.get(row['item_id'], row['date'])).days,
            axis=1
        )
        
        # Store performance relative to other stores
        df['store_performance'] = df.groupby(['date', 'item_id'])[target_col].transform(
            lambda x: x / (x.mean() + 1e-8)
        )
        
        # Category competition (how many items in same category in same store)
        df['category_competition'] = df.groupby(['store_id', 'cat_id', 'date'])['item_id']\
            .transform('nunique')
        
        # Seasonal item indicator (higher sales in certain months)
        monthly_sales = df.groupby(['item_id', 'month'])[target_col].mean()
        df['seasonal_strength'] = df.apply(
            lambda row: monthly_sales.get((row['item_id'], row['month']), 0) / 
                       (monthly_sales.groupby('item_id').mean().get(row['item_id'], 1e-8)),
            axis=1
        )
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between different variables"""
        logger.info("Creating interaction features...")
        
        df = df.copy()
        
        # Price-time interactions
        if 'sell_price' in df.columns:
            df['price_month_interaction'] = df['sell_price'] * df['month']
            df['price_weekend_interaction'] = df['sell_price'] * df['is_weekend']
            df['price_holiday_interaction'] = df['sell_price'] * df.get('is_holiday', 0)
        
        # Temporal-category interactions
        df['month_cat_interaction'] = df['month'] * df.get('cat_id_encoded', 0)
        df['dow_store_interaction'] = df['dayofweek'] * df.get('store_id_encoded', 0)
        
        # Lag-price interactions
        if 'sell_price' in df.columns and 'demand_lag_7' in df.columns:
            df['lag7_price_interaction'] = df['demand_lag_7'] * df['sell_price']
        
        return df
    
    def _create_event_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create features for events and holidays"""
        logger.info("Creating event features...")
        
        df = df.copy()
        
        # Event indicators (now these should be encoded columns)
        event_cols = ['event_name_1_encoded', 'event_type_1_encoded', 'event_name_2_encoded', 'event_type_2_encoded']
        
        for col in event_cols:
            if col in df.columns:
                df[f'{col}_exists'] = (df[col] > 0).astype(int)
        
        # SNAP benefits (important for M5)
        snap_cols = ['snap_CA', 'snap_TX', 'snap_WI']
        for col in snap_cols:
            if col in df.columns:
                df[col] = df[col].astype(int)
        
        # Note: Holiday proximity features would need the original event names
        # For now, we'll skip this to avoid object columns
        # This could be enhanced later with a mapping of encoded values to holiday names
        
        return df
    
    def prepare_features_for_modeling(self, df: pd.DataFrame, 
                                    target_col: str) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare final feature set for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Select feature columns (exclude identifiers, target, and original categorical columns)
        exclude_cols = [
            'id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id',
            'd', 'date', target_col, 'wm_yr_wk', 'weekday', 'event_name_1', 'event_type_1',
            'event_name_2', 'event_type_2', 'state_cat', 'store_cat', 'store_dept'
        ]
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values and ensure all columns are numeric
        X = df[feature_cols].copy()
        y = df[target_col].copy() if target_col in df.columns else None
        
        # Convert all object/string columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                # Try to convert to numeric first
                X[col] = pd.to_numeric(X[col], errors='ignore')
                
                # If still object, encode as categorical
                if X[col].dtype == 'object':
                    if not self.fitted and col not in self.label_encoders:
                        self.label_encoders[col] = LabelEncoder()
                        X[col] = self.label_encoders[col].fit_transform(X[col].astype(str))
                    elif col in self.label_encoders:
                        # Transform using fitted encoder
                        try:
                            X[col] = self.label_encoders[col].transform(X[col].astype(str))
                        except ValueError:
                            # Handle unseen categories
                            X[col] = 0
                    else:
                        # Fallback: simple numeric encoding
                        X[col] = pd.Categorical(X[col]).codes
        
        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))
        X = X.fillna(0)  # For any remaining missing values
        
        # Ensure all columns are numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.Categorical(X[col]).codes
        
        # Convert to appropriate numeric types
        X = X.astype('float32')
        
        # Final safety check - ensure no object columns remain
        object_cols = X.select_dtypes(include=['object']).columns
        if len(object_cols) > 0:
            logger.warning(f"Found remaining object columns: {list(object_cols)}")
            for col in object_cols:
                X[col] = pd.Categorical(X[col]).codes
            X = X.astype('float32')
        
        # Store feature names
        self.feature_names = X.columns.tolist()
        
        logger.info(f"Final feature set: {X.shape[1]} features")
        logger.info(f"Features: {', '.join(self.feature_names[:10])}...")
        logger.info(f"Data types: {X.dtypes.value_counts().to_dict()}")
        
        return X, y
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Group features by type for analysis"""
        groups = {
            'temporal': [f for f in self.feature_names if any(x in f for x in ['month', 'day', 'year', 'week', 'sin', 'cos'])],
            'lag': [f for f in self.feature_names if 'lag_' in f],
            'rolling': [f for f in self.feature_names if 'rolling_' in f or 'ewm_' in f],
            'price': [f for f in self.feature_names if 'price' in f or 'discount' in f],
            'demand_patterns': [f for f in self.feature_names if any(x in f for x in ['intermittency', 'intensity', 'velocity'])],
            'categorical': [f for f in self.feature_names if 'encoded' in f],
            'business': [f for f in self.feature_names if any(x in f for x in ['performance', 'competition', 'age'])],
            'events': [f for f in self.feature_names if any(x in f for x in ['event', 'snap', 'holiday'])]
        }
        
        return groups


def create_sample_features_for_testing():
    """Create a small sample dataset for testing feature engineering"""
    np.random.seed(42)
    
    # Sample data similar to M5 structure
    dates = pd.date_range('2020-01-01', '2020-03-31', freq='D')
    items = ['ITEM_001', 'ITEM_002']
    stores = ['STORE_001', 'STORE_002']
    
    data = []
    for date in dates:
        for item in items:
            for store in stores:
                demand = np.random.poisson(1.5) if np.random.random() > 0.3 else 0
                data.append({
                    'date': date,
                    'd': f"d_{(date - dates[0]).days + 1}",
                    'item_id': item,
                    'store_id': store,
                    'dept_id': 'DEPT_001',
                    'cat_id': 'FOODS',
                    'state_id': 'CA',
                    'demand': demand,
                    'wm_yr_wk': date.isocalendar()[1],
                    'sell_price': np.random.uniform(1.0, 10.0)
                })
    
    df = pd.DataFrame(data)
    
    # Create minimal calendar data
    calendar_data = []
    for i, date in enumerate(dates):
        calendar_data.append({
            'd': f"d_{i + 1}",
            'date': date,
            'wm_yr_wk': date.isocalendar()[1],
            'weekday': date.strftime('%A'),
            'wday': date.weekday() + 1,
            'month': date.month,
            'year': date.year,
            'event_name_1': None,
            'event_type_1': None,
            'event_name_2': None,
            'event_type_2': None,
            'snap_CA': 0,
            'snap_TX': 0,
            'snap_WI': 0
        })
    
    calendar_df = pd.DataFrame(calendar_data)
    
    # Create prices data
    prices_data = []
    for item in items:
        for store in stores:
            for week in range(1, 14):  # 13 weeks
                prices_data.append({
                    'store_id': store,
                    'item_id': item,
                    'wm_yr_wk': week,
                    'sell_price': np.random.uniform(1.0, 10.0)
                })
    
    prices_df = pd.DataFrame(prices_data)
    
    return df, calendar_df, prices_df


if __name__ == "__main__":
    # Test the feature engineering
    print("Testing Feature Engineering...")
    
    # Create sample data
    sales_df, calendar_df, prices_df = create_sample_features_for_testing()
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Create features
    features_df = feature_engineer.create_all_features(
        sales_df, calendar_df, prices_df, target_col='demand'
    )
    
    # Prepare for modeling
    X, y = feature_engineer.prepare_features_for_modeling(features_df, 'demand')
    
    print(f"Original data shape: {sales_df.shape}")
    print(f"Features shape: {X.shape}")
    print(f"Number of features: {X.shape[1]}")
    print("\nFeature groups:")
    for group, features in feature_engineer.get_feature_importance_groups().items():
        print(f"  {group}: {len(features)} features")
    
    print("\nFeature engineering test completed successfully!")
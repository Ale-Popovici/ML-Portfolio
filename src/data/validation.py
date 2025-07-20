# Data Validation and Quality Checks
# File: src/data/validation.py

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Tuple
import great_expectations as ge
from great_expectations.dataset import PandasDataset

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation for demand forecasting pipeline"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.validation_results = {}
    
    def validate_m5_data(self) -> bool:
        """Validate M5 competition data integrity"""
        logger.info("Validating M5 data...")
        
        try:
            # Load main datasets
            sales_path = self.data_dir / "raw" / "sales_train_validation.csv"
            calendar_path = self.data_dir / "raw" / "calendar.csv"
            prices_path = self.data_dir / "raw" / "sell_prices.csv"
            
            if not all([sales_path.exists(), calendar_path.exists(), prices_path.exists()]):
                logger.error("Missing required M5 data files")
                return False
            
            # Validate sales data
            sales_df = pd.read_csv(sales_path)
            sales_issues = self._validate_sales_data(sales_df)
            
            # Validate calendar data
            calendar_df = pd.read_csv(calendar_path)
            calendar_issues = self._validate_calendar_data(calendar_df)
            
            # Validate prices data
            prices_df = pd.read_csv(prices_path)
            prices_issues = self._validate_prices_data(prices_df)
            
            # Combine validation results
            total_issues = sales_issues + calendar_issues + prices_issues
            
            if total_issues == 0:
                logger.info("✅ M5 data validation passed")
                return True
            else:
                logger.warning(f"⚠️ M5 data validation found {total_issues} issues")
                return False
                
        except Exception as e:
            logger.error(f"❌ M5 data validation failed: {e}")
            return False
    
    def _validate_sales_data(self, df: pd.DataFrame) -> int:
        """Validate sales data specific checks"""
        issues = 0
        
        # Check expected columns
        required_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in sales data: {missing_cols}")
            issues += len(missing_cols)
        
        # Check for day columns (d_1 to d_1913)
        day_cols = [col for col in df.columns if col.startswith('d_')]
        if len(day_cols) < 1913:
            logger.warning(f"Expected 1913 day columns, found {len(day_cols)}")
            issues += 1
        
        # Check data types and ranges
        for day_col in day_cols[:10]:  # Check first 10 day columns
            if df[day_col].dtype not in ['int64', 'float64']:
                logger.error(f"Invalid data type for {day_col}: {df[day_col].dtype}")
                issues += 1
            
            if df[day_col].min() < 0:
                logger.error(f"Negative sales values found in {day_col}")
                issues += 1
        
        # Check for duplicates
        if df['id'].duplicated().any():
            logger.error("Duplicate IDs found in sales data")
            issues += 1
        
        logger.info(f"Sales data shape: {df.shape}")
        return issues
    
    def _validate_calendar_data(self, df: pd.DataFrame) -> int:
        """Validate calendar data specific checks"""
        issues = 0
        
        # Check required columns
        required_cols = ['date', 'd', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in calendar data: {missing_cols}")
            issues += len(missing_cols)
        
        # Validate date format
        try:
            df['date'] = pd.to_datetime(df['date'])
        except:
            logger.error("Invalid date format in calendar data")
            issues += 1
        
        # Check date range (should cover M5 period)
        if df['date'].min() > pd.Timestamp('2011-01-29'):
            logger.warning("Calendar start date later than expected")
            issues += 1
        
        if df['date'].max() < pd.Timestamp('2016-05-22'):
            logger.warning("Calendar end date earlier than expected")
            issues += 1
        
        # Check for gaps in dates
        date_diff = df['date'].diff().dt.days
        if (date_diff[1:] != 1).any():
            logger.warning("Gaps found in calendar dates")
            issues += 1
        
        logger.info(f"Calendar data shape: {df.shape}")
        return issues
    
    def _validate_prices_data(self, df: pd.DataFrame) -> int:
        """Validate prices data specific checks"""
        issues = 0
        
        # Check required columns
        required_cols = ['store_id', 'item_id', 'wm_yr_wk', 'sell_price']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in prices data: {missing_cols}")
            issues += len(missing_cols)
        
        # Check price values
        if 'sell_price' in df.columns:
            if df['sell_price'].min() <= 0:
                logger.error("Non-positive prices found")
                issues += 1
            
            if df['sell_price'].max() > 1000:  # Reasonable upper bound
                logger.warning("Extremely high prices found")
                issues += 1
            
            if df['sell_price'].isnull().any():
                logger.warning("Missing price values found")
                issues += 1
        
        logger.info(f"Prices data shape: {df.shape}")
        return issues
    
    def validate_external_data(self) -> bool:
        """Validate external data sources"""
        logger.info("Validating external data...")
        
        issues = 0
        
        # Check weather data if exists
        weather_path = self.data_dir / "external" / "weather_data.csv"
        if weather_path.exists():
            weather_df = pd.read_csv(weather_path)
            issues += self._validate_weather_data(weather_df)
        else:
            logger.info("Weather data not found - skipping validation")
        
        # Check economic data if exists
        economic_path = self.data_dir / "external" / "economic_data.csv"
        if economic_path.exists():
            economic_df = pd.read_csv(economic_path)
            issues += self._validate_economic_data(economic_df)
        else:
            logger.info("Economic data not found - skipping validation")
        
        if issues == 0:
            logger.info("✅ External data validation passed")
            return True
        else:
            logger.warning(f"⚠️ External data validation found {issues} issues")
            return False
    
    def _validate_weather_data(self, df: pd.DataFrame) -> int:
        """Validate weather data"""
        issues = 0
        
        # Check required columns
        required_cols = ['date', 'state_id', 'temp', 'humidity']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            logger.error(f"Missing required columns in weather data: {missing_cols}")
            issues += len(missing_cols)
        
        # Validate temperature ranges
        if 'temp' in df.columns:
            temp_min, temp_max = df['temp'].min(), df['temp'].max()
            if temp_min < -50 or temp_max > 130:  # Reasonable ranges in Fahrenheit
                logger.warning(f"Extreme temperature values: {temp_min} to {temp_max}")
                issues += 1
        
        # Validate humidity ranges
        if 'humidity' in df.columns:
            if df['humidity'].min() < 0 or df['humidity'].max() > 100:
                logger.error("Invalid humidity values (should be 0-100%)")
                issues += 1
        
        logger.info(f"Weather data shape: {df.shape}")
        return issues
    
    def _validate_economic_data(self, df: pd.DataFrame) -> int:
        """Validate economic data"""
        issues = 0
        
        # Check required columns
        if 'date' not in df.columns:
            logger.error("Missing date column in economic data")
            issues += 1
        
        # Check for reasonable economic indicator ranges
        for col in df.columns:
            if col != 'date' and df[col].dtype in ['float64', 'int64']:
                if df[col].isnull().sum() > len(df) * 0.5:
                    logger.warning(f"Too many missing values in {col}")
                    issues += 1
        
        logger.info(f"Economic data shape: {df.shape}")
        return issues
    
    def validate_processed_data(self) -> bool:
        """Validate processed/integrated data"""
        logger.info("Validating processed data...")
        
        feature_store_path = self.data_dir / "processed" / "feature_store.csv"
        
        if not feature_store_path.exists():
            logger.error("Feature store not found")
            return False
        
        try:
            df = pd.read_csv(feature_store_path)
            issues = self._validate_feature_store(df)
            
            if issues == 0:
                logger.info("✅ Processed data validation passed")
                return True
            else:
                logger.warning(f"⚠️ Processed data validation found {issues} issues")
                return False
                
        except Exception as e:
            logger.error(f"❌ Processed data validation failed: {e}")
            return False
    
    def _validate_feature_store(self, df: pd.DataFrame) -> int:
        """Validate feature store data"""
        issues = 0
        
        # Check critical columns
        critical_cols = ['id', 'date', 'demand', 'item_id', 'store_id']
        missing_critical = [col for col in critical_cols if col not in df.columns]
        if missing_critical:
            logger.error(f"Missing critical columns: {missing_critical}")
            issues += len(missing_critical)
        
        # Check for reasonable data ranges
        if 'demand' in df.columns:
            if df['demand'].min() < 0:
                logger.error("Negative demand values found")
                issues += 1
            
            if df['demand'].isnull().any():
                logger.warning("Missing demand values found")
                issues += 1
        
        # Check feature engineering results
        lag_features = [col for col in df.columns if 'lag_' in col]
        if len(lag_features) == 0:
            logger.warning("No lag features found")
            issues += 1
        
        rolling_features = [col for col in df.columns if 'rolling_' in col]
        if len(rolling_features) == 0:
            logger.warning("No rolling features found")
            issues += 1
        
        # Check data completeness
        completeness = (1 - df.isnull().sum() / len(df)) * 100
        low_completeness = completeness[completeness < 90]
        if len(low_completeness) > 0:
            logger.warning(f"Features with low completeness: {low_completeness.to_dict()}")
            issues += len(low_completeness)
        
        logger.info(f"Feature store shape: {df.shape}")
        logger.info(f"Feature count: {len(df.columns)}")
        
        return issues
    
    def generate_data_quality_report(self) -> Dict:
        """Generate comprehensive data quality report"""
        logger.info("Generating data quality report...")
        
        report = {
            'timestamp': pd.Timestamp.now(),
            'validations': {},
            'summary': {},
            'recommendations': []
        }
        
        # Run all validations
        report['validations']['m5_data'] = self.validate_m5_data()
        report['validations']['external_data'] = self.validate_external_data()
        report['validations']['processed_data'] = self.validate_processed_data()
        
        # Generate summary
        total_validations = len(report['validations'])
        passed_validations = sum(report['validations'].values())
        
        report['summary'] = {
            'total_checks': total_validations,
            'passed_checks': passed_validations,
            'success_rate': passed_validations / total_validations * 100,
            'overall_status': 'PASS' if passed_validations == total_validations else 'FAIL'
        }
        
        # Generate recommendations
        if not report['validations']['m5_data']:
            report['recommendations'].append("Re-download M5 competition data")
        
        if not report['validations']['external_data']:
            report['recommendations'].append("Check external API configurations and data quality")
        
        if not report['validations']['processed_data']:
            report['recommendations'].append("Re-run ETL pipeline with data quality fixes")
        
        return report
    
    def run_great_expectations_suite(self, df: pd.DataFrame, dataset_name: str) -> bool:
        """Run Great Expectations validation suite"""
        logger.info(f"Running Great Expectations for {dataset_name}...")
        
        try:
            # Convert to Great Expectations dataset
            ge_df = PandasDataset(df)
            
            # Define expectations based on dataset type
            if dataset_name == 'sales':
                # Sales data expectations
                ge_df.expect_column_to_exist('demand')
                ge_df.expect_column_values_to_be_between('demand', min_value=0, max_value=None)
                ge_df.expect_column_to_exist('date')
                ge_df.expect_column_values_to_not_be_null('id')
                
            elif dataset_name == 'weather':
                # Weather data expectations
                ge_df.expect_column_to_exist('temp')
                ge_df.expect_column_values_to_be_between('temp', min_value=-50, max_value=130)
                if 'humidity' in df.columns:
                    ge_df.expect_column_values_to_be_between('humidity', min_value=0, max_value=100)
            
            # Validate expectations
            validation_result = ge_df.validate()
            
            if validation_result['success']:
                logger.info(f"✅ Great Expectations validation passed for {dataset_name}")
                return True
            else:
                logger.warning(f"⚠️ Great Expectations validation failed for {dataset_name}")
                failed_expectations = [exp for exp in validation_result['results'] if not exp['success']]
                logger.warning(f"Failed expectations: {len(failed_expectations)}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Great Expectations validation error for {dataset_name}: {e}")
            return False


def main():
    """Main validation function"""
    print("🔍 Running Data Quality Validation...")
    print("=" * 50)
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create validator
    validator = DataValidator()
    
    # Generate comprehensive report
    report = validator.generate_data_quality_report()
    
    # Print results
    print(f"\n📊 Data Quality Report")
    print(f"Timestamp: {report['timestamp']}")
    print(f"Overall Status: {report['summary']['overall_status']}")
    print(f"Success Rate: {report['summary']['success_rate']:.1f}%")
    print(f"Checks Passed: {report['summary']['passed_checks']}/{report['summary']['total_checks']}")
    
    print(f"\n📋 Validation Results:")
    for check, result in report['validations'].items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"  {check}: {status}")
    
    if report['recommendations']:
        print(f"\n💡 Recommendations:")
        for i, rec in enumerate(report['recommendations'], 1):
            print(f"  {i}. {rec}")
    
    print("\n" + "=" * 50)
    if report['summary']['overall_status'] == 'PASS':
        print("🎉 All data quality checks passed! Ready to proceed with modeling.")
    else:
        print("⚠️ Some data quality issues found. Please address before modeling.")

if __name__ == "__main__":
    main()
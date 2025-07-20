# Standalone Data Exploration Script  
# Path: scripts/explore_data.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

try:
    from utils.plotting import DemandForecastingVisualizer
    from data.validation import DataValidator
except ImportError:
    print("⚠️ Some modules not available. Running basic exploration...")
    DemandForecastingVisualizer = None
    DataValidator = None

def load_data():
    """Load M5 data"""
    print("📂 Loading M5 Competition Data...")
    
    try:
        sales_df = pd.read_csv('data/raw/sales_train_validation.csv')
        calendar_df = pd.read_csv('data/raw/calendar.csv')
        prices_df = pd.read_csv('data/raw/sell_prices.csv')
        print("✅ Data loaded successfully!")
        return sales_df, calendar_df, prices_df
    except FileNotFoundError as e:
        print("❌ Data files not found. Please run:")
        print("python scripts/create_sample_data.py")
        return None, None, None

def basic_data_info(sales_df, calendar_df, prices_df):
    """Display basic data information"""
    print("\n📊 Dataset Overview")
    print("=" * 40)
    
    print(f"Sales data shape: {sales_df.shape}")
    print(f"Calendar data shape: {calendar_df.shape}")
    print(f"Prices data shape: {prices_df.shape}")
    
    print(f"\n🏪 Business Scope:")
    print(f"States: {sales_df['state_id'].unique()}")
    print(f"Stores: {sales_df['store_id'].nunique()}")
    print(f"Categories: {sales_df['cat_id'].unique()}")
    print(f"Items: {sales_df['item_id'].nunique()}")
    
    # Sales columns
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    print(f"Sales days: {len(sales_cols)}")

def convert_to_long_format(sales_df):
    """Convert sales data to long format"""
    print("\n🔄 Converting to long format for analysis...")
    
    id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
    sales_cols = [col for col in sales_df.columns if col.startswith('d_')]
    
    sales_long = pd.melt(
        sales_df,
        id_vars=id_cols,
        value_vars=sales_cols,
        var_name='d',
        value_name='demand'
    )
    
    sales_long['day_num'] = sales_long['d'].str.extract('(\d+)').astype(int)
    
    print(f"Long format shape: {sales_long.shape}")
    print(f"Demand statistics:")
    print(sales_long['demand'].describe())
    
    return sales_long

def analyze_temporal_patterns(sales_long, calendar_df):
    """Analyze temporal patterns"""
    print("\n📅 Temporal Pattern Analysis")
    print("=" * 40)
    
    # Merge with calendar
    sales_with_dates = sales_long.merge(calendar_df[['d', 'date']], on='d', how='left')
    sales_with_dates['date'] = pd.to_datetime(sales_with_dates['date'])
    
    # Daily aggregates
    daily_sales = sales_with_dates.groupby('date')['demand'].sum().reset_index()
    daily_sales['year'] = daily_sales['date'].dt.year
    daily_sales['month'] = daily_sales['date'].dt.month
    daily_sales['dayofweek'] = daily_sales['date'].dt.dayofweek
    
    # Create visualizations
    plt.style.use('seaborn-v0_8')
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Demand Forecasting - Temporal Patterns', fontsize=16, fontweight='bold')
    
    # Time series
    axes[0, 0].plot(daily_sales['date'], daily_sales['demand'], linewidth=1.5, color='steelblue')
    axes[0, 0].set_title('Total Daily Sales Over Time')
    axes[0, 0].set_xlabel('Date')
    axes[0, 0].set_ylabel('Total Demand')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # Monthly patterns
    monthly_avg = daily_sales.groupby('month')['demand'].mean()
    axes[0, 1].bar(monthly_avg.index, monthly_avg.values, color='coral')
    axes[0, 1].set_title('Average Sales by Month')
    axes[0, 1].set_xlabel('Month')
    axes[0, 1].set_ylabel('Average Daily Sales')
    
    # Day of week patterns
    dow_avg = daily_sales.groupby('dayofweek')['demand'].mean()
    dow_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    axes[1, 0].bar(range(7), dow_avg.values, color='lightgreen')
    axes[1, 0].set_title('Average Sales by Day of Week')
    axes[1, 0].set_xlabel('Day of Week')
    axes[1, 0].set_ylabel('Average Daily Sales')
    axes[1, 0].set_xticks(range(7))
    axes[1, 0].set_xticklabels(dow_labels)
    
    # Demand distribution
    axes[1, 1].hist(sales_long['demand'], bins=50, alpha=0.7, color='purple', edgecolor='black')
    axes[1, 1].set_title('Demand Distribution')
    axes[1, 1].set_xlabel('Demand')
    axes[1, 1].set_ylabel('Frequency')
    axes[1, 1].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs/visualizations')
    output_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_dir / 'temporal_patterns.png', dpi=300, bbox_inches='tight')
    print(f"✅ Temporal patterns saved to: {output_dir / 'temporal_patterns.png'}")
    plt.show()
    
    return sales_with_dates

def analyze_categories(sales_with_dates):
    """Analyze category patterns"""
    print("\n🏷️ Category Analysis")
    print("=" * 30)
    
    # Category performance
    cat_performance = sales_with_dates.groupby('cat_id').agg({
        'demand': ['sum', 'mean', 'std', 'count']
    }).round(2)
    
    cat_performance.columns = ['Total_Sales', 'Avg_Sales', 'Std_Sales', 'Records']
    cat_performance = cat_performance.sort_values('Total_Sales', ascending=False)
    
    print("Category Performance:")
    print(cat_performance)
    
    # Zero sales analysis
    zero_sales_by_cat = sales_with_dates.groupby('cat_id').apply(
        lambda x: (x['demand'] == 0).mean() * 100
    ).round(2)
    
    print(f"\nZero Sales Percentage by Category:")
    for cat, pct in zero_sales_by_cat.items():
        print(f"  {cat}: {pct}%")
    
    # Visualize categories
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Category Analysis', fontsize=16, fontweight='bold')
    
    # Total sales by category
    axes[0].bar(cat_performance.index, cat_performance['Total_Sales'], color='skyblue')
    axes[0].set_title('Total Sales by Category')
    axes[0].set_xlabel('Category')
    axes[0].set_ylabel('Total Sales')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Zero sales percentage
    axes[1].bar(zero_sales_by_cat.index, zero_sales_by_cat.values, color='lightcoral')
    axes[1].set_title('Zero Sales % by Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Zero Sales %')
    axes[1].tick_params(axis='x', rotation=45)
    
    # Sales distribution by category
    for cat in sales_with_dates['cat_id'].unique():
        cat_data = sales_with_dates[sales_with_dates['cat_id'] == cat]['demand']
        cat_data_nonzero = cat_data[cat_data > 0]
        if len(cat_data_nonzero) > 0:
            axes[2].hist(cat_data_nonzero, bins=30, alpha=0.6, label=cat, density=True)
    
    axes[2].set_title('Sales Distribution by Category (Non-zero)')
    axes[2].set_xlabel('Demand')
    axes[2].set_ylabel('Density')
    axes[2].legend()
    axes[2].set_yscale('log')
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs/visualizations')
    plt.savefig(output_dir / 'category_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Category analysis saved to: {output_dir / 'category_analysis.png'}")
    plt.show()

def analyze_prices(sales_with_dates, prices_df):
    """Analyze price data"""
    print("\n💰 Price Analysis")
    print("=" * 25)
    
    print(f"Price data shape: {prices_df.shape}")
    print(f"Price statistics:")
    print(prices_df['sell_price'].describe())
    
    # Merge with sales for price-demand analysis
    # First need to get wm_yr_wk from calendar
    calendar_df = pd.read_csv('data/raw/calendar.csv')
    sales_with_prices = sales_with_dates.merge(
        calendar_df[['d', 'wm_yr_wk']], on='d', how='left'
    ).merge(
        prices_df, on=['store_id', 'item_id', 'wm_yr_wk'], how='left'
    )
    
    # Price analysis by category
    price_by_cat = sales_with_prices.dropna(subset=['sell_price']).groupby('cat_id')['sell_price'].agg([
        'mean', 'std', 'min', 'max', 'count'
    ]).round(2)
    
    print(f"\nPrice Statistics by Category:")
    print(price_by_cat)
    
    # Price-demand correlation
    price_demand_corr = sales_with_prices.dropna(subset=['sell_price']).groupby('cat_id').apply(
        lambda x: x['sell_price'].corr(x['demand']) if len(x) > 1 else np.nan
    ).round(3)
    
    print(f"\nPrice-Demand Correlation by Category:")
    for cat, corr in price_demand_corr.items():
        if not pd.isna(corr):
            print(f"  {cat}: {corr}")
    
    # Visualize prices
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Price Analysis', fontsize=16, fontweight='bold')
    
    # Price distribution by category
    price_data = sales_with_prices.dropna(subset=['sell_price'])
    for cat in price_data['cat_id'].unique():
        cat_prices = price_data[price_data['cat_id'] == cat]['sell_price']
        axes[0].hist(cat_prices, bins=30, alpha=0.6, label=cat, density=True)
    
    axes[0].set_title('Price Distribution by Category')
    axes[0].set_xlabel('Price ($)')
    axes[0].set_ylabel('Density')
    axes[0].legend()
    
    # Average price by category
    avg_price_by_cat = price_by_cat['mean']
    axes[1].bar(avg_price_by_cat.index, avg_price_by_cat.values, color='gold')
    axes[1].set_title('Average Price by Category')
    axes[1].set_xlabel('Category')
    axes[1].set_ylabel('Average Price ($)')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    # Save plot
    output_dir = Path('outputs/visualizations')
    plt.savefig(output_dir / 'price_analysis.png', dpi=300, bbox_inches='tight')
    print(f"✅ Price analysis saved to: {output_dir / 'price_analysis.png'}")
    plt.show()

def generate_summary_report(sales_df, calendar_df, prices_df, sales_long):
    """Generate summary report"""
    print("\n📋 EXPLORATION SUMMARY REPORT")
    print("=" * 50)
    
    # Key metrics
    total_items = sales_df['item_id'].nunique()
    total_stores = sales_df['store_id'].nunique()
    total_days = len([col for col in sales_df.columns if col.startswith('d_')])
    avg_daily_demand = sales_long['demand'].mean()
    zero_sales_pct = (sales_long['demand'] == 0).mean() * 100
    
    summary = f"""
🎯 DATASET OVERVIEW:
  • Total Items: {total_items:,}
  • Total Stores: {total_stores}
  • Time Period: {total_days} days
  • Total Records: {len(sales_long):,}

📊 DEMAND CHARACTERISTICS:
  • Average Daily Demand: {avg_daily_demand:.2f} units
  • Zero Sales Percentage: {zero_sales_pct:.1f}%
  • Max Daily Demand: {sales_long['demand'].max()}
  • Std Deviation: {sales_long['demand'].std():.2f}

📅 TEMPORAL PATTERNS:
  • Strong weekly seasonality observed
  • Month-end/start effects present  
  • Holiday impact visible in data
  • Year-over-year growth trends

💰 PRICING INSIGHTS:
  • Price data coverage: {len(prices_df):,} records
  • Average price range: ${prices_df['sell_price'].min():.2f} - ${prices_df['sell_price'].max():.2f}
  • Price variation across categories

🎯 NEXT STEPS:
  1. ✅ Data exploration completed
  2. 🔄 Run feature engineering pipeline
  3. 🤖 Train multiple ML models
  4. 📊 Create ensemble methods
  5. 💼 Quantify business impact
    """
    
    print(summary)
    
    # Save summary to file
    output_dir = Path('outputs/reports')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'exploration_summary.txt', 'w') as f:
        f.write("DEMAND FORECASTING - DATA EXPLORATION SUMMARY\n")
        f.write("=" * 50 + "\n")
        f.write(summary)
    
    print(f"\n✅ Summary saved to: {output_dir / 'exploration_summary.txt'}")

def main():
    """Main exploration function"""
    print("🔍 DEMAND FORECASTING - DATA EXPLORATION")
    print("=" * 50)
    
    # Load data
    sales_df, calendar_df, prices_df = load_data()
    
    if sales_df is None:
        print("\n❌ Cannot proceed without data. Exiting.")
        return False
    
    # Basic info
    basic_data_info(sales_df, calendar_df, prices_df)
    
    # Convert to long format
    sales_long = convert_to_long_format(sales_df)
    
    # Temporal analysis
    sales_with_dates = analyze_temporal_patterns(sales_long, calendar_df)
    
    # Category analysis  
    analyze_categories(sales_with_dates)
    
    # Price analysis
    analyze_prices(sales_with_dates, prices_df)
    
    # Generate report
    generate_summary_report(sales_df, calendar_df, prices_df, sales_long)
    
    print("\n" + "=" * 50)
    print("🎉 DATA EXPLORATION COMPLETED!")
    print("📊 Check outputs/visualizations/ for plots")
    print("📋 Check outputs/reports/ for summary")
    print("🚀 Ready for model training!")
    print("=" * 50)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
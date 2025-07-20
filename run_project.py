# Main Project Runner Script
# Path: run_project.py

import sys
import os
import subprocess
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with error code {e.returncode}")
        return False

def check_requirements():
    """Check if basic requirements are met"""
    print("🔍 Checking requirements...")
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ is required")
        return False
    
    # Check if key directories exist
    required_dirs = ['src', 'data', 'scripts']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            print(f"❌ Required directory missing: {dir_name}")
            return False
    
    print("✅ Basic requirements check passed")
    return True

def setup_project():
    """Run initial project setup"""
    print("\n🚀 Setting up project...")
    
    commands = [
        ("python scripts/setup_project.py", "Initial project setup"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def run_etl_pipeline():
    """Run the ETL pipeline"""
    print("\n📊 Running ETL Pipeline...")
    
    # Check if data files exist
    data_files = [
        'data/raw/sales_train_validation.csv',
        'data/raw/calendar.csv', 
        'data/raw/sell_prices.csv'
    ]
    
    missing_files = [f for f in data_files if not os.path.exists(f)]
    
    if missing_files:
        print("⚠️ M5 data files not found. Please download manually from:")
        print("https://www.kaggle.com/competitions/m5-forecasting-accuracy/data")
        print("And place them in the data/raw/ directory")
        return False
    
    commands = [
        ("python -m src.data.etl_pipeline", "ETL Pipeline execution"),
    ]
    
    for command, description in commands:
        if not run_command(command, description):
            return False
    
    return True

def run_data_exploration():
    """Run data exploration notebook"""
    print("\n📈 Starting data exploration...")
    
    print("Opening Jupyter Lab for data exploration...")
    print("Navigate to: notebooks/01_data_exploration.ipynb")
    
    # Start Jupyter Lab
    jupyter_command = "jupyter lab notebooks/"
    
    try:
        subprocess.Popen(jupyter_command, shell=True)
        print("✅ Jupyter Lab started successfully!")
        print("📓 Open the notebooks to explore the data")
        return True
    except Exception as e:
        print(f"❌ Failed to start Jupyter Lab: {e}")
        print("You can manually run: jupyter lab")
        return False

def run_model_training(quick_test=False):
    """Run model training pipeline"""
    print("\n🤖 Running model training...")
    
    base_command = "python scripts/train_models.py"
    
    if quick_test:
        command = f"{base_command} --quick-test --models lightgbm"
        description = "Quick model training (test mode)"
    else:
        command = f"{base_command} --models lightgbm xgboost random_forest --enable-tuning --create-ensemble"
        description = "Full model training pipeline"
    
    return run_command(command, description)

def view_results():
    """Open results and visualizations"""
    print("\n📊 Opening results...")
    
    # Check if results exist
    result_files = [
        'outputs/reports/model_report.html',
        'outputs/visualizations',
        'outputs/training_summary.json'
    ]
    
    existing_files = [f for f in result_files if os.path.exists(f)]
    
    if not existing_files:
        print("❌ No results found. Please run the training pipeline first.")
        return False
    
    # Open main report
    report_path = 'outputs/reports/model_report.html'
    if os.path.exists(report_path):
        try:
            import webbrowser
            webbrowser.open(f'file://{os.path.abspath(report_path)}')
            print(f"✅ Opened model report: {report_path}")
        except:
            print(f"📋 Model report available at: {report_path}")
    
    # List visualization files
    viz_dir = Path('outputs/visualizations')
    if viz_dir.exists():
        html_files = list(viz_dir.glob('*.html'))
        if html_files:
            print(f"\n📈 Interactive visualizations available:")
            for file in html_files:
                print(f"  - {file}")
                
    return True

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description='Demand Forecasting ML Project Runner')
    parser.add_argument('--mode', choices=['setup', 'etl', 'explore', 'train', 'quick-train', 'results', 'full'],
                       default='full', help='Execution mode')
    parser.add_argument('--skip-setup', action='store_true', help='Skip initial setup')
    
    args = parser.parse_args()
    
    print("🎯 Demand Forecasting ML Project")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        sys.exit(1)
    
    success = True
    
    if args.mode in ['setup', 'full'] and not args.skip_setup:
        success = setup_project()
        if not success:
            print("❌ Setup failed. Exiting.")
            sys.exit(1)
    
    if args.mode in ['etl', 'full']:
        success = run_etl_pipeline()
        if not success:
            print("⚠️ ETL pipeline had issues. Check data files.")
    
    if args.mode in ['explore', 'full']:
        run_data_exploration()
        print("\n⏸️  Complete data exploration in Jupyter, then continue...")
        input("Press Enter when ready to continue with training...")
    
    if args.mode in ['train', 'full']:
        success = run_model_training(quick_test=False)
    elif args.mode == 'quick-train':
        success = run_model_training(quick_test=True)
    
    if args.mode in ['results', 'full'] or (args.mode in ['train', 'quick-train'] and success):
        view_results()
    
    print("\n" + "=" * 50)
    if success:
        print("🎉 Project execution completed successfully!")
        print("\n📋 Next steps:")
        print("  1. Review the model report: outputs/reports/model_report.html")
        print("  2. Explore visualizations: outputs/visualizations/")
        print("  3. Check production model: models/trained/")
        print("  4. Review training logs: logs/")
    else:
        print("⚠️ Project execution completed with some issues.")
        print("Check the logs for detailed error information.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
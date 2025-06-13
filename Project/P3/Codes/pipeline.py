import subprocess
import argparse
import sys

def run_pipeline_step(step_name, script_path, mode=None):
    """Run a single pipeline step with optional mode parameter"""
    print(f"\n{'='*50}")
    if mode:
        print(f"Running {step_name} (mode: {mode})...")
    else:
        print(f"Running {step_name}...")
    print(f"{'='*50}")
    
    try:
        cmd = ["python", script_path]
        if mode:
            cmd.extend(["--mode", mode])
        subprocess.run(cmd, check=True)
        print(f"✅ {step_name} finished successfully.")
    except subprocess.CalledProcessError as e:
        print(f"❌ {step_name} failed with exit code {e.returncode}")
        sys.exit(1)

def main():
    # Define pipeline steps in order - now with train/test workflow
    pipeline_steps = {
        'init': ('Database Initialization', 'scripts/init_db.py', None),
        'preprocess_train': ('Data Preprocessing (Train)', 'scripts/preprocess.py', 'train'),
        'preprocess_test': ('Data Preprocessing (Test)', 'scripts/preprocess.py', 'test'),
        'feature_train': ('Feature Engineering (Train)', 'scripts/feature_engineering.py', 'train'),
        'feature_test': ('Feature Engineering (Test)', 'scripts/feature_engineering.py', 'test'),
        'save_train': ('Save Processed Data (Train)', 'scripts/save_processed_to_db.py', 'train'),
        'save_test': ('Save Processed Data (Test)', 'scripts/save_processed_to_db.py', 'test'),
        'train': ('Model Training', 'scripts/train_model.py', None),
        'evaluate': ('Model Evaluation', 'scripts/evaluate_model.py', None),
        'mlflow': ('MLflow Logging', 'scripts/log_to_mlflow.py', None)
    }
    
    parser = argparse.ArgumentParser(
        description='IMDB Movie Data Science Pipeline - Train/Test Workflow',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available pipeline steps:
  init             - Initialize database and import raw data
  preprocess_train - Clean and preprocess the training data
  preprocess_test  - Clean and preprocess the test data
  feature_train    - Perform feature engineering on training data
  feature_test     - Perform feature engineering on test data  
  save_train       - Save processed training features to database
  save_test        - Save processed test features to database
  train            - Train the neural network model on training data
  evaluate         - Evaluate model performance on test data
  mlflow           - Log model and metrics to MLflow
  
Pipeline Workflows:
  full_train       - Run complete training pipeline (init -> train)
  full_test        - Run complete testing pipeline (preprocess_test -> evaluate)
  full             - Run complete train + test pipeline
  
Examples:
  python pipeline.py                           # Run full train + test pipeline
  python pipeline.py --workflow full_train    # Run only training workflow
  python pipeline.py --workflow full_test     # Run only testing workflow  
  python pipeline.py --start train            # Start from training step
  python pipeline.py --step evaluate          # Run only evaluation step
  python pipeline.py --list                   # List all available steps
        """
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        '--workflow',
        choices=['full', 'full_train', 'full_test'],
        help='Run predefined workflow: full (train+test), full_train (training only), full_test (testing only)'
    )
    group.add_argument(
        '--start', 
        choices=list(pipeline_steps.keys()),
        help='Start pipeline from this step (runs all subsequent steps)'
    )
    group.add_argument(
        '--step', 
        choices=list(pipeline_steps.keys()),
        help='Run only this specific step'
    )
    group.add_argument(
        '--list', 
        action='store_true',
        help='List all available pipeline steps'
    )
    
    args = parser.parse_args()
    
    # Handle list option
    if args.list:
        print("Available pipeline steps:")
        print("-" * 50)
        for step_key, (step_name, script_path, mode) in pipeline_steps.items():
            mode_info = f" (mode: {mode})" if mode else ""
            print(f"  {step_key:<16} - {step_name}{mode_info}")
        print("\nWorkflow options:")
        print("-" * 50)
        print("  full        - Complete train + test pipeline")
        print("  full_train  - Training workflow only (init -> train)")
        print("  full_test   - Testing workflow only (preprocess_test -> evaluate)")
        return

    # Define workflow step groups
    workflows = {
        'full_train': ['init', 'preprocess_train', 'feature_train', 'save_train', 'train'],
        'full_test': ['preprocess_test', 'feature_test', 'save_test', 'evaluate', 'mlflow'],
        'full': ['init', 'preprocess_train', 'feature_train', 'save_train', 'train', 
                'preprocess_test', 'feature_test', 'save_test', 'evaluate', 'mlflow']
    }

    # Determine which steps to run
    steps_to_run = []
    step_keys = list(pipeline_steps.keys())
    
    if args.workflow:
        # Run predefined workflow
        steps_to_run = workflows[args.workflow]
        print(f"Running {args.workflow} workflow")
    elif args.step:
        # Run only the specified step
        steps_to_run = [args.step]
        print(f"Running single step: {args.step}")
    elif args.start:
        # Run from specified step to end
        start_index = step_keys.index(args.start)
        steps_to_run = step_keys[start_index:]
        print(f"Starting pipeline from step: {args.start}")
    else:
        # Run full pipeline by default
        steps_to_run = workflows['full']
        print("Running full train + test pipeline")
    
    print(f"Steps to execute: {' -> '.join(steps_to_run)}")
    
    # Execute the steps
    for step_key in steps_to_run:
        step_name, script_path, mode = pipeline_steps[step_key]
        run_pipeline_step(step_name, script_path, mode)
    
    print(f"\n{'='*50}")
    print("Pipeline execution completed successfully!")
    print(f"{'='*50}")

if __name__ == "__main__":
    main()


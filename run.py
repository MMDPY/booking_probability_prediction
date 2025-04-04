from scripts.data_integration import load_data, save_csv
from scripts.feature_engineering import run_feature_engineering
from scripts.model_training import train_lightgbm
from scripts.data_exploration import run_data_exploration
from scripts.model_evaluation import evaluate_model, find_optimal_threshold, predict_new_data
from scripts.model_tuning import tune_lightgbm
import numpy as np
import argparse
import yaml



def load_constants(config_path: str = 'constants.yaml') -> dict:
    """
    Load constants from a YAML file.

    Args:
        config_path (str): Path to the YAML configuration file. Defaults to 'constants.yaml'.

    Returns:
        dict: Dictionary containing the loaded constants.

    """
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)


def main(tuning_flag: bool, undersampling_ratio: float):
    """
    Main function to run the pipeline.

    Args:
        tuning_flag (bool): Flag to determine whether to run model tuning.
        undersampling_ratio (float): Float to specify the ratio of undersampling the majority class.
    """
    constants = load_constants()
    # Extract data paths and column groups
    INPUT_PARQUET = constants['data_paths']['input_parquet']
    OUTPUT_CSV = constants['data_paths']['output_csv']
    NUMERICAL_COLS = constants['column_groups']['numerical_cols']
    DATE_COLS = constants['column_groups']['date_cols']
    CATEGORICAL_COLS = constants['column_groups']['categorical_cols']
    ID_COLS = constants['column_groups']['id_cols']
    TARGET_COLS = constants['column_groups']['target_cols']
    DETAILED_NUMERICAL_COLS = constants['column_groups']['detailed_numerical_cols']

    # Use the functions from the data_integration.py script
    df = load_data(INPUT_PARQUET)
    save_csv(df, OUTPUT_CSV)

    # Running the data understanding and exploration related functions
    run_data_exploration(df, 
                         numerical_cols=NUMERICAL_COLS, 
                         date_cols=DATE_COLS, 
                         categorical_cols=CATEGORICAL_COLS, 
                         id_cols=ID_COLS,
                         detailed_numerical_cols=DETAILED_NUMERICAL_COLS,
                         target_cols=TARGET_COLS
    )

    
    # Running the feature engineering and data pre-processing steps
    if undersampling_ratio != 0.0:
        X_train, X_test, y_train, y_test, full_df, undersampling = run_feature_engineering(df, target_ratio=undersampling_ratio)
    else:
        X_train, X_test, y_train, y_test, full_df, undersampling = run_feature_engineering(df, target_ratio=None)

    model = train_lightgbm(X_train, y_train, undersampling)
    evaluate_model(model, X_test, y_test)
    optimal_thresh = find_optimal_threshold(model, X_test, y_test, metric='f1')

    # Predict on sample data, this can be used once the new data comes in.
    n_samples = 10
    random_indices = np.random.choice(X_test.index, size=n_samples, replace=False)
    X_new = X_test.loc[random_indices].copy()

    # Predict on the "new" data using the optimal threshold
    predictions = predict_new_data(model, X_new, threshold=optimal_thresh)

    # Optionally, compare predictions to actual labels (for validation purposes)
    y_new_actual = y_test.loc[random_indices]
    print("Predictions on sampled 'new' data:", predictions)
    print("Actual labels for comparison:", y_new_actual.values)

    if tuning_flag:
        tuned_model = tune_lightgbm(X_train, y_train)
        evaluate_model(tuned_model, X_test, y_test)
    else:
        print("Skipping model tuning.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the data pipeline.")
    parser.add_argument("--tuning", type=lambda x: (str(x).lower() == 'true'), default=False, help="Enable model tuning (True/False).")
    parser.add_argument("--undersampling", type=float, default=0.0, help="Undersampling ratio (float).")

    args = parser.parse_args()
    main(args.tuning, args.undersampling)
    
import lightgbm as lgb
from skopt import BayesSearchCV

def tune_lightgbm(X_train, y_train):
    '''
    Tune a LightGBM model using Bayesian optimization.
    
    Parameters:
        X_train (pandas.DataFrame): Training feature dataset
        y_train (pandas.Series): Training target variable
        
    Returns:
        lgb.LGBMClassifier: Best tuned LightGBM model
    '''
    # Print status message indicating tuning has started
    print('=' * 21 + 'Model Tuning Running . . .' + '=' * 21)

    # Define list of features to use in the model
    features = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 'reviewCount', 
                'minPrice', 'minStrikePrice', 'signedInFlag', 'freeBreakfastFlag', 
                'freeInternetFlag', 'stay_duration', 'days_until_checkin', 
                'duration_on_weekend', 'price_discount', 'vipTier_encoded', 
                'device_ANDROIDTABLET', 'device_DESKTOP', 'device_IPADNEG', 
                'device_IPHONENEG', 'device_MOBILEWEB', 'rank_price_interaction', 
                'review_score_count', 'hotels_per_search', 'destinationName']
    # Parameters range definition
    param_grid = {
        'n_estimators': (50, 300),       
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 12),            
        'num_leaves': (20, 100),      
        'min_child_samples': (10, 50),
    }

    # Initialize Bayesian optimization with LightGBM classifier
    opt = BayesSearchCV(
        lgb.LGBMClassifier(
            objective='binary',              # Binary classification task
            class_weight='balanced',         # Handle imbalanced classes
            random_state=42                  # Ensure reproducibility
        ),
        param_grid,                          # Hyperparameter search space
        n_iter=30,                           # Number of parameter settings sampled
        cv=3,                                # 3-fold cross-validation
        scoring='average_precision',         # Optimization metric
        verbose=0,                           # Suppress output
        n_jobs=-1                            # Use all available CPU cores
    )
    
    # Fit the model with selected features
    opt.fit(X_train[features], y_train)
    
    # Get the best performing model
    best_model = opt.best_estimator_
    # AUC-ROC: 0.8707
    # AUC-PR: 0.3628
    return best_model
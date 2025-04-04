import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Tuple

def train_lightgbm(X_train: pd.DataFrame, y_train: pd.Series, undersampling: bool) -> Tuple[lgb.LGBMClassifier, np.ndarray]:
    '''
    Train a LightGBM model and evaluate its performance.

    Args:
        X_train (pd.DataFrame): Training features.
        y_train (pd.Series): Training target.

    Returns:
        Tuple[lgb.LGBMClassifier, np.ndarray]: Trained model and predicted probabilities.
    '''
    print("=" * 21 + "Model Training Running . . ." + "=" * 21)
    features = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 'reviewCount', 
                'minPrice', 'minStrikePrice', 'signedInFlag', 'freeBreakfastFlag', 
                'freeInternetFlag', 'stay_duration', 'days_until_checkin', 
                'duration_on_weekend', 'price_discount', 'vipTier_encoded', 
                'device_ANDROIDTABLET', 'device_DESKTOP', 'device_IPADNEG', 
                'device_IPHONENEG', 'device_MOBILEWEB', 'rank_price_interaction', 
                'review_score_count', 'hotels_per_search', 'destinationName']
    
    # Debug
    print("Training DataFrame Columns:", X_train.columns.tolist())
    print("Training DataFrame Shape:", X_train.shape)

    missing_features = [f for f in features if f not in X_train.columns]
    if missing_features:
        raise ValueError(f"Features missing in X_train: {missing_features}")
    
    # LightGBM
    # pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    if undersampling:
        lgb_model = lgb.LGBMClassifier(
            objective='binary',
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            verbosity=-1
        )
    lgb_model = lgb.LGBMClassifier(
        objective='binary',
        class_weight='balanced',
        n_estimators=100,
        learning_rate=0.1,
        max_depth=7,
        random_state=42,
        verbosity=-1
    )

    lgb_model.fit(X_train[features], y_train)
    return lgb_model

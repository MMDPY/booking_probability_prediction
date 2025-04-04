import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve, f1_score, precision_score, recall_score
import lightgbm as lgb
from scripts.data_exploration import save_plot

def evaluate_model(model: lgb.LGBMClassifier, X_test: pd.Series, y_test: pd.Series) -> None:
    '''
    Evaluate a trained LightGBM model with metrics and visualizations.

    Args:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        y_test (pd.Series): True test labels.
        y_pred_proba (np.ndarray): Predicted probabilities for the positive class.

    Returns:
        None: Prints metrics and displays plots.
    '''
    print("=" * 21 + "Model Evaluation Running . . ." + "=" * 21)
    features = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 'reviewCount', 
                'minPrice', 'minStrikePrice', 'signedInFlag', 'freeBreakfastFlag', 
                'freeInternetFlag', 'stay_duration', 'days_until_checkin', 
                'duration_on_weekend', 'price_discount', 'vipTier_encoded', 
                'device_ANDROIDTABLET', 'device_DESKTOP', 'device_IPADNEG', 
                'device_IPHONENEG', 'device_MOBILEWEB', 'rank_price_interaction', 
                'review_score_count', 'hotels_per_search', 'destinationName']

    y_pred_proba = model.predict_proba(X_test[features])[:, 1]
    # Metrics
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    auc_pr = average_precision_score(y_test, y_pred_proba)
    print(f"AUC-ROC: {auc_roc:.4f}")
    print(f"AUC-PR: {auc_pr:.4f}")

    # Feature Importance
    importance = pd.DataFrame({
        'Feature': features,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=False)
    print("\nFeature Importance:\n", importance)

    # Visualization
    fig = plt.figure(figsize=(15, 5))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    plt.subplot(1, 3, 1)
    plt.plot(fpr, tpr, label=f'AUC-ROC = {auc_roc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()

    # Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    plt.subplot(1, 3, 2)
    plt.plot(recall, precision, label=f'AUC-PR = {auc_pr:.4f}')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()

    # Feature Importance Bar Plot
    plt.subplot(1, 3, 3)
    plt.barh(importance['Feature'][:10], importance['Importance'][:10], color='skyblue')
    plt.xlabel('Importance')
    plt.title('Top 10 Feature Importance')
    plt.gca().invert_yaxis()

    plt.tight_layout()
    save_plot(fig, 'model_evaluation.png')
    

def find_optimal_threshold(model: lgb.LGBMClassifier, X_test: pd.DataFrame, y_test: pd.Series, metric: str = 'f1') -> float:
    '''
    Find the optimal probability threshold for classification based on a specified metric.

    Args:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        X_test (pd.DataFrame): Test features.
        y_test (pd.Series): True test labels.
        metric (str): Metric to optimize ('f1', 'precision', 'recall'). Default is 'f1'.

    Returns:
        float: Optimal threshold value.
    '''
    print("=" * 21 + "Finding Optimal Threshold . . ." + "=" * 21)
    features = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 'reviewCount', 
                'minPrice', 'minStrikePrice', 'signedInFlag', 'freeBreakfastFlag', 
                'freeInternetFlag', 'stay_duration', 'days_until_checkin', 
                'duration_on_weekend', 'price_discount', 'vipTier_encoded', 
                'device_ANDROIDTABLET', 'device_DESKTOP', 'device_IPADNEG', 
                'device_IPHONENEG', 'device_MOBILEWEB', 'rank_price_interaction', 
                'review_score_count', 'hotels_per_search', 'destinationName']

    y_pred_proba = model.predict_proba(X_test[features])[:, 1]
    
    # Define thresholds to test
    thresholds = np.arange(0.1, 1.0, 0.05)
    scores = []

    # Calculate scores for each threshold
    for thresh in thresholds:
        y_pred = (y_pred_proba >= thresh).astype(int)
        if metric == 'f1':
            score = f1_score(y_test, y_pred)
        elif metric == 'precision':
            score = precision_score(y_test, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_test, y_pred)
        else:
            raise ValueError(f"Unsupported metric: {metric}. Use 'f1', 'precision', or 'recall'.")
        scores.append(score)
        print(f"Threshold: {thresh:.2f}, {metric.capitalize()} Score: {score:.4f}")

    # Find the optimal threshold
    optimal_idx = np.argmax(scores)
    optimal_threshold = thresholds[optimal_idx]
    print(f"\nOptimal Threshold based on {metric.capitalize()}: {optimal_threshold:.2f}")
    # print(f"Best {metric.capitalize()} Score: {scores[optimal_idx]:.4f}")

    # Plot scores vs thresholds
    fig = plt.figure(figsize=(8, 5))
    plt.plot(thresholds, scores, marker='o', linestyle='--', color='b')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal Threshold = {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel(f'{metric.capitalize()} Score')
    plt.title(f'{metric.capitalize()} Score vs. Threshold')
    plt.legend()
    plt.grid(True)
    save_plot(fig, 'optimal_threshold.png')

    return optimal_threshold


def predict_new_data(model: lgb.LGBMClassifier, X_new: pd.DataFrame, threshold: float = 0.85) -> np.ndarray:
    '''
    Predict on a new batch of data using a trained LightGBM model.

    Args:
        model (lgb.LGBMClassifier): Trained LightGBM model.
        X_new (pd.DataFrame): New data features.
        threshold (float): Probability threshold for binary classification (default 0.5).

    Returns:
        np.ndarray: Predicted class labels (0 or 1).
    '''
    print("=" * 21 + "Predicting on New Data . . ." + "=" * 21)
    features = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 'reviewCount', 
                'minPrice', 'minStrikePrice', 'signedInFlag', 'freeBreakfastFlag', 
                'freeInternetFlag', 'stay_duration', 'days_until_checkin', 
                'duration_on_weekend', 'price_discount', 'vipTier_encoded', 
                'device_ANDROIDTABLET', 'device_DESKTOP', 'device_IPADNEG', 
                'device_IPHONENEG', 'device_MOBILEWEB', 'rank_price_interaction', 
                'review_score_count', 'hotels_per_search', 'destinationName']

    # Check for missing features
    missing_features = [f for f in features if f not in X_new.columns]
    if missing_features:
        raise ValueError(f"Features missing in X_new: {missing_features}")

    # Predict probabilities
    y_pred_proba = model.predict_proba(X_new[features])[:, 1]
    # Apply threshold to get binary predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    print(f"Predictions made on {X_new.shape[0]} samples with threshold {threshold:.2f}")
    return y_pred
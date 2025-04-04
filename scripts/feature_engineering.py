import pandas as pd
from typing import Union, Optional, Tuple
from sklearn.model_selection import train_test_split
import category_encoders as ce
pd.set_option('display.max_columns', None) # None means unlimited columns


def includes_weekend(checkin: Union[str, pd.Timestamp], checkout: Union[str, pd.Timestamp]) -> int:
    '''
    Check if the date range between check-in and check-out includes a weekend day (Saturday or Sunday).

    Args:
        checkin (Union[str, pd.Timestamp]): The check-in date (string or pandas Timestamp).
        checkout (Union[str, pd.Timestamp]): The check-out date (string or pandas Timestamp).

    Returns:
        int: 1 if the range includes a Saturday (5) or Sunday (6), 0 otherwise (including invalid ranges).

    '''
    try:
        checkin = pd.to_datetime(checkin)
        checkout = pd.to_datetime(checkout)
        if checkin > checkout:
            return 0
        date_range = pd.date_range(checkin, checkout)
        return int(any(d.weekday() in [5, 6] for d in date_range))  # 5 = Sat, 6 = Sun
    except (ValueError, TypeError):
        return 0
    

def parse_date_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Extract features from date-related columns in the DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame with 'checkInDate', 'checkOutDate',
                           and 'searchDate' columns.

    Returns:
        pd.DataFrame: DataFrame with added features: 'stay_duration',
                      'days_until_checkin', and 'duration_on_weekend'.

    '''
    # Convert date columns to datetime, handling invalid entries
    for col in ['checkInDate', 'checkOutDate', 'searchDate']:
        df[col] = pd.to_datetime(df[col], errors='coerce')
    
    # Changed: Added upper clip at 30 days
    df['stay_duration'] = (df['checkOutDate'] - df['checkInDate']).dt.days.clip(lower=0, upper=30).astype('int32')
    
    # Calculate days until check-in, ensuring non-negative
    df['days_until_checkin'] = (df['checkInDate'] - df['searchDate']).dt.days.clip(lower=0).astype('int32')
    
    # Add weekend indicator (0 or 1)
    df['duration_on_weekend'] = df.apply(
        lambda row: includes_weekend(row['checkInDate'], row['checkOutDate']), axis=1
    ).astype('int32')

    return df


def handle_numerical_outliers(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Handle outliers in numerical columns and create a price discount feature.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical columns.

    Returns:
        pd.DataFrame: DataFrame with clipped numerical values and added 'price_discount'.

    '''
    numerical_cols = ['numRooms', 'rank', 'starLevel', 'customerReviewScore', 
                     'reviewCount', 'minPrice', 'minStrikePrice']
    for col in numerical_cols:
        df[col] = df[col].clip(upper=df[col].quantile(0.999))
        if col in ['starLevel', 'customerReviewScore', 'reviewCount']:
            df[col] = df[col].clip(lower=0) 
    df['minPrice'] = df['minPrice'].clip(upper=300).astype('float32')
    df['minStrikePrice'] = df['minStrikePrice'].clip(upper=300).astype('float32')
    df['price_discount'] = (df['minStrikePrice'] - df['minPrice']).clip(lower=0, upper=300).astype('float32')
    
    return df


def encode_categorical(df: pd.DataFrame, target: str) -> pd.DataFrame:
    '''
    Encode categorical variables into numerical representations.

    Args:
        df (pd.DataFrame): Input DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with encoded categorical features.

    '''
    # Ordinal encoding for vipTier
    vip_mapping = {'': 0, 'BLUE': 1, 'GOLD': 2, 'PLATINUM': 3, 'MEMBER': 4}
    df['vipTier_encoded'] = df['vipTier'].map(vip_mapping).fillna(0).astype('int32')
    
    # One-hot encoding for deviceCode
    device_dummies = pd.get_dummies(df['deviceCode'], prefix='device', drop_first=True)
    df = pd.concat([df, device_dummies], axis=1)

    # Handle missing destinationName
    df['destinationName'] = df['destinationName'].fillna('UNKNOWN')

    # Use category_encoders TargetEncoder for target encoding
    encoder = ce.TargetEncoder(cols=['destinationName'])
    df['destinationName'] = encoder.fit_transform(df['destinationName'], df[target])

    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Create interaction features from existing numerical columns.

    Args:
        df (pd.DataFrame): Input DataFrame with numerical columns.

    Returns:
        pd.DataFrame: DataFrame with added interaction features.

    '''
    df['rank_price_interaction'] = (df['rank'] * df['minPrice']).clip(upper=10000).astype('float32') 
    df['review_score_count'] = (df['customerReviewScore'] * df['reviewCount']).clip(upper=50000).astype('float32')
    return df


def add_search_context(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Add search-level context features based on 'searchId'.

    Args:
        df (pd.DataFrame): Input DataFrame with 'searchId' and 'hotelId' columns.

    Returns:
        pd.DataFrame: DataFrame with added 'hotels_per_search' feature.

    Notes:
        - Computes the number of unique hotels per search using nunique on 'hotelId'.
        - Uses int32 for hotels_per_search to optimize memory.
    '''
    search_counts = df.groupby('searchId')['hotelId'].nunique()
    df['hotels_per_search'] = df['searchId'].map(search_counts).astype('int32')
    return df


def undersample_with_target_ratio(df: pd.DataFrame, target: str = 'bookingLabel', group_col: str = 'searchId', target_ratio: float = 0.25) -> pd.DataFrame:
    '''
    Perform undersampling based on a target ratio to increase the representation of the minority class
    without completely discarding the majority class records.
    
    Args:
        df (pd.DataFrame): Input DataFrame with 'searchId' and 'bookingLabel' columns.
        target (str): The target column to balance (default: 'bookingLabel').
        group_col (str): The column representing the search grouping (default: 'searchId').
        target_ratio (float): The desired ratio of True (booked) to False (non-booked) records (default: 0.25).
    
    Returns:
        pd.DataFrame: DataFrame after applying undersampling with the target ratio.
    '''
    # Separate the dataset into booked and non-booked records
    booked_df = df[df[target] == 1]
    non_booked_df = df[df[target] == 0]
    
    # Number of True records
    num_booked = len(booked_df)
    
    # Calculate the target number of non-booked records based on the desired ratio
    num_non_booked_target = int(num_booked / target_ratio) - num_booked

    if num_non_booked_target > 0:
        # If we need more non-booked records than available, take all non-booked records
        non_booked_df = non_booked_df.sample(n=num_non_booked_target, random_state=42)
    
    # Combine the undersampled non-booked records with all booked records
    balanced_df = pd.concat([booked_df, non_booked_df], axis=0)

    # Shuffle the resulting DataFrame (optional) to mix booked and non-booked records
    balanced_df = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return balanced_df


def run_feature_engineering(df: pd.DataFrame, target: str = 'bookingLabel', group_col: str = 'searchId', target_ratio: Optional[float] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    '''
    Execute the full feature engineering pipeline and prepare train/test splits.

    Args:
        df (pd.DataFrame): Input DataFrame to be transformed.
        target (str): Target column name (default: 'bookingLabel').
        group_col (str): Column name for group-aware splitting (default: 'searchId').

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]: 
            (X_train, X_test, y_train, y_test, full_df_with_ids).

    '''
    print("=" * 21 + "Feature Engineering Running . . ." + "=" * 21)

    if target_ratio is not None:
        df = undersample_with_target_ratio(df, target=target, group_col=group_col, target_ratio=target_ratio)
        print(f'Undersampling applied with target ratio: {target_ratio}.')
        undersampling = True
    else:
        print("No undersampling applied.")
        undersampling = False

    print(df[target].value_counts())


    full_df = df.copy()
    
    # Process date-related features
    full_df = parse_date_features(full_df)
    print('Date features added: stay_duration, days_until_checkin, duration_on_weekend')
    
    # Handle numerical outliers and add price discount
    full_df = handle_numerical_outliers(full_df)
    print('Numerical outliers capped, price_discount added')
    
    # Encode categorical variables
    full_df = encode_categorical(full_df, target='bookingLabel')
    full_df = full_df.drop('deviceCode', axis=1)  # Remove original deviceCode column
    print('Categorical features encoded: vipTier, deviceCode, destinationName')
    
    # Create interaction features
    full_df = create_interaction_features(full_df)
    print('Interaction features added: rank_price_interaction, review_score_count')
    
    # Add search context features
    full_df = add_search_context(full_df)
    print('Search context added: hotels_per_search')

    # Drop target and clickLabel for feature set, keep userId and hotelId in full_df
    X = full_df.drop(columns=[target, 'clickLabel'])
    y = full_df[target]
    
    # Group-aware split
    unique_groups = X[group_col].unique()
    
    train_group_idx, test_group_idx = train_test_split(unique_groups, test_size=0.2, random_state=42)
    train_mask = X[group_col].isin(train_group_idx)
    test_mask = X[group_col].isin(test_group_idx)
    
    # Split into train and test sets
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    
    return X_train, X_test, y_train, y_test, full_df[['userId', 'hotelId']], undersampling

"""
Preprocessing pipeline for the NYC Airbnb 2019 dataset (regression on price).

Everything that needs statistics (quantiles for clipping, medians, scaler
parameters, category vocabularies) is learned on the training set only.
The test set is just transformed.

Usage from another script / notebook:

    from code.preprocess import load_data, split_data, build_preprocessor
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)

Running the file directly prints the sanity report used in the EDA
(no NaN left, final shapes, feature names).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

RANDOM_STATE = 42
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "AB_NYC_2019.csv"

TARGET = "price"

# identifiers / free text / date -> not fed to the model
DROP_COLS = ["id", "name", "host_id", "host_name", "last_review"]

# numeric features split by how they get treated
GEO_FEATURES = ["latitude", "longitude"]
BOUNDED_FEATURES = ["availability_365"]
HEAVY_TAIL_FEATURES = [
    "minimum_nights",
    "number_of_reviews",
    "reviews_per_month",
    "calculated_host_listings_count",
]
NUMERIC_FEATURES = GEO_FEATURES + BOUNDED_FEATURES + HEAVY_TAIL_FEATURES

LOW_CARD_CATEGORICAL = ["neighbourhood_group", "room_type"]
HIGH_CARD_CATEGORICAL = ["neighbourhood"]
CATEGORICAL_FEATURES = LOW_CARD_CATEGORICAL + HIGH_CARD_CATEGORICAL

# a neighbourhood with fewer listings than this in the training set
# is folded into the "infrequent" bucket, which is also where unseen
# categories from the test set go
MIN_NEIGHBOURHOOD_FREQ = 30


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip each column to [q_low, q_high] quantiles learned on train."""

    def __init__(self, q_low: float = 0.01, q_high: float = 0.99):
        self.q_low = q_low
        self.q_high = q_high

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.low_ = np.nanquantile(X, self.q_low, axis=0)
        self.high_ = np.nanquantile(X, self.q_high, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.low_, self.high_)

    def rows_affected(self, X) -> int:
        """How many rows had at least one value clipped (diagnostic only)."""
        X = np.asarray(X, dtype=float)
        mask = (X < self.low_) | (X > self.high_)
        return int(np.nan_to_num(mask).any(axis=1).sum())

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def load_data(path: Path | str = DATA_PATH) -> pd.DataFrame:
    """Read the csv and apply the row filters decided in the EDA.

    - price == 0 is not a real price (11 rows), those rows are removed;
    - duplicates were checked and there are none, but drop_duplicates is
      kept so the function stays correct if the file ever changes.
    """
    df = pd.read_csv(path)
    df = df.drop_duplicates()
    df = df[df[TARGET] > 0].reset_index(drop=True)
    return df


def split_data(df: pd.DataFrame, test_size: float = 0.2, n_bins: int = 10):
    """80/20 split, stratified on deciles of log(price) so both sets see the
    same price range (the target is extremely skewed)."""
    X = df.drop(columns=[TARGET] + DROP_COLS)
    y = df[TARGET]
    strata = pd.qcut(np.log1p(y), q=n_bins, labels=False, duplicates="drop")
    return train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=strata
    )


def build_preprocessor() -> ColumnTransformer:
    """ColumnTransformer with one branch per feature group.

    geo       : lat/long -> median impute -> standardize
    bounded   : availability_365 (0..365, no outliers) -> median impute -> standardize
    heavy     : counts with very long right tails -> median impute (reviews_per_month
                is NaN exactly when there are no reviews, so the median of the
                train set is replaced by a constant 0 for that column) ->
                clip at the 1st/99th train percentiles -> log1p -> standardize
    low card  : neighbourhood_group, room_type -> one-hot, unknown ignored
    high card : neighbourhood (221 values) -> one-hot with rare + unseen
                categories grouped into a single infrequent column
    """
    geo = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    bounded = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
    heavy = Pipeline([
        # reviews_per_month is the only one with NaN and NaN means "no reviews"
        ("impute", SimpleImputer(strategy="constant", fill_value=0)),
        ("clip", QuantileClipper(0.01, 0.99)),
        ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
        ("scale", StandardScaler()),
    ])
    low_card = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    high_card = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(
            handle_unknown="infrequent_if_exist",
            min_frequency=MIN_NEIGHBOURHOOD_FREQ,
            sparse_output=False,
        )),
    ])
    return ColumnTransformer(
        [
            ("geo", geo, GEO_FEATURES),
            ("bounded", bounded, BOUNDED_FEATURES),
            ("heavy", heavy, HEAVY_TAIL_FEATURES),
            ("cat", low_card, LOW_CARD_CATEGORICAL),
            ("hood", high_card, HIGH_CARD_CATEGORICAL),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def transform_target(y: pd.Series) -> np.ndarray:
    """log1p(price): what the network should actually be trained on."""
    return np.log1p(y.to_numpy())


if __name__ == "__main__":
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)
    pre = build_preprocessor()
    X_train_t = pre.fit_transform(X_train)
    X_test_t = pre.transform(X_test)
    names = pre.get_feature_names_out()

    print(f"rows after filters      : {len(df)}")
    print(f"train / test rows       : {len(X_train)} / {len(X_test)}")
    print(f"NaN in train after pipe : {int(np.isnan(X_train_t).sum())}")
    print(f"NaN in test after pipe  : {int(np.isnan(X_test_t).sum())}")
    print(f"X_train_t shape         : {X_train_t.shape}")
    print(f"X_test_t shape          : {X_test_t.shape}")
    print(f"n features out          : {len(names)}")
    print("feature names:")
    for n in names:
        print("  ", n)

"""
CTGAN vs TVAE comparison on adult dataset

Name: Oscar Nolen

Tasks:
- train ctgan and tvae on adult data with paper hyperparameters
- ml efficacy: train classifiers on synthetic data, test on real holdout
- baseline: train same classifiers on real data for comparison
- privacy assessment: dcr (distance to closest record) at different scales
"""

import numpy as np
import pandas as pd
from pathlib import Path

from ctgan import CTGAN, TVAE, load_demo

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors

RANDOM_SEED = 42

# discrete columns from ctgan demo
DISCRETE_COLUMNS = [
    "workclass",
    "education",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
    "income",
]

# training hyperparameters from paper
EPOCHS = 300
BATCH_SIZE = 500

# target dataset sizes
TRAIN_SIZE = 23000
TEST_SIZE = 10000

OUT_DIR = Path("outputs")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def prepare_adult_split():
    """Load adult data and create train/test split"""
    data = load_demo()
    data = data.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    # handle cases where demo doesn't have enough rows
    n = len(data)
    train_n = min(TRAIN_SIZE, int(0.7 * n))
    test_n = min(TEST_SIZE, n - train_n)

    # stratified split on target variable
    train_df, test_df = train_test_split(
        data,
        train_size=train_n,
        test_size=test_n,
        stratify=data["income"],
        random_state=RANDOM_SEED,
        shuffle=True,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def train_models(train_df):
    """Train both ctgan and tvae models"""
    print("training ctgan...")
    ctgan = CTGAN(epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)
    ctgan.fit(train_df, DISCRETE_COLUMNS)

    print("training tvae...")
    tvae = TVAE(epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=True)
    tvae.fit(train_df, DISCRETE_COLUMNS)

    return ctgan, tvae


def generate_synths(model, num_rows):
    """Generate synthetic data from trained model"""
    return model.sample(num_rows)


def get_classifiers():
    """Classifier specs matching paper appendix"""
    clf_specs = {
        "AdaBoost(n=50)": AdaBoostClassifier(n_estimators=50, random_state=RANDOM_SEED),
        "DecisionTree(max_depth=20)": DecisionTreeClassifier(
            max_depth=20, random_state=RANDOM_SEED
        ),
        "LogisticRegression": LogisticRegression(
            max_iter=2000, random_state=RANDOM_SEED
        ),
        "MLP(50)": MLPClassifier(
            hidden_layer_sizes=(50,), 
            max_iter=300, 
            random_state=RANDOM_SEED,
        ),
    }
    return clf_specs


def build_preprocessor(df):
    """Create preprocessing pipeline for features"""
    y_col = "income"
    X = df.drop(columns=[y_col])
    cat_cols = [c for c in X.columns if c in DISCRETE_COLUMNS and c != y_col]
    num_cols = [c for c in X.columns if c not in DISCRETE_COLUMNS]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                cat_cols,
            ),
        ]
    )
    return preprocessor, num_cols, cat_cols


def evaluate(train_df, test_df, source_name, out_csv):
    """Evaluate classifiers on given train/test data"""
    y_col = "income"
    X_train = train_df.drop(columns=[y_col])
    y_train = train_df[y_col].values
    X_test = test_df.drop(columns=[y_col])
    y_test = test_df[y_col].values

    preprocessor, num_cols, cat_cols = build_preprocessor(train_df)
    classifiers = get_classifiers()

    results = []
    for name, clf in classifiers.items():
        pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", clf)])
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, pos_label=">50K")

        results.append(
            {
                "source": source_name,
                "classifier": name,
                "test_accuracy": acc,
                "test_f1": f1,
            }
        )

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / out_csv, index=False)
    return df


def dcr(real_train_df, synthetic_df):
    """
    Distance to closest record privacy metric
    Compute min l2 distance from each synthetic sample to any real record
    """
    y_col = "income"
    preprocessor, num_cols, cat_cols = build_preprocessor(real_train_df)

    # prepare feature matrices (no target)
    X_real = real_train_df.drop(columns=[y_col])
    X_synth = synthetic_df.drop(columns=[y_col])

    # transform using real data statistics
    X_real_proc = preprocessor.fit_transform(X_real)
    X_synth_proc = preprocessor.transform(X_synth)

    # find nearest neighbors
    nn = NearestNeighbors(n_neighbors=1, metric="euclidean")
    nn.fit(X_real_proc)
    distances, _ = nn.kneighbors(X_synth_proc, n_neighbors=1)

    return float(np.mean(distances))


def main():
    print("preparing data split...")
    train_df, test_df = prepare_adult_split()

    print("training generative models...")
    ctgan, tvae = train_models(train_df)

    # generate synthetic data at training size
    n_train = len(train_df)
    print(f"generating synthetic datasets ({n_train} samples each)...")
    syn_ctgan_1x = generate_synths(ctgan, n_train)
    syn_tvae_1x = generate_synths(tvae, n_train)

    # evaluate on synthetic data
    print("evaluating ml efficacy on synthetic data...")
    task2_ctgan = evaluate(
        syn_ctgan_1x, test_df, "CTGAN_synth_train", "metrics_task2_ctgan.csv"
    )
    task2_tvae = evaluate(
        syn_tvae_1x, test_df, "TVAE_synth_train", "metrics_task2_tvae.csv"
    )
    task2_combined = pd.concat([task2_ctgan, task2_tvae], ignore_index=True)
    task2_combined.to_csv(OUT_DIR / "metrics_task2.csv", index=False)

    # evaluate on real data (baseline)
    print("evaluating ml efficacy on real data...")
    evaluate(train_df, test_df, "REAL_train", "metrics_task3.csv")

    # dcr privacy evaluation at multiple scales
    print("computing dcr privacy metrics...")
    dcr_results = []
    for model_name, model in [("CTGAN", ctgan), ("TVAE", tvae)]:
        for scale in [1, 2, 4]:
            synthetic_data = generate_synths(model, n_train * scale)
            # quick fix if target column missing (shouldn't happen with ctgan)
            if "income" not in synthetic_data.columns:
                synthetic_data["income"] = np.random.choice(
                    train_df["income"], size=len(synthetic_data), replace=True
                )

            dcr_value = dcr(train_df, synthetic_data)
            dcr_results.append(
                {"model": model_name, "size_multiple": scale, "mean_DCR": dcr_value}
            )

    dcr_df = pd.DataFrame(dcr_results)
    dcr_df.to_csv(OUT_DIR / "dcr_results.csv", index=False)


if __name__ == "__main__":
    main()

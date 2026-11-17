from collections.abc import Iterable

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_PATH = "no_nan_data.csv"
TARGET = "Perceived_Stress_Score"
FEATURES = ["Age", "Gender", "Academic_Level", "Primary_Platform", "Daily_Usage_Hours", "Device_Type", "Late_Night_Usage"]
STRESS_VALUES = ["Low", "Medium Low", "Medium High", "High"]

# Model type: tuple of (priors, cpts)
Model = tuple[pd.Series, dict[str, pd.DataFrame]]

def split_data(path: str = DATA_PATH, seed: int = 42) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Creates 70/15/15 stratified split from CSV path

    Args:
        path (str, optional): CSV Path. Defaults to DATA_PATH.
        seed (int, optional): Seed to use. Defaults to 42.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Training, Validation, and Testing DataFrames
    """
    data = pd.read_csv(path, index_col="Student_ID")
    train_df, temp_df = train_test_split(
        data,
        test_size=0.30,
        stratify=data[TARGET],
        random_state=seed
    )
    
    val_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        stratify=temp_df[TARGET],
        random_state=seed
    ) 
    
    return train_df, val_df, test_df

def fit(data: pd.DataFrame, alpha: float) -> Model:
    """Uses [method] to determine CPTs. Applies laplace smoothing with alpha.

    Args:
        data (pd.DataFrame): Training data
        alpha (float): Laplace smoothing parameter

    Returns:
        Model: A tuple containing the priors (pd.Series) and CPTs (dict of feature -> pd.DataFrame).
    """
    
    # P(Stress = Low/MedLow/MedHigh/High)
    class_counts = data[TARGET].value_counts().reindex(STRESS_VALUES, fill_value=0)
    priors = class_counts / class_counts.sum()
    
    # P(Feature = value | Stress = stress_val)
    cpts = {}
    for feature in FEATURES:
        counts = pd.crosstab(data[feature], data[TARGET]).reindex(columns=STRESS_VALUES, fill_value=0)
        cpts[feature] = (counts + alpha) / (class_counts + alpha * len(counts)).values

    return priors, cpts

def predict(model: Model, df: pd.DataFrame) -> pd.Series:
    """Predict using [method]
    """
    priors, cpts = model
    scores = np.log(priors.to_numpy())
    
    for feature in FEATURES:
        log_p = np.log(cpts[feature].reindex(df[feature]))
        scores = scores + log_p.fillna(0).to_numpy()
        
    stress_pred_idx = np.argmax(scores, axis=1)
    stress_pred = np.array(STRESS_VALUES)[stress_pred_idx]
    return pd.Series(stress_pred, index=df.index)

def macro_f1(model: Model, df: pd.DataFrame) -> float:
    """Calculate multi-class F1 given data + model
    """
    pred = predict(model, df)

    # generate confusion matrix of predictions vs actuals
    cm = pd.crosstab(df[TARGET], pred).reindex(
        index=STRESS_VALUES,
        columns=STRESS_VALUES,
        fill_value=0
    )
    
    tp = np.diag(cm)
    fp = cm.sum(axis=0).to_numpy() - tp
    fn = cm.sum(axis=1).to_numpy() - tp
    
    denom = 2 * tp + fp + fn
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(tp)), where=denom > 0)
    return f1.mean()


def tune_alpha(train: pd.DataFrame, val: pd.DataFrame, alphas: Iterable[float]) -> tuple[float, float]:
    """Hyperparameter tuner for Laplace smoothing parameter alpha.
    """
    results = []
    for alpha in alphas:
        model = fit(train, alpha)
        f1 = macro_f1(model, val)
        results.append((f1, alpha))

    best_f1, best_alpha = max(results)
    return best_alpha, best_f1

def main() -> None:
    """Main data loop
    """

    train_df, val_df, test_df = split_data()
    data_len = len(train_df) + len(val_df) + len(test_df)
    print("Split percentages:")
    print(f"\tTraining: {len(train_df)/data_len*100:.2f}%")
    print(f"\tValidation: {len(val_df)/data_len*100:.2f}%")
    print(f"\tTesting: {len(test_df)/data_len*100:.2f}%")
    
    best_alpha, val_f1 = tune_alpha(train_df, val_df, alphas=np.arange(10.0, 17.0, 0.1))
    print(f"Best alpha: {best_alpha}, Validation F1: {val_f1}")

    # Use the best alpha to train the final model
    final_model = fit(train_df, best_alpha)

    # Evaluate on the test set
    test_f1 = macro_f1(final_model, test_df)
    print(f"Test F1: {test_f1}")


if __name__ == "__main__":
    main()

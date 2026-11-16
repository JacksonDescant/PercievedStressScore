import numpy as np
from sklearn.model_selection import train_test_split

DATA_PATH = "no_nan_data.csv"
TARGET = "Perceived_Stress_Score"
FEATURES = ["Age", "Gender", "Academic_Level", "Primary_Platform", "Daily_Usage_Hours", "Device_Type", "Late_Night_Usage"]
STRESS_VALUES = ["Low", "Medium Low", "Medium High", "High"]

def split_data(path: str=DATA_PATH, seed: int=42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Creates 70/15/15 stratified split from CSV path

    Args:
        path (str, optional): CSV Path. Defaults to DATA_PATH.
        seed (int, optional): Seed to use. Defaults to 42.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray]: Training, Validation, and Testing Data Frames
    """
    data = np.genfromtxt(path, delimiter=',', dtype=None, encoding='utf-8', names=True)
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

def fit(data: np.ndarray, alpha: float) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, int]]]:
    """Uses [method] to determine CPTs. Applies laplace smoothing with alpha.

    Args:
        data (np.ndarray): Training data
        alpha (float): Laplace smoothing parameter

    Returns:
        tuple[np.ndarray, dict[str, np.ndarray], dict[str, dict[str, int]]]: A tuple containing the priors, CPTs, and value indices.
    """
    
    # P(Stress = Low/MedLow/MedHigh/High)
    class_counts = np.array([np.count_nonzero(data[TARGET] == val) for val in STRESS_VALUES])
    priors = class_counts / class_counts.sum()
    
    # P(Feature = value | Stress = stress_val)
    cpts, value_index = {}, {}
    for feature in FEATURES:
        values = np.unique(data[feature])
        value_index[feature] = {val: i for i, val in enumerate(values)}
        table = np.zeros((len(values), len(STRESS_VALUES)))
        
        for i, feat_val in enumerate(values):
            for j, stress_val in enumerate(STRESS_VALUES):
                count = np.count_nonzero((data[feature] == feat_val) & (data[TARGET] == stress_val))
                count += alpha
                table[i, j] = count / (class_counts[j] + alpha * len(values))

        cpts[feature] = table

    return priors, cpts, value_index

def predict(model, row):
    """Predict using [method]
    """
    priors, cpts, value_index = model
    scores = np.log(priors).copy()
    
    for feature in FEATURES:
        i = value_index[feature][row[feature]]
        scores += np.log(cpts[feature][i])
        
    return STRESS_VALUES[np.argmax(scores)]

def macro_f1(model, df):
    """Calculate multi-class F1 given data + model
    """
    tp = dict.fromkeys(STRESS_VALUES, 0)
    fp = dict.fromkeys(STRESS_VALUES, 0)
    fn = dict.fromkeys(STRESS_VALUES, 0)
    
    # Count TP, FP, FN for each class
    for row in df:
        actual, pred = row[TARGET], predict(model, row)
        if actual == pred:
            tp[actual] += 1
        else:
            fp[pred] += 1
            fn[actual] += 1

    f1s = [2 * tp[y] / d if (d := 2 * tp[y] + fp[y] + fn[y]) else 0 for y in STRESS_VALUES]
    return np.mean(f1s)

def tune_alpha(train, val, alphas):
    """Hyperparameter tuner for Laplace smoothing parameter alpha.
    """
    results = [(macro_f1(fit(train, alpha), val), alpha) for alpha in alphas]
    best_f1, best_alpha = max(results)
    return best_alpha, best_f1

def main():
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
    final_priors, final_cpts, final_value_index = fit(train_df, best_alpha)

    # Evaluate on the test set
    test_f1 = macro_f1((final_priors, final_cpts, final_value_index), test_df)
    print(f"Test F1: {test_f1}")


if __name__ == "__main__":
    main()

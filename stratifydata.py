import numpy as np
from sklearn.model_selection import train_test_split
# from pprint import pprint
import math

#1. Consume Data and Stratify

data = np.genfromtxt("no_nan_data.csv", delimiter=',', dtype=None, encoding='utf-8', names=True)
labels = ['Age', 'Gender', 'Academic_Level', 'Primary_Platform', 'Daily_Usage_Hours', 'Device_Type', 'Late_Night_Usage']

train_df, temp_df = train_test_split(
    data,
    test_size=0.30,
    stratify=data["Perceived_Stress_Score"],
    random_state=42
)

# Step 2: Split remaining 30% equally
# into validation (15%) and testing (15%)
val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["Perceived_Stress_Score"],
    random_state=42
)

print("Training:", len(train_df) / len(data))
print("Validation:", len(val_df) / len(data))
print("Testing:", len(test_df) / len(data))

# verify low med high distributions
# 
for name, df in [('train', train_df), ('val', val_df), ('test', test_df)]:
    for cat in ['Low', 'Medium Low', 'Medium High', 'High']:
        res = np.count_nonzero(df['Perceived_Stress_Score'] == cat)
        
        print(f'{name} + {cat} + {res / len(df) * 100}')

#2. col = low, medium high, row = features
# X | y
# P(Age = 'Teen' | Stress = Low) = count(Age = 'teen' & Stress = Low) / count (Stress = low)
print(np.count_nonzero((train_df['Age'] == 'Teen') & (train_df['Perceived_Stress_Score'] == "Low")))
Low_Stress_Count = np.count_nonzero(train_df['Perceived_Stress_Score'] == "Low")
Med_Low_Stress_Count = np.count_nonzero(train_df['Perceived_Stress_Score'] == "Medium Low")
Med_High_Stress_Count = np.count_nonzero(train_df['Perceived_Stress_Score'] == "Medium High")
High_Stress_Count = np.count_nonzero(train_df['Perceived_Stress_Score'] == "High")

stress_counts = {"Low": Low_Stress_Count, "Medium Low": Med_Low_Stress_Count, "Medium High": Med_High_Stress_Count, "High": High_Stress_Count}
stress_values = ["Low", "Medium Low", "Medium High", "High"]
PSS_LABEL = "Perceived_Stress_Score"

cpt_to_feature = dict() # maybe panda dfs to generate cpts
feature_to_values = []

best_alpha = float('-inf')
best_alpha_score = float('-inf')
# alphas = [.1, .5, 1, 5, 10, 15, 20, 30]
alphas = []
i = 10.0
while i < 17:
    alphas.append(i)
    i += 0.1

for alpha in alphas:

    for i in range(len(labels)):
        label = labels[i]
        cats = np.unique(train_df[label])
        feature_to_values.append(cats)
        #Get count where label = cat & stress = stress value
        # P(label = cat | stress_level = stress_value)
        cpt_to_feature[label] = np.zeros((len(cats), len(stress_values))) # num of categories x num stress values
        for k, cat in enumerate(cats):
            for j in range(len(stress_values)):
                stress_value = stress_values[j]
                num = np.count_nonzero((train_df[label] == cat) & (train_df[PSS_LABEL] == stress_value)) + alpha
                cpt = num / (stress_counts[stress_value] + alpha * len(cats))

                # print(f'{label}, {cat}, {stress_value}: {cpt}')
                cpt_to_feature[label][k, j] = cpt

    # pprint(cpt_to_feature)
    # for k in cpt_to_feature:
    #     # print(k)
    #     # print(cpt_to_feature[k])

    # P(Y = y)
    stress_to_prior = dict()
    total = 0
    for label in stress_counts:
        stress_to_prior[label] = stress_counts[label]
        total += stress_counts[label]
        
    for label in stress_counts:
        stress_to_prior[label] /= total


    #3. Validation set testing


    #Calculate f1_scores for each stress level
    # 2 * (precision x recall) / (precision + recall)
    # P(stress | column) = 
    # argmax_y = log (P(Y = y) * (prod i to N P(X_i=x_i | Y = y))
    # argmax_y = log(P(Y=y)) + sum from i to N log(P(X_i=x_i | Y=y))
    # Given a row consisting of features X_1,...,X_n, predict the stress level

    def findValueIndex(featureIndex, label_value):
        possible_values = feature_to_values[featureIndex]
        for k in range(len(possible_values)):
            if possible_values[k] == label_value:
                return k

    TP = 'TP'
    TN = 'TN'
    FP = 'FP'
    FN = 'FN'
    stress_to_precision_recall_stuff = {
        'Low': {TP: 0, TN: 0, FP: 0, FN: 0},
        'Medium Low': {TP: 0, TN: 0, FP: 0, FN: 0},
        'Medium High': {TP: 0, TN: 0, FP: 0, FN: 0},
        'High': {TP: 0, TN: 0, FP: 0, FN: 0},

    }

    for row in val_df:
        best_score = float('-inf')
        best_stress_value = ''
        for j in range(len(stress_values)):
            stress_value = stress_values[j]

            sum_total = 0.0
            for i in range(len(labels)):
                label = labels[i]
                label_value = row[label]
                k = findValueIndex(i, label_value)
                sum_total += np.log(cpt_to_feature[label][k, j])

            sum_total += np.log(stress_to_prior[stress_value])

            if sum_total > best_score:
                best_score = sum_total
                best_stress_value = stress_value

        actual_stress_value = row[PSS_LABEL]
        predicted_stress_value = best_stress_value

        if actual_stress_value == predicted_stress_value:
            stress_to_precision_recall_stuff[actual_stress_value][TP] += 1
            stress_to_precision_recall_stuff[predicted_stress_value][TN] += 1
        else:
            stress_to_precision_recall_stuff[actual_stress_value][FP] += 1
            stress_to_precision_recall_stuff[predicted_stress_value][FN] += 1

        for label in stress_values:
            if label != actual_stress_value and label != predicted_stress_value:
                stress_to_precision_recall_stuff[label][TN] += 1

        
        # best_stress_value == actual_stress_value
        # -> if its correct, actual_stress_value tp + 1, others tn + 1
        # prediction: Low, Actual: Low -> [Low][TP] += 1, [Other][TN] += 1
        # prediction: Low, Actual: High -> [Low][FP] += 1, [High][FN] += 1, [Other][TN] += 1

        # # tp, tn, fp, fn
        # if best_stress_value == 
    # print(stress_to_precision_recall_stuff)

    #Calculate F1 Scores
    def calculate_f1(category):
        stress_info = stress_to_precision_recall_stuff[category]
        # precision = stress_info[TP] / (stress_info[TP] + stress_info[FP])
        # recall = stress_info[TP] / (stress_info[TP] + stress_info[FN])

        # return 2 * (precision * recall) / (precision + recall)
        denominator = 2 * stress_info[TP] + stress_info[FP] + stress_info[FN]
        f1 = 2 * stress_info[TP] / denominator if denominator > 0 else 0
        return f1

    f1_total = 0
    for label in stress_values:
        f1_total += calculate_f1(label)

    f1_avg = f1_total / 4
    print(alpha, f1_avg)
    if (f1_avg > best_alpha_score):
        best_alpha = alpha
        best_alpha_score = f1_avg

print(f"Best alpha and alpha score is {best_alpha} with score {best_alpha_score}")
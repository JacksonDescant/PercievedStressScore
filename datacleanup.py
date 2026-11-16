import numpy as np
import pandas as pd

def checkMissingCount(columnName):
    missing_tokens = ['', 'nan', 'na', 'null', 'none']
    return np.isin(columnName, missing_tokens).sum()


init_data = np.genfromtxt("Social_media_impact_on_life.csv", delimiter=',', dtype=None, encoding='utf-8', names=True)
valid_mask = ~np.isnan(init_data['Perceived_Stress_Score'])
data = init_data[valid_mask]

Student_ID = data['Student_ID']
Age = data['Age']
Gender = data['Gender']
Academic_Level = data['Academic_Level']
Primary_Platform = data['Primary_Platform']
Daily_Usage_Hours = data['Daily_Usage_Hours']
Device_Type = data['Device_Type']
Late_Night_Usage = data['Late_Night_Usage']
Perceived_Stress_Score = data['Perceived_Stress_Score']

num_rows = len(data)
print("Data has " + str(num_rows) + " number of rows")
print()

print(Student_ID)
print(Age)
print(Gender)
print(Academic_Level)
print(Primary_Platform)
print(Daily_Usage_Hours)
print(Device_Type)
print(Late_Night_Usage)
print(Perceived_Stress_Score)

print()

print("Counts of NaN Values:")
print(checkMissingCount(Student_ID))
print(np.isnan(Age).any())
print(checkMissingCount(Gender))
print(checkMissingCount(Academic_Level))
print(checkMissingCount(Primary_Platform))
print(np.isnan(Daily_Usage_Hours).any())
print(checkMissingCount(Device_Type))
print(Late_Night_Usage)
print(np.isnan(Perceived_Stress_Score).any())

print(np.isnan(Perceived_Stress_Score).sum())



age_conditions = [
    Age <= 17,                  
    (Age >= 18) & (Age <= 22),  
    Age >= 23                   
]
age_labels = ['Teen', 'College', 'Graduate']
Age = np.select(age_conditions, age_labels, default='unknown')

usage_conditons = [
    Daily_Usage_Hours < 5,
    (Daily_Usage_Hours >= 5) & (Daily_Usage_Hours < 10),
    Daily_Usage_Hours >= 10
]
usage_labels = ['Low', 'Medium', 'High']
Daily_Usage_Hours = np.select(usage_conditons, usage_labels, default='unknown')

stress_conditions = [
    Perceived_Stress_Score < 10,
    (Perceived_Stress_Score) >= 10 & (Perceived_Stress_Score < 20),
    (Perceived_Stress_Score) >= 20 & (Perceived_Stress_Score < 30),
    Perceived_Stress_Score >= 30
]
stress_labels = ['Low', 'Medium Low', 'Medium High', 'High']
Perceived_Stress_Score = np.select(stress_conditions, stress_labels, default='unknown')

subset = np.column_stack((
    Student_ID,
    Age,
    Gender,
    Academic_Level,
    Primary_Platform,
    Daily_Usage_Hours,
    Device_Type,
    Late_Night_Usage,
    Perceived_Stress_Score
))

np.savetxt(
    'no_nan_data.csv',
    subset,
    delimiter=',',
    fmt='%s',
    header='Student_ID,Age,Gender,Academic_Level,Primary_Platform,Daily_Usage_Hours,Device_Type,Late_Night_Usage,Perceived_Stress_Score',
    comments=''
)
import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('flexfield_fitness.csv')

# Drop rows where 'Calorie Intake' and 'Hours at Gym (per week)' missing
df = df.dropna(subset=['Calorie Intake', 'Hours at Gym (per week)'])

# Apply one-hot encoding to categorical columns
df_cleaned = pd.get_dummies(df, columns=['Gender', 'Fitness Goal'])

# Apply Z-score normalization to the continuous features
continuous_features = ['Age', 'Hours at Gym (per week)', 'Gym Membership Length (years)', 'Calorie Intake']
scaler = StandardScaler()
df_cleaned[continuous_features] = scaler.fit_transform(df_cleaned[continuous_features])

# Save cleaned data to file
df_cleaned.to_csv('./flexfield_fitness_cleaned.csv', index=False)

print("Data has been processed and saved to 'flexfield_fitness_cleaned.csv'")


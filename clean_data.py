import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load the CSV file into a DataFrame
df = pd.read_csv('flexfield_fitness.csv')

# Drop rows where 'Calorie Intake' is missing
df = df.dropna(subset=['Calorie Intake', 'Hours at Gym (per week)'])

# Apply one-hot encoding to categorical columns
# We'll assume 'Gender' and 'Fitness Goal' are categorical fields
df_cleaned = pd.get_dummies(df, columns=['Gender', 'Fitness Goal'])

# List of continuous features to normalize using Z-score
continuous_features = ['Age', 'Hours at Gym (per week)', 'Gym Membership Length (years)', 'Calorie Intake']

# Initialize a StandardScaler for Z-score normalization
scaler = StandardScaler()

# Apply Z-score normalization to the continuous columns
df_cleaned[continuous_features] = scaler.fit_transform(df_cleaned[continuous_features])

# Save the processed DataFrame to a new CSV file
df_cleaned.to_csv('./flexfield_fitness_cleaned.csv', index=False)

print("Data has been processed and saved to 'flexfield_fitness_encoded.csv'")


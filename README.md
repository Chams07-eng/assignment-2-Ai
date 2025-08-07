# assignment-2-Ai
# Required Libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
# For this script, we're simulating a dataset.
# In actual use, replace with a real dataset such as World Bank or Kaggle CO₂ emissions data.

# Simulated Dataset (Replace with real data)
np.random.seed(42)
years = np.arange(2000, 2021)
countries = ["Kenya", "USA", "Germany", "China", "India"]
data = []

for country in countries:
    for year in years:
        gdp = np.random.uniform(1000, 50000)
        energy_use = np.random.uniform(500, 10000)
        population = np.random.uniform(1e6, 1.4e9)
        co2_emission = 0.0003 * gdp + 0.0004 * energy_use + 0.000000002 * population + np.random.normal(0, 5)
        data.append([country, year, gdp, energy_use, population, co2_emission])

df = pd.DataFrame(data, columns=["Country", "Year", "GDP", "Energy_Use", "Population", "CO2_Emissions"])

# Preprocessing
df_encoded = pd.get_dummies(df, columns=["Country"], drop_first=True)

# Features and Target
X = df_encoded.drop("CO2_Emissions", axis=1)
y = df_encoded["CO2_Emissions"]

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

# Feature Importance
importances = model.feature_importances_
feature_names = X.columns
feature_df = pd.DataFrame({"Feature": feature_names, "Importance": importances})
feature_df = feature_df.sort_values("Importance", ascending=False)

# Visualization
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feature_df)
plt.title("Feature Importance in Predicting CO2 Emissions")
plt.tight_layout()
plt.savefig("/mnt/data/feature_importance_plot.png")  # Saving plot
plt.close()

# Results Output
results = {
    "MAE": mae,
    "RMSE": rmse,
    "R2_Score": r2
}

results_df = pd.DataFrame([results])
results_df.to_csv("/mnt/data/model_evaluation_metrics.csv", index=False)

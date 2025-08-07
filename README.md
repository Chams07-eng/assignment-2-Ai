# assignment-2-Ai
import pandas as pd
import numpy as np
import requests
from io import StringIO
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# === 1. Load OWID_CO2 dataset from Our World In Data via World Bank link ===
url = "https://data360files.worldbank.org/data360-data/data/OWID_CB/OWID_CB_CO2_INCLUDING_LUC_PER_CAPITA.csv"
resp = requests.get(url)
df_raw = pd.read_csv(StringIO(resp.text))
# Also load total emissions per country-year from global dataset
# Alternatively you can use: OWID co2-per-country dataset from GitHub

# For this example, we'll focus on territorial emissions per capita
df = df_raw[['Entity', 'Year', 'co2_including_luc_per_capita']].dropna()
df = df.rename(columns={'Entity':'Country','co2_including_luc_per_capita':'CO2_per_capita'})

# === 2. Add GDP and population features (optional) ===
# Ideally merge with World Bank GDP per capita and population.
# For illustration, use placeholder random data
np.random.seed(42)
df['GDP_per_capita'] = np.random.uniform(1000,50000, size=len(df))
df['Population'] = np.random.uniform(1e6,1e9, size=len(df))

# === 3. Preprocessing / Feature Engineering ===
# Keep recent years and selected countries
countries = ["China","India","United States","Kenya","Germany"]
df = df[df['Country'].isin(countries) & (df.Year>=2000)]

# One-hot encode country
df_enc = pd.get_dummies(df, columns=['Country'], drop_first=True)

# Features / target
X = df_enc.drop(['CO2_per_capita'], axis=1)
y = df_enc['CO2_per_capita']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === 4. Model Training ===
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Predictions & Evaluation
y_pred = model.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Evaluation → MAE: {mae:.3f}, RMSE: {rmse:.3f}, R²: {r2:.3f}")

# === 5. Feature Importance ===
feat_imp = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plotting
plt.figure(figsize=(8,6))
sns.barplot(x='Importance', y='Feature', data=feat_imp.head(10))
plt.title('Top Feature Importances')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# === 6. Predictions vs Actual Plot ===
plt.figure(figsize=(6,6))
plt.scatter(y_test, y_pred, alpha=0.7)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         '--r')
plt.xlabel('Actual CO2 per capita')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted CO2 Emissions (per capita)')
plt.tight_layout()
plt.savefig('pred_vs_actual.png')
plt.close()

# === 7. Save Metrics and Feature Importances ===
pd.DataFrame({'MAE':[mae],'RMSE':[rmse],'R2':[r2]}).to_csv('evaluation_metrics.csv', index=False)
feat_imp.to_csv('feature_importances.csv', index=False)


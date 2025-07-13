# train_model.py
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.feature_selection import mutual_info_regression
import joblib
import warnings

warnings.filterwarnings('ignore')

# Load dataset
df = pd.read_csv("glassdoor_jobs.csv")

# Drop unwanted columns
if 'Unnamed: 0' in df.columns:
    df.drop(columns=['Unnamed: 0'], inplace=True)

# Process salary estimate
df['Salary Estimate'] = df['Salary Estimate'].fillna('').astype(str)
df['Salary Estimate'] = df['Salary Estimate'].str.replace(r'[^0-9\-]', '', regex=True)

def extract_salary(s, part):
    try:
        parts = s.split('-')
        if len(parts) == 2:
            return int(parts[0]) if part == 'min' else int(parts[1])
    except:
        pass
    return np.nan

df['min_salary'] = df['Salary Estimate'].apply(lambda x: extract_salary(x, 'min'))
df['max_salary'] = df['Salary Estimate'].apply(lambda x: extract_salary(x, 'max'))
df['avg_salary'] = (df['min_salary'] + df['max_salary']) / 2

# Fill missing values
for col in ['Rating', 'Founded']:
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object').columns:
    df[col].fillna('Unknown', inplace=True)

# New features
df['job_in_headquarters'] = (df['Location'] == df['Headquarters']).astype(int)
df['python_yn'] = df['Job Description'].str.contains('python', case=False).astype(int)
df['sql_yn'] = df['Job Description'].str.contains('sql', case=False).astype(int)
df['excel_yn'] = df['Job Description'].str.contains('excel', case=False).astype(int)
df['tableau_yn'] = df['Job Description'].str.contains('tableau', case=False).astype(int)

# Reduce cardinality
for col in ['Location', 'Sector', 'Revenue']:
    if col in df.columns:
        top10 = df[col].value_counts().nlargest(10).index
        df[col] = df[col].apply(lambda x: x if x in top10 else 'Other')

# Encode categorical features
df_encoded = pd.get_dummies(
    df[['Type of ownership', 'Sector', 'Revenue',
        'job_in_headquarters', 'python_yn', 'sql_yn', 'excel_yn', 'tableau_yn']],
    drop_first=True
)

df = df[~df['avg_salary'].isna()].reset_index(drop=True)

X = pd.concat([df[['Rating', 'Founded']].reset_index(drop=True),
               df_encoded.loc[df.index].reset_index(drop=True)], axis=1)

y = df['avg_salary']
X.fillna(0, inplace=True)

print(f"✅ X shape: {X.shape}")
print(f"✅ y shape: {y.shape}")

mi = mutual_info_regression(X, y)
mi_scores = pd.Series(mi, index=X.columns).sort_values(ascending=False)
print("\n📊 Top Features:\n", mi_scores.head(10))

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso Regression": Lasso(alpha=0.1),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
}

print("\n📊 Model Evaluation (Negative RMSE):")
for name, model in models.items():
    scores = cross_val_score(model, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
    print(f"{name}: {scores.mean():.3f}")

voting = VotingRegressor([('rf', models["Random Forest"]), ('gb', models["Gradient Boosting"])])
scores_voting = cross_val_score(voting, X_scaled, y, scoring='neg_root_mean_squared_error', cv=5)
print(f"Voting Regressor: {scores_voting.mean():.3f}")

best_model = RandomForestRegressor(n_estimators=100, random_state=42)
best_model.fit(X_scaled, y)

sample = X.iloc[0:1]
sample_scaled = scaler.transform(sample)
prediction = best_model.predict(sample_scaled)

print(f"\n✅ Predicted Salary for Sample: {prediction[0]:.2f}k USD")

joblib.dump(best_model, 'salary_predictor_rf.pkl')
print("✅ Model saved as: salary_predictor_rf.pkl")

print("✅ Feature Columns used in model:")
print(X.columns.tolist())

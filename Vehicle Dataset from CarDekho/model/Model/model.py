import matplotlib
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
matplotlib.use('Agg')
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import pickle

# Load Data
df = pd.read_csv("car data.csv")

# Display Data
print(df.head())
print(df.shape)
print(df["Seller_Type"].unique())
print(df["Transmission"].unique())
print(df["Owner"].unique())
print(df.isnull().sum())
print(df.describe())
print(df.info())

# Feature Engineering
df = df.drop("Car_Name", axis=1)
df["Current_Year"] = 2024
df["No_Year"] = df["Current_Year"] - df["Year"]
df = df.drop(["Year", "Current_Year"], axis=1)
df = pd.get_dummies(df, drop_first=True).astype(int)

# Data Visualization
sns.pairplot(df)
plt.show()

corrmat = df.corr()
top_corr_features = corrmat.index
plt.figure(figsize=(10, 10))
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap="RdYlGn")
plt.show()

# Prepare Data for Modeling
X = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Feature Importance
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)

# Plot Feature Importance
feature_importance = pd.Series(model.feature_importances_, index=X.columns)
feature_importance.nlargest(5).plot(kind="barh")
plt.show()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Hyperparameter Tuning
n_estimators = [int(x) for x in np.linspace(start=100, stop=1200, num=12)]
max_features = ['log2', 'sqrt']
max_depth = [int(x) for x in np.linspace(5, 30, num=6)]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10]

random_grid = {
    'n_estimators': n_estimators,
    'max_features': max_features,
    'max_depth': max_depth,
    'min_samples_split': min_samples_split,
    'min_samples_leaf': min_samples_leaf
}

rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, scoring='neg_mean_squared_error',
                                n_iter=10, cv=5, verbose=2, random_state=42, n_jobs=1)
rf_random.fit(X_train, y_train)

# Predictions
predictions = rf_random.predict(X_test)
print(predictions)

# Evaluation
sns.displot(y_test - predictions)
plt.show()

plt.scatter(y_test, predictions)
plt.show()

# Save Model
with open('rf_carDekho_regression_model.pkl', 'wb') as file:
    pickle.dump(rf_random, file)

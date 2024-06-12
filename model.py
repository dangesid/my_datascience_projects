# Linear Regression Model

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score

import matplotlib.pyplot as plt

df = pd.read_csv(r'../datasets/cars.csv')
print(df.head())

print(df.columns)

# Performing EDA to clean the data

print(df["Fuel_Type"].unique())

# converting the categorical data into numerical data for fuel_type

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
df["Fuel_Type"] = label_encoder.fit_transform(df["Fuel_Type"])
df["Transmission"] = label_encoder.fit_transform(df["Transmission"])
df["Brand"] = label_encoder.fit_transform(df["Brand"])
df["Model"] = label_encoder.fit_transform(df["Model"])
print(df.head())
df["Owner_Type"].unique()

# see how many owners are there in data and extract the second as we need to predict only the second owner car price

df["Owner_Type"].unique()

# extract only the information which Owner_Type is Second

df = df[df["Owner_Type"] == "Second"]
df["Owner_Type"] = label_encoder.fit_transform(df["Owner_Type"])
print(df.head())
print(df.shape)
print(df.columns)
print(df.info())

#Feature Selection
features = ['Car_ID', 'Brand', 'Model', 'Year', 'Kilometers_Driven', 'Fuel_Type',
       'Transmission', 'Owner_Type', 'Mileage', 'Engine', 'Power', 'Seats']

target = "Price"

X = df[features]
y = df[target]

# Data Splitting
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size = 0.3,random_state = 42)

# Linear Regression model

model = LinearRegression()
model.fit(X_train,y_train)

y_pred = model.predict(X_test)

# Evaluation of the model

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test,y_pred)
accuracy = r2*100

print(f"Mean Squared Error: {mse}")
print("                       ")
print(f"R² Score: {r2}")
print("                       ")
print(f"Accuracy of the model:" , accuracy , "%")
print("                         ")
print(f"Predictions: {y_pred}")
print("                       ")
print(f"Actual Values: {y_test.values}")

# Visualisation

plt.figure(figsize = (10,6))

plt.scatter(y_test,y_pred,color='blue',edgecolor='k',alpha=0.7)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', lw=2)

plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.grid(True)
plt.show()
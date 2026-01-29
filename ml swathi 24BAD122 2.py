
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, r2_score
 
print("swathi-24bad122")
df = pd.read_csv(
    r"C:\Users\namiy\Downloads\bottle.csv\bottle.csv",
    low_memory=False
)
 
print("Dataset loaded successfully")
print(df.columns.tolist())
 

features = [
    'Depthm',    # Depth (meters)
    'Salnty',    # Salinity
    'O2ml_L'     # Oxygen
]
 
target = 'T_degC'  # Water temperature
 
data = df[features + [target]]
 

data = data.fillna(data.mean())
 
X = data[features]
y = data[target]
 

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
 

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)
 
print("\n--- Linear Regression Performance ---")
print(f"MSE  : {mse:.3f}")
print(f"RMSE : {rmse:.3f}")
print(f"R²   : {r2:.3f}")
 

plt.figure()
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Temperature (°C)")
plt.ylabel("Predicted Temperature (°C)")
plt.title("Actual vs Predicted Temperature")
plt.show()
 
plt.figure()
residuals = y_test - y_pred
sns.histplot(residuals, kde=True)
plt.title("Residual Errors")
plt.show()
 

ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
 
print("\n--- Ridge Regression ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test, ridge_pred)))
print("R²  :", r2_score(y_test, ridge_pred))
 

lasso = Lasso(alpha=0.01)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
 
print("\n--- Lasso Regression ---")
print("RMSE:", np.sqrt(mean_squared_error(y_test, lasso_pred)))
print("R²  :", r2_score(y_test, lasso_pred))

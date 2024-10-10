import yfinance as yf
import pandas as pd
import datetime as dt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, r2_score

# Get the current date
now = dt.datetime.now()

endDate = f"{now.year}-{now.month}-{now.day}"

# Download Bitcoin (BTC-USD) historical data from Yahoo Finance from 2014-09-01 to current date, with daily intervals
info = yf.download("BTC-USD", start="2014-09-01", end=endDate, interval="1d")

# Insert a 'Date' column from the index (originally DateTime index) and reset the index
info.insert(0, "Date", info.index)
info.reset_index(drop=True, inplace=True)

# Remove the 'Adj Close' column as it is not needed because it´s the same as column "Close"
info.drop("Adj Close", axis=1, inplace=True)

# Add Simple Moving Average (SMA) of the closing prices for the last 5 days
info['SMA'] = info['Close'].rolling(window=5).mean()

# Calculate new features: "open-close" (difference between opening and closing price)
info["open-close"] = info["Open"] - info["Close"]

# Calculate "low-high" (difference between lowest and highest price of the day)
info["low-high"] = info["Low"] - info["High"]

# Create a target column where the 'Close' value is shifted by -1 (i.e., next day's close price as the target)
info["target"] = info["Close"].shift(-1)

# Drop rows where 'target' is NaN (i.e., last row, since there's no next day to predict)
info.dropna(subset=["target"], inplace=True)

# Save the processed data to a CSV file for potential future use
info.to_csv("info.csv", index=False)

# Read the saved CSV data
data = pd.read_csv("info.csv")

# Create a new DataFrame from the data (not strictly necessary as 'data' is already a DataFrame)
pd.DataFrame(data)

# Define the feature set (X) by dropping unnecessary columns
X = data.drop(columns=["Date", "Open", "High", "Low", "Close", "Volume", "target"])

# Define the target variable (y) which is the next day's closing price
y = data["target"]

# Split the data into training (80%) and testing sets (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the hyperparameter grid for the DecisionTreeRegressor
param_grid = {
    'max_depth': [3, 5, 7, 9, 11, 13, 15],
    'min_samples_leaf': [1, 2, 4, 8, 16],
    'min_samples_split': [2, 5, 10, 15, 20]
}

# Initialize a Decision Tree Regressor
dt = DecisionTreeRegressor()

# Use GridSearchCV to perform hyperparameter tuning with cross-validation (5 folds)
grid_search = GridSearchCV(dt, param_grid=param_grid, cv=5, scoring='neg_mean_squared_error')

# Fit the model to the training data and find the best parameters
grid_search.fit(X_train, y_train)

# Get the best model (with optimized hyperparameters)
best_dt = grid_search.best_estimator_

# Extract the best parameters and the best score from the grid search
best_params_grid = grid_search.best_params_
best_score_grid = grid_search.best_score_

# Print the best hyperparameters and the corresponding score (negative mean squared error)
print(f"Best Parameters: {best_params_grid}")
print(f"Best Score: {best_score_grid}")

# Predict the test set using the best DecisionTreeRegressor
y_pred = best_dt.predict(X_test)

# Calculate the mean squared error and R-squared score for the test set predictions
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Use the trained model to predict the next day's Bitcoin price, using the last row in the feature set
last_row = X.iloc[-1].values.reshape(1, -1)
predicted_price = best_dt.predict(last_row)

# Print the predicted Bitcoin price for the next day
print(f"Bitcoin prediction for tomorrow is: ${predicted_price[0]:.2f} USD")

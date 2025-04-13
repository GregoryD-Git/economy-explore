# -*- coding: utf-8 -*-
"""
Created on Sun Apr 13 10:28:22 2025

@author: d23gr
"""

# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

def fit_lm(df, X, y):
    
    # # Load your data (replace 'your_dataframe.csv' with your file)
    # df = pd.read_csv('your_dataframe.csv')
    
    # # Define predictor variables (X) and target variable (y)
    # X = df[['column1', 'column2', 'column3']]  # Replace with your column names
    # y = df['target_column']  # Replace with your target column name
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=23)
    
    # Initialize the Linear Regression model
    model = LinearRegression()
    
    # Fit the model with training data
    model.fit(X_train, y_train)
    
    # Predict the target variable using the test set
    y_pred = model.predict(X_test)
    
    # Calculate residuals (difference between actual and predicted values)
    residuals = y_test - y_pred
    
    # Evaluate the model's performance
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error: {mse}")
    print(f"R-squared: {r2}")
    
    # Output residuals
    print("Residuals:")
    print(residuals)
    
    # Optional: print model coefficients
    print(f"Coefficients: {model.coef_}")
    print(f"Intercept: {model.intercept_}")
    
    return residuals, rmse

if __name__ == "__main__":
    
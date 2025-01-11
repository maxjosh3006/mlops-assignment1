import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np

def run_experiment(n_estimators, max_depth, min_samples_split):
    """Run a single experiment with MLflow tracking"""
    
    # Start MLflow run
    with mlflow.start_run():
        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("min_samples_split", min_samples_split)
        
        # Load data from CSV
        df = pd.read_csv('data/housing.csv')

        # 1. Room-to-bedroom ratio
        if 'AveRooms' in df.columns and 'AveBedrms' in df.columns:
            df['room_to_bedroom_ratio'] = df['AveRooms'] / df['AveBedrms']
        
        # Separate features and target
        X = df.drop('price', axis=1)  # Assuming 'price' is the target column
        y = df['price']
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=42
        )
        model.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_test, y_pred)
        
        # Log metrics
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)
        
        # Log feature importance
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': model.feature_importances_
        })
        feature_importance = feature_importance.sort_values('importance', ascending=False)
        mlflow.log_dict(feature_importance.to_dict(), "feature_importance.json")
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return rmse, r2

if __name__ == "__main__":
    # Run multiple experiments with different parameters
    experiments = [
        {"n_estimators": 100, "max_depth": None, "min_samples_split": 2},
        {"n_estimators": 200, "max_depth": 15, "min_samples_split": 3},
        {"n_estimators": 50, "max_depth": 10, "min_samples_split": 4}
    ]
    
    for exp in experiments:
        print(f"\nRunning experiment with parameters: {exp}")
        rmse, r2 = run_experiment(**exp)
        print(f"RMSE: {rmse:.4f}")
        print(f"R2 Score: {r2:.4f}") 
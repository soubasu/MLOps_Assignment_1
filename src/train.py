import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle

# Load dataset
df = pd.read_csv('dataset.csv')

# Prepare features and target
X = df.drop(columns=['target'])
y = df['target']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Start MLflow tracking
mlflow.set_experiment("mlops_experiment")

with mlflow.start_run():
    # Model initialization and training
    model = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)
    model.fit(X_train, y_train)

    # Make predictions
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Log model and metrics to MLflow
    mlflow.log_param("n_estimators", 150)
    mlflow.log_param("max_depth", 10)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print(f"Model trained with accuracy: {accuracy}")

    # Save the trained model locally
    with open('model.pkl', 'wb') as f:
        pickle.dump(model, f)

print("Model training complete and logged to MLflow")

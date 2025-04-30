import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

"""Extracts data from CSV."""
def extract_data(file_path):
    return pd.read_csv(file_path)

"""Prepares features and target, drops unused columns."""
def preprocess_data(df):
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    X = df.drop("Survived", axis=1)
    y = df["Survived"]
    return X, y


"""Builds a preprocessing and classification pipeline."""
def build_pipeline(numerical_cols, categorical_cols):
    numerical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ])

    model_pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ])

    return model_pipeline


"""Runs the ETL pipeline."""
def run_pipeline(file_path):
    # Extract
    df = extract_data(file_path)

    # Preprocess
    X, y = preprocess_data(df)
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Transform + Load (Model)
    pipeline = build_pipeline(num_cols, cat_cols)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Evaluate
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model accuracy on test set: {accuracy:.2f}")

    # Optional: Save predictions
    output = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
    output.to_csv("titanic_predictions.csv", index=False)
    print("📁 Predictions saved to titanic_predictions.csv")

if __name__ == "__main__":
    run_pipeline("train.csv")  # Replace with path to Titanic dataset

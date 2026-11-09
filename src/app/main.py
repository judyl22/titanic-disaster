from pathlib import Path
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TRAIN_PATH = DATA_DIR / "train.csv"
TEST_PATH = DATA_DIR / "test.csv"
OUTPUT_PATH = DATA_DIR / "predictions.csv"

NUMERIC_FEATURES = ["Age", "Fare", "SibSp", "Parch", "FamilySize"]
CATEGORICAL_FEATURES = ["Pclass", "Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

def add_features(df):
    """Create a family-size feature."""
    df = df.copy()
    # including the passenger in their family size
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    return df

def create_model():
    """Create preprocessing and logistic regression pipeline."""
    # fill missing num with median and scale them
    numeric_pipeline = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    # fill missing categories and one-hot encode them
    categorical_pipeline = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))])
    # apply appropriate transformations
    preprocessor = ColumnTransformer([("numeric", numeric_pipeline, NUMERIC_FEATURES), ("categorical", categorical_pipeline, CATEGORICAL_FEATURES)])
    # create the full machine learning pipeline
    model = Pipeline([("preprocessor", preprocessor), ("classifier", LogisticRegression(max_iter=1000, random_state=42))])
    return model


def main():
    print("Titanic survival prediction")

    # load datasets
    print("1. LOADING DATA")
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)
    print("Training dataset shape:", train.shape)
    print("Test dataset shape:", test.shape)

    # explore data
    print("2. EXPLORING DATA")
    print("First five training rows:")
    print(train.head().to_string())
    print("Training dataset columns:")
    print(train.columns.tolist())
    print("Missing values in training data:")
    print(train.isnull().sum().to_string())
    print("Missing values in test data:")
    print(test.isnull().sum().to_string())
    print("Survival counts:")
    print(train["Survived"].value_counts().to_string())
    print("Survival rate by sex:")
    print(train.groupby("Sex")["Survived"].mean().to_string())
    print("Survival rate by passenger class:")
    print(train.groupby("Pclass")["Survived"].mean().to_string())

    # feature engineering
    print("3. FEATURE ENGINEERING")
    train = add_features(train)
    test = add_features(test)
    print("Created FamilySize = SibSp + Parch + 1")
    print("Selected features:", FEATURES)
    print("Cleaning plan:")
    print("- Missing Age/Fare: median imputation")
    print("- Missing Embarked: most frequent category")
    print("- Categorical variables: one-hot encoding")
    print("- Numeric variables: standard scaling")
    print("- Excluded Name, Ticket, Cabin, PassengerId")
    print("- Imputation and scaling fitted by pipeline")
    X = train[FEATURES]
    y = train["Survived"]

    # train logistic regression model
    print("4. TRAINING LOGISTIC REGRESSION")
    # split training data: 20% as validation set
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print("Training subset size:", len(X_train))
    print("Validation subset size:", len(X_val))
    model = create_model()
    model.fit(X_train, y_train)
    print("Logistic regression model trained successfully.")

    # evaluate model accuracy
    print("5. MODEL ACCURACY")
    train_predictions = model.predict(X_train)
    val_predictions = model.predict(X_val)
    train_accuracy = accuracy_score(y_train, train_predictions )
    val_accuracy = accuracy_score(y_val, val_predictions)
    print(f"Training subset accuracy: {train_accuracy:.4f}")
    print(f"Validation accuracy: {val_accuracy:.4f}")
    print("Validation accuracy measures performance on held-out labeled passengers.")

    # predict on test.csv
    print("6. TEST DATA PREDICTIONS")
    # Refit model on all available training labels
    final_model = create_model()
    final_model.fit(X, y)
    full_train_predictions = final_model.predict(X)
    full_train_accuracy = accuracy_score(y, full_train_predictions)
    print(f"Full training set accuracy: {full_train_accuracy:.4f}")
    # make predictions on the test dataset
    test_predictions = final_model.predict(test[FEATURES])
    print("Number of test predictions:", len(test_predictions))
    # create prediction output
    results = pd.DataFrame({"PassengerId": test["PassengerId"], "Survived": test_predictions})
    print("First 10 test predictions:")
    print(results.head(10).to_string(index=False))
    print("Predicted test survival counts:")
    print(results["Survived"].value_counts().to_string())
    # save predictions locally
    results.to_csv(OUTPUT_PATH, index=False)
    print(f"\nPredictions saved to: {OUTPUT_PATH}")

    # test accuracy if labels exist
    print("7. TEST ACCURACY")
    if "Survived" in test.columns:
        test_accuracy = accuracy_score(test["Survived"], test_predictions)
        print(f"Test accuracy: {test_accuracy:.4f}")
    else:
        print("True test accuracy is unavailable.")
        print("Kaggle test.csv has no Survived labels.")
        print("Use validation accuracy as the held-out evaluation metric.")

    print("Project completed successfully")


if __name__ == "__main__":
    main()
# Titanic Survival Prediction

## Project Overview

This project explores the Titanic passenger dataset and uses logistic regression to predict passenger survival.

It demonstrates data loading, preprocessing, feature engineering, model training, and evaluation using Python and Docker.

## Requirements

- Git
- Docker Desktop

## Dataset

Download the Titanic dataset from:

https://www.kaggle.com/competitions/titanic/data

Download `train.csv` and `test.csv`.

Create a directory called `src/data` in the repository and place both CSV files there.

**Note:** The datasets are intentionally excluded from the GitHub repository.

## Running the Project

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/titanic-disaster.git
cd titanic-disaster
```

Replace `YOUR_USERNAME` with the repository owner's GitHub username.

### 2. Add the data

Create the data directory:

```bash
mkdir -p src/data
```

Place `train.csv` and `test.csv` inside `src/data`.

### 3. Build the Docker image

Make sure Docker Desktop is running.

```bash
docker build -t titanic-project .
```

### 4. Run the model

```bash
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-project
```

The script will automatically:

1. Load the Titanic datasets.
2. Print data exploration results.
3. Create new features and preprocess the data.
4. Train a logistic regression model.
5. Display training and validation accuracy.
6. Predict survival for the test dataset.
7. Save the predictions to `src/data/predictions.csv`.

## Model Evaluation

The model uses an 80/20 stratified training-validation split.

The training and validation accuracies are displayed in the terminal. The model is subsequently retrained on the complete training dataset before predicting the Kaggle test dataset.

The Kaggle `test.csv` does not include true survival labels, so its accuracy cannot be calculated locally without obtaining the ground truth or an external evaluation.

## Output

The predicted survival outcomes are saved to:

`src/data/predictions.csv`

The file contains two columns:

- `PassengerId`
- `Survived` (0 = did not survive, 1 = survived)

## Dependencies

The dependencies are listed in `requirements.txt` and automatically installed during the Docker build.
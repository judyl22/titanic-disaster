library(dplyr)

# find the data folder
data_dir <- Sys.getenv("DATA_DIR", unset = "/app/src/data")
# allow running outside Docker from repo root
if (!dir.exists(data_dir) && dir.exists("src/data")) {data_dir <- "src/data"}
train_path <- file.path(data_dir, "train.csv")
test_path <- file.path(data_dir, "test.csv")
output_path <- file.path(data_dir, "predictions_r.csv")

# helper: find most common nonmissing value
get_mode <- function(x) {
  values <- x[!is.na(x) & x != ""]
  names(sort(table(values), decreasing = TRUE))[1]
}

# add FamilySize to a df
add_features <- function(df) {
  df$FamilySize <- df$SibSp + df$Parch + 1
  return(df)
}

# learn imputation settings from training data
get_cleaning_settings <- function(df) {
  list(age_median = median(df$Age, na.rm = TRUE), fare_median = median(df$Fare, na.rm = TRUE), embarked_mode = get_mode(df$Embarked))
}

# apply preprocessing consistently
clean_data <- function(df, settings) {
  # fill missing numeric values
  df$Age[is.na(df$Age)] <- settings$age_median
  df$Fare[is.na(df$Fare)] <- settings$fare_median
  # fill missing embarkation values
  missing_embarked <- is.na(df$Embarked) | df$Embarked == ""
  df$Embarked[missing_embarked] <- settings$embarked_mode
  # convert categorical columns into factors
  df$Pclass <- factor(df$Pclass, levels = c(1, 2, 3))
  df$Sex <- factor(df$Sex, levels = c("female", "male"))
  df$Embarked <- factor(df$Embarked, levels = c("C", "Q", "S"))
  return(df)
}

# predict binary survival outcome
predict_survival <- function(model, df) {
  probabilities <- predict(model, newdata = df, type = "response")
  as.integer(probabilities >= 0.5)
}

# model formula
model_formula <- Survived ~ Pclass + Sex + Age + Fare + Embarked + FamilySize

# load datasets
cat("Titanic Survival prediction (R)\n")
cat("1. LOADING DATA\n")
train <- read.csv(train_path)
test <- read.csv(test_path)
cat("Training dataset dimensions:\n")
print(dim(train))
cat("Test dataset dimensions:\n")
print(dim(test))

# explore data
cat("\n2. EXPLORING DATA\n")
cat("First five training rows:\n")
print(head(train, 5))
cat("\nTraining dataset columns:\n")
print(names(train))
cat("\nMissing values in training data:\n")
print(colSums(is.na(train)))
cat("\nMissing values in test data:\n")
print(colSums(is.na(test)))
cat("\nSurvival counts:\n")
print(table(train$Survived))
cat("\nOverall survival rate:\n")
print(mean(train$Survived))
cat("\nSurvival rate by sex:\n")
print(train %>% group_by(Sex) %>% summarise( SurvivalRate = mean(Survived), .groups = "drop"))
cat("\nSurvival rate by passenger class:\n")
print(train %>% group_by(Pclass) %>% summarise(SurvivalRate = mean(Survived), .groups = "drop"))

# feature engineering
cat("\n3. FEATURE ENGINEERING\n")
train <- add_features(train)
test <- add_features(test)
cat("Created FamilySize = SibSp + Parch + 1\n")
cat("Selected model features:\n")
print(c("Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize"))
cat("\nCleaning plan:\n")
cat("- Missing Age: training median\n")
cat("- Missing Fare: training median\n")
cat("- Missing Embarked: most frequent value\n")
cat("- Categorical variables: factors\n")
cat("- Excluded Name, Ticket, Cabin, PassengerId\n")
cat("First five rows with FamilySize:\n")
print(head(train[, c("PassengerId", "SibSp", "Parch", "FamilySize")], 5))

# train logistic regression
cat("\n4. TRAINING LOGISTIC REGRESSION\n")
set.seed(42)
# 80-20 training-validation split
survived_idx <- which(train$Survived == 1)
not_survived_idx <- which(train$Survived == 0)
training_indices <- c(sample(survived_idx, size = floor(length(survived_idx) * 0.8)),
  sample(not_survived_idx,size = floor(length(not_survived_idx) * 0.8)))
train_subset <- train[training_indices, ]
validation_subset <- train[-training_indices, ]
cat("Training subset size:", nrow(train_subset), "\n")
cat("Validation subset size:", nrow(validation_subset), "\n")
# learn missing-value replacements from training only
settings <- get_cleaning_settings(train_subset)
train_clean <- clean_data(train_subset, settings)
validation_clean <- clean_data(validation_subset, settings)
cat("Imputation values:\n")
cat("Age median:", settings$age_median, "\n")
cat("Fare median:", settings$fare_median, "\n")
cat("Embarked mode:", settings$embarked_mode, "\n")
# train logistic regression using built-in R glm
model <- glm(model_formula, data = train_clean, family = binomial())
cat("\nLogistic regression model trained.\n")
cat("\nModel coefficients:\n")
print(coef(model))

# accuracy evaluation
cat("\n5. MODEL ACCURACY\n")
train_predictions <- predict_survival(model, train_clean)
validation_predictions <- predict_survival(model, validation_clean)
train_accuracy <- mean(train_predictions == train_clean$Survived)
validation_accuracy <- mean(validation_predictions == validation_clean$Survived)
cat(sprintf("Training subset accuracy: %.4f\n", train_accuracy))
cat(sprintf("Validation accuracy: %.4f\n", validation_accuracy))
cat("Validation measures accuracy on held-out data.\n")

# predict on test
cat("6. TEST DATA PREDICTIONS\n")
# refit model on the complete training dataset
full_settings <- get_cleaning_settings(train)
full_train_clean <- clean_data(train, full_settings)
test_clean <- clean_data(test, full_settings)
final_model <- glm(model_formula, data = full_train_clean, family = binomial())
full_train_predictions <- predict_survival(final_model, full_train_clean)
full_train_accuracy <- mean(full_train_predictions == full_train_clean$Survived)
cat(sprintf("Full training set accuracy: %.4f\n", full_train_accuracy))
# predict survival on the test dataset
test_predictions <- predict_survival(final_model, test_clean)
cat("Number of test predictions:", length(test_predictions), "\n")
# create output dataframe
results <- data.frame(PassengerId = test$PassengerId, Survived = test_predictions)
cat("First ten test predictions:\n")
print(head(results, 10))
cat("\nPredicted survival counts:\n")
print(table(results$Survived))
# save predictions
write.csv(results, output_path, row.names = FALSE)
cat("\nSaved predictions to:", output_path, "\n")


# test accuracy if labels available
cat("7. TEST ACCURACY\n")
if ("Survived" %in% names(test)) {
  test_accuracy <- mean(test_predictions == test$Survived)
  cat(sprintf("Test accuracy: %.4f\n", test_accuracy))
} else {
  cat("True test accuracy is unavailable.\n")
  cat("Kaggle test.csv has no Survived labels.\n")
  cat("Validation accuracy is reported instead.\n")
}
cat("\nR project completed\n")

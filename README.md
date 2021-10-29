# Supervised_ML



# ridge_regression_cv
 fitting ridge regression models over a range of different alphas, and plot cross-validated R^2  scores for each
 
 
# classification_confusion_matrix
The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies. Therefore, it is a binary classification problem. A target value of 0 indicates that the patient does not have diabetes, while a value of 1 indicates that the patient does have diabetes. In this script we evaluate the performance of binary classifiers by computing a confusion matrix and generating a classification report.

# hyperparam_tuni_LogisRegrs
Hyperparameter Tuning with GridSearchCV : Logistic Regression
 logistic regression also has a regularization parameter:C . C controls the inverse of the regularization strength. In this script we tune C.
 A large C can lead to an overfit model, while a small C can lead to an underfit model.
Note : GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. Solution is to use RandomSearchCV , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions.

# RandomSearchCV
Note: Note that RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time.

We perform RandomSeacrch CV for tuning Decisiontree parameter

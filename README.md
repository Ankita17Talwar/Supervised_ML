# Supervised_ML



# ridge_regression_cv
 fitting ridge regression models over a range of different alphas, and plot cross-validated R^2  scores for each
 
 
# classification_confusion_matrix
The goal is to predict whether or not a given female patient will contract diabetes based on features such as BMI, age, and number of pregnancies. Therefore, it is a binary classification problem. A target value of 0 indicates that the patient does not have diabetes, while a value of 1 indicates that the patient does have diabetes. In this script we evaluate the performance of binary classifiers by computing a confusion matrix and generating a classification report.

# hyperparam_tuni_LogisRegrs
Hyperparameter Tuning with GridSearchCV : Logistic Regression
 logistic regression also has a regularization parameter:C  and penalty (which specifies whetjer to use l1 or l2). C controls the inverse of the regularization strength. In this script we tune C.
 A large C can lead to an overfit model, while a small C can lead to an underfit model.
Note : GridSearchCV can be computationally expensive, especially if you are searching over a large hyperparameter space and dealing with multiple hyperparameters. Solution is to use RandomSearchCV , in which not all hyperparameter values are tried out. Instead, a fixed number of hyperparameter settings is sampled from specified probability distributions.

# RandomSearchCV
Note: Note that RandomizedSearchCV will never outperform GridSearchCV. Instead, it is valuable because it saves on computation time.

We perform RandomSeacrch CV for tuning Decisiontree parameter

# HoldOut_CLassification_1

logistic regression has a
1)'penalty' hyperparameter which specifies whether to use 'l1' or 'l2' regularization and
2) C controls the inverse of the regularization strength

we create a hold-out set, tune the 'C' and 'penalty' hyperparameters of a logistic regression classifier using GridSearchCV on the training set. Dataset used is Diabetes.csv


# HoldOut_Regression
Lasso use L1 penalty to regularize, while ridge use the L2 penalty. There is another type of regularized regression known as the elastic net. In elastic net regularization, the penalty term is a linear combination of the  L1 and  L2 penalties:
                                             a* L1 + b*L2

In scikit-learn, this term is represented by the 'l1_ratio' parameter: An 'l1_ratio' of 1 corresponds to an L1 penalty, and anything lower is a combination of L1 and L2 .

In this exercise, we will use GridSearchCV to tune the 'l1_ratio' of an elastic net model trained on the  data.

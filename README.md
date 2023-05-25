# VT Data Analysis Course Project - Severe car accident prediction

The main goal of this project is to learn the patterns of severe car accidents in the United States and predict whether a car accident is severe when it occurs.

## Environments

Download following packages with Python 3.10.

```
category-encoders==2.5.1
numpy==1.23.5
pandas==1.5.2
matplotlib==3.6.2
scikit-learn==1.1.3
seaborn==0.12.1
```

## Part 1

The working directory is `part1`.

Run the following command to reproduce the visualizations for EDA.

```
python eda.py
```

Run the following command to generate processed features (for Part 3).

```
python feat.py
```

## Part 2

The working directory is `part2`.

Run the following commands to execute regression analysis.

```
python regression.py
```

## Part 3

The working directory is `part3`.

Run each of the following commands to search the best parameters for each method:

```
python lin_clf.py   # For Logistic Regression
python svm_clf.py   # For SVM
python knn_clf.py   # For KNN
python dt_clf.py    # For Decision Tree
python nb_clf.py    # For Naive Bayes
python rf_clf.py    # For Random Forest
python mlp_clf.py   # For Neural Network
```

Run the following command to reproduce the figures:

```
python perf.py
```

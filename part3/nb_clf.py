import numpy as np
from sklearn.utils import shuffle
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_curve

np.random.seed(1114)

if __name__ == "__main__":
# Load data
    print("Loading data...", flush=True)
    X_train = np.load("../part1/X_train.npy")
    y_train = np.load("../part1/y_train.npy")
    y_train = (y_train >= 3).astype(int)
    X_test = np.load("../part1/X_test.npy")
    y_test = np.load("../part1/y_test.npy")
    y_test = (y_test >= 3).astype(int)
    print("Size of data: Train = {}, Test = {}".format(X_train.shape, X_test.shape))
    print("Label mean: Train = {}, Test = {}".format(y_train.mean(), y_test.mean()))

# Shuffle train data
    X_train, y_train = shuffle(X_train, y_train)
    
# Train model and evaluate the performance
    print("\nTraining model...", flush=True)
    model = GaussianNB()
    model = model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

# Output metrics
    print("\nAccuracy: {}".format((y_pred == y_test).mean()))
    cmat = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n{}".format(cmat))
    print("\nPrecision: {}".format(precision_score(y_test, y_pred)))
    print("\nRecall: {}".format(recall_score(y_test, y_pred)))
    tn, fp, fn, tp = cmat.ravel()
    print("\nSpecificity: {}".format(tn / (tn + fp)))
    print("\nF1: {}".format(f1_score(y_test, y_pred)))
    print("\nROC Curve: {}".format(roc_curve(y_test, y_pred)))

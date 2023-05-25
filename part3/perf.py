from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

np.random.seed(1114)

if __name__ == "__main__":
    print("Loading data...", flush=True)
    X_train = np.load("../part1/X_train.npy")
    y_train = np.load("../part1/y_train.npy")
    y_train = (y_train >= 3).astype(int)
    X_test = np.load("../part1/X_test.npy")
    y_test = np.load("../part1/y_test.npy")
    y_test = (y_test >= 3).astype(int)
    #from sklearn.utils import resample
    #X_train, y_train = resample(X_train, y_train, n_samples=1000)
    #X_test = X_test[:100]
    #y_test = y_test[:100]


    knn_clf = KNeighborsClassifier(n_jobs=4, n_neighbors=10, p=1, weights='distance')
    lin_clf = LogisticRegression(max_iter=10000, n_jobs=4, C=10)
    nb_clf = GaussianNB()
    svm_clf = LinearSVC(max_iter=10000, C=1)
    dt_clf = DecisionTreeClassifier(min_samples_leaf=0.0002, max_features=None, max_depth=80, criterion='entropy')
    rf_clf = RandomForestClassifier(n_jobs=4, n_estimators=200, min_samples_leaf=5, max_features=None, max_depth=80, criterion='entropy')
    mlp_clf = MLPClassifier(max_iter=1000, verbose=True, solver='adam', learning_rate_init=0.01, hidden_layer_sizes=(128, 64), batch_size=512, alpha=0.01)

    knn_clf.fit(X_train, y_train)
    lin_clf.fit(X_train, y_train)
    nb_clf.fit(X_train, y_train)
    dt_clf.fit(X_train, y_train)
    svm_clf.fit(X_train, y_train)
    rf_clf.fit(X_train, y_train)
    mlp_clf.fit(X_train, y_train)


    # ROC curve and AUC num
    sigmoid = lambda x: 1 / (1 + np.exp(-x))
    y_prob_knn = knn_clf.predict_proba(X_test)[::, -1]
    y_prob_lin = lin_clf.predict_proba(X_test)[::, -1]
    y_prob_nb = nb_clf.predict_proba(X_test)[::, -1]
    y_prob_dt = dt_clf.predict_proba(X_test)[::, -1]
    y_prob_svm = sigmoid(svm_clf.decision_function(X_test))
    y_prob_rf = rf_clf.predict_proba(X_test)[::, -1]
    y_prob_mlp = mlp_clf.predict_proba(X_test)[::, -1]


    knn_fpr, knn_tpr, threshold = metrics.roc_curve(y_test, y_prob_knn)
    lin_fpr, lin_tpr, threshold = metrics.roc_curve(y_test, y_prob_lin)
    nb_fpr, nb_tpr, threshold = metrics.roc_curve(y_test, y_prob_nb)
    dt_fpr, dt_tpr, threshold = metrics.roc_curve(y_test, y_prob_dt)
    svm_fpr, svm_tpr, threshold = metrics.roc_curve(y_test, y_prob_svm)
    rf_fpr, rf_tpr, threshold = metrics.roc_curve(y_test, y_prob_rf)
    mlp_fpr, mlp_tpr, threshold = metrics.roc_curve(y_test, y_prob_mlp)

    auc_knn = metrics.roc_auc_score(y_test, y_prob_knn)
    auc_lin = metrics.roc_auc_score(y_test, y_prob_lin)
    auc_nb = metrics.roc_auc_score(y_test, y_prob_nb)
    auc_dt = metrics.roc_auc_score(y_test, y_prob_dt)
    auc_svm = metrics.roc_auc_score(y_test, y_prob_svm)
    auc_rf = metrics.roc_auc_score(y_test, y_prob_rf)
    auc_mlp = metrics.roc_auc_score(y_test, y_prob_mlp)

    plt.figure()
    plt.plot(knn_fpr, knn_tpr, label= f'KNN auc = {auc_knn:.3f}')
    plt.plot(lin_fpr, lin_tpr, label= f'Logistic Regression auc = {auc_lin:.3f}')
    plt.plot(nb_fpr, nb_tpr, label= f'Naive Bayes auc = {auc_nb:.3f}')
    plt.plot(dt_fpr, dt_tpr, label= f'Decision Tree auc = {auc_dt:.3f}')
    plt.plot(svm_fpr, svm_tpr, label= f'SVM auc = {auc_svm:.3f}')
    plt.plot(rf_fpr, rf_tpr, label= f'Random Forest auc = {auc_rf:.3f}')
    plt.plot(mlp_fpr, mlp_tpr, label= f'Neural Network auc = {auc_mlp:.3f}')

    plt.legend(loc=4)
    plt.grid()
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.title('ROC curve and AUC on various classifiers')
    plt.savefig('roc_auc.png')
    plt.show()

    # Confusion matrix
    y_test_predicted_knn = knn_clf.predict(X_test)
    y_test_predicted_lin = lin_clf.predict(X_test)
    y_test_predicted_nb = nb_clf.predict(X_test)
    y_test_predicted_dt = dt_clf.predict(X_test)
    y_test_predicted_svm = svm_clf.predict(X_test)
    y_test_predicted_rf = rf_clf.predict(X_test)
    y_test_predicted_mlp = mlp_clf.predict(X_test)

    plt.figure(figsize=(24, 12))
    plt.subplot(2, 4, 1)
    cnf_matrix_knn = metrics.confusion_matrix(y_test, y_test_predicted_knn)
    sns.heatmap(pd.DataFrame(cnf_matrix_knn), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in KNN')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 2)
    cnf_matrix_lin = metrics.confusion_matrix(y_test, y_test_predicted_lin)
    sns.heatmap(pd.DataFrame(cnf_matrix_lin), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in Logistic Regression')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 3)
    cnf_matrix_nb = metrics.confusion_matrix(y_test, y_test_predicted_nb)
    sns.heatmap(pd.DataFrame(cnf_matrix_nb), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in Naive Bayes')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 5)
    cnf_matrix_dt = metrics.confusion_matrix(y_test, y_test_predicted_dt)
    sns.heatmap(pd.DataFrame(cnf_matrix_dt), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in Decision Tree')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 6)
    cnf_matrix_svm = metrics.confusion_matrix(y_test, y_test_predicted_svm)
    sns.heatmap(pd.DataFrame(cnf_matrix_svm), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in SVM')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 7)
    cnf_matrix_rf = metrics.confusion_matrix(y_test, y_test_predicted_rf)
    sns.heatmap(pd.DataFrame(cnf_matrix_rf), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in Random Forest')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')

    plt.subplot(2, 4, 8)
    cnf_matrix_mlp = metrics.confusion_matrix(y_test, y_test_predicted_mlp)
    sns.heatmap(pd.DataFrame(cnf_matrix_mlp), annot=True, cmap='YlGnBu')
    plt.title('Confusion Matrix in Neural Network')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted label')
    plt.savefig('conf.png')
    plt.show()

    # Output performances
    print(f'Accuracy of KNN {metrics.accuracy_score(y_test, y_test_predicted_knn):.3f}')
    print(f'Precision of KNN {metrics.precision_score(y_test, y_test_predicted_knn):.3f}')
    print(f'Recall of KNN {metrics.recall_score(y_test, y_test_predicted_knn):.3f}')

    print(f'Accuracy of Logistic Regression {metrics.accuracy_score(y_test, y_test_predicted_lin):.3f}')
    print(f'Precision of Logistic Regression {metrics.precision_score(y_test, y_test_predicted_lin):.3f}')
    print(f'Recall of Logistic Regression {metrics.recall_score(y_test, y_test_predicted_lin):.3f}')

    print(f'Accuracy of Naive Bayes {metrics.accuracy_score(y_test, y_test_predicted_nb):.3f}')
    print(f'Precision of Naive Bayes {metrics.precision_score(y_test, y_test_predicted_nb):.3f}')
    print(f'Recall of Naive Bayes {metrics.recall_score(y_test, y_test_predicted_nb):.3f}')

    print(f'Accuracy of Decision Tree {metrics.accuracy_score(y_test, y_test_predicted_dt):.3f}')
    print(f'Precision of Decision Tree {metrics.precision_score(y_test, y_test_predicted_dt):.3f}')
    print(f'Recall of Decision Tree {metrics.recall_score(y_test, y_test_predicted_dt):.3f}')

    print(f'Accuracy of SVM {metrics.accuracy_score(y_test, y_test_predicted_svm):.3f}')
    print(f'Precision of SVM {metrics.precision_score(y_test, y_test_predicted_svm):.3f}')
    print(f'Recall of SVM {metrics.recall_score(y_test, y_test_predicted_svm):.3f}')

    print(f'Accuracy of Random Forest {metrics.accuracy_score(y_test, y_test_predicted_rf):.3f}')
    print(f'Precision of Random Forest {metrics.precision_score(y_test, y_test_predicted_rf):.3f}')
    print(f'Recall of Random Forest {metrics.recall_score(y_test, y_test_predicted_rf):.3f}')

    print(f'Accuracy of Neural Network {metrics.accuracy_score(y_test, y_test_predicted_mlp):.3f}')
    print(f'Precision of Neural Network {metrics.precision_score(y_test, y_test_predicted_mlp):.3f}')
    print(f'Recall of Neural Network {metrics.recall_score(y_test, y_test_predicted_mlp):.3f}')

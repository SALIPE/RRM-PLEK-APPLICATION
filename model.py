from typing import List

import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score,
                             classification_report, confusion_matrix,
                             multilabel_confusion_matrix)
from sklearn.model_selection import ShuffleSplit, cross_val_score


def cross_val_model(X,
          Y:List[str]):
     
    # X_normalized = preprocessing.normalize(X, norm='l2')

    clf = tree.DecisionTreeClassifier()
    # clf.fit(X,Y)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, X, Y, cv=cv)

    max_accuracy = max(scores)
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        if(scores[i] == max_accuracy):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            clf.fit([X[i] for i in train_index],
               [Y[i] for i in train_index])

    return clf, scores

def save_model(model, filename:str):
    plt.figure(figsize=(300,100), dpi=80)
    class_names = model.classes_
    tree.plot_tree(model, fontsize=14, class_names=class_names)
    plt.savefig(filename)
    plt.close("all")


def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    class_names = clf.classes_
    # cm = multilabel_confusion_matrix(y, y_pred, labels=class_names)
    # clf_rep = classification_report(y, y_pred, target_names=class_names)

    accuracy = accuracy_score(y, y_pred,)
    return accuracy


def evaluate_bin_model(X_bins, y_bins, X, Y):
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    X_rnd, Y_rnd = np.array(X)[indices], np.array(Y)[indices]

    clf = tree.DecisionTreeClassifier()
    clf.fit(X_bins,y_bins)

    scores = confusion_matrix_scorer(clf, X_rnd, Y_rnd)

    return clf, scores




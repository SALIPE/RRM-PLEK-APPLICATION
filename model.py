from typing import List

from sklearn import preprocessing, tree
from sklearn.model_selection import ShuffleSplit, cross_val_score


def model(X,
          Y:List[str]):
     
    # X_normalized = preprocessing.normalize(X, norm='l2')

    clf = tree.DecisionTreeClassifier()
    clf.fit(X,Y)
    # cv = ShuffleSplit(n_splits=5, test_size=0.3, random_state=0)
    # scores = cross_val_score(clf, X, Y, cv=5)
    # print(scores)

    return clf


def predict_sequences(classification_model, to_predict:List[List[float]]):

    return classification_model.predict(to_predict)
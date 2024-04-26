from typing import List

from sklearn import preprocessing, tree
from sklearn.model_selection import ShuffleSplit, cross_val_score


def model(X,
          Y:List[str]):
     
    # X_normalized = preprocessing.normalize(X, norm='l2')

    clf = tree.DecisionTreeClassifier()
    # clf.fit(X,Y)
    cv = ShuffleSplit(n_splits=10, test_size=0.3, random_state=42)
    scores = cross_val_score(clf, X, Y, cv=cv)
    print(scores)

    max_accuracy = max(scores)
    for i, (train_index, test_index) in enumerate(cv.split(X)):
        if(scores[i] == max_accuracy):
            print(f"Fold {i}:")
            print(f"  Train: index={train_index}")
            clf.fit([X[i] for i in train_index],
               [Y[i] for i in train_index])


    return clf


def predict_sequences(classification_model, to_predict:List[List[float]]):

    return classification_model.predict(to_predict)
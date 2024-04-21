from typing import List

from sklearn import preprocessing, tree


def model(mrna_bins:List[float],
          ncrna_bins:List[float]):
     
    X = [ncrna_bins,mrna_bins]
    X_normalized = preprocessing.normalize(X, norm='l2')
    print(X_normalized)
    
    Y = ['ncRNA', 'mRNA']
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(X_normalized, Y)

    return clf
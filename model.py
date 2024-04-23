from typing import List

from Bio.Seq import Seq
from sklearn import preprocessing, tree
from sklearn.model_selection import train_test_split

import io_utils as iou

Y = ['ncRNA', 'mRNA']

def model(mrna_bins:List[float],
          ncrna_bins:List[float]):
     
    X = [ncrna_bins,mrna_bins]

    # X_normalized = preprocessing.normalize(X, norm='l2')

    clf = tree.DecisionTreeClassifier(
        random_state=42)
    clf = clf.fit(X,Y)

    return clf

def split_data(sequence_path:str, class_name:str):
    sequences = iou.buffer_sequences(sequence_path=sequence_path)

    rna_sequences: List[Seq] = []

    for key in sequences:
        seq = sequences[key]
        if(len(seq)>=200):
            rna_sequences.append(seq.seq)
            # rna_sequences.append(transcribe(seq.seq))


    X_train, X_test, y_train, y_test = train_test_split(
    rna_sequences, [class_name for i in rna_sequences], test_size=0.3, random_state=0)

    return X_train, X_test, y_test

def predict_sequences(classification_model, to_predict:List[List[float]]):

    return classification_model.predict(to_predict)
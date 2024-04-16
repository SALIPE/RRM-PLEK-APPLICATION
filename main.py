from typing import List

import numpy as np
import numpy.typing as npt
from Bio import SeqIO

from transformation_utils import to_dft

folders = ["Bos Tauros",
           "Danio rerio",
           "Gorilla gorilla"]

eiip_values ={
    "A":0.1260,
    "G":0.0806,
    "T":0.1335,
    "C":0.1340,
}

def reference_sequence(seq_path: str) -> npt.ArrayLike:
    file_sequences = []

    with open(seq_path, encoding='utf8') as reference:
        for record in SeqIO.parse(reference, 'fasta'):
            seq_char = str(record.seq).upper()
            file_sequences.append(seq_char)
    
    return file_sequences
        
def value_map(seq:str)-> npt.ArrayLike:
    return [eiip_values[char] for char in seq]
     

def to_numeric_char_value(seq_list: npt.ArrayLike)-> npt.ArrayLike:
     return [value_map(seq) for seq in seq_list]

        
def main():
    path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia1.fasta"
    sequence = reference_sequence(path_loc)
   
    sequence = to_numeric_char_value(sequence)
    print(sequence)
    # dft = to_dft(sequence)
    # print(dft)


if __name__ == "__main__":
    main()
    
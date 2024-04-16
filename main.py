from typing import List

import numpy as np
import numpy.typing as npt
from Bio import SeqIO

from io_utils import buffer_sequences, to_numeric_char_value
from transformation_utils import cross_spectrum, to_dft

folders = ["Bos Tauros",
           "Danio rerio",
           "Gorilla gorilla"]

        
def main():
    path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia2.txt"
    sequence = buffer_sequences(sequence_path=path_loc)

    first_key = list(sequence.keys())

    firts_sequence = sequence[first_key[0]]
    second_sequence = sequence[first_key[1]]

    sequence_1 = to_numeric_char_value([firts_sequence.seq])
    sequence_1 = np.array(sequence_1[0], dtype=float)

    seq_size = len(sequence_1)

    sequence_2 = to_numeric_char_value([second_sequence.seq])
    sequence_2 = np.array(sequence_2[0], dtype=float)
    sequence_2 = sequence_2[0:seq_size]

    
  
    dtf1 = to_dft(sequence_1,seq_size)
    dtf2 = to_dft(sequence_2,seq_size)

    seq_size = seq_size-1
    freq = np.fft.fftfreq(seq_size)
    freq = np.fft.fftshift(freq)

    cross_spectrum(dtf1,dtf2,freq,seq_size)



if __name__ == "__main__":
    main()
    
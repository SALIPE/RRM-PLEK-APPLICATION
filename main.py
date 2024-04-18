from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from Bio import Seq, SeqIO
from scipy.fft import rfftfreq

import io_utils as iou
import transformation_utils as tfu


def main():
    path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia2.txt"
    sequence = iou.buffer_sequences(sequence_path=path_loc)

    first_key = list(sequence.keys())
    firts_sequence = sequence[first_key[0]]
    firts_sequence.seq = Seq.transcribe(firts_sequence.seq)

    amn_values = iou.to_aminoacid_char_value([firts_sequence.seq])[0]

    freq  = rfftfreq(len(amn_values))
    dft = tfu.to_dft(seq=amn_values)

    plt.plot(freq[1:], np.abs(dft)[1:])
    plt.show()
   
   

if __name__ == "__main__":
    main()
    
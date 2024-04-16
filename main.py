from typing import List

import numpy as np
import numpy.typing as npt
from Bio import SeqIO

from io_utils import buffer_sequences, to_numeric_char_value
from transformation_utils import to_dft

folders = ["Bos Tauros",
           "Danio rerio",
           "Gorilla gorilla"]

        
def main():
    path_loc = "..\dataset-plek\Dados\Human\human_rna_fna_refseq_mRNA_22389"
    sequence = buffer_sequences(sequence_path=path_loc)

    first_key = list(sequence.keys())

    firts_sequence = sequence[first_key[60]]

    sequence = to_numeric_char_value([firts_sequence.seq])
    sequence = np.array(sequence[0])
  
    to_dft(sequence)


if __name__ == "__main__":
    main()
    
from typing import Dict, Union

import numpy.typing as npt
from Bio import File, SeqIO, SeqRecord

eiip_values: Dict ={
    "A":0.1260,
    "G":0.0806,
    "T":0.1335,
    "C":0.1340,
}

def buffer_sequences(
    sequence_path: str, reference: bool = False
) -> Union[Dict[str, SeqRecord.SeqRecord],File._IndexedSeqFileDict]:

    if reference:
        return SeqIO.to_dict(SeqIO.parse(sequence_path, 'fasta'))
    return SeqIO.index(sequence_path, 'fasta')



def find_sequences(seq_path: str) -> npt.ArrayLike:
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
from typing import Dict, List, Union

import numpy as np
import numpy.typing as npt
from Bio import File, SeqIO, SeqRecord
from Bio.Seq import Seq

EIIP_NUCLEOTIDE: Dict[str,float] ={
    "A":0.1260,
    "G":0.0806,
    "T":0.1335,
    "C":0.1340,
}

EIIP_AMINOACID: Dict[str,float] = {
    'L': 0.0000,
    'I': 0.0000,
    'N': 0.0036,
    'G': 0.0050,
    'V': 0.0057,
    'E': 0.0058,
    'P': 0.0198,
    'H': 0.0242,
    'K': 0.0371,
    'A': 0.0373,
    'Y': 0.0516,
    'W': 0.0548,
    'Q': 0.0761,
    'M': 0.0823,
    'S': 0.0829,
    'C': 0.0829,
    'T': 0.0941,
    'F': 0.0946,
    'R': 0.0959,
    'D': 0.1263,
    'X': 0.0000,
    'J': 0.0000,
    '*': 0.0000
}
AMINOACID_MAP: Dict[str,List[str]]={
    'A': ['GCU', 'GCC','GCA','GCG'],
    'I': ['AUU', 'AUC','AUA'],
    'N': ['AAU', 'AAC'],
    'G': ['GGU', 'GGC','GGA','GGG'],
    'V': ['GUU', 'GUC','GUA','GUG'],
    'E': ['GAA', 'GAG'],
    'P': ['CCU', 'CCC','CCA','CCG'],
    'H': ['CAU', 'CAC'],
    'K': ['AAA', 'AAG'],
    'L': ['UUA', 'UUG','CUU','CUC','CUA','CUG'],
    'Y': ['UAU', 'UAC'],
    'W': ['UGG'],
    'Q': ['CAA', 'CAG'],
    'M': ['AUG'],
    'S': ['UCU', 'UCC','UCA','UCG','AGU','AGC'],
    'C': ['UGU', 'UGC'],
    'T': ['ACU','ACC','ACA','ACG'],
    'F': ['UUU', 'UUC'],
    'R': ['CGU', 'CGC','CGA','CGG','AGA','AGG'],
    'D': ['GAU', 'GAC'],
    '*': ['UGA','UAA','UAG',
          'NNN','NN','N',
          'ANN','NNA','AAN','ANA','NAA',
          'UNN','NNU','UUN','UNU','NUU',
          'CNN','NNC','CCN','CNC','NCC',
          'UNN','NNU','UUN','UNU','NUU',
          'GNN','NNG','GGN','GNG','NGG',
          'NCA','NCG','NCU',
          'NGA','NGC','NGU',
          'NAC','NAG','NAU',
          'NUC','NUG','NUA',
          'ANG','ANC','ANU',
          'UNG','UNC','UNA',
          'GNU','GNC','GNA',
          'CNU','CNG','CNA',
          'CUN','CGN','CAN',
          'GUN','GCN','GAN',
          'AUN','ACN','AGN',
          'UGN','UCN','UAN',
          'A','C','U','G',
          'AA','AC','AG','AU',
          'CA','CU','CG','CC',
          'UA','UU','UC','UG',
          'GA','GU','GG','GC']
}

def translate(seq: str) -> str:
    sequence = Seq(seq)
    return str(sequence.translate()).replace('*', '').replace('X', '').replace('J', '')

def create_aminoacid_map()->Dict[str,str]:
    amn_map = {}
    for k, v in AMINOACID_MAP.items():
        for char in v:
            amn_map[char] = k
    return amn_map

AMINOACIDS:Dict[str,str] = create_aminoacid_map()


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


def nucleotide_map(seq:str)-> npt.ArrayLike:
    return [EIIP_NUCLEOTIDE[char] for char in seq]

def aminoacid_map(seq:str)-> List[float]:
    # amin_char = [AMINOACIDS[seq[i:i+3]] for i in range(0, len(seq), 3)]
    amin_char = translate(seq)
    return  [EIIP_AMINOACID[char] for char in amin_char]

def rna_aminoacid_map(seq:str)-> List[float]:
    return  [EIIP_AMINOACID[char] for char in seq]
     
def to_nucleotide_char_value(seq_list: npt.ArrayLike)-> npt.ArrayLike:
     return [nucleotide_map(seq) for seq in seq_list]

def to_aminoacid_char_value(seq_list: npt.ArrayLike)-> List[List[float]]:
     return [aminoacid_map(seq) for seq in seq_list]

def rna_to_aminoacid_char_value(seq_list: npt.ArrayLike)-> List[List[float]]:
     return [rna_aminoacid_map(seq) for seq in seq_list]

def min_max_norm(x:List[float])->List[float]:
    # Min-Max Normalization
    min_val = np.min(x)
    max_val = np.max(x)
    print(np.nan_to_num(max_val - min_val))
    normalized_data = (x - min_val) / np.nan_to_num(max_val - min_val)

    return normalized_data
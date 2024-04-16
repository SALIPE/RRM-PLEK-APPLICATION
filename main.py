from typing import List

from Bio import SeqIO

folders = ["Bos Tauros",
           "Danio rerio",
           "Gorilla gorilla"]

eiip_values ={
    "A":0.1260,
    "G":0.0806,
    "T":0.1335,
    "C":0.1340,
}

def reference_sequence(seq_path: str) -> str:

    with open(seq_path, encoding='utf8') as reference:
        for record in SeqIO.parse(reference, 'fasta'):
            return str(record.seq).upper()
        
def value_map(char:str)->float:
    return eiip_values[char]

def to_numeric_char_value(dna_seq: str)-> List[float]:
    return list(map(value_map,dna_seq))
        
def main():
    path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia1.txt"
    sequence = reference_sequence(path_loc)
    print(to_numeric_char_value(sequence))


if __name__ == "__main__":
    main()
    
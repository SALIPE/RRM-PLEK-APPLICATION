import numpy as np
from Bio import SeqIO

file_name = ""

seqs_len = []
with open(file_name) as handle:
    for record in SeqIO.parse(handle, "fasta"):
        seqs_len.append(len(record.seq))
        
seqs_mean = np.mean(seqs_len)
seqs_sd = np.std(seqs_len)
variation = seqs_sd/seqs_mean
seqs_min = min(seqs_len)
seqs_max = max(seqs_len)

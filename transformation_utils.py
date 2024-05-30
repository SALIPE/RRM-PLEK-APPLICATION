from typing import List

import numpy as np
from Bio.Seq import Seq
from scipy.fft import rfft

import io_utils as iou


## produto direto
def element_wise_product(dft_list:List[List[float]])->List[float]:
    """
    Performs element-wise product on a list of lists,
    even with varying sublist lengths.
    """

    res:List[float] = dft_list[0]
    for b in dft_list[1:]:
        res = [np.multiply(x1,x2) for x1, x2 in zip(res,b)]

    return res

def internal_prod(lst:List[float])->float:

    if(len(lst)==0):
        return 0.
    
    res = lst[0]
    for b in lst[1:]:
        if(np.isinf(np.multiply(res,b))):
            break
        res = np.multiply(res,b)

    return res

def to_fft_collection(sequences:List[Seq],
                      seq_size:int=None)->List[List[float]]:
  
    # amn_values = iou.to_nucleotide_char_value([seq for seq in sequences])
    amn_values = iou.to_aminoacid_char_value([seq for seq in sequences])
    coeff_FFT_zip = []

    for pseq in amn_values:
        n = len(pseq)
        if(seq_size is not None):
            n = seq_size
        dtf_to_zip = rfft(x=pseq, n=n)
        coeff_FFT_zip.append(np.abs(dtf_to_zip)[1:])

    return coeff_FFT_zip



def handle_data(sequence_path:str,
                class_name:str, 
                to_dft:bool=True,
                seq_size:int=None):
    sequences = iou.buffer_sequences(sequence_path=sequence_path)

    rna_sequences: List[Seq] = []

    for key in sequences:
        seq = sequences[key]
        if(len(seq.seq)>0):
            rna_sequences.append(seq.seq)

    eiip_sequences: List[List[float]]=[]

    if to_dft:
        eiip_sequences = to_fft_collection(sequences=rna_sequences,seq_size=seq_size)
    else:
        eiip_sequences = iou.to_aminoacid_char_value(rna_sequences)
        # eiip_sequences = iou.to_nucleotide_char_value(rna_sequences)

    labels =  [class_name for i in eiip_sequences]
    return eiip_sequences, labels


def prepare_data(m_path_loc:str,
                 nc_path_loc:str, 
                 to_dft:bool, 
                 seq_size:int=None,
                 specie:str=""):
    print("Loading and transforming data...")
   
    # mRNA data
    Mx, My = handle_data(m_path_loc, "mRNA_"+specie,to_dft,seq_size)
    
    # ncRNA data
    NCx,NCy = handle_data(nc_path_loc, "ncRNA_"+specie,to_dft,seq_size)
    
    # Balance sequences proportions
    Mx_size = len(Mx)
    NCx_size = len(NCx)
    # if(not to_dft):
    print(f'{Mx_size} original size -> mRNA_{specie}')
    print(f'{NCx_size} original size -> ncRNA_{specie}')

    if(Mx_size > NCx_size):
        Mx = Mx[0:NCx_size]
        My = My[0:NCx_size]
        
    elif(NCx_size>Mx_size):
        NCx = NCx[0:Mx_size]
        NCy = NCy[0:Mx_size]
        
    print(f'{len(Mx)} sequences -> mRNA_{specie}')
    print(f'{len(NCx)} sequences -> ncRNA_{specie}')

    return Mx,My,NCx,NCy


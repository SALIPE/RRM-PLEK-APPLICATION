

from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from Bio.Seq import Seq
from numpy import absolute, array
from scipy.fft import rfft, rfftfreq

import io_utils as iou


def cross_spectrum(seq_x: npt.ArrayLike, 
                   seq_y:npt.ArrayLike,
                   size:int):
    spectrum = []
    n = range(size)
    for i in n:
        spectrum.append(seq_x[i].real*seq_y[i].imag)

    spectrum = array(spectrum)
    spectrum = absolute(spectrum)

    return spectrum

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

def to_fft_collection(sequences:List[Seq])->List[List[float]]:
  
    amn_values = iou.to_aminoacid_char_value([seq for seq in sequences])
    coeff_FFT_zip = []

    for pseq in amn_values:
        dtf_to_zip = rfft(x=pseq)
        coeff_FFT_zip.append(np.abs(dtf_to_zip)[1:])

    return coeff_FFT_zip

def collect_bins(sequences,
                 seq_size:int,
                 class_name:str=""):
    
    freq:List[float] = rfftfreq((seq_size*2)-1, d=1)
    # arr_list = np.array(sequences, dtype=np.float32)
    arr_list = np.array(sequences)
    cross_spectral = np.nan_to_num(np.prod(a=arr_list, axis=0))
    # print(cross_spectral)

    # plt.plot(freq,cross_spectral)
    # plt.title(f'Cross-Spectral {class_name}\nTamanho da serie {freq.size}')
    # plt.show()

    return cross_spectral


def handle_data(sequence_path:str, class_name:str, to_eiip:bool=True):
    sequences = iou.buffer_sequences(sequence_path=sequence_path)

    rna_sequences: List[Seq] = []

    for key in sequences:
        seq = sequences[key]
        rna_sequences.append(seq.seq)
    
    if to_eiip:
        eiip_sequences: List[List[float]] = to_fft_collection(sequences=rna_sequences )
        labels =  [class_name for i in eiip_sequences]
        return eiip_sequences, labels
    else:
        protein_sequences = [iou.translate(seq) for seq in rna_sequences]
        labels =  [class_name for i in rna_sequences]
        return protein_sequences, labels

def prepare_data(m_path_loc:str,nc_path_loc:str, to_dft:bool):
    print("Loading and transforming data...")
   
    # mRNA data
    Mx, My = handle_data(m_path_loc, "mRNA",to_dft)
    # ncRNA data
    NCx,NCy = handle_data(nc_path_loc, "ncRNA",to_dft)

    return Mx,My,NCx,NCy
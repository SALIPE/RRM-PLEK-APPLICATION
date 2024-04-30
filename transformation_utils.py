

import bisect
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from Bio.Seq import Seq
from numpy import absolute, array
from scipy.fft import irfft, rfft, rfftfreq

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
    # cross_spectral = iou.min_max_norm(cross_spectral)
    # print(cross_spectral)

    # plt.plot(freq,cross_spectral)
    # plt.title(f'Cross-Spectral {class_name}\nTamanho da serie {freq.size}')
    # plt.show()

    return cross_spectral


def handle_data(sequence_path:str, class_name:str, to_dft:bool=True, have_limit:bool=True):
    sequences = iou.buffer_sequences(sequence_path=sequence_path)

    rna_sequences: List[Seq] = []

    for key in sequences:
        if(have_limit and len(rna_sequences)==359):
            break
        seq = sequences[key]
        rna_sequences.append(seq.seq)

    eiip_sequences: List[List[float]]=[]

    if to_dft:
        eiip_sequences = to_fft_collection(sequences=rna_sequences )
    else:
        eiip_sequences = iou.to_aminoacid_char_value([seq for seq in rna_sequences])

    labels =  [class_name for i in eiip_sequences]
    return eiip_sequences, labels


def prepare_data(m_path_loc:str,nc_path_loc:str, to_dft:bool, specie:str="",have_limit:bool=True):
    print("Loading and transforming data...")
   
    # mRNA data
    Mx, My = handle_data(m_path_loc, "mRNA_"+specie,to_dft,have_limit)
    print(f'{len(Mx)} sequences -> {"mRNA_"+specie}')
    
    # ncRNA data
    NCx,NCy = handle_data(nc_path_loc, "ncRNA_"+specie,to_dft,have_limit)
    print(f'{len(NCx)} sequences -> {"ncRNA_"+specie}')

    return Mx,My,NCx,NCy

def return_to_eiip(fft_sequences,frequencies,selected_freq_indexes):
    eiip_seqs= []
    # Suppose selected_freq_indexes contains the indexes of frequencies you're interested in
    selected_frequencies = np.array(frequencies)[selected_freq_indexes]

    for dft_seq in fft_sequences:
        freqs:List[float] = rfftfreq((len(dft_seq)*2)-1, d=1)

        # Find the indexes in the DFT output corresponding to these frequencies
        corresponding_indexes = [bisect.bisect(freqs, freq)-1 for freq in selected_frequencies]
        corresponding_indexes = list(set(corresponding_indexes))
        corresponding_indexes.sort()

        #Use the iDFT to reconstruct the original sequence from these frequency components
        reconstructed_dft_seq = np.zeros_like([0,*dft_seq])
        reconstructed_dft_seq[corresponding_indexes] = np.array(dft_seq)[corresponding_indexes]
        reconstructed_seq = irfft(reconstructed_dft_seq)
        reconstructed_seq = [float("{:.4f}".format(x)) for x in reconstructed_seq]
        reconstructed_seq = np.abs(reconstructed_seq)

    
        eiip_seqs.append(reconstructed_seq)

    
    return eiip_seqs
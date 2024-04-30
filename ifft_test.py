from typing import List

import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

import io_utils as iou
import main as mn
import transformation_utils as tfu

if __name__ == "__main__":

    file = {
            "specie":"Gorilla_gorilla",
            "m_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia2.txt",
            "nc_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia1.txt"
        }

    Mx,My,NCx,NCy = mn.prepare_dft_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=file["specie"],
        have_limit=True
    )
        
    p_Mx,p_My,p_NCx,p_NCy = mn.prepare_protein_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=file["specie"],
        have_limit=True
    )

    X,Y,seq_size,size_ls = mn.evaluate_diff_sequences(
            Mx=Mx,
            My=My,
            NCx=NCx,
            NCy=NCy)


    # NORMALIZING DFT
    #GET BINS FITTED FOR SE SEQUENCE LENGTH
    nc_bins, m_bins = mn.get_cross_spectrum(Mx,NCx,True,size_ls,seq_size)

    nc_bins = iou.min_max_norm(nc_bins)

    plt.plot(rfftfreq((len(nc_bins)*2)-1),nc_bins)
    plt.title(f'Histograma NC individual')
    plt.show()

    m_bins = iou.min_max_norm(m_bins)

    plt.plot(rfftfreq((len(m_bins)*2)-1),m_bins)
    plt.title(f'Histograma M individual')
    plt.show()
    
    m_idxs = []
    for i in range(len(m_bins)):
        if(m_bins[i] > 0.1):
            m_idxs.append(i)

    nc_idxs = []
    for i in range(len(nc_bins)):
        if(nc_bins[i] > 0.1):
            nc_idxs.append(i)
    
    selected_freq_indexes = list(set(m_idxs+nc_idxs))
    selected_freq_indexes.sort()

    
        
    m_dftseq = X[2]
    print(Mx[2])
    pm_dftseq = p_Mx[2][0:len(m_dftseq)]

    print(f'\nIndexes:')
    print(selected_freq_indexes)

    print(f'\nOriginal DFT sequence: {len(m_dftseq)}\n')
    print(m_dftseq)
    print(f'\nOriginal PROTEIN sequence: {len(pm_dftseq)}\n')
    print(pm_dftseq)

    reconstructed_dft_seq = np.zeros_like(m_dftseq)
    reconstructed_dft_seq[selected_freq_indexes] = np.array(m_dftseq)[selected_freq_indexes]

    print(f'\nNew DFT sequence: {len(reconstructed_dft_seq)}\n')
    print(reconstructed_dft_seq)

    reconstructed_seq = irfft(reconstructed_dft_seq, n=len(pm_dftseq))
    reconstructed_seq = [float("{:.4f}".format(x)) for x in reconstructed_seq]
    reconstructed_seq = np.abs(reconstructed_seq)
   
    print(f'\nNew PROTEIN sequence: {len(reconstructed_seq)}\n')
    print(reconstructed_seq)
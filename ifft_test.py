from typing import List

import numpy as np
from scipy.fft import irfft, rfft, rfftfreq

import io_utils as iou
import main as mn
import transformation_utils as tfu


def extract_determinisct_proteins(fft_sequences,
                                  p_sequences,
                                  selected_freq_indexes):
    
    filtered_sequences = []
    ifft_sequences = []
    eiip_seqs= []

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


    for p_seq, f_seq in zip(p_sequences,ifft_sequences):
        filtered:List[float] = []
        #Filtering protein sequences
        size = len(min([p_seq, f_seq],key=len))
        for i in range(size):
            pvalue = p_seq[i]
            fvalue = f_seq[i]
            if(float("{:.3f}".format(fvalue))==float("{:.3f}".format(pvalue))):
                # print(f'{i} = {float("{:.3f}".format(fvalue))}')
                filtered.append(pvalue)
        filtered_sequences.append(filtered)

    return filtered_sequences

if __name__ == "__main__":

    file = {
            "specie":"Gorilla_gorilla",
            "m_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia2.txt",
            "nc_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia1.txt"
        },

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
    m_bins = iou.min_max_norm(m_bins)
    
    m_idxs = []
    for i in range(m_bins):
        if(m_bins[i] > 0.1):
            m_idxs.append(i)

    nc_idxs = []
    for i in range(nc_bins):
        if(nc_bins[i] > 0.1):
            nc_idxs.append(i)
    
    selected_freq_indexes = list(set(m_idxs+nc_idxs))
    selected_freq_indexes.sort()
        
    fp_NCx = extract_determinisct_proteins(NCx,p_NCx,selected_freq_indexes)

    fp_Mx = extract_determinisct_proteins(Mx,p_Mx,selected_freq_indexes)

    reconstructed_dft_seq = np.zeros_like([0,*dft_seq])
    reconstructed_dft_seq[corresponding_indexes] = np.array(dft_seq)[corresponding_indexes]
    reconstructed_seq = irfft(reconstructed_dft_seq)
    reconstructed_seq = [float("{:.4f}".format(x)) for x in reconstructed_seq]
    reconstructed_seq = np.abs(reconstructed_seq)
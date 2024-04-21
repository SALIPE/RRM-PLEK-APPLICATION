import bisect
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from Bio.Seq import Seq, transcribe
from scipy.fft import rfft, rfftfreq

import io_utils as iou
import model
import transformation_utils as tfu

ncrna_file = "\\sequencia1.txt"
mrna_file = "\\sequencia2.txt"


def data_bin_collect(sequences:List[Seq], print_chart:bool=False)->List[float]:
    # path_loc = "..\dataset-plek\Gorilla_gorilla"+class_file
    # sequences = iou.buffer_sequences(sequence_path=path_loc)

    # rna_sequences: List[Seq] = []

    # for key in sequences:
    #     seq = sequences[key]
    #     if(len(seq)>=200):
    #         rna_sequences.append(seq.seq)
    #         # rna_sequences.append(transcribe(seq.seq))
        
    amn_values = iou.to_aminoacid_char_value([seq for seq in sequences])

    max_freq = 0.5
    hist_bins = 150
    coeff_FFT = []
    coeff_FFT_zip = []
    coeff_FFT_mean_len = []
    intervals = np.linspace(0, max_freq, hist_bins)
    hist = [[] for _ in range(hist_bins)]
    histogram = []

    # max_val = len(max(amn_values,key=len))
    min_val = len(min(amn_values,key=len))
    # mean_value = np.mean([len(i) for i in amn_values])
    # std_value = np.std([len(i) for i in amn_values])
    # mean_len = int(mean_value + std_value)

    for pseq in amn_values:
        # Max size sequences
        # dft = rfft(x=pseq, n=max_val)  
        # coeff_FFT.append(np.abs(dft))

        # Original size sequences
        dtf_to_zip = rfft(pseq, norm="ortho")
        coeff_FFT_zip.append(np.abs(dtf_to_zip)[1:])
        
        # histograma
        freq_hist = rfftfreq(len(pseq), d=1)
        fft_freq = [(fft,freqs) for fft, freqs in zip(dtf_to_zip,freq_hist)]
        
        # mean_len
        # fft_eiip_mean_len = rfft(pseq, n=mean_len)
        # coeff_FFT_mean_len.append(np.abs(fft_eiip_mean_len))

        for val in fft_freq:
            hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))
       

    #     plt.plot(freq_hist[1:],np.abs(dtf_to_zip)[1:])
        
    # plot_name = 'Tamanho original'
    # plt.title(plot_name)
    # plt.show()

    ### Produto direto zipando para o menor valor de sequencia
    # cross_spectral_zip = tfu.element_wise_product(coeff_FFT_zip)
    # freq_zip = rfftfreq(min_val,d=1)

    # plt.plot(freq_zip[1:],cross_spectral_zip)
    # plt.title(f'Produto direto entre coeficientes zipado\nTamanho da serie {freq_zip.size}')
    # plt.show()
    
    ### Produto direto tomando sequencias de mesmo tamanho
    # freq = rfftfreq(max_val, d=1)
    # cross_spectral = np.prod(coeff_FFT, axis=0)

    # plt.plot(freq[1:],cross_spectral[1:])
    # plt.title(f'Series de mesmo tamanho\nTamanho da serie {freq.size}')
    # plt.show()
    
    # ### mean len
    # freq_mean_len = rfftfreq(mean_len, d=1)
    # cross_spectral_mean_len = np.prod(coeff_FFT_mean_len, axis=0)

    # plt.plot(freq_mean_len[1:],cross_spectral_mean_len[1:])
    # plt.title(f'Tamanho: média + desvio padrão\nTamanho da serie {freq.size}')
    # plt.show()
    
    ### histogram
    for lst in hist:
        histogram.append(np.prod(lst))
    
    if(print_chart):
        plt.plot(intervals[1:],histogram[1:])
        plt.title(f'Histograma\nNumero de Bins (0-{max_freq}): {intervals.size}')
        plt.show()

    return histogram[1:]
    

if __name__ == "__main__":

    m_path_loc = "..\dataset-plek\Gorilla_gorilla"+mrna_file
    nc_path_loc = "..\dataset-plek\Gorilla_gorilla"+ncrna_file
   
    # mRNA data
    X1_train, X1_test = model.split_data(m_path_loc, "mRNA")
    mrna_train_bins:List[float] = data_bin_collect(X1_train,True)

    mrna_test_bins:List[float] = data_bin_collect([X1_test[0],X1_test[1]],True)

    # ncRNA data
    X2_train, X2_test = model.split_data(nc_path_loc, "ncRNA")
    ncrna_train_bins:List[float] = data_bin_collect(X2_train,True)

    ncrna_test_bins:List[float] = data_bin_collect([X2_test[0],X2_test[1]],True)

    #    print(mrna_bins)
    #    print(ncrna_bins)

    classification_model = model.model(
        ncrna_bins=ncrna_train_bins,
        mrna_bins=mrna_train_bins)
    
    
    pred_mrna = model.predict_sequences(
        classification_model=classification_model,
        to_predict=[mrna_test_bins,ncrna_test_bins])
  
    print(pred_mrna)

    
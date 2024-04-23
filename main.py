import bisect
from typing import List

import matplotlib.pyplot as plt
import numpy as np
from Bio.Seq import Seq
from scipy.fft import rfft, rfftfreq

import io_utils as iou
import model


def data_bin_collect(sequences:List[Seq],
                     class_name:str,
                      print_chart:bool=False)->List[float]:
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
    # min_val = len(min(amn_values,key=len))
    # mean_value = np.mean([len(i) for i in amn_values])
    # std_value = np.std([len(i) for i in amn_values])
    # mean_len = int(mean_value + std_value)

    if(len(amn_values)==1):
            pseq = amn_values[0]

            dtf = rfft(pseq,n=hist_bins*2, norm="ortho")
            # dtf = iou.min_max_norm(np.abs(dtf))
 
            histogram = np.abs(dtf)[1:]
    else:            
        for pseq in amn_values:
            # Max size sequences
            # dft = rfft(x=pseq, n=max_val)  
            # coeff_FFT.append(np.abs(dft))

            # Original size sequences
            dtf_to_zip = rfft(pseq, norm="ortho")
            # dtf_to_zip = iou.min_max_norm(np.abs(dtf_to_zip))
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
    if(len(amn_values)>1):
        for lst in hist:
            histogram.append(np.prod(lst))
        # histogram = iou.min_max_norm(histogram)
        
    
    if(print_chart):
        plt.plot(intervals[1:],histogram[1:])
        plt.title(f'Histograma {class_name}\nNumero de Bins (0-{max_freq}): {intervals.size}')
        plt.show()

    return histogram[1:]
    

if __name__ == "__main__":

    # m_path_loc = "..\dataset-plek\Dados\Human\human_rna_fna_refseq_mRNA_22389"
    # nc_path_loc = "..\dataset-plek\Dados\Human\human_gencode_v17_lncRNA_22389"

    m_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia1.txt"
    nc_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia2.txt"

    # m_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia2.txt"
    # nc_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia1.txt"
   
    # mRNA data
    X1_train, X1_test,y1_test = model.split_data(m_path_loc, "mRNA")
    mrna_train_bins:List[float] = data_bin_collect(X1_train,
                                                   class_name=" train mRNA",
                                                   print_chart=True)
    
    # ncRNA data
    X2_train, X2_test,y2_test = model.split_data(nc_path_loc, "ncRNA")
    ncrna_train_bins:List[float] = data_bin_collect(X2_train,
                                                    class_name="train ncRNA",
                                                   print_chart=True)
    
    min_val = len(min([y2_test,y1_test],key=len))
    print(min_val)
    # mrna_test = zip(X1_test,y1_test)
    mrna_test_bins:List[List[float]] = []
    for i in range(min_val):
        mrna_test_bins.append(data_bin_collect([X1_test[i],X1_test[i]],
                                                  class_name="test mRNA",
                                                   print_chart=False))

    # ncrna_test = zip(X2_test,y2_test)
    ncrna_test_bins:List[List[float]] = []
    for i in range(min_val):
        ncrna_test_bins.append(data_bin_collect([X2_test[i],X2_test[i]],
                                                   class_name="test ncRNA",
                                                   print_chart=False))

 
    classification_model = model.model(
        ncrna_bins=ncrna_train_bins,
        mrna_bins=mrna_train_bins)
    
    to_predict_list:List[List[float]] = []

    for i in range(min_val):
        to_predict_list.append(ncrna_test_bins[i])
        to_predict_list.append(mrna_test_bins[i])
    
    pred_mrna = model.predict_sequences(
        classification_model=classification_model,
        to_predict=to_predict_list)
  
    t_mrna = 0
    f_mrna = 0
    t_ncrna = 0
    f_ncrna = 0
    for i in range(len(pred_mrna)):
        if(pred_mrna[i]=="mRNA"):
            if(i%2!=0):
                t_mrna+=1
            else:
                f_mrna+=1
        elif(pred_mrna[i]=="ncRNA"):
            if(i%2==0):
                t_ncrna+=1
            else:
                f_ncrna+=1

    accuracy = (t_mrna+t_ncrna)/len(pred_mrna)
    mrna_acurracy = t_mrna/(t_mrna+f_ncrna)
    ncrna_acurracy = t_ncrna/(t_ncrna+f_mrna)

    print(f'Acurracy mRNA {mrna_acurracy*100}%')
    print(f'Acurracy ncRNA {ncrna_acurracy*100}%')
    print(f'Acurracy {accuracy*100}%')
    print(pred_mrna)


    
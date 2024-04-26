import bisect
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.fft import rfft, rfftfreq
from sklearn import tree
from sklearn.model_selection import train_test_split

import io_utils as iou
import model


def data_bin_collect(sequences:List[Seq],
                    seq_size:int,
                    class_name:str = "",
                    to_bin:bool = False,
                    print_chart:bool=False):
  
    amn_values = iou.to_aminoacid_char_value([seq for seq in sequences])

    max_freq = 0.5
    hist_bins = 150
    # coeff_FFT = []
    coeff_FFT_zip = []
    # coeff_FFT_mean_len = []
    intervals = np.linspace(0, max_freq, hist_bins)
    hist = [[] for _ in range(hist_bins)]
    histogram = []

    # max_val = len(max(amn_values,key=len))
    # min_val = len(min(amn_values,key=len))
    # mean_value = np.mean([len(i) for i in amn_values])
    # std_value = np.std([len(i) for i in amn_values])
    # mean_len = int(mean_value + std_value)
   
    for pseq in amn_values:
        # Max size sequences
        # dft = rfft(x=pseq, n=max_val)  
        # coeff_FFT.append(np.abs(dft))

        # Original size sequences
        n = len(pseq) if to_bin else seq_size
        
        dtf_to_zip = rfft(x=pseq,n=n ,norm="ortho")
        # dtf_to_zip = iou.min_max_norm(np.abs(dtf_to_zip))
        coeff_FFT_zip.append(np.abs(dtf_to_zip)[1:])
        
        # mean_len
        # fft_eiip_mean_len = rfft(pseq, n=mean_len)
        # coeff_FFT_mean_len.append(np.abs(fft_eiip_mean_len))
        if(to_bin):
            # histograma
            freq_hist = rfftfreq(len(pseq), d=1)
            fft_freq = [(fft,freqs) for fft, freqs in zip(dtf_to_zip,freq_hist)]

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
    if(to_bin):
        for lst in hist:
            histogram.append(np.prod(lst))
        
        if(print_chart):
            plt.plot(intervals[1:],histogram[1:])
            plt.title(f'Histograma {class_name}\nNumero de Bins (0-{max_freq}): {intervals.size}')
            plt.show()
        
        return histogram[1:]
      

    return coeff_FFT_zip


def handle_data(sequence_path:str, class_name:str):
    sequences = iou.buffer_sequences(sequence_path=sequence_path)

    rna_sequences: List[Seq] = []

    for key in sequences:
        seq = sequences[key]
        rna_sequences.append(seq.seq)
 
    labels =  [class_name for i in rna_sequences]

    return rna_sequences, labels


def train_model():

    # m_path_loc = "..\dataset-plek\Dados\Human\human_rna_fna_refseq_mRNA_22389"
    # nc_path_loc = "..\dataset-plek\Dados\Human\human_gencode_v17_lncRNA_22389"

    # m_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia1.txt"
    # nc_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia2.txt"

    m_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia2.txt"
    nc_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia1.txt"

    print("Carregando e transformando dados...")
   
    # mRNA data
    Mx, My = handle_data(m_path_loc, "mRNA")
   
    # ncRNA data
    NCx,NCy = handle_data(nc_path_loc, "ncRNA")
   

    min_val = len(min([*Mx,NCx],key=len))
 
    Mx = data_bin_collect(sequences=Mx, seq_size=min_val)
    print("mRNAs transformados.")
    NCx = data_bin_collect(sequences=NCx, seq_size=min_val)
    print("ncRNAs transformados.")

    df = pd.DataFrame({'sequences': [*Mx,*NCx], 
                       'labels': [*My,*NCy]})
    
    X = df['sequences'].to_list()
    Y = df['labels'].to_list()

    # X_train, X_test, y_train, y_test = train_test_split(
    #     X, Y, test_size=0.4, random_state=7,shuffle=True)


    # print([y_train[0],y_train[1]])
    filename = "rrna_decisiontree.pickle"

    # load model
    # loaded_model = pickle.load(open(filename, "rb"))

    print("Carregando modelo...")
    classification_model = model.model(X=X, Y=Y)

    # save model
    pickle.dump(classification_model, open(filename, "wb"))

if __name__ == "__main__":
    # filename = "rrna_decisiontree.pickle"
    # loaded_model = pickle.load(open(filename, "rb"))

    
    
    # # plt.figure(figsize=(100,100), dpi=80)
    # plt.show(tree.plot_tree(loaded_model, fontsize=10))
    # plt.savefig('foo.png')
    # plt.savefig('foo.pdf')
    # plt.close()

    train_model()
    
  


    
import bisect
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.fft import rfft, rfftfreq
from sklearn import tree
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import ShuffleSplit, cross_val_score

import io_utils as iou
import model
import transformation_utils as tfu

LABELS = ["mRNA","ncRNA"]


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



def prepare_dft_data():

    # m_path_loc = "..\dataset-plek\Dados\Human\human_rna_fna_refseq_mRNA_22389"
    # nc_path_loc = "..\dataset-plek\Dados\Human\human_gencode_v17_lncRNA_22389"

    # m_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia1.txt"
    # nc_path_loc = "..\dataset-plek\Macaca_mulatta\sequencia2.txt"

    m_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia2.txt"
    nc_path_loc = "..\dataset-plek\Gorilla_gorilla\sequencia1.txt"

    print("Loading and transforming data...")
   
    # mRNA data
    Mx, My = tfu.handle_data(m_path_loc, "mRNA")
    # ncRNA data
    NCx,NCy = tfu.handle_data(nc_path_loc, "ncRNA")

    return Mx,My,NCx,NCy

def next_power_of_2(x:int)->int:  
    return 1 if x == 0 else 2**(x - 1).bit_length()

def evaluate_diff_sequences(Mx,My,NCx,NCy,
                            min_size:bool = True,
                            max_size:bool = False,
                            mean_size:bool = False):
    X = [*Mx,*NCx]

    dfts:List[List[float]] = []
    size_ls:List[int]= []
    seq_size:int = 0

    if min_size:
        min_value = len(min(X,key=len))
        print(f'Sequence Min length value: {min_value}')
        size_ls = [i for i in range(min_value)]
        seq_size=min_value

    elif max_size:
        max_value = len(max(X,key=len))
        # max_value = next_power_of_2(max_value)
        print(f'Sequence Max length value: {max_value}')
        size_ls = [i for i in range(max_value)]
        seq_size=max_value
        
    
    elif mean_size:
        mean_value = np.mean([len(i) for i in X])
        std_value = np.std([len(i) for i in X])
        mean_len = int(mean_value + std_value)
        print(f'Sequence Mean length value: {mean_len}')
        size_ls = [i for i in range(mean_len)]
        seq_size=mean_len


    for eiip_seq in X:
        if(max_size or mean_size):
            t = seq_size - len(eiip_seq)
            t = 0 if t<0 else t
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
            dfts.append([fft for fft, freqs in zip(eiip_seq.tolist(),size_ls)])
        else:
            dfts.append([fft for fft, freqs in zip(eiip_seq,size_ls)])


    df = pd.DataFrame({'sequences': dfts, 
                        'labels': [*My,*NCy]})

    X = df['sequences'].to_list()
    Y = df['labels'].to_list()

    X  = np.nan_to_num(np.array(X, dtype=np.float32))

    return X,Y,seq_size,size_ls

    # filename = "rrna_decisiontree.pickle"

    # # load model
    # # loaded_model = pickle.load(open(filename, "rb"))

    # print("Carregando modelo...")
    # classification_model = model.model(X=X, Y=Y)

    # # save model
    # pickle.dump(classification_model, open(filename, "wb"))


def get_cross_spectrum(Mx:List[List[float]],
                       NCx:List[List[float]],
                       is_min:bool,
                       seq_size_ls:List[int],
                       seq_size:int):
    mrnas_zip = []
    ncrnas_zip = []
    
    for eiip_seq in Mx:
        if is_min:
            mrnas_zip.append([fft for fft, freqs in zip(eiip_seq,seq_size_ls)])
        else:
            t = seq_size - len(eiip_seq)
            t = 0 if t<0 else t
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
            mrnas_zip.append([fft for fft, freqs in zip(eiip_seq.tolist(),seq_size_ls)])
       

    for eiip_seq in NCx:
        if is_min:
            ncrnas_zip.append([fft for fft, freqs in zip(eiip_seq,seq_size_ls)])
        else:
            t = seq_size - len(eiip_seq)
            t = 0 if t<0 else t
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
            ncrnas_zip.append([fft for fft, freqs in zip(eiip_seq.tolist(),seq_size_ls)])


    nc_bins = tfu.collect_bins(sequences=ncrnas_zip,
                       seq_size=seq_size,
                       class_name="ncRNA")
    m_bins = tfu.collect_bins(sequences=mrnas_zip,
                       seq_size=seq_size,
                       class_name="mRNA")
    
    nc_bins = np.nan_to_num(np.array(nc_bins, dtype=np.float32))
    m_bins = np.nan_to_num(np.array(m_bins, dtype=np.float32))

    return nc_bins, m_bins

def confusion_matrix_scorer(clf, X, y):
    y_pred = clf.predict(X)
    cm = confusion_matrix(y, y_pred, labels=LABELS)
    return {'tn': cm[0, 0], 'fp': cm[0, 1],
            'fn': cm[1, 0], 'tp': cm[1, 1]}


def evaluate_bin_model(nc_bins, m_bins,
                       X, Y):
    indices = np.arange(len(Y))
    np.random.shuffle(indices)
    X, Y = np.array(X)[indices], np.array(Y)[indices]

    clf = tree.DecisionTreeClassifier()
    clf.fit([nc_bins, m_bins],LABELS)

    print([nc_bins, m_bins])

    scores = confusion_matrix_scorer(clf, X, Y)
    print(scores)


if __name__ == "__main__":
    Mx,My,NCx,NCy = prepare_dft_data()

    X,Y,seq_size,size_ls = evaluate_diff_sequences(
        Mx=Mx,
        My=My,
        NCx=NCx,
        NCy=NCy,
        max_size=False,
        min_size=False,
        mean_size=True
    )

    nc_bins, m_bins = get_cross_spectrum(Mx,NCx,False,size_ls,seq_size)

    evaluate_bin_model(nc_bins, m_bins,X, Y)

    

    

    
  


    
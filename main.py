import bisect
import math
import pickle
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from Bio.Seq import Seq
from scipy.fft import rfft, rfftfreq

import io_utils as iou
import model
import transformation_utils as tfu

hist_bins = 512
max_freq = 0.5 # aminoacids
# max_freq = 0.56 # nucleotideo
intervals = np.linspace(0, max_freq, hist_bins)

def prepare_dft_data(m_path_loc:str,
                     nc_path_loc:str, 
                     specie:str, 
                     seq_size:int = None):
    return tfu.prepare_data(m_path_loc,nc_path_loc,True,seq_size,specie)

    
def prepare_protein_data(m_path_loc:str,
                         nc_path_loc:str, 
                         specie:str,
                         seq_size:int = None):
    return tfu.prepare_data(m_path_loc,nc_path_loc,False,seq_size,specie)

def next_power_of_2(x:int)->int:  
    return 1 if x == 0 else 2**(x - 1).bit_length()


def get_histogram_bins(sequences:List[List[float]],
                       class_name:str=""):

    hist = [[] for _ in range(hist_bins)]
    histogram = []

    decision_freq_idx =[]

    for eiip_seq in sequences:
            freq_hist = rfftfreq((len(eiip_seq)*2)-1, d=1)
            fft_freq = [(fft,freqs) for fft, freqs in zip(eiip_seq,freq_hist)]
            
            for val in fft_freq:
                hist[bisect.bisect(intervals, val[1])-1].append(abs(val[0]))
    ### histogram
    for lst in hist:
        histogram.append(np.prod(lst))

    histogram = iou.min_max_norm(histogram)

    for i in range(hist_bins):
        if(histogram[i] > 0.1):
            decision_freq_idx.append(i)

    
    # print(f'\n{class_name}Descision Freqs idxs:')
    # print(decision_freq_idx)

    plt.plot(intervals,histogram)
    plt.title(f'Histograma {class_name}\nNumero de Bins (0-{max_freq}): {intervals.size}')
    plt.show()

    return histogram,decision_freq_idx

def to_histogram_bins(fft_seq:List[float]):
    
    hist = [[] for _ in range(hist_bins)]
    histogram = []

    freq_hist = rfftfreq((len(fft_seq)*2)-1, d=1)
    fft_freq = [(fft,freqs) for fft, freqs in zip(fft_seq,freq_hist)]
            
    for val in fft_freq:
        hist[bisect.bisect(intervals, val[1])-1].append(abs(val[0]))

    ### histogram
    for lst in hist:
        histogram.append(np.prod(lst))

    histogram = iou.min_max_norm(histogram)
    # plt.plot(intervals,histogram)
    # plt.title(f'Histograma individual')
    # plt.show()

    return histogram


def normalize_sequences_to_bins(sequences:List[List[float]])->List[List[float]]:
    histograms:List[List[float]] = []
    for fft in sequences:
        histograms.append(to_histogram_bins(fft))

    return histograms

    
def single_specie_valuate(file:dict, conclusion:dict):
    specie:str = file["specie"]

    conclusion["sequence_type"].append(specie)
    conclusion["seq_size"].append(hist_bins)

    print(f'\n{specie} - Getting FFT data...')
    Mx,My,NCx,NCy = prepare_dft_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=specie)
        
    m_hist_bins,m_idxs = get_histogram_bins(sequences=Mx, 
                           class_name="mRNA_"+specie)
      
    nc_hist_bins,nc_idxs = get_histogram_bins(sequences=NCx, 
                        class_name="ncRNA_"+specie)
    
    conclusion["m_freq_peak_idxs"].append(m_idxs)
    conclusion["nc_freq_peak_idxs"].append(nc_idxs)
    
    most_id_idxs = list(set(m_idxs + nc_idxs))
    most_id_idxs.sort()

    norm_Mx = normalize_sequences_to_bins(Mx)
    norm_NCx = normalize_sequences_to_bins(NCx)

    '''
    Training with normalize spectre 0-1 for n-bins size,
    for classify by dft.

    '''
    print(f'\n{specie} - Valuating DFT model...')
    X_fft=[*norm_Mx,*norm_NCx]
    y_fft=[*My,*NCy]
    clf, dft_model_score = model.cross_val_model(X=X_fft,Y=y_fft)
    conclusion["dft_model_scores"].append(dft_model_score)
    model.save_model(clf,specie+"_dft_model_tree.png")

    
    '''
    Training looking only for the most valuables frequences,
    extracting the ffts sequences values by index.

    '''
    print(f'\n{specie} - Valuating most valuables indexes...')
    X_most_valuable=[np.array(fft)[most_id_idxs] for fft in X_fft]

    clf, cosic_model_score = model.cross_val_model(X=X_most_valuable,Y=y_fft)
    conclusion["cossic_model_score"].append(cosic_model_score)
    model.save_model(clf,specie+"_most_valuable_tree.png")
    model.confusion_matrix_scorer(clf, X_most_valuable, y_fft)
    '''
    Training only with the protein eiip sequences

    '''
    p_Mx,p_My,p_NCx,p_NCy = prepare_protein_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=specie)

    seqs = [*p_Mx,*p_NCx]
    p_y = np.array([*p_My,*p_NCy])

    eiip_zip:List[List[float]] = []

    min_len = len(min(seqs,key=len))
    mean_value = np.mean([len(i) for i in seqs])
    std_value = np.std([len(i) for i in seqs])
    mean_len = int(mean_value + std_value)

    for eiip_seq in seqs:
        t = mean_len - len(eiip_seq)
        if(t>0):
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
        eiip_zip.append(eiip_seq[0:mean_len])

    eiip_zip = np.array(eiip_zip,dtype=np.float32)

    print(f'\n{specie} - Valuating EIIP model...')
    clf, protein_model_score = model.cross_val_model(X=eiip_zip, Y=p_y)
    conclusion["protein_model_score"].append(protein_model_score)
    # model.save_model(clf,label+"protein_model_tree.png")



if __name__ == "__main__":

    files =[
        {
            "specie":"Gorilla_gorilla",
            "m_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia2.txt",
            "nc_path_loc":"..\dataset-plek\Gorilla_gorilla\sequencia1.txt"
        },
        {
            "specie":"Macaca_mulatta",
            "m_path_loc":"..\dataset-plek\Macaca_mulatta\sequencia1.txt",
            "nc_path_loc":"..\dataset-plek\Macaca_mulatta\sequencia2.txt"
        }
    ]

    conclusions = {
            "sequence_type":[],
            "seq_size":[],
            "m_freq_peak_idxs":[],
            "nc_freq_peak_idxs":[],
            "dft_model_scores":[],
            "protein_model_score":[],
            "cossic_model_score":[]
        }
    
    for file in files:
        single_specie_valuate(file, conclusion=conclusions)
    
  

    conclusions_df = pd.DataFrame.from_dict(conclusions)
    conclusions_df.to_csv('conclusions_most_valuable_indexes512.csv', index=True) 
    



    

    

    
  


    
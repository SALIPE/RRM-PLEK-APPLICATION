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

hist_bins = 256
max_freq = 0.5 # aminoacids
# max_freq = 0.56 # nucleotideo
intervals = np.linspace(0, max_freq, hist_bins)

options=[
    {
        "label":"mean_length",
        "max_size":False,
        "min_size":False,
        "mean_size":True
    },
    {
        "label":"min_length",
        "max_size":False,
        "min_size":True,
        "mean_size":False
    },
    # {
    #     "label":"Sequence max length",
    #     "max_size":True,
    #     "min_size":False,
    #     "mean_size":False
    # }
]


def prepare_dft_data(m_path_loc:str,nc_path_loc:str, specie:str, have_limit:bool=True):
    return tfu.prepare_data(m_path_loc,nc_path_loc,True,specie,have_limit)

    
def prepare_protein_data(m_path_loc:str,nc_path_loc:str, specie:str,have_limit:bool=True):
    return tfu.prepare_data(m_path_loc,nc_path_loc,False,specie,have_limit)

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

    # plt.plot(intervals,histogram)
    # plt.title(f'Histograma {class_name}\nNumero de Bins (0-{max_freq}): {intervals.size}')
    # plt.show()

    return histogram,decision_freq_idx

def to_histogram_bins(eiip_seq:List[float]):
    
    hist = [[] for _ in range(hist_bins)]
    histogram = []

    freq_hist = rfftfreq((len(eiip_seq)*2)-1, d=1)
    fft_freq = [(fft,freqs) for fft, freqs in zip(eiip_seq,freq_hist)]
            
    for val in fft_freq:
        hist[bisect.bisect(intervals, val[1])-1].append(abs(val[0]))
    ### histogram
    for lst in hist:
        hist_value = lst[0] if(len(lst)>0) else 0.0
        histogram.append(hist_value)

    histogram = iou.min_max_norm(histogram)
    # plt.plot(intervals,histogram)
    # plt.title(f'Histograma individual')
    # plt.show()

    return histogram


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
    
    return nc_bins, m_bins


def normalize_sequences_to_bins(sequences:List[List[float]]):
    histograms = []
    for fft in sequences:
        histograms.append(to_histogram_bins(fft))

    return histograms

def extract_determinisct_proteins(fft_sequences,
                                  p_sequences,
                                  selected_freq_indexes):
    
    filtered_sequences = []
    ifft_sequences =  tfu.return_to_eiip(fft_sequences,intervals,selected_freq_indexes)
    # print("\n")
    # print(ifft_sequences[0])
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
    
def single_specie_valuate(file:dict, conclusion:dict):
    specie:str = file["specie"]

    Mx,My,NCx,NCy = prepare_dft_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=specie,
        have_limit=True
    )
        
    p_Mx,p_My,p_NCx,p_NCy = prepare_protein_data(
        m_path_loc=file["m_path_loc"],
        nc_path_loc=file["nc_path_loc"],
        specie=specie,
        have_limit=True
    )

    m_hist_bins,m_idxs = get_histogram_bins(sequences=Mx, 
                           class_name="mRNA_"+specie)
        
    nc_hist_bins,nc_idxs = get_histogram_bins(sequences=NCx, 
                        class_name="ncRNA_"+specie)
    
    X_bins=[m_hist_bins,nc_hist_bins]
    y_bins=["mRNA_"+specie,"ncRNA_"+specie]
    
    hist_Mx = normalize_sequences_to_bins(Mx)
    hist_NCx = normalize_sequences_to_bins(NCx)

    X_bins_test=[*hist_Mx,*hist_NCx]
    y_bins_test=[*My,*NCy]

    print(f'\nValuating {specie} original size cross spectrum model...\n')
    clf, cossic_scores =  model.evaluate_bin_model(X_bins, y_bins, X_bins_test, y_bins_test)
    # model.save_model(clf,specie+"_cossic_histogram_tree.png")

    conclusion["sequence_type"].append("Cross-Spectrum Validation 2-classes-"+specie)
    conclusion["seq_size"].append(None)
    conclusion["m_freq_peak_idxs"].append(m_idxs)
    conclusion["nc_freq_peak_idxs"].append(nc_idxs)
    conclusion["dft_model_scores"].append(None)
    conclusion["spectrum_model_scores"].append(cossic_scores)
    conclusion["protein_model_score"].append(None)
    conclusion["cossic_model_score"].append(None)

    for option in options:

        label:str = specie+option["label"]

        print(f'\nRunning {label} two classes option:\n')

        X,Y,seq_size,size_ls = evaluate_diff_sequences(
            Mx=Mx,
            My=My,
            NCx=NCx,
            NCy=NCy,
            max_size=option["max_size"],
            min_size=option["min_size"],
            mean_size=option["mean_size"]
        )

        #DFT SEQUENCE VALUATION
        print(f'\nValuating {label} DFT model...\n')
        clf, dft_model_score = model.cross_val_model(X=X,Y=Y)
        # model.save_model(clf,label+"dft_model_tree.png")

        #GET BINS FITTED FOR SE SEQUENCE LENGTH
        nc_bins, m_bins = get_cross_spectrum(Mx,NCx,option["min_size"],size_ls,seq_size)

        nc_idx = []
        m_idx = []

        '''
        Training only with the same size sequences cross apectrum spectre,
        for classify by dft spectre.

        '''
        X_bins=[m_bins,nc_bins]

        print(f'\nValuating {label} cross spectrum model...\n')
        clf, spectrum_scores =  model.evaluate_bin_model(X_bins, y_bins, X, Y)
        # model.save_model(clf,label+"cossic_cross_spectrum_tree.png")


        for i in range(seq_size):
            if(nc_bins[i] > 0):
                nc_idx.append(i)
            if(m_bins[i]> 0):
                m_idx.append(i)

    
        most_id_idxs = list(set(m_idx + nc_idx))

        # PROTEIN EIIP VALUATION
        p_X = [*p_Mx,*p_NCx]
        p_y = np.array([*p_My,*p_NCy])

        eiip_zip:List[List[float]] = []

        for eiip_seq in p_X:
            t = seq_size - len(eiip_seq)
            t = 0 if t<0 else t
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
            eiip_zip.append([eiip for eiip, lbl in zip(eiip_seq.tolist(),size_ls)])

        eiip_zip = np.array(eiip_zip,dtype=np.float32)

        print(f'\nValuating {label} EIIP model...\n')
        clf, protein_model_score = model.cross_val_model(X=eiip_zip,
                                                    Y=p_y)
        # model.save_model(clf,label+"protein_model_tree.png")

        '''
        Extract the most valuable indexes from the eiip seqs to see if with 
        the peaks is possible to classify

        '''
        eiip_formatted = [np.array(seq)[most_id_idxs] for seq in p_X]
        eiip_formatted = np.array(eiip_formatted,dtype=np.float32)

        print(f'\nValuating {label} EIIP formatted model...\n')
        clf, cossic_model_score = model.cross_val_model(
            X=eiip_formatted,
            Y=p_y)        
        # model.save_model(clf,label+"eiip_formatted_tree.png")


        conclusion["sequence_type"].append("2-classes: "+label)
        conclusion["seq_size"].append(seq_size)
        conclusion["m_freq_peak_idxs"].append(m_idx)
        conclusion["nc_freq_peak_idxs"].append(nc_idx)
        conclusion["dft_model_scores"].append(dft_model_score)
        conclusion["spectrum_model_scores"].append(spectrum_scores)
        conclusion["protein_model_score"].append(protein_model_score)
        conclusion["cossic_model_score"].append(cossic_model_score)



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
            "spectrum_model_scores":[],
            "protein_model_score":[],
            "cossic_model_score":[]
        }
    
    for file in files:
        single_specie_valuate(file, conclusion=conclusions)
    
    Mx1,My1,NCx1,NCy1 = prepare_dft_data(
            m_path_loc=files[0]["m_path_loc"],
            nc_path_loc=files[0]["nc_path_loc"],
            specie=files[0]["specie"]
        )

    Mx2,My2,NCx2,NCy2 = prepare_dft_data(
        m_path_loc=files[1]["m_path_loc"],
        nc_path_loc=files[1]["nc_path_loc"],
        specie=files[1]["specie"]
    )

    p_Mx1,p_My1,p_NCx1,p_NCy1 = prepare_protein_data(
            m_path_loc=files[0]["m_path_loc"],
            nc_path_loc=files[0]["nc_path_loc"],
            specie=files[0]["specie"]
        )

    p_Mx2,p_My2,p_NCx2,p_NCy2 = prepare_protein_data(
        m_path_loc=files[1]["m_path_loc"],
        nc_path_loc=files[1]["nc_path_loc"],
        specie=files[1]["specie"]
    )

    m_hist_bins1,m1_idxs = get_histogram_bins(sequences=Mx1, 
                           class_name="mRNA_"+files[0]["specie"])
        
    nc_hist_bins1,nc1_idxs = get_histogram_bins(sequences=NCx1, 
                        class_name="ncRNA_"+files[0]["specie"])
    
    m_hist_bins2,m2_idxs = get_histogram_bins(sequences=Mx2, 
                        class_name="mRNA_"+files[1]["specie"])
    
    nc_hist_bins2,nc2_idxs = get_histogram_bins(sequences=NCx2, 
                        class_name="ncRNA_"+files[1]["specie"])
    
    selected_freq_indexes = list(set(m1_idxs + nc1_idxs + m2_idxs + nc2_idxs))

    # fp_Mx1 = extract_determinisct_proteins(Mx1,p_Mx1,selected_freq_indexes)
    # print(p_Mx1[0])
    # print(fp_Mx1[0])
    # fp_NCx1 = extract_determinisct_proteins(NCx1,p_NCx1,selected_freq_indexes)
    # fp_Mx2 = extract_determinisct_proteins(Mx2,p_Mx2,selected_freq_indexes)
    # fp_NCx2 = extract_determinisct_proteins(NCx2,p_NCx2,selected_freq_indexes)

    '''

    Validate histogram as input for train the classificator,
    the ideia here is to esse if with only the cross spectrums is possible to 
    classify the four classes.

    '''
   
    X_bins=[m_hist_bins1,nc_hist_bins1,m_hist_bins2,nc_hist_bins2]
    y_bins=["mRNA_"+files[0]["specie"],
            "ncRNA_"+files[0]["specie"],
            "mRNA_"+files[1]["specie"],
            "ncRNA_"+files[1]["specie"]]
    
    hist_Mx1 = normalize_sequences_to_bins(Mx1)
    hist_NCx1 = normalize_sequences_to_bins(NCx1)

    hist_Mx2 = normalize_sequences_to_bins(Mx2)
    hist_NCx2 = normalize_sequences_to_bins(NCx2)

    X_bins_test=[*hist_Mx1,*hist_NCx1,*hist_Mx2,*hist_NCx2]
    y_bins_test=[*My1,*NCy1,*My2,*NCy2]

    print(f'\nValuating original size cross spectrum model...\n')
    clf, cossic_scores =  model.evaluate_bin_model(X_bins, y_bins, X_bins_test, y_bins_test)
    model.save_model(clf,"cossic_histogram_tree.png")

    conclusions["sequence_type"].append("Cross-Spectrum Validation 4-classes")
    conclusions["seq_size"].append(None)
    conclusions["m_freq_peak_idxs"].append(list(set(m1_idxs + m2_idxs)))
    conclusions["nc_freq_peak_idxs"].append(list(set(nc1_idxs  + nc2_idxs)))
    conclusions["dft_model_scores"].append(None)
    conclusions["spectrum_model_scores"].append(cossic_scores)
    conclusions["protein_model_score"].append(None)
    conclusions["cossic_model_score"].append(None)


    '''
    Validate differents classifications, dft, eiip protein and cross_spectrum as train input,
    the ideia here is to esse which aproach get the better performance to
    classify the four classes.

    '''

    for option in options:

        label:str = option["label"]

        print(f'\nRunning {label} four classes option:\n')

        Mx =[*Mx1,*Mx2]
        My =[*My1,*My2]
        NCx=[*NCx1,*NCx2]
        NCy=[*NCy1,*NCy2]

        X,Y,seq_size,size_ls = evaluate_diff_sequences(
            Mx=Mx,
            My=My,
            NCx=NCx,
            NCy=NCy,
            max_size=option["max_size"],
            min_size=option["min_size"],
            mean_size=option["mean_size"]
        )

        #DFT SEQUENCE VALUATION
        print(f'\nValuating {label} DFT model...\n')
        clf, dft_model_score = model.cross_val_model(X=X,Y=Y)
        model.save_model(clf,label+"dft_model_tree.png")

        #GET BINS FITTED FOR SE SEQUENCE LENGTH
        nc_bins1, m_bins1 = get_cross_spectrum(Mx1,NCx1,option["min_size"],size_ls,seq_size)

        nc_bins2, m_bins2 = get_cross_spectrum(Mx2,NCx2,option["min_size"],size_ls,seq_size)

        nc_idx = []
        m_idx = []

        '''
        Training only with the same size sequences cross apectrum spectre,
        for classify by dft spectre.

        '''
        X_bins=[m_bins1,nc_bins1,m_bins2,nc_bins2]

        print(f'\nValuating {label} cross spectrum model...\n')
        clf, spectrum_scores =  model.evaluate_bin_model(X_bins, y_bins, X, Y)
        model.save_model(clf,label+"cossic_cross_spectrum_tree.png")

        # S/N CALCULATION

        # nc_spectrum_mean1 = np.mean(nc_bins1)
        # m_spectrum_mean1 = np.mean(m_bins1)

        # print(f'mRNA {files[0]["specie"]} cross-spectrum mean value: {m_spectrum_mean1}') 
        # print(f'ncRNA {files[0]["specie"]} cross-spectrum mean value: {nc_spectrum_mean1}')

        # nc_spectrum_mean2 = np.mean(nc_bins2)
        # m_spectrum_mean2 = np.mean(m_bins2)

        # print(f'mRNA {files[1]["specie"]} cross-spectrum mean value: {m_spectrum_mean2}') 
        # print(f'ncRNA {files[1]["specie"]} cross-spectrum mean value: {nc_spectrum_mean2}')

        # for i in range(seq_size):
        #     if(nc_bins1[i]/nc_spectrum_mean1 > 10) or (nc_bins2[i]/nc_spectrum_mean2 > 10):
        #         nc_idx.append(i)
        #     if(m_bins1[i]/m_spectrum_mean1> 10) or (m_bins2[i]/m_spectrum_mean2 > 10):
        #         m_idx.append(i)

        for i in range(seq_size):
            if(nc_bins1[i] > 0) or (nc_bins2[i]> 0):
                nc_idx.append(i)
            if(m_bins1[i]> 0) or (m_bins2[i] > 0):
                m_idx.append(i)

    
        most_id_idxs = list(set(m_idx + nc_idx))


        # PROTEIN EIIP VALUATION

        p_Mx =[*p_Mx1,*p_Mx2]
        p_My =[*p_My1,*p_My2]
        p_NCx=[*p_NCx1,*p_NCx2]
        p_NCy=[*p_NCy1,*p_NCy2]

        p_X = [*p_Mx,*p_NCx]
        p_y = np.array([*p_My,*p_NCy])

        eiip_zip:List[List[float]] = []

        for eiip_seq in p_X:
            t = seq_size - len(eiip_seq)
            t = 0 if t<0 else t
            eiip_seq = np.pad(eiip_seq, pad_width=(0, t), mode='constant')
            eiip_zip.append([eiip for eiip, lbl in zip(eiip_seq.tolist(),size_ls)])

        eiip_zip = np.array(eiip_zip,dtype=np.float32)

        print(f'\nValuating {label} EIIP model...\n')
        clf, protein_model_score = model.cross_val_model(X=eiip_zip,
                                                    Y=p_y)
        model.save_model(clf,label+"protein_model_tree.png")

        '''
        Extract the most valuable indexes from the eiip seqs to see if with 
        the peaks is possible to classify

        '''

        eiip_formatted = [np.array(seq)[most_id_idxs] for seq in p_X]
        eiip_formatted = np.array(eiip_formatted,dtype=np.float32)

        print(f'\nValuating {label} EIIP formatted model...\n')
        clf, cossic_model_score = model.cross_val_model(
            X=eiip_formatted,
            Y=p_y)        
        model.save_model(clf,label+"eiip_formatted_tree.png")


        conclusions["sequence_type"].append("4-classes: "+label)
        conclusions["seq_size"].append(seq_size)
        conclusions["m_freq_peak_idxs"].append(m_idx)
        conclusions["nc_freq_peak_idxs"].append(nc_idx)
        conclusions["dft_model_scores"].append(dft_model_score)
        conclusions["spectrum_model_scores"].append(spectrum_scores)
        conclusions["protein_model_score"].append(protein_model_score)
        conclusions["cossic_model_score"].append(cossic_model_score)

    conclusions_df = pd.DataFrame.from_dict(conclusions)
    conclusions_df.to_csv('conclusions_w_single_hist.csv', index=True) 
    



    

    

    
  


    
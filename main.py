import bisect
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from Bio import SeqIO
from scipy.fft import rfft, rfftfreq
from scipy.stats import gaussian_kde
from sklearn.model_selection import train_test_split

import clean_sequences as clseq
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
            # norm = iou.min_max_norm(eiip_seq)
            fft_freq = [(fft,freqs) for fft, freqs in zip(eiip_seq,freq_hist)]
            
            for val in fft_freq:
                hist[bisect.bisect(intervals, val[1])-1].append(abs(val[0]))
    ### histogram
    for lst in hist:
        histogram.append(tfu.internal_prod(lst))

    histogram = iou.min_max_norm(histogram)

    for i in range(hist_bins):
        if(histogram[i] > 0.1):
            decision_freq_idx.append(i)

    
    # print(f'\n{class_name}Descision Freqs idxs:')
    # print(decision_freq_idx)

    # plt.plot(intervals,histogram)
    # plt.title(f'Histograma {class_name}\nNumero de Bins (0-{max_freq}): {intervals.size}')
    # plt.xlabel("FREQUENCY")
    # # plt.xlim([0,max_freq])
    # plt.show()

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
    # # plt.xlim([0,max_freq])
    # plt.show()

    return histogram


def normalize_sequences_to_bins(sequences:List[List[float]])->List[List[float]]:
    histograms:List[List[float]] = []
    for fft in sequences:
        histograms.append(to_histogram_bins(fft))

    return histograms


def cross_dataset_valuate(file_specie1:dict,
                          file_specie2:dict,
                          conclusion:dict):

    specie:str = file["specie"]
    conclusion["sequence_type"].append(specie)

    mfile1:str = file_specie1["m_path_loc"]
    ncfile1:str = file_specie1["nc_path_loc"]

    mfile2:str = file_specie2["m_path_loc"]
    ncfile2:str = file_specie2["nc_path_loc"]

    print(f'\n{specie} - Getting FFT data...')
    Mx1,My1,NCx1,NCy1 = prepare_dft_data(
        m_path_loc = mfile1,
        nc_path_loc = ncfile1,
        specie=specie)
    
    Mx2,My2,NCx2,NCy2 = prepare_dft_data(
        m_path_loc = mfile2,
        nc_path_loc = ncfile2,
        specie=specie)
    
    m_hist_bins,m_idxs1 = get_histogram_bins(sequences=Mx1, 
                           class_name="mRNA_"+specie)
      
    nc_hist_bins,nc_idxs1 = get_histogram_bins(sequences=NCx1, 
                        class_name="ncRNA_"+specie)
    
    m_hist_bins,m_idxs2 = get_histogram_bins(sequences=Mx2, 
                           class_name="mRNA_"+specie)
      
    nc_hist_bins,nc_idxs2 = get_histogram_bins(sequences=NCx2, 
                        class_name="ncRNA_"+specie)
    
    m_idxs = list(set(m_idxs1 + m_idxs2))
    m_idxs.sort()

    nc_idxs = list(set(nc_idxs1 + nc_idxs2))
    nc_idxs.sort()
    
    conclusion["m_freq_peak_idxs"].append(m_idxs)
    conclusion["nc_freq_peak_idxs"].append(nc_idxs)
    
    most_id_idxs = list(set(m_idxs + nc_idxs))
    most_id_idxs.sort()
    conclusion["N"].append(len(most_id_idxs))
    conclusion["frequences"].append(np.array(intervals)[most_id_idxs])

    norm_Mx1 = normalize_sequences_to_bins(Mx1)
    norm_NCx1 = normalize_sequences_to_bins(NCx1)
    X_fft=[*norm_Mx1,*norm_NCx1]
    y_fft=[*My1,*NCy1]

    norm_Mx2 = normalize_sequences_to_bins(Mx2)
    norm_NCx2 = normalize_sequences_to_bins(NCx2)
    X_test = [*norm_Mx2,*norm_NCx2]
    y_test = [*My2,*NCy2]

    print(f'\n{specie} - Valuating DFT model...')
    clf = model.simple_train(X=X_fft,Y=y_fft)
    conclusion["dft_model_scores"].append(None)
    conclusion["acc_fft"].append(model.confusion_matrix_scorer(clf, X_test, y_test))

    '''
    Training looking only for the most valuables frequences,
    extracting the ffts sequences values by index.

    '''
    print(f'\n{specie} - Valuating most valuables indexes...')
    X_most_valuable_train=[np.array(fft)[most_id_idxs] for fft in X_fft]
    X_most_valuable_test=[np.array(fft)[most_id_idxs] for fft in X_test]

    clf = model.simple_train(X=X_most_valuable_train,Y=y_fft)
    conclusion["cossic_model_score"].append(None)
    conclusion["acc_mv"].append(model.confusion_matrix_scorer(clf, X_most_valuable_test, y_test))

    
def single_specie_valuate(file:dict, 
                          conclusion:dict):
    specie:str = file["specie"]
    mfile:str = file["m_path_loc"]
    ncfile:str = file["nc_path_loc"]

    conclusion["sequence_type"].append(specie)

    print(f'\n{specie} - Getting FFT data...')
    Mx,My,NCx,NCy = prepare_dft_data(
        m_path_loc = mfile,
        nc_path_loc = ncfile,
        specie=specie)
        
    m_hist_bins,m_idxs = get_histogram_bins(sequences=Mx, 
                           class_name="mRNA_"+specie)
      
    nc_hist_bins,nc_idxs = get_histogram_bins(sequences=NCx, 
                        class_name="ncRNA_"+specie)
    
    conclusion["m_freq_peak_idxs"].append(m_idxs)
    conclusion["nc_freq_peak_idxs"].append(nc_idxs)
    
    most_id_idxs = list(set(m_idxs + nc_idxs))
    most_id_idxs.sort()
    conclusion["N"].append(len(most_id_idxs))

    conclusion["frequences"].append(np.array(intervals)[most_id_idxs])

    norm_Mx = normalize_sequences_to_bins(Mx)
    norm_NCx = normalize_sequences_to_bins(NCx)

    X_fft=[*norm_Mx,*norm_NCx]
    y_fft=[*My,*NCy]

    '''
    Training with normalize spectre 0-1 for n-bins size,
    for classify by dft.

    '''
    print(f'\n{specie} - Valuating DFT model...')
    
    X_train, X_test, y_train, y_test = train_test_split(
     X_fft, y_fft, test_size=0.3, random_state=42)
    
    clf, dft_model_score = model.cross_val_model(X=X_train,Y=y_train)
    conclusion["dft_model_scores"].append(dft_model_score)
    conclusion["acc_fft"].append(model.confusion_matrix_scorer(clf, X_test, y_test))

    # model.save_model(clf,specie+"_dft_model_tree.png")

    
    '''
    Training looking only for the most valuables frequences,
    extracting the ffts sequences values by index.

    '''
    print(f'\n{specie} - Valuating most valuables indexes...')
    X_most_valuable=[np.array(fft)[most_id_idxs] for fft in X_fft]

    X_train, X_test, y_train, y_test = train_test_split(
     X_most_valuable, y_fft, test_size=0.3, random_state=42)

    clf, cosic_model_score = model.cross_val_model(X=X_train,Y=y_train)
    conclusion["cossic_model_score"].append(cosic_model_score)
    # model.save_model(clf,specie+"_most_valuable_tree.png")
    conclusion["acc_mv"].append(model.confusion_matrix_scorer(clf, X_test, y_test))

    # model.save_model(clf,label+"protein_model_tree.png")

    # _, ax = plt.subplots()
   
    # x1 = [value for seq in X_fft for value in seq]
    # x2 = [value for seq in X_most_valuable for value in seq]
    # x2 = [x2[i] if i< len(x2) else None for i in range(len(x1))]

    # d = {'original': x1, 'reduced': x1, 'hue':["hue" for i in range(len(x1))]}
    # df = pd.DataFrame(data=d)

    # sns.scatterplot(data=df, x="original", y="reduced", hue="hue")
    # plt.show()


def toymodel():

    # m_gorilla1='ATGTACCTCTTTTCTCTAGGCTCAGAGTCCCCCAAAGGGGCCATTGGCCACATTGTCCCTACTGAGAAGACCATTCTGGCTGTAGAGAGGAACAAAGTGCTGCTGCCTCCTCTCTGGAACAGGACCTTCAGCTGGGGCTTTGATGACTTCAGCTGCTGCTTGGGGAGCTACGGCTCCGACAAGGTCCTGATGACATTCGAGAACCTGGCTGCCTGGGGCCGCTGTCTGTGCGCCGTGTGCCCGTCCCCAACAATGATTGTCACCTCTGGGACCAGCACTGTGGTGTGTGTGTGGGAGCTCAGCATGACCAAAGGCCGCCCGAGGGGCTTGCGCCTCCAGCAG'
    # m_gorilla2='ATGGCGCACTCGGCTGCCGCCGTGCCGCTGGGCGCGCTGGAGCAGGGCTGCCCCATCCGCGTGGAGCACGACCGGAGGAGGGCTTACCTGTCTCCACAACCCCCAGGATGTCATGACCGGGCCGTCCTGCTCTATGAGTACGTGGGCAAGCGGATCGTGGACCTGCAGCACACCGAGGTCCCAGATGCCTACCGTGGGCGTGGCATCGCCAAGCACCTTGCCAAGGCCGCCCTGGACTTCGTGGTGGAGGAGGACCTGAAGGCCCATCTCACCTGCTGGTACATCCAGAAGTACGTCAAGGAGAACCCCCTGCCGCAGTACCTGGAGCGCCTGCAGCCGTAA'
    m_gorilla1='ATGAAATTATTGACAACCATATGTAGACTGAAGCTTGAAAAAATGTACTCAAAGACAAATACATCTTCCACAATATCTGAAAAGGCCTGACATGGGACAGAGAAAATCAGCACAGCCAGAAGTGAGGGGCACCATATCACCTTTAGTAGGTGGAAGGCATGTACAGCGATTGGAGGTCGATGTAAAAATCAATGTGATGATAGTGAATTTAGGATTTCATACTGTGCAAGACCTACAACTCGTTGCTGCGTGACAGAATGTGACCCTATGGACCCAAATAATTGGATCCCAAAGGACTCAGTAGGGACTCAAGAATGGTACCCTAAAGACTCACGTCATTGA'
    m_gorilla2='ACCAACGCCGTGGCGCACGTGGATGACATGCCCAACGCGCTGTCCGCCCTGAGCGACCTGCACGCGCACAAGCTTCGGGTGGACCCGGTCAACTTCAAGCTCCTAAGCCACTGCCTGCTGGTGACCCTGGCCGCCCACCTCCCCGCCGAGTTCACCCCTGCGGTGCACGCCTCCCTGGACAAGTTCCTGGCTTCTGTGAGCACCGTGCTGACCTCCAAATACCGTTAAGCTGGAGACTCGCTGGCCGTTCCTCCTGCCCGCTGGGCCTCCCAACGGGCCCTCTTCCCCTTCCTGCACCCGTACCCCCCTGGTCTTTGAATAAAGTCTGAGTGGGCGGCAGCC'
    
    # nc_gorilla1 = 'AAGACTATATTTTCAGGGATCATTTCTACAGTGCACTACTAGAGAAGTTTCTGTGAACTTGTAGAGCACCGGAAACCATGAGCAGGAAGTGCAGCGTTCTCTCCTGAGCATGAAGCCGGCTCTTGGTGTGGCTTCGCTGCAACTGCCATTGGCCATTGATGATCGTTCTTCTCTTCTCTGGGACAGTAAGAGAGAGAGGACACAGTCTGAGTGG'
    # nc_gorilla2 = 'AAGATTATACTTTCAGTGATCTTTTTTTAGTTTGTTACTAGAAAAGTGTCTCTGAACCTGGAGAGCACCAGAAACCATGAGGAGGAGATGTAGCGCTCTCTCCTGAGCTTAAAGCTGGCTCTTGCTTTTGCTTTGCTGCAACTGCTTTTTGCCATTGATGATCATTCTTCTCTTCCTCCTGGGAAGTAAGAGAGAGAAGATGCAGCACGAATGG'

   
    print(len(m_gorilla1))

    eiip1 = iou.aminoacid_map(m_gorilla1)

    freq_hist = rfftfreq(len(eiip1), d=1)

    plt.plot(eiip1)
    plt.ylabel("EIIP")
    plt.xlabel("SEQUENCE")
    plt.show()

    fft1 = rfft(x=eiip1)
    fft1 = np.abs(fft1)[1:]
    fft1_norm = iou.min_max_norm(fft1)

    fft_freq = [freqs for fft, freqs in zip(fft1,freq_hist)]

    plt.plot(fft_freq,fft1_norm)
    plt.xlabel("FREQUENCY")
    plt.show()
    
  
    eiip2 = iou.aminoacid_map(m_gorilla2)

    

    plt.plot(eiip2)
    plt.ylabel("EIIP")
    plt.xlabel("SEQUENCE")
    plt.show()

    fft2 = rfft(x=eiip2)
    fft2 = np.abs(fft2)[1:]
    fft2_norm = iou.min_max_norm(fft2)

    fft_freq = [freqs for fft, freqs in zip(fft2,freq_hist)]


    plt.plot(fft_freq,fft2_norm)
    plt.xlabel("FREQUENCY")
    plt.show()

   
    cs = tfu.element_wise_product([fft1,fft2])
    cs_norm = iou.min_max_norm(cs)

    plt.plot(fft_freq,cs_norm)
    plt.xlabel("FREQUENCY")
    plt.show()
    
    # sequences = iou.buffer_sequences(sequence_path="..\dataset-plek\Gorilla_gorilla\sequencia2.txt")


    # for key in sequences:
    #     seq = sequences[key]
      
    #     if(len(seq.seq) == len(m_gorilla1)):
    #         print(seq.seq)



def toymodel_diff_lengths(diff_legths:bool):
    seqs = []

    if(diff_legths):
        m_gorilla_path = os.path.join('..', 'datasets', 'dataset-plek','Gorilla_gorilla','mRNA_cleaned')
        seqs = iou.find_sequences(m_gorilla_path)
    else:
        m_gorilla1='ATGTACCTCTTTTCTCTAGGCTCAGAGTCCCCCAAAGGGGCCATTGGCCACATTGTCCCTACTGAGAAGACCATTCTGGCTGTAGAGAGGAACAAAGTGCTGCTGCCTCCTCTCTGGAACAGGACCTTCAGCTGGGGCTTTGATGACTTCAGCTGCTGCTTGGGGAGCTACGGCTCCGACAAGGTCCTGATGACATTCGAGAACCTGGCTGCCTGGGGCCGCTGTCTGTGCGCCGTGTGCCCGTCCCCAACAATGATTGTCACCTCTGGGACCAGCACTGTGGTGTGTGTGTGGGAGCTCAGCATGACCAAAGGCCGCCCGAGGGGCTTGCGCCTCCAGCAG'
        m_gorilla2='ATGGCGCACTCGGCTGCCGCCGTGCCGCTGGGCGCGCTGGAGCAGGGCTGCCCCATCCGCGTGGAGCACGACCGGAGGAGGGCTTACCTGTCTCCACAACCCCCAGGATGTCATGACCGGGCCGTCCTGCTCTATGAGTACGTGGGCAAGCGGATCGTGGACCTGCAGCACACCGAGGTCCCAGATGCCTACCGTGGGCGTGGCATCGCCAAGCACCTTGCCAAGGCCGCCCTGGACTTCGTGGTGGAGGAGGACCTGAAGGCCCATCTCACCTGCTGGTACATCCAGAAGTACGTCAAGGAGAACCCCCTGCCGCAGTACCTGGAGCGCCTGCAGCCGTAA'
        m_gorilla3='ATGAAATTATTGACAACCATATGTAGACTGAAGCTTGAAAAAATGTACTCAAAGACAAATACATCTTCCACAATATCTGAAAAGGCCTGACATGGGACAGAGAAAATCAGCACAGCCAGAAGTGAGGGGCACCATATCACCTTTAGTAGGTGGAAGGCATGTACAGCGATTGGAGGTCGATGTAAAAATCAATGTGATGATAGTGAATTTAGGATTTCATACTGTGCAAGACCTACAACTCGTTGCTGCGTGACAGAATGTGACCCTATGGACCCAAATAATTGGATCCCAAAGGACTCAGTAGGGACTCAAGAATGGTACCCTAAAGACTCACGTCATTGA'
        m_gorilla4='ACCAACGCCGTGGCGCACGTGGATGACATGCCCAACGCGCTGTCCGCCCTGAGCGACCTGCACGCGCACAAGCTTCGGGTGGACCCGGTCAACTTCAAGCTCCTAAGCCACTGCCTGCTGGTGACCCTGGCCGCCCACCTCCCCGCCGAGTTCACCCCTGCGGTGCACGCCTCCCTGGACAAGTTCCTGGCTTCTGTGAGCACCGTGCTGACCTCCAAATACCGTTAAGCTGGAGACTCGCTGGCCGTTCCTCCTGCCCGCTGGGCCTCCCAACGGGCCCTCTTCCCCTTCCTGCACCCGTACCCCCCTGGTCTTTGAATAAAGTCTGAGTGGGCGGCAGCC'
        
        
        seqs = [m_gorilla1,m_gorilla2,m_gorilla3,m_gorilla4]

    print(len(seqs))
    group_size = len(seqs)
    fft_list = []
    fft_freq = []

    for i in range(group_size):

        eiip1 = iou.aminoacid_map(seqs[i])
        fft1 = rfft(x=eiip1)
        fft1 = np.abs(fft1)[1:]
        fft_list.append(fft1)

        if not diff_legths or (i in [0,1,group_size-1]):
            freq_hist = rfftfreq(len(eiip1), d=1)

            plt.plot(eiip1)
            plt.title("Sequence:"+str(i))
            plt.ylabel("EIIP")
            plt.xlabel("SEQUENCE")
            plt.show()


            if diff_legths:
                hist = to_histogram_bins(fft1)
                plt.plot(intervals,hist)
                plt.title("Histogram:"+str(i))
            else:
                fft1_norm = iou.min_max_norm(fft1)
                fft_freq = [freqs for fft, freqs in zip(fft1,freq_hist)]
                plt.plot(fft_freq,fft1_norm)
                plt.title("FFT:"+str(i))
            plt.xlabel("FREQUENCY")
            plt.show()
    

 
    if diff_legths:
        hist_list,m_idxs = get_histogram_bins(sequences=fft_list)
        cs_norm = iou.min_max_norm(hist_list)
        plt.plot(intervals,cs_norm)
    else:
        cs = tfu.element_wise_product(fft_list)
        cs_norm = iou.min_max_norm(cs)
        plt.plot(fft_freq,cs_norm)

    plt.xlabel("FREQUENCY")
    plt.show()
    

if __name__ == "__main__":
    
    toymodel_diff_lengths(True)
    # cpc2_root = os.path.join('..', 'datasets', 'CPC2')
    # plek_root = os.path.join('..', 'datasets', 'dataset-plek')

    # plek_files =[
    #     {
    #         "specie":"Gorilla_gorilla",
    #         "fold":'Gorilla_gorilla',
    #         "m_file":'sequencia2.txt',
    #         "nc_file":'sequencia1.txt',
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Macaca_mulatta",
    #         "fold":'Macaca_mulatta',
    #         "m_file":'sequencia1.txt',
    #         "nc_file":'sequencia2.txt',
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Bos_taurus",
    #         "fold":'Bos_taurus',
    #         "m_file":"sequencia2.txt",
    #         "nc_file":"sequencia1.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Danio_rerio",
    #         "fold":'Danio_rerio',
    #         "m_file":"sequencia2.txt",
    #         "nc_file":"sequencia1.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Mus_musculus",
    #         "fold":'Mus_musculus',
    #         "m_file":"sequencia2.txt",
    #         "nc_file":"sequencia1.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Pan_troglodytes",
    #         "fold":'Pan_troglodytes',
    #         "m_file":"sequencia1.txt",
    #         "nc_file":"sequencia2.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Pongo_abelii",
    #         "fold":'Pongo_abelii',
    #         "m_file":"sequencia1.txt",
    #         "nc_file":"sequencia2.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #     {
    #         "specie":"Sus_scrofa",
    #         "fold":'Sus_scrofa',
    #         "m_file":"sequencia1.txt",
    #         "nc_file":"sequencia2.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     },
    #       {
    #         "specie":"Xenopus_tropicalis",
    #         "fold":'Xenopus_tropicalis',
    #         "m_file":"sequencia2.txt",
    #         "nc_file":"sequencia1.txt",
    #         "m_path_loc":'mRNA_cleaned',
    #         "nc_path_loc":'ncRNA_cleaned'
    #     }
    # ]

    # cpc2_files = [
    #     {
    #         "specie":"Arabidopsis_thaliana",
    #         "fold":'arabidopsis',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     },
    #      {
    #         "specie":"Drosophila_melanogaster",
    #         "fold":'fruitfly',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     },
    #      {
    #         "specie":"Homo_sapiens",
    #         "fold":'human',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     },
    #      {
    #         "specie":"Mus_musculus",
    #         "fold":'mouse',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     },
    #     #  {
    #     #     "specie":"Rattus_norvegicus",
    #     #     "fold":'rat',
    #     #     "m_file":"mRNA.fasta",
    #     #     "nc_file":"lncRNA.fasta",
    #     #     "m_path_loc":'mRNA_clean',
    #     #     "nc_path_loc":'ncRNA_clean'
    #     # },
    #     #  {
    #     #     "specie":"Oryza_sativa_japonica",
    #     #     "fold":'rice',
    #     #     "m_file":"mRNA.fasta",
    #     #     "nc_file":"lncRNA.fasta",
    #     #     "m_path_loc":'mRNA_clean',
    #     #     "nc_path_loc":'ncRNA_clean'
    #     # },
    #      {
    #         "specie":"Caenorhabditis_elegans",
    #         "fold":'worm',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     },
    #      {
    #         "specie":"Danio_rerio",
    #         "fold":'zebrafish',
    #         "m_file":"mRNA.fasta",
    #         "nc_file":"lncRNA.fasta",
    #         "m_path_loc":'mRNA_clean',
    #         "nc_path_loc":'ncRNA_clean'
    #     }
    # ]
   
    # cpc2_train_inner_files=[
    #       {
    #         "specie":"Danio_rerio",
    #         "plek_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','m_filtered_plek.fasta'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','nc_filtered_plek.fasta'),
    #         },
    #         "cpc2_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','mRNA_cleaned_cpc2'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','ncRNA_cleaned_cpc2'),
    #         }
            
    #     },
    #     {
    #         "specie":"Mus_musculus",
    #          "plek_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','m_filtered_plek.fasta'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','nc_filtered_plek.fasta'),
    #         },
    #         "cpc2_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','mRNA_cleaned_cpc2'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','ncRNA_cleaned_cpc2'),
    #         }
    #     }
    # ]

    # plek_train_inner_files=[
    #       {
    #         "specie":"Danio_rerio",
    #         "plek_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','mRNA_cleaned_plek'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','ncRNA_cleaned_plek'),
    #         },
    #         "cpc2_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','m_filtered_cpc2.fasta'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Danio_rerio','nc_filtered_cpc2.fasta'),
    #         }
            
    #     },
    #     {
    #         "specie":"Mus_musculus",
    #          "plek_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','mRNA_cleaned_plek'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','ncRNA_cleaned_plek'),
    #         },
    #         "cpc2_files":{
    #             "m_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','m_filtered_cpc2.fasta'),
    #             "nc_path_loc":os.path.join('..', 'datasets', 'inner_organisms','Mus_musculus','nc_filtered_cpc2.fasta'),
    #         }
    #     }
    # ]
    # conclusions = {
    #         "sequence_type":[],
    #         "acc_fft":[],
    #         "acc_mv":[],
    #         "N":[],
    #         "m_freq_peak_idxs":[],
    #         "nc_freq_peak_idxs":[],
    #         "frequences":[],
    #         "dft_model_scores":[],
    #         "cossic_model_score":[]
    #     }
    
    # # for file in plek_train_inner_files:
    # #     cross_dataset_valuate(file_specie1=file["plek_files"],
    # #                       file_specie2=file["cpc2_files"] ,
    # #                       conclusion= conclusions)
    # #     conclusions_df = pd.DataFrame.from_dict(conclusions)
    # #     conclusions_df.to_csv('crossdataset_conclusions_most_valuable_plektrain.csv', index=True) 

    # # conclusions = {
    # #         "sequence_type":[],
    # #         "acc_fft":[],
    # #         "acc_mv":[],
    # #         "N":[],
    # #         "m_freq_peak_idxs":[],
    # #         "nc_freq_peak_idxs":[],
    # #         "frequences":[],
    # #         "dft_model_scores":[],
    # #         "cossic_model_score":[]
    # #     }

    # # for file in cpc2_train_inner_files:
    # #     cross_dataset_valuate(file_specie1=file["cpc2_files"],
    # #                       file_specie2=file["plek_files"] ,
    # #                       conclusion= conclusions)
    #     # conclusions_df = pd.DataFrame.from_dict(conclusions)
    #     # conclusions_df.to_csv('crossdataset_conclusions_most_valuable_cpc2train.csv', index=True)
    
    # for file in plek_files:
    #     fold = file["fold"]
    # #     clseq.clean_sequences(class_name="ncRNA",
    # #                       root = os.path.join(cpc2_root, fold),
    # #                       filename= file["nc_file"])
    # #     clseq.clean_sequences(class_name="mRNA",
    # #                       root = os.path.join(cpc2_root, fold),
    # #                       filename= file["m_file"])
   
    #     file["m_path_loc"] = os.path.join(plek_root, fold, file["m_path_loc"])
    #     file["nc_path_loc"] = os.path.join(plek_root, fold, file["nc_path_loc"])
    #     single_specie_valuate(file, conclusion=conclusions)
    #     # conclusions_df = pd.DataFrame.from_dict(conclusions)
    #     # conclusions_df.to_csv('./cpc2_results/cpc2_conclusions_most_valuable2.csv', index=True) 

    #     # print(f'\n {file["specie"]}')
    #     # print("mRNA values:")
    #     # seqs_len = []
    #     # with open(plek_root+file["m_path_loc"]) as handle:
    #     #     for record in SeqIO.parse(handle, "fasta"):
    #     #         seqs_len.append(len(record.seq))
        

    #     # seqs_mean = np.mean(seqs_len)
    #     # seqs_sd = np.std(seqs_len)
    #     # print(f'mean: {seqs_mean} +- dev: {seqs_sd}')

    #     # print("ncRNA values:")
    #     # seqs_len = []
    #     # with open(plek_root+ file["nc_path_loc"]) as handle:
    #     #     for record in SeqIO.parse(handle, "fasta"):
    #     #         seqs_len.append(len(record.seq))
                
    #     # seqs_mean = np.mean(seqs_len)
    #     # seqs_sd = np.std(seqs_len)
    #     # print(f'mean: {seqs_mean} +- dev: {seqs_sd}')

        
    
    
  

  
    



    

    

    
  


    
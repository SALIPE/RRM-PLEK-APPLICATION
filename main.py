import bisect
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.fft import rfft, rfftfreq
from sklearn.model_selection import train_test_split

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

    
def single_specie_valuate(root:str,
                          file:dict, 
                          conclusion:dict):
    specie:str = file["specie"]

    conclusion["sequence_type"].append(specie)

    print(f'\n{specie} - Getting FFT data...')
    Mx,My,NCx,NCy = prepare_dft_data(
        m_path_loc=root+file["m_path_loc"],
        nc_path_loc=root+file["nc_path_loc"],
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

if __name__ == "__main__":
    # toymodel()
    plek_root = '..\datasets\dataset-plek'
    cpc2_root = '..\datasets\CPC2'

    plek_files =[
        {
            "specie":"Gorilla_gorilla",
            "m_path_loc":"\Gorilla_gorilla\sequencia2.txt",
            "nc_path_loc":"\Gorilla_gorilla\sequencia1.txt"
        },
        {
            "specie":"Macaca_mulatta",
            "m_path_loc":"\Macaca_mulatta\sequencia1.txt",
            "nc_path_loc":"\Macaca_mulatta\sequencia2.txt"
        },
        {
            "specie":"Bos_taurus",
            "m_path_loc":"\Bos_taurus\sequencia2.txt",
            "nc_path_loc":"\Bos_taurus\sequencia1.txt"
        },
        {
            "specie":"Danio_rerio",
            "m_path_loc":"\Danio_rerio\sequencia2.txt",
            "nc_path_loc":"\Danio_rerio\sequencia1.txt"
        },
        {
            "specie":"Mus_musculus",
            "m_path_loc":"\Mus_musculus\sequencia2.txt",
            "nc_path_loc":"\Mus_musculus\sequencia1.txt"
        },
        {
            "specie":"Pan_troglodytes",
            "m_path_loc":"\Pan_troglodytes\sequencia1.txt",
            "nc_path_loc":"\Pan_troglodytes\sequencia2.txt"
        },
        {
            "specie":"Pongo_abelii",
            "m_path_loc":"\Pongo_abelii\sequencia1.txt",
            "nc_path_loc":"\Pongo_abelii\sequencia2.txt"
        },
        {
            "specie":"Sus_scrofa",
            "m_path_loc":"\Sus_scrofa\sequencia1.txt",
            "nc_path_loc":"\Sus_scrofa\sequencia2.txt"
        },
          {
            "specie":"Xenopus_tropicalis",
            "m_path_loc":"\Xenopus_tropicalis\sequencia2.txt",
            "nc_path_loc":"\Xenopus_tropicalis\sequencia1.txt"
        }
    ]

    cpc2_files = [
        {
            "specie":"Arabidopsis_thaliana",
            "m_path_loc":"\\arabidopsis\mRNA.fasta",
            "nc_path_loc":"\\arabidopsis\lncRNA.fasta"
        },
         {
            "specie":"Drosophila_melanogaster",
            "m_path_loc":"\\fruitfly\mRNA.fasta",
            "nc_path_loc":"\\fruitfly\lncRNA.fasta"
        },
         {
            "specie":"Homo_sapiens",
            "m_path_loc":"\human\mRNA.fasta",
            "nc_path_loc":"\human\lncRNA.fasta"
        },
         {
            "specie":"Mus_musculus",
            "m_path_loc":"\mouse\mRNA.fasta",
            "nc_path_loc":"\mouse\lncRNA.fasta"
        },
         {
            "specie":"Rattus_norvegicus",
            "m_path_loc":"\\rat\mRNA.fasta",
            "nc_path_loc":"\\rat\lncRNA.fasta"
        },
         {
            "specie":"Oryza_sativa_japonica",
            "m_path_loc":"\\rice\mRNA.fasta",
            "nc_path_loc":"\\rice\lncRNA.fasta"
        },
         {
            "specie":"Caenorhabditis_elegans",
            "m_path_loc":"\worm\mRNA.fasta",
            "nc_path_loc":"\worm\lncRNA.fasta"
        },
         {
            "specie":"Danio_rerio",
            "m_path_loc":"\zebrafish\mRNA.fasta",
            "nc_path_loc":"\zebrafish\lncRNA.fasta"
        }
    ]
    conclusions = {
            "sequence_type":[],
            "acc_fft":[],
            "acc_mv":[],
            "N":[],
            "m_freq_peak_idxs":[],
            "nc_freq_peak_idxs":[],
            "frequences":[],
            "dft_model_scores":[],
            "cossic_model_score":[]
        }
    
    for file in cpc2_files:
        single_specie_valuate(cpc2_root,file, conclusion=conclusions)
        conclusions_df = pd.DataFrame.from_dict(conclusions)
        conclusions_df.to_csv('./cpc2_results/cpc2_conclusions_most_valuable.csv', index=True) 
    #     # filenames=[
    #     #     file["m_path_loc"],
    #     #     file["nc_path_loc"]
    #     # ]
        
    #     # print(f'\n {file["specie"]}')
    #     # print("mRNA values:")
    #     # seqs_len = []
    #     # with open( file["m_path_loc"]) as handle:
    #     #     for record in SeqIO.parse(handle, "fasta"):
    #     #         seqs_len.append(len(record.seq))
                
    #     # seqs_mean = np.mean(seqs_len)
    #     # seqs_sd = np.std(seqs_len)
    #     # print(f'mean: {seqs_mean} +- dev: {seqs_sd}')

    #     # print("ncRNA values:")
    #     # seqs_len = []
    #     # with open( file["nc_path_loc"]) as handle:
    #     #     for record in SeqIO.parse(handle, "fasta"):
    #     #         seqs_len.append(len(record.seq))
                
    #     # seqs_mean = np.mean(seqs_len)
    #     # seqs_sd = np.std(seqs_len)
    #     # print(f'mean: {seqs_mean} +- dev: {seqs_sd}')

          
    #         # variation = seqs_sd/seqs_mean
    #         # seqs_min = min(seqs_len)
    #         # seqs_max = max(seqs_len)
    
    
  

  
    



    

    

    
  


    
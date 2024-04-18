from scipy.fft import rfft, rfftfreq
import numpy as np

from Bio import SeqIO, SeqRecord
from Bio.Seq import Seq

import matplotlib.pyplot as plt

import os
from fnmatch import fnmatch

import bisect

## Funcoes para obtencao do EIIP
EIIP_NUCLEOTIDE: dict[str, float] = {
    'A': 0.1260,
    'G': 0.0806,
    'T': 0.1335,
    'C': 0.1340}

EIIP_AMINOACID: dict[str,float] = {
    'L': 0.0000,
    'I': 0.0000,
    'N': 0.0036,
    'G': 0.0050,
    'V': 0.0057,
    'E': 0.0058,
    'P': 0.0198,
    'H': 0.0242,
    'K': 0.0371,
    'A': 0.0373,
    'Y': 0.0516,
    'W': 0.0548,
    'Q': 0.0761,
    'M': 0.0823,
    'S': 0.0829,
    'C': 0.0829,
    'T': 0.0941,
    'F': 0.0946,
    'R': 0.0959,
    'D': 0.1263}

# Traducao da sequencia de RNA/DNA para sequencia de aminoacidos
# exclui os stop codons
def translate(seq: str) -> str:
    sequence = Seq(seq)
    return str(sequence.translate()).replace('*', '')

def get_eiip(seq: SeqRecord.SeqRecord, step: int, translate_seq: bool = False):
    seq_translate = seq
    eiip_seq = []
    # if translate_seq:
    #     # seq_translate = translate(seq)
        
    #     for index in range(0,len(seq_translate),step):
    #         if seq_translate[index] == '*':
    #             eiip_seq.append(0)
    #         else:
    #             eiip_seq.append(EIIP_AMINOACID.get(seq_translate[index]))
    #         print(seq_translate[index], EIIP_AMINOACID.get(seq_translate[index]),'\n')
    #     return eiip_seq
    
    for index in range(0,len(seq_translate),step):
        if seq_translate[index] == '*':
            eiip_seq.append(0)
        else:
            eiip_seq.append(EIIP_NUCLEOTIDE.get(seq_translate[index]))
        print(seq_translate[index], EIIP_NUCLEOTIDE.get(seq_translate[index]),'\n')
    return eiip_seq

## produto direto
def element_wise_product(data):
    """
    Performs element-wise product on a list of lists,
    even with varying sublist lengths.
    """

    res = data[0]
    for b in data[1:]:
        res = [x1*x2 for x1, x2 in zip(res,b)]

    return res

# parametros
dir_path = r'/home/matheus/Dropbox/08_packages/07_BASiNET_v2/data/100_seq'
step = 1
hist_bins = 100
# max_freq = 0.5 # aminoacids
max_freq = 0.56 # nucleotideo


## load seqs
file_list = [name for name in os.listdir(dir_path) if fnmatch(name, "*.fasta")]
files = [dir_path + "/" + file_name for file_name in file_list]


for file_in in files:
    coeff_FFT = []
    coeff_FFT_zip = []
    coeff_FFT_mean_len = []
    intervals = np.linspace(0, max_freq, hist_bins)
    hist = [[] for _ in range(hist_bins)]
    histogram = []
    file_name = file_in.split('/')[-1].split('.')[0]
    ## traduzindo as sequencias
    # with open(file_in, encoding="utf-8") as handle:
    #     seqs = [translate(str(record.seq).upper()) for record in SeqIO.parse(handle, "fasta")]
    # translate_seq = True
    ##
    ### se nao for traduzir as sequencias
    with open(file_in, encoding="utf-8") as handle:
        seqs = [str(record.seq).upper() for record in SeqIO.parse(handle, "fasta")]
    translate_seq=False    
    
    
    max_val = len(max(seqs,key=len))
    min_val = len(min(seqs,key=len))
    mean_value = np.mean([len(i) for i in seqs])
    std_value = np.std([len(i) for i in seqs])
    mean_len = int(mean_value + std_value)
    ### padding valores ate o tamanho maximo das sequencias
    for seq in seqs:
        eiip_seqs = get_eiip(seq, step, translate_seq=translate_seq)
        fft_eiip = rfft(eiip_seqs, n=max_val)
        coeff_FFT.append(np.abs(fft_eiip))
        
        ### produto direto
        fft_eiip_zip = rfft(eiip_seqs)
        coeff_FFT_zip.append(np.abs(fft_eiip_zip)[1:])
        
        ### histograma
        freq_hist = rfftfreq(len(eiip_seqs), d=1)
        fft_freq = [(fft,freqs) for fft, freqs in zip(fft_eiip_zip,freq_hist)]
        
        #### mean_len
        fft_eiip_mean_len = rfft(eiip_seqs, n=mean_len)
        coeff_FFT_mean_len.append(np.abs(fft_eiip_mean_len))
        
        for val in fft_freq:
            hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))
        
        # Plot mesmo valor
        # plt.plot(rfftfreq(max_val, d=1)[1:],np.abs(fft_eiip)[1:])
        # plot_name = f'{file_name}\nMesmo tamanho (padding zero)'
        
        # Plot cada sequencia com seu tamanho original
        plt.plot(freq_hist[1:],np.abs(fft_eiip_zip)[1:])
        plot_name = f'{file_name}\nTamanho original'
        
    plt.title(plot_name)
    plt.show()
    ### produto direto
    cross_spectral_zip = element_wise_product(coeff_FFT_zip)
    
    ### padding valores ate o taamnho maximo das sequencias
    freq = rfftfreq(max_val, d=1)
    cross_spectral = np.prod(coeff_FFT, axis=0)
    
    ### mean len
    freq_mean_len = rfftfreq(mean_len, d=1)
    cross_spectral_mean_len = np.prod(coeff_FFT_mean_len, axis=0)
    
    ### histogram
    for lst in hist:
        histogram.append(np.prod(lst))
    
    plt.plot(freq[1:],cross_spectral[1:])
    plt.title(f'{file_name}\nMesmo tamanho\nTamanho da serie {freq.size}')
    plt.show()

    ### produto direto
    freq_zip = rfftfreq(min_val,d=1)
    plt.plot(freq_zip[1:],cross_spectral_zip)
    plt.title(f'{file_name}\nProduto direto entre coeficientes\nTamanho da serie {freq_zip.size}')
    plt.show()

    ### histogram
    plt.plot(intervals[1:],histogram[1:])
    plt.title(f'{file_name}\nHistograma\nNumero de Bins (0-{max_freq}): {intervals.size}')
    plt.show()
    
    ### mean len
    plt.plot(freq_mean_len[1:],cross_spectral_mean_len[1:])
    plt.title(f'{file_name}\nTamanho: média + desvio padrão\nTamanho da serie {freq.size}')
    plt.show()
 
    
#########
#### TOY MODEL
##########
step = 1
hist_bins = 100
max_freq = 0.5 # aminoacids

coeff_FFT = []
coeff_FFT_zip = []
coeff_FFT_mean_len = []
intervals = np.linspace(0, max_freq, hist_bins)
hist = [[] for _ in range(hist_bins)]
histogram = []
file_name = 'TOY MODEL - COSIC ARTIGO'

seq1 = 'PALPEDGGSGAFPPGHFKDPKRLYCKNGGFFLRIHPDGRVDGVREKSDPHIKLQLQAEERGWSIKGVCANRYLAMKEDGRLLASKCVTDECFFFERLESNNYNTYRSRKYSSWYVALKRTGQYKLGPKTGPGQKAILFLPMSAKS'
seq2 = 'FNLPLGNYKKPKLLYCSNGGYFLRILPDGTVDGTKDRSDQHIQLQLCAESIGEVYIKSTETGQFLAMDTDGLLYGSQTPNEECLFLERLEENHYNTYISKKHAEKHWFVGLKKNGRSKLGPRTHFGQKAILFLPLPVSSD'

seqs = [seq1, seq2]
max_val = len(max(seqs,key=len))
min_val = len(min(seqs,key=len))
mean_value = np.mean([len(i) for i in seqs])
std_value = np.std([len(i) for i in seqs])
mean_len = int(mean_value + std_value)

eiip_seq1 = get_eiip(seq1, step, True)
eiip_seq2 = get_eiip(seq2, step, True)

coeff_FFT.append(np.abs(rfft(eiip_seq1, n=max_val)))
coeff_FFT.append(np.abs(rfft(eiip_seq2, n=max_val)))

coeff_FFT_zip.append(np.abs(rfft(eiip_seq1))[1:])

freq_hist = rfftfreq(len(eiip_seq1), d=1)
fft_freq = [(fft,freqs) for fft, freqs in zip(rfft(eiip_seq1),freq_hist)]
for val in fft_freq:
    hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))

plt.plot(freq_hist[1:],np.abs(rfft(eiip_seq1))[1:])


coeff_FFT_zip.append(np.abs(rfft(eiip_seq2))[1:])

freq_hist = rfftfreq(len(eiip_seq2), d=1)
fft_freq = [(fft,freqs) for fft, freqs in zip(rfft(eiip_seq2),freq_hist)]
for val in fft_freq:
    hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))

    
plt.plot(freq_hist[1:],np.abs(rfft(eiip_seq2))[1:])
plot_name = f'ToyModel\nTamanho original'

plt.show()

coeff_FFT_mean_len.append(np.abs(rfft(eiip_seq1, n=mean_len)))
coeff_FFT_mean_len.append(np.abs(rfft(eiip_seq2, n=mean_len)))

cross_spectral_zip = element_wise_product(coeff_FFT_zip)

freq = rfftfreq(max_val, d=1)
cross_spectral = np.prod(coeff_FFT, axis=0)

freq_mean_len = rfftfreq(mean_len, d=1)
cross_spectral_mean_len = np.prod(coeff_FFT_mean_len, axis=0)

### histogram
for lst in hist:
    histogram.append(np.prod(lst))

plt.plot(eiip_seq1)
plt.title(f'{file_name}\nEIIP seq1')
plt.show()

plt.plot(eiip_seq2)
plt.title(f'{file_name}\nEIIP seq2')
plt.show()


plt.plot(freq[1:],cross_spectral[1:])
plt.title(f'{file_name}\nMesmo tamanho\nTamanho da serie {freq.size}')
plt.show()

### produto direto
freq_zip = rfftfreq(min_val,d=1)
plt.plot(freq_zip[1:],cross_spectral_zip)
plt.title(f'{file_name}\nProduto direto entre coeficientes\nTamanho da serie {freq_zip.size}')
plt.show()

### histogram
plt.plot(intervals[1:],histogram[1:])
plt.title(f'{file_name}\nHistograma\nNumero de Bins (0-0.5): {intervals.size}')
plt.show()

### mean len
plt.plot(freq_mean_len[1:],cross_spectral_mean_len[1:])
plt.title(f'{file_name}\nTamanho: média + desvio padrão\nTamanho da serie {freq.size}')
plt.show()


##### model 2

step = 1
hist_bins = 100
max_freq = 0.5 # aminoacids

coeff_FFT = []
coeff_FFT_zip = []
coeff_FFT_mean_len = []
intervals = np.linspace(0, max_freq, hist_bins)
hist = [[] for _ in range(hist_bins)]
histogram = []
file_name = 'TOY MODEL - COSIC LIVRO'

seq1 = 'VLSPADKTNVKAAWGKVGAHAGEYGAEALERMFLSFPTTKTYFPHFDLSHGSAQVKGHGKKVADALTNAVAHVDDMPNALSALSDLHAHKLRVDPVNFKLLSHCLLVTLAAHLPAEFTPAVHASLDKFLASVSTVLTSKYR'
seq2 = 'VHLTPEEKSAVTALWGKVNVDEVGGEALGRLLVVYPWTQRFFESFGDLSTPDAVMGNPKVKAHGKKVLGAFSDGLAHLDNLKGTFATLSELHCDKLHVDPENFRLLGNVLVCVLAHHFGKEFTPPVQAAYQKVVAGVANALAHKYH'

seqs = [seq1, seq2]
max_val = len(max(seqs,key=len))
min_val = len(min(seqs,key=len))
mean_value = np.mean([len(i) for i in seqs])
std_value = np.std([len(i) for i in seqs])
mean_len = int(mean_value + std_value)

eiip_seq1 = get_eiip(seq1, step, True)
eiip_seq2 = get_eiip(seq2, step, True)

coeff_FFT.append(np.abs(rfft(eiip_seq1, n=max_val)))
coeff_FFT.append(np.abs(rfft(eiip_seq2, n=max_val)))

coeff_FFT_zip.append(np.abs(rfft(eiip_seq1))[1:])

freq_hist = rfftfreq(len(eiip_seq1), d=1)
fft_freq = [(fft,freqs) for fft, freqs in zip(rfft(eiip_seq1),freq_hist)]
for val in fft_freq:
    hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))

plt.plot(freq_hist[1:],np.abs(rfft(eiip_seq1))[1:])

coeff_FFT_zip.append(np.abs(rfft(eiip_seq2))[1:])

freq_hist = rfftfreq(len(eiip_seq2), d=1)
fft_freq = [(fft,freqs) for fft, freqs in zip(rfft(eiip_seq2),freq_hist)]
for val in fft_freq:
    hist[bisect.bisect_right(intervals, val[1])-1].append(abs(val[0]))
    
plt.plot(freq_hist[1:],np.abs(rfft(eiip_seq2))[1:])
plot_name = f'ToyModel\nTamanho original'
plt.show()

coeff_FFT_mean_len.append(np.abs(rfft(eiip_seq1, n=mean_len)))
coeff_FFT_mean_len.append(np.abs(rfft(eiip_seq2, n=mean_len)))

cross_spectral_zip = element_wise_product(coeff_FFT_zip)

freq = rfftfreq(max_val, d=1)
cross_spectral = np.prod(coeff_FFT, axis=0)

freq_mean_len = rfftfreq(mean_len, d=1)
cross_spectral_mean_len = np.prod(coeff_FFT_mean_len, axis=0)

### histogram
for lst in hist:
    histogram.append(np.prod(lst))

plt.plot(eiip_seq1)
plt.title(f'{file_name}\nEIIP seq1')
plt.show()

plt.plot(eiip_seq2)
plt.title(f'{file_name}\nEIIP seq2')
plt.show()


plt.plot(freq[1:],cross_spectral[1:])
plt.title(f'{file_name}\nMesmo tamanho\nTamanho da serie {freq.size}')
plt.show()

### produto direto
freq_zip = rfftfreq(min_val,d=1)
plt.plot(freq_zip[1:],cross_spectral_zip)
plt.title(f'{file_name}\nProduto direto entre coeficientes\nTamanho da serie {freq_zip.size}')
plt.show()

### histogram
plt.plot(intervals[1:],histogram[1:])
plt.title(f'{file_name}\nHistograma\nNumero de Bins (0-0.5): {intervals.size}')
plt.show()

### mean len
plt.plot(freq_mean_len[1:],cross_spectral_mean_len[1:])
plt.title(f'{file_name}\nTamanho: média + desvio padrão\nTamanho da serie {freq.size}')
plt.show()
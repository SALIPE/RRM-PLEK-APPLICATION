

import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy import absolute, array, delete, fft, ndarray


def to_dft(seq: npt.ArrayLike, freq_size: int)-> ndarray:
 

    discrete_transform = fft.fft(a=seq, n=freq_size)
    discrete_transform = delete(discrete_transform, 0)

    # freq = fft.rfftfreq(seq.shape[-1]-2)

    # plt.subplot(212)
    # plt.plot(freq,absolute(discrete_transform))
    # plt.show()
    return discrete_transform

def cross_spectrum(seq_x: npt.ArrayLike, 
                   seq_y:npt.ArrayLike, 
                   freq:npt.ArrayLike,
                   size:int):
    spectrum = []
    n = range(size)
    for i in n:
        spectrum.append(seq_x[i].real*seq_y[i].imag)

    spectrum = array(spectrum)
    spectrum = absolute(spectrum)

    plt.subplot(311)
    plt.plot(freq, fft.fftshift(absolute(seq_x))) 


    plt.subplot(312)
    plt.plot(freq,  fft.fftshift(absolute(seq_y)))

    plt.subplot(313)
    plt.plot(freq, fft.fftshift(spectrum))
    plt.show()

    return spectrum

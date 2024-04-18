

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import absolute, array
from scipy.fft import rfft


def to_dft(seq: npt.ArrayLike)-> tuple:
    data = rfft(x=seq,norm="ortho")
    return data*1000

def cross_spectrum(seq_x: npt.ArrayLike, 
                   seq_y:npt.ArrayLike,
                   size:int):
    spectrum = []
    n = range(size)
    for i in n:
        spectrum.append(seq_x[i].real*seq_y[i].imag)

    spectrum = array(spectrum)
    spectrum = absolute(spectrum)

    return spectrum

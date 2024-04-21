

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from numpy import absolute, array
from scipy.fft import rfft


def to_dft(seq: npt.ArrayLike, size:int)-> tuple:
    data = rfft(x=seq, n=size, norm="ortho")
    return data

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

## produto direto
def element_wise_product(dft_list):
    """
    Performs element-wise product on a list of lists,
    even with varying sublist lengths.
    """

    res = dft_list[0]
    for b in dft_list[1:]:
        res = [x1*x2 for x1, x2 in zip(res,b)]

    return res

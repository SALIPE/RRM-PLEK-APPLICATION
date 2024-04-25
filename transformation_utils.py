

from typing import List

import numpy as np
import numpy.typing as npt
from numpy import absolute, array
from scipy.fft import rfft


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
def element_wise_product(dft_list:List[List[float]]):
    """
    Performs element-wise product on a list of lists,
    even with varying sublist lengths.
    """

    res = dft_list[0]
    for b in dft_list[1:]:
        res = [np.multiply(x1,x2) for x1, x2 in zip(res,b)]

    return res

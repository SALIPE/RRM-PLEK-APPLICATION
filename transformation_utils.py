

import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy import fft, ndarray


def to_dft(seq: npt.ArrayLike)-> ndarray:
    n:int =seq.size

    plt.subplot(121)
    plt.plot( range(n),seq)

    discrete_transform = fft.rfft(seq)
    freq = fft.rfftfreq(seq.shape[-1])

    plt.subplot(122)
    plt.plot(freq, discrete_transform.real, freq, discrete_transform.imag)
    plt.show()
    return discrete_transform

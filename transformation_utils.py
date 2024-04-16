

import matplotlib.pyplot as plt
import numpy.typing as npt
from numpy import absolute, delete, fft, mean, ndarray


def to_dft(seq: npt.ArrayLike)-> ndarray:
    n:int =len(seq)

    plt.subplot(211)
    plt.plot(range(n),seq)

    discrete_transform = fft.rfft(seq)
    discrete_transform = delete(discrete_transform, 0)
    dtf_abs = absolute(discrete_transform)

    mean_value = mean(dtf_abs)
    sn_amp = [amp/mean_value for amp in dtf_abs]
    freq = fft.rfftfreq(seq.shape[-1]-2)

    plt.subplot(212)
    plt.plot(freq,sn_amp)
    plt.show()
    return discrete_transform

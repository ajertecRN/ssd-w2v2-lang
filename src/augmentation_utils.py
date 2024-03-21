import numpy as np
from scipy import signal


def random_range(low, high, to_integer: bool):
    y = np.random.uniform(low=low, high=high, size=(1,))
    if to_integer:
        y = int(y)
    return y


def normalize_wav(x, use_always: bool):
    if use_always:
        x = x / np.amax(abs(x))
    elif np.amax(abs(x)) > 1:
        x = x / np.amax(abs(x))
    return x


def gen_notch_coeffs(
    nBands, minF, maxF, minBW, maxBW, minCoeff, maxCoeff, minG, maxG, fs
):
    b = 1
    for i in range(0, nBands):
        fc = random_range(minF, maxF, False)
        bw = random_range(minBW, maxBW, False)
        c = random_range(minCoeff, maxCoeff, True)

        if c / 2 == int(c / 2):
            c = c + 1
        f1 = fc - bw / 2
        f2 = fc + bw / 2
        if f1 <= 0:
            f1 = 1 / 1000
        if f2 >= fs / 2:
            f2 = fs / 2 - 1 / 1000
        b = np.convolve(
            signal.firwin(c, [float(f1), float(f2)], window="hamming", fs=fs), b
        )

    G = random_range(minG, maxG, False)
    _, h = signal.freqz(b, 1, fs=fs)
    b = pow(10, G / 20) * b / np.amax(abs(h))
    return b


def filter_FIR(x, b):
    N = b.shape[0] + 1
    xpad = np.pad(x, (0, N), "constant")
    y = signal.lfilter(b, 1, xpad)
    y = y[int(N / 2) : int(y.shape[0] - N / 2)]
    return y

# testFFT.py - An FFT test program
# (c) 2021 @RR_Inyo
# Released under the MIT license
# https://opensource.org/licenses/mit-license.php

import sys
import time
import cmath
import numpy as np

# DFT, straight-forward Discrete Fourier Transform
# Caution! Original input data is overwritten and destroyed.
def DFT(f):
    N = len(f)
    k = np.arange(N)
    y = np.zeros_like(f)  # To contain calculation results

    # DFT core calculation
    for i in range(N):
        # Prepare complex sinusoid
        W = np.exp(-2j * np.pi * i * k / N)

        # multiply-accumulate computation
        y[i] = W @ f

    # Overwrite results to input
    f[:] = y

# DFT, straight-forward Discrete Fourier Transform
# * Not relying on NumPy
# Caution! Original input data is overwritten and destroyed.
def DFTcmath(f):
    N = len(f)
    y = [0 + 1j * 0] * N  # Define list to contain calculation results, 0-initialized

    # DFT core calculation
    for i in range(N):
        for j in range(N):
            # Prepare complex sinusoid
            W = cmath.exp(-2j * cmath.pi * i * j / N)

            # multiply-accumulate computation
            y[i] += W * f[j]

    # Overwrite the results to input
    f[:] = y

# FFT, recursive, ported from Interface, CQ Publishing, Sep. 2021, p. 53, List 1
# Decimation-in-frequency version
# Caution! Original input data is overwritten and destroyed.
def FFTrs(f):
    N = len(f)
    if N > 1:
        n = N // 2  # Caution! Operator '/' in Python3 returns float even for integer divided by integer.
        k = np.arange(n)

        # Prepare complex sinusoid
        W = np.exp(-2j * np.pi * k / N)

        # Butterfly computation
        f_tmp = f[:n] - f[n:]
        f[:n] += f[n:]
        f[n:] = W * f_tmp

        # Recursively call this function
        FFTrs(f[:n])  # First half
        FFTrs(f[n:])  # Second half

        # Simple permutation
        y = np.empty_like(f)
        y[0::2] = f[:n]
        y[1::2] = f[n:]

        # Overwrite results to input
        f[:] = y

# FFT, recursive, ported from Interface, CQ Publishing, Sep. 2021, p. 53, List 2
# Decimation-in-frequency version
# Bit-reversal permutation
# Caution! Original input data is overwritten and destroyed.
def _FFTrb_butterfly(f):
    N = len(f)
    if N > 1:
        n = N // 2  # Caution! Operator '/' in Python3 returns float even for integer divided by integer.
        k = np.arange(n)

        # Prepare complex sinusoid
        W = np.exp(-2j * np.pi * k / N)

        # Butterfly computation
        f_tmp = f[:n] - f[n:]
        f[:n] += f[n:]
        f[n:] = W * f_tmp

        # Recursively call this function
        _FFTrb_butterfly(f[:n])  # First half
        _FFTrb_butterfly(f[n:])  # Second half

def FFTrb(f):
    # Butterfly computations
    _FFTrb_butterfly(f)

    # Bit-reversal permutation
    i = 0
    for j in range(1, N - 1):
        k = N >> 1
        while(True):
            i ^= k
            if k > i:
                k >>= 1
                continue
            else:
                break

        if j < i:
            f_tmp = f[j]
            f[j] = f[i]
            f[i] = f_tmp

# FFT, non-recursive, decimation-in-frequency
# Caution! Original input data is overwritten and destroyed.
def FFTnr(f):
    N = len(f)
    p = int(np.log2(N))

    # Butterfly computations
    for i in range(0, p):       # Number of time-domain data divisions
        n = N // 2**(i + 1)

        # Prepare complex sinusoid, rotation operator
        k = np.arange(n)
        W = np.exp(-2j * np.pi * k / (2 * n))

        for j in range(2 ** i): # Counter for divided time-domain data segments
            # Butterfly computation of each segment
            f_tmp = f[n * 2 * j : n * (2 * j + 1)] - f[n * (2 * j + 1) : n * (2 * j + 2)]
            f[n * 2 * j : n * (2 * j + 1)] += f[n * (2 * j + 1) : n * (2 * j + 2)]
            f[n * (2 * j + 1) : n * (2 * j + 2)] = W * f_tmp

    # Bit-reversal permutation
    i = 0
    for j in range(1, N - 1):
        k = N >> 1
        while(True):
            i ^= k
            if k > i:
                k >>= 1
                continue
            else:
                break
        if j < i:
            f_tmp = f[j]
            f[j] = f[i]
            f[i] = f_tmp

# FFT, non-recursive, decimation-in-frequency
# ** Not relying on NumPy **
# Caution! Original input data is overwritten and destroyed.
def FFTcmath(f):
    N = len(f)
    p = int(cmath.log(N, 2).real)

    # FFT core calculation
    for i in range(0, p):       # Number of time-domain data divisions
        n = N // (2**(i + 1))
        for j in range(2 ** i): # Counter for divided time-domain data
            for k in range(n):  # Counter for each item in divided time-domain data
                # Prepare complex sinusoid, rotation operator
                W = cmath.exp(-2j * np.pi * k / (2 * n))

                # Butterfly computation
                f_tmp = f[n * 2 * j + k] - f[n * (2 * j + 1) + k]
                f[n * 2 * j  + k] = f[n * 2 * j  + k] + f[n * (2 * j + 1) + k]
                f[n * (2 * j + 1) + k] = W * f_tmp

    # Bit-reversal permutation
    i = 0
    for j in range(1, N - 1):
        k = N >> 1
        while(True):
            i ^= k
            if k > i:
                k >>= 1
                continue
            else:
                break
        if j < i:
            f_tmp = f[j]
            f[j] = f[i]
            f[i] = f_tmp

# Main routine
if __name__ == '__main__':
    # Debug flag
    DEBUG = False

    # Change number from cardinal to ordinal
    def ordinal(i):
        j = i % 100;
        k = i % 10;
        if k == 1 and j != 11:
            return str(i) + 'st'
        elif k == 2 and j != 12:
            return str(i) + 'nd'
        elif k == 3 and j != 13:
            return str(i) + 'rd'
        else:
            return str(i) + 'th'

    # Show spectrum on console
    def showSpectrum(f, alg):
        print(f'\r<Spectrum by {alg}>')
        for i in range(N):
            th = 1e-6
            if np.abs(F[i]) > th:
                print(f'{ordinal(i)}:\t{f[i]}')

    # Get number of datapoints from arguments
    args = sys.argv
    if len(args) < 2:
        N = 64  # Default number of datapoints
    else:
        N = int(args[1])

        # Check if power of 2
        if N & (N - 1) != 0:
            print('Error: Not a power of 2', file = sys.stderr)
            sys.exit(1)

    # Number of computation attempts per algorithm
    M = 50

    # Prepare sample input data, as NumPy ndarray
    k = np.arange(N)
    f =   0.8 * np.exp(1 * 2j * np.pi * k / N) \
        + 0.6 * np.exp(7 * 2j * np.pi * k / N + 1j * np.pi / 2) \
        + 0.4 * np.exp(12 * 2j * np.pi * k / N + 1j * np.pi / 2) \
        + 0.3 * np.exp(17 * 2j * np.pi * k / N - 1j * np.pi / 6)

    # Prepare sample input data, as Python's list
    g = [ \
          0.8 * np.exp(1 * 2j * np.pi * k / N) \
        + 0.6 * np.exp(7 * 2j * np.pi * k / N + 1j * np.pi / 2) \
        + 0.4 * np.exp(12 * 2j * np.pi * k / N + 1j * np.pi / 2) \
        + 0.3 * np.exp(17 * 2j * np.pi * k / N - 1j * np.pi / 6) \
        for k in range(N) \
    ]

    # Print message
    print(f'<FFT execution times by Python for {N} datapoints from average of {M} attempts>')
    print('<Note for abbreviation>')
    print('np:\tMaking Use of NumPy vector computation')
    print('cmath:\tNot importing NumPy, using only cmath')
    print('rs:\tRecursive, simple permutation')
    print('rb:\tRecursive, bit-reversal permutation')
    print('nr:\tNon-recursive, bit-reversal permutation')

    # Perform Fourier Transform
    # DFT
    # Rerlying on NumPy
    if DEBUG: print('-' * 64)
    alg = 'DFT.np'
    t1 = 0
    for i in range(M):
        F = f.copy()
        t0 = time.perf_counter()
        DFT(F)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(F, alg)

    # DFT
    # ** Not relying on NumPy **
    if DEBUG: print('-' * 64)
    alg = 'DFT.cmath'
    t1 = 0
    for i in range(M):
        G = g.copy()
        t0 = time.perf_counter()
        DFTcmath(G)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(G, alg)

    # FFT, recursive, simple permutation
    if DEBUG: print('-' * 64)
    alg = 'FFT.np.rs'
    t1 = 0
    for i in range(M):
        F = f.copy()
        t0 = time.perf_counter()
        FFTrs(F)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(F, alg)

    # FFT, recursive, bit-reversal permutation
    # Relying on NumPy
    if DEBUG: print('-' * 64)
    alg = 'FFT.np.rb'
    t1 = 0
    for i in range(M):
        F = f.copy()
        t0 = time.perf_counter()
        FFTrb(F)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(F, alg)

    # FFT, non-recursive, bit-reversal permutation
    # Relying on umPy
    if DEBUG: print('-' * 64)
    alg = 'FFT.np.nr'
    t1 = 0
    for i in range(M):
        F = f.copy()
        t0 = time.perf_counter()
        FFTnr(F)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(F, alg)

    # FFT, non-recursive, bit-reversal permutation
    # ** Not relying on NumPy **
    if DEBUG: print('-' * 64)
    alg = 'FFT.cmath.nr'
    t1 = 0
    for i in range(M):
        G = g.copy()
        t0 = time.perf_counter()
        FFTcmath(G)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    if DEBUG: showSpectrum(G, alg)

    # FFT by numpy.fft.fft
    if DEBUG: print('-' * 64)
    alg = 'numpy.fft.fft'
    t1 = 0
    for i in range(M):
        t0 = time.perf_counter()
        F = np.fft.fft(f)
        t1 += time.perf_counter() - t0
    t1 /= M
    print(f'{alg}:\t{t1 * 1e6:.1f} microsec.')
    showSpectrum(F, alg)

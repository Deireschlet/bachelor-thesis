import os
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_process import ArmaProcess
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller

def path_constructor(folder_path, file_name):
    return os.path.join(folder_path, file_name)

def randomwalk1D(n, start=0):
    xposition = [start]
    values = [-1, 1]
    for i in range(1, n + 1):
        start += random.choice(values)
        xposition.append(start)
    return pd.Series(xposition)

def adf_test(timeseries):
    print("Results of Dickey-Fuller Test:")
    dftest = adfuller(timeseries)
    dfoutput = pd.Series(
        dftest[0:4],
        index=[
            "Test Statistic",
            "p-value",
            "#Lags Used",
            "Number of Observations Used",
        ],
    )
    for key, value in dftest[4].items():
        dfoutput["Critical Value (%s)" % key] = value
    return dfoutput


def plot_acf_data(data, model, lags=20):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plot_acf(data, lags=lags, ax=ax)
    ax.set_title(f'ACF of {model}')
    plt.show()

def plot_pacf_data(data, model, lags=20):
    plt.figure(figsize=(10, 6))
    ax = plt.gca()
    plot_pacf(data, lags=lags, ax=ax)
    ax.set_title(f'PACF of {model}')
    plt.show()

def AR_generator(phi: np.array, n):
    ar = np.r_[1, -phi]
    ma = np.r_[1]
    AR_object = ArmaProcess(ar, ma)
    AR_data = AR_object.generate_sample(nsample=n)
    return AR_data

def MA_generator(theta: np.array, n):
    ar = np.array([1])
    ma = np.array([1, *theta])
    MA_object = ArmaProcess(ar, ma)
    MA_data = MA_object.generate_sample(nsample=n)
    return MA_data

def ARMA_generator(phi: np.array, theta: np.array, n):
    ar = np.r_[1, -phi]
    ma = np.r_[1, *theta]
    ARMA_object = ArmaProcess(ar, ma)
    ARMA_data = ARMA_object.generate_sample(nsample=n)
    return ARMA_data

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def manual_ar_process(ar_coeffs, noise_var, n):
    noise = np.random.normal(0, np.sqrt(noise_var), n)
    p = len(ar_coeffs)
    ar_series = np.zeros(n)
    
    for t in range(p, n):
        ar_series[t] = sum(ar_coeffs[i] * ar_series[t - i - 1] for i in range(p)) + noise[t]
    
    return ar_series

def manual_ma_process(ma_coeffs, noise_var, n):
        noise = np.random.normal(0, np.sqrt(noise_var), n)
        q = len(ma_coeffs)
        ma_series = np.zeros(n)
        
        for t in range(q, n):
            ma_series[t] = sum(ma_coeffs[i] * noise[t - i - 1] for i in range(q)) + noise[t]
        
        return ma_series

def manual_arma_process(ar_coeffs, ma_coeffs, noise_var, n):
    noise = np.random.normal(0, np.sqrt(noise_var), n)
    p = len(ar_coeffs)
    q = len(ma_coeffs)
    arma_series = np.zeros(n)
    
    for t in range(p, n):
        arma_series[t] = sum(ar_coeffs[i] * arma_series[t - i - 1] for i in range(p)) + noise[t] + sum(ma_coeffs[i] * noise[t - i - 1] for i in range(q))
    
    return arma_series

def yule_walker(data, order):
    
    # Compute sample autocovariances
    autocovariances = [np.cov(data[:-k], data[k:])[0, 1] if k > 0 else np.var(data, ddof=0) 
                       for k in range(order + 1)]
    
    # Create the autocovariance matrix R
    R = np.array([[autocovariances[abs(i - j)] for j in range(order)] for i in range(order)])

    # Create the autocovariance vector r
    r = np.array(autocovariances[1:order + 1])

    ar_params = np.linalg.solve(R, r)
    
    # Compute noise variance
    noise_variance = autocovariances[0] - np.dot(ar_params, r)

    return ar_params, noise_variance


def calculate_unit_roots(ar_coeffs):
    
    poly_coeffs = [1] + [-coeff for coeff in ar_coeffs]
    
    # Find the roots of the polynomial
    roots = np.roots(poly_coeffs)
    
    unit_roots = 1 / roots
    
    return roots, unit_roots


def plot_roots(unit_roots, padding=1):
    fig = plt.figure(figsize=(6, 6))
    circle = plt.Circle((0, 0), 1, color='blue', fill=False, linewidth=0.8, linestyle='--')
    ax = plt.gca()
    ax.add_artist(circle)
    ax.scatter(unit_roots.real, unit_roots.imag, color='red', label='Unit Roots')
    ax.set_xlabel("Re")
    ax.set_ylabel("Im")
    ax.set_title("Unit Roots of the Lag Polynomial")
    ax.set_aspect('equal', adjustable='box')
    ax.legend()
    ax.grid()
    
    # Autoscale and add padding
    ax.autoscale()
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax.set_xlim([xlim[0] - padding, xlim[1] + padding])
    ax.set_ylim([ylim[0] - padding, ylim[1] + padding])
    
    return fig

def main():
    print('Helper functions for time series analysis')

if __name__ == '__main__':
    main()
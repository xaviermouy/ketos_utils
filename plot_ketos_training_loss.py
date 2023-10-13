# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:07:31 2022

@author: xavier.mouy
"""

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('TkAgg')

infile = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\databases\spectro-0.5s_fft-0.06_step-0.008_fmin-0_fmax-800_augmented\results\bs32_ep50\log.csv"

df = pd.read_csv(infile)
pd.pivot_table(
    df.reset_index(), index="epoch", columns="dataset", values="loss"
).plot(subplots=False, grid=True, ylabel="Loss", xlabel="Epoch")
pd.pivot_table(
    df.reset_index(), index="epoch", columns="dataset", values="Precision"
).plot(subplots=False, grid=True, ylabel="Precision", xlabel="Epoch")
pd.pivot_table(
    df.reset_index(), index="epoch", columns="dataset", values="Recall"
).plot(subplots=False, grid=True, ylabel="Recall", xlabel="Epoch")

plt.show()
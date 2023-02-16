# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 10:07:31 2022

@author: xavier.mouy
"""

import pandas as pd
import matplotlib.pyplot as plt

infile = r"D:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\databases\spectro-0.4s_fft-0.128_step-0.005_fmin-0_fmax-1200_no-norm\model\bs256_ep50\log.csv"

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

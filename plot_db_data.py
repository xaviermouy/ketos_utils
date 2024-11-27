# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 15:07:05 2022

@author: xavier.mouy
"""
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.database_interface import open_file, open_table
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
matplotlib.use('TkAgg')


#db_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\ketos_databases\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800\database.h5"
#db_file = r"D:\NOAA\2022_BC_fish_detector\ketos\dataset_20230202T162110\databases\spectro-0.4s_fft-0.128_step-0.005_fmin-0_fmax-800_no-norm\database.h5"
db_file = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\noise_MA-RI\database.h5"
db_file = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\noise-StantonBank\database.h5"
db_file = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\database.h5'

db_file = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_with_more_noise\database.h5'

db_file = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_with_more_noise_and_more_MW\database.h5'

db_file =r'G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\databases\spectro-0.2s_fft-0.064_step-0.01_fmin-0_fmax-1200_no-norm_denoised_fulldataset\database.h5'
db = dbi.open_file(db_file, "r")
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/test/data")


#idx_1 = dbi.filter_by_label(train_data, 1)

idx = 30113
spectro = train_data[idx]["data"]
label = train_data[idx]["label"]
filename = train_data[idx]["filename"]

# labels = train_data[:]["label"]
#plt.figure()
#plt.plot(labels)

plt.figure()
plt.imshow(spectro.T)
plt.title(str(label) + ' - ' + str(filename))
plt.gca().invert_yaxis()
# plt.ylim([0, 200])
# plt.gca().invert
plt.show()
print('s')
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
import pandas as pd
import os
#matplotlib.use('TkAgg')
matplotlib.use("Agg")


db_file = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_full-dataset_and_more-noise\database.h5'
out_dir = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_full-dataset_and_more-noise\db_spectrograms'
target_class_id = 1

# load db
db = dbi.open_file(db_file, "r")
train_data = dbi.open_table(db, "/train/data")
val_data = dbi.open_table(db, "/test/data")

# only_select target labels
train_labels = train_data[:]["label"]
train_labels_target_idx = [i for i, elem in enumerate(train_labels) if elem==target_class_id]
train_indices = pd.DataFrame({'indices': train_labels_target_idx})
train_indices.to_csv(os.path.join(out_dir,'target_indices.csv'))

# Go through train table
for idx in train_labels_target_idx:
    outfilename =os.path.join(out_dir, str(idx) + '.png')
    if os.path.isfile(outfilename) == False:

        spectro = train_data[idx]["data"]
        label = train_data[idx]["label"]
        filename = train_data[idx]["filename"]

        #plt.figure()
        plt.imshow(spectro.T)
        plt.title(str(label) + ' - ' + str(filename))
        plt.gca().invert_yaxis()
        plt.savefig(outfilename)
        #plt.show()
        plt.close('all')

# close file
db.close()
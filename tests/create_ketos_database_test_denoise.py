# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:00:17 2022

This script takes the annotation logs for the train and test sets and creates
a database of spectrograms that will be used to train/test the neural net

@author: xavier.mouy
"""


import os
import csv
import pandas as pd
from ketos.data_handling import selection_table as sl
import ketos.data_handling.database_interface as dbi
from ketos.data_handling.parsing import load_audio_representation
from ketos.audio.spectrogram import MagSpectrogram

import sys
sys.path.append("..") # Adds higher directory to python modules path.
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.visualization.grapher_builder import GrapherFactory
import time
import copy 
infile = r'C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\data\fish\AMAR173.4.20190916T131248Z.wav'
out_dir = r'C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\data\test_manual_db'


# Spectrogram parameters
frame = 512
nfft = 512
step = 50
#ovlp = 2500
fmin = 0 
fmax = 1000
window_type = 'hann'
freq_min_hz = 0
freq_max_hz = 1000

denoiser_name = 'median_equalizer'
denoiser_window_duration_sec= 10
denoiser_use_dask = True
denoise_dask_chunks= [50,50000]


# start and stop time of wavfile to analyze
t1 = 0
t2 = 60


## ###########################################################################
tic = time.perf_counter()

# load audio data
sound = Sound(infile)
fs = sound.waveform_sampling_frequency
#sound.read(channel=0, chunk=[round(t1*fs), round(t2*fs)])
sound.read(channel=0)

# Calculates  spectrogram
Spectro = Spectrogram(frame, window_type, nfft, step, fs, unit='samp')
Spectro.compute(sound, dB=True)





# Crop unused frequencies
Spectro.crop(frequency_min=freq_min_hz,
              frequency_max=freq_max_hz,
              inplace=True,
              )

Spectro1 = copy.copy(Spectro)

# Denoise
print('Denoise')
Spectro.denoise(denoiser_name,
                window_duration=denoiser_window_duration_sec,
                use_dask=denoiser_use_dask,
                dask_chunks=denoise_dask_chunks,
                inplace=True)

toc = time.perf_counter()
print(f"Executed in {toc - tic:0.4f} seconds")

# # Plot
# graph = GrapherFactory('SoundPlotter', title='Recording', frequency_max=1000)
# graph.add_data(sound)
# graph.add_data(Spectro1)
# graph.add_data(Spectro)
# graph.colormap = 'jet'
# graph.show()

# # Detector


print("---- Creating database of spectrograms ----")


ketos.data_handling.database_interface.write_repres_attrs(table, x)


## creatae spectrogram database
db_file = os.path.join(out_dir, "database.h5")
dbi.create_database(
    output_file=db_file,
    # table_name="data",
    data_dir=params["data_train_dir"],
    dataset_name="train",
    selections=std_annot_train_aug,
    audio_repres=spec_cfg,
    verbose=True,
    progress_bar=True,
    discard_wrong_shape=False,
    # mode="w",
)

# dbi.create_database(
#     output_file=db_file,
#     data_dir=params["data_test_dir"],
#     dataset_name="test",
#     selections=std_annot_test_aug,
#     audio_repres=spec_cfg,
# )

# # db = dbi.open_file("database.h5", 'r')

# print("---- Saving parameters to log file ----")
# ## writes input parameters as csv file for book keeping purposes
# log_file = open(
#     os.path.join(current_out_dir, "ketos_database_parameters.csv"), "w"
# )
# writer = csv.writer(log_file)
# for key, value in params.items():
#     writer.writerow([key, value])
# log_file.close()

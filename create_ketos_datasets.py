# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 09:05:25 2022

This script prepares audio and annotation data for creating a ketos database

- splits the annotations into a train and test dataset
    - split ratio controlled by user
    - Stratified split between train and test
    - Group based split: Annotations coming the same deployment and same hour
      must stay together in either the train or test set (avoids background 
      matching issue during training).

- creates a train.csv and test.csv file to be used by Ketos
- moves audio files into a train and test folder
- decimates audio files based on user parameters.

@author: xavier.mouy
"""

from ecosound.core.annotation import Annotation
from ecosound.core.measurement import Measurement

from ecosound.core.audiotools import Sound
import soundfile as sf
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import scipy
import os
import csv


def decimate(infile, outfile, channel, params):
    # init audio file
    audio_data = Sound(infile)
    # load audio data
    audio_data.read(channel=channel-1, detrend=True)
    # bandpass filter
    #filter_type='bandpass'
    #cutoff_frequencies=params["bandpass_filter_fc"]
    #audio_data.filter(filter_type, cutoff_frequencies, order=params["filter_order"], verbose=True)
    # decimate
    audio_data.decimate(params["sanpling_rate_hz"])
    ## detrend
    audio_data.detrend()
    # normalize
    #audio_data.normalize(method='std')
    audio_data.normalize(method='amplitude')
    # write new file
    #outfilename = os.path.basename(os.path.splitext(train_file)[0]) + ".wav"
    audio_data.write(outfile)

## ############################################################################
params = dict()
params["dataset_file_path"] = r"G:\NOAA\2022_Minke_whale_detector\manual_annotations\discrete_datasets\UK-SAMS-WestScotland-201711-StantonBank_set2\Annotations_dataset_UK-SAMS-WestScotland-201711-StantonBank_set2 annotations.nc"
params["out_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos"
params["group_timeframe"] = "6H"
params["train_ratio"] = 0.75
params["sanpling_rate_hz"] = 2000
params["use_subclass_label"] = False # if True, startifies using [label_class + label_subclass]


# params = dict()
# params[
#     "dataset_file_path"
# ] = r"D:\MANUAL_ANNOTATIONS\Annotations_dataset_HK-NN-MW-HB_20230928T100147.nc"
# params["out_dir"] = r"C:\Users\xavie\OneDrive\Desktop\KETOS"
# params["group_timeframe"] = "6H"
# params["train_ratio"] = 0.75
#
# params["sanpling_rate_hz"] = 2000
# params["bandpass_filter_fc"]=[20,800]
# params["filter_type"] = "iir"
# params["filter_order"] = 8
#
# # if True, startifies using [label_class + label_subclass]
# params["use_subclass_label"] = False


# params = dict()
# params[
#     "dataset_file_path"
# ] = r"D:\NOAA\2022_BC_fish_detector\manual_annotations\Annotations_dataset_FS-NN-UN-HS-KW_20230202T135354_withSNRgt2dB.nc"
# params["out_dir"] = r"D:\NOAA\2022_BC_fish_detector\ketos"
# params["group_timeframe"] = "1H"
# params["train_ratio"] = 0.75
#
# params["sanpling_rate_hz"] = 4000
# params["filter_type"] = "iir"
# params["filter_order"] = 8
#
# # if True, startifies using [label_class + label_subclass]
# params["use_subclass_label"] = False

## ############################################################################

# create output folders for this dataset
current_dir_name = "dataset_" + datetime.strftime(
    datetime.now(), "%Y%m%dT%H%M%S"
)
current_out_dir = os.path.join(params["out_dir"], current_dir_name)
os.mkdir(current_out_dir)
# current_out_dir = params['out_dir']

# writes input parameters as csv file for book keeping purposes
log_file = open(os.path.join(current_out_dir, "dataset_parameters.csv"), "w")
writer = csv.writer(log_file)
for key, value in params.items():
    writer.writerow([key, value])
log_file.close()

# Load dataset
dataset = Annotation()
# dataset = Measurement()
dataset.from_netcdf(params["dataset_file_path"])
if params["use_subclass_label"]:
    dataset.data["label"] = (
        dataset.data["label_class"] + dataset.data["label_subclass"]
    )
else:
    dataset.data["label"] = dataset.data["label_class"]


# # filter
# print("d")
# dataset.filter("(label_class =='FS' & snr >= 1) | (label_class =='NN')",inplace=True)


data = dataset.data

# Add group ID for train/test splits (annotations with the same groupID can't be in different train or test sets)
data.insert(
    0,
    "TimeLabel",
    data["time_min_date"]
    .dt.round(params["group_timeframe"])
    .apply(lambda x: x.strftime("%Y%m%d%H%M%S")),
)
# data.insert(0,'group_label',data['label_class'] + '_' + data['TimeLabel'] + '_' + data['deployment_ID'])
data.insert(0, "group_label", data["TimeLabel"] + "_" + data["deployment_ID"])
labels = list(set(data["group_label"]))
IDs = [*range(0, len(labels))]
data.insert(0, "group_ID", -1)
for n, label in enumerate(labels):
    data.loc[data["group_label"] == label, "group_ID"] = IDs[n]
data.drop(columns=["TimeLabel"])

# Add class-subclass ID for train/test stratification
if params["use_subclass_label"]:
    data.insert(0, "class_ID", data["label_class"] + data["label_subclass"])
else:
    data.insert(0, "class_ID", data["label_class"])


# Splits data into train and test sets
n_splits = round(1 / (1 - params["train_ratio"]))
skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=None)
for train_index, test_index in skf.split(
    data, data["class_ID"], groups=data["group_ID"]
):
    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    break

print('----------------------------------')
print('Train set:')
print(' ')
print(data_train.pivot_table(index='deployment_ID', columns='class_ID', aggfunc="size", fill_value=0))

print('----------------------------------')
print('Test set:')
print(' ')
print(data_test.pivot_table(index='deployment_ID', columns='class_ID', aggfunc="size", fill_value=0))

# Add channel number to filename so it can handle multichannel files
data_train["audio_file_name"]= data_train["audio_file_name"] + '_CH' + data_train["audio_channel"].astype(str)
data_test["audio_file_name"]= data_test["audio_file_name"] + '_CH' + data_test["audio_channel"].astype(str)

# save to train and test sets to CSV
data_train.to_csv(os.path.join(current_out_dir, "train.csv"))
data_test.to_csv(os.path.join(current_out_dir, "test.csv"))

# create train folder and decimate audio file
train_files = (
    data_train["audio_file_dir"]
    + "\\"
    + data_train["audio_file_name"]
    + data_train["audio_file_extension"]
)
train_files = list(set(train_files))
train_dir = os.path.join(current_out_dir, "train_data")
os.mkdir(train_dir)
for train_file in train_files:
    print(train_file)
    split_tup = os.path.splitext(train_file)
    file_name = split_tup[0]
    file_extension = split_tup[1]
    ch_idx = file_name.rfind('_CH')
    infile = file_name[:ch_idx] + file_extension
    outfile = os.path.join(train_dir,os.path.basename(file_name)) + ".wav"
    channel = int(file_name[ch_idx+3:])
    decimate(infile, outfile, channel, params)

# create test folder and decimate audio file
test_files = (
    data_test["audio_file_dir"]
    + "\\"
    + data_test["audio_file_name"]
    + data_test["audio_file_extension"]
)
test_files = list(set(test_files))
test_dir = os.path.join(current_out_dir, "test_data")
os.mkdir(test_dir)
for test_file in test_files:
    #decimate(test_file, test_dir, params)
    print(test_file)
    split_tup = os.path.splitext(test_file)
    file_name = split_tup[0]
    file_extension = split_tup[1]
    ch_idx = file_name.rfind('_CH')
    infile = file_name[:ch_idx] + file_extension
    outfile = os.path.join(test_dir,os.path.basename(file_name)) + ".wav"
    channel = int(file_name[ch_idx+3:])
    decimate(infile, outfile, channel, params)

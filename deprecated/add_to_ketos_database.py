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
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram, adjust_FFT_size
import matplotlib.pyplot as plt
import numpy as np
import tables
from tqdm import tqdm
import warnings

class SpectroTable(tables.IsDescription):
    data = tables.Float32Col()
    filename = tables.StringCol(100)
    id = tables.UInt32Col()
    label = tables.UInt8Col()
    offset = tables.Float64Col()    # 32-bit integer


# def write_to_database(db_file, dataset_name, selections, data_dir, params, db_mode="w"):
#     """ Create spectrogram database
#
#
#     """
#
#     h5file = tables.open_file(db_file, mode=db_mode, title="Database")
#     # create table
#     group = h5file.create_group("/", dataset_name, '')  # create group
#     filters = tables.Filters(complevel=1, fletcher32=True)
#     # create table
#     SpectroTable.columns['data'] = tables.Float32Col(shape=(win_dur_bins, win_bw_bins,))
#     table = h5file.create_table(group, 'data', SpectroTable, '', filters=filters, chunkshape=21)
#     table_item = table.row
#     files_list = list(set(selections.index.get_level_values(0)))
#
#     for file in files_list:
#         print(file)
#         # annotations for that file
#         tmp = selections.loc[file]
#         # load audio data
#         sound = Sound(os.path.join(data_dir, file))
#         t1 = tmp.start.min() - file_buffer_sec
#         t2 = tmp.end.max() + file_buffer_sec
#         clip_start_sec = file_buffer_sec
#         if t1 < 0:
#             clip_start_sec = file_buffer_sec + t1
#             t1 = 0
#         if t2 > sound.file_duration_sec:
#             t2 = sound.file_duration_sec
#         sound.read(channel=0, unit='sec', chunk=[t1, t2])
#         # Calculates  spectrogram
#         spectro = Spectrogram(params["spec_cfg"]['window'],
#                               params["spec_cfg"]['window_func'],
#                               params["spec_cfg"]['window'],
#                               params["spec_cfg"]['step'],
#                               sound.waveform_sampling_frequency,
#                               unit='sec',
#                               )
#         spectro.compute(sound,
#                         params["spec_cfg"]['dB'],
#                         params["spec_cfg"]['use_dask'],
#                         params["spec_cfg"]['dask_chunks'],
#                         )
#         # Crop unused frequencies
#         spectro.crop(frequency_min=params["spec_cfg"]['freq_min'],
#                      frequency_max=params["spec_cfg"]['freq_max'],
#                      inplace=True,
#                      )
#         # Denoise
#         print('Denoise')
#         spectro.denoise(params["spec_cfg"]['denoiser'][0]['name'],
#                         window_duration=params["spec_cfg"]['denoiser'][0]['window_duration_sec'],
#                         use_dask=params["spec_cfg"]['denoiser'][0]['use_dask'],
#                         dask_chunks=tuple(params["spec_cfg"]['denoiser'][0]['dask_chunks']),
#                         inplace=True)
#
#         # go through each annotation
#         for annot_index, annot_row in tmp.iterrows():
#             #print('ss')
#             # time_idx = np.array(abs(spectro.axis_times - annot_row['start']-t1)).argmin()
#             time_idx = np.array(abs(spectro.axis_times - clip_start_sec)).argmin()
#             table_item['data'] = spectro.spectrogram[:, time_idx:time_idx + win_dur_bins].T
#             table_item['filename'] = file
#             table_item['id'] = int(annot_index)
#             table_item['label'] = int(annot_row['label'])
#             # Insert a new record to table
#             table_item.append()
#         # writes to file
#         table.flush()
#     # close file
#     h5file.close()

def write_to_database(db_file, dataset_name, selections, data_dir, params, db_mode="w"):
    """ Create spectrogram database


    """

    h5file = tables.open_file(db_file, mode=db_mode, title="Database")
    # create table
    group = h5file.create_group("/", dataset_name, '')  # create group
    filters = tables.Filters(complevel=1, fletcher32=True)
    # create table
    SpectroTable.columns['data'] = tables.Float32Col(shape=(params['spec_cfg']['win_dur_bins'], params['spec_cfg']['win_bw_bins'],))
    table = h5file.create_table(group, 'data', SpectroTable, '', filters=filters, chunkshape=21)
    table_item = table.row

    # buffer before and after annot needed for the denoiser
    try:
        file_buffer_sec = 2*params["spec_cfg"]['denoiser'][0]['window_duration_sec']
    except:
        file_buffer_sec =0
    pbar = tqdm(total=selections.shape[0],colour="green")
    idxx=0
    for annot_index, annot_row in selections.iterrows():
        try:
            # load audio data
            file = annot_row.name[0]
            sound = Sound(os.path.join(data_dir, file))
            t1 = annot_row.start.min() - file_buffer_sec
            t2 = annot_row.end.max() + file_buffer_sec
            clip_start_sec = file_buffer_sec
            if t1 < 0:
                clip_start_sec = file_buffer_sec + t1
                t1 = 0
            if t2 > sound.file_duration_sec:
                t2 = sound.file_duration_sec
            sound.read(channel=0, unit='sec', chunk=[t1, t2])
            # Calculates  spectrogram
            spectro = Spectrogram(params["spec_cfg"]['window'],
                                  params["spec_cfg"]['window_func'],
                                  params["spec_cfg"]['window'],
                                  params["spec_cfg"]['step'],
                                  sound.waveform_sampling_frequency,
                                  unit='sec',
                                  verbose=False,
                                  )
            spectro.compute(sound,
                            params["spec_cfg"]['dB'],
                            params["spec_cfg"]['use_dask'],
                            params["spec_cfg"]['dask_chunks'],
                            )
            # Crop unused frequencies
            spectro.crop(frequency_min=params["spec_cfg"]['freq_min'],
                         frequency_max=params["spec_cfg"]['freq_max'],
                         inplace=True,
                         )
            # Denoise
            #print('Denoise')
            spectro.denoise(params["spec_cfg"]['denoiser'][0]['name'],
                            window_duration=params["spec_cfg"]['denoiser'][0]['window_duration_sec'],
                            use_dask=params["spec_cfg"]['denoiser'][0]['use_dask'],
                            dask_chunks=tuple(params["spec_cfg"]['denoiser'][0]['dask_chunks']),
                            inplace=True)

            time_idx = np.array(abs(spectro.axis_times - clip_start_sec)).argmin()
            spectro_matrix = spectro.spectrogram[:, time_idx:time_idx + params['spec_cfg']['win_dur_bins']].T
            if spectro_matrix.shape == (params['spec_cfg']['win_dur_bins'], params['spec_cfg']['win_bw_bins']):
                table_item['data'] = spectro_matrix
                table_item['filename'] = file
                table_item['id'] = int(idxx)
                table_item['label'] = int(annot_row['label'])
                idxx+=1
                # Insert a new record to table
                table_item.append()
                # writes to file
                table.flush()
            else:
                print('Wrong spectrogram dimensions - data sample ignored.' + 'Spectro shape=' +str(spectro_matrix.shape) + ', expected shape= ' +str((params['spec_cfg']['win_dur_bins'], params['spec_cfg']['win_bw_bins'])))
        except:
            print('something went wrong...')
            continue
        pbar.update(1)
    # close file
    pbar.close()
    h5file.close()

def run():

    # # Input parameters #################################################################

    # params = dict()
    # params[
    #     "train_annot_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train.csv"
    # params[
    #     "test_annot_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test.csv"
    #
    # params[
    #     "data_train_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train_data"
    # params[
    #     "data_test_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test_data"
    # params[
    #     "out_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-10s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm"
    # params[
    #     "spectro_config_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-10s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\spec_config.json"
    #
    #
    # params["class_labels"] = []
    # params["class_labels"].append(
    #     ["NN", "HB", "HK", "HKP", "NNS", "HKPT"]
    # )  # class 0
    # params["class_labels"].append(["MW"])  # class 1
    #
    # params["max_samples_per_class_train"] = [400000, None]
    # params["max_samples_per_class_test"] = [None, None]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 10  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 0.90

    # params = dict()
    # params[
    #     "train_annot_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train.csv"
    # params[
    #     "test_annot_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test.csv"
    #
    # params[
    #     "data_train_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train_data"
    # params[
    #     "data_test_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test_data"
    # params[
    #     "out_dir"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-10s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm"
    # params[
    #     "spectro_config_file"
    # ] = r"D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-10s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\spec_config.json"
    #
    #
    # params["class_labels"] = []
    # params["class_labels"].append(
    #     ["NN", "HB", "HK", "HKP", "NNS", "HKPT"]
    # )  # class 0
    # params["class_labels"].append(["MW"])  # class 1
    #
    # params["max_samples_per_class_train"] = [400000, None]
    # params["max_samples_per_class_test"] = [None, None]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 10  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 0.90

    # params = dict()
    # params["train_annot_file"] = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\train.csv"
    # params["test_annot_file"] = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\test.csv"
    # params["data_train_dir"] = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\train_data"
    # params["data_test_dir"] = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\test_data"
    # params["out_dir"] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\tests"
    # params["spectro_config_file"] = r"G:\NOAA\2023_Haddock_detector\ketos\dataset_20230928T100401\databases\spectro-0.5s_fft-0.06_step-0.008_fmin-0_fmax-800\spec_config.json"
    #
    # params["class_labels"] = []
    # params["class_labels"].append(["NN", "HB", "MW"])  # class 0
    # params["class_labels"].append(["HK"])  # class 1
    # params["max_samples_per_class_train"] = [10, 20]
    # params["max_samples_per_class_test"] = [10, 20]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 0.5  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 0 #0.1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 1 #0.90


    # ## BC Fish ----------------------------------------------------------------------------
    # params = dict()
    # params["train_annot_file"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\train.csv"
    # params["test_annot_file"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\test.csv"
    # params["data_train_dir"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\train_data"
    # params["data_test_dir"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\test_data"
    # params["out_dir"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\databases\spectro-0.2s_fft-0.064_step-0.01_fmin-0_fmax-1200_no-norm_denoised_fulldataset"
    # params["spectro_config_file"] = r"G:\NOAA\2022_BC_fish_detector\ketos\dataset_20230206T120453\databases\spectro-0.2s_fft-0.064_step-0.01_fmin-0_fmax-1200_no-norm_denoised_fulldataset\spec_config_custom.json"
    #
    # params["class_labels"] = []
    # params["class_labels"].append(["NN", "KW", "HS"])  # class 0
    # params["class_labels"].append(["FS"])  # class 1
    # #params["max_samples_per_class_train"] = [40000, 10000]
    # #params["max_samples_per_class_test"] = [20000, 5000]
    # params["max_samples_per_class_train"] = [None, None]
    # params["max_samples_per_class_test"] = [None, None]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 0.2  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 0.1 #0.1 #0.1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 1

    ## BC Fish ----------------------------------------------------------------------------
    params = dict()
    params["train_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train.csv"
    params["test_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test.csv"
    params["data_train_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train_data"
    params["data_test_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test_data"
    params[
        "out_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s"
    params[
        "spectro_config_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\spec_config_custom.json"

    params["class_labels"] = []
    params["class_labels"].append(["NN", "HB", "HK", "HKP", "NNS", "HKPT"])  # class 0
    params["class_labels"].append(["MW"])  # class 1

    #params["max_samples_per_class_train"] = [70000, 71207]
    #params["max_samples_per_class_test"] = [70000, 16435]

    params["max_samples_per_class_train"] = [2000, 2000]
    params["max_samples_per_class_test"] = [2000, 2000]

    # segemnt duration in sec for the CNN:
    params["classif_window_sec"] = 5  # 0.4
    # step between consecutive windows in sec (0: no augmentation):
    params["aug_win_step_sec"] = 1  # 0.1 #0.1
    # windows must contain at least x% of the annotation:
    params["aug_min_annot_ovlp"] = 1



    ## ############################################################################

    # load spectro_config_file
    spec_cfg = load_audio_representation(
        params["spectro_config_file"], name="spectrogram"
    )
    if params["classif_window_sec"] != spec_cfg["duration"]:
        raise Exception(
            "classif_window_sec value does not match the one in the spectro_config_file."
        )
    params["spec_cfg"] = spec_cfg

    current_out_dir = params["out_dir"]

    # calculate freq and time bins of spectrogram classif windows
    frame_samp = params['spec_cfg']['window']*params['spec_cfg']['rate']
    nfft = adjust_FFT_size(int(frame_samp), verbose=True)
    freq_bw_hz = params['spec_cfg']['freq_max'] - params['spec_cfg']['freq_min']
    freq_res_hz = params['spec_cfg']['rate'] / nfft
    time_res_s = np.ceil(params['spec_cfg']['step']*params['spec_cfg']['rate'])/params['spec_cfg']['rate']
    
    axis_frequencies = np.arange(0, params['spec_cfg']['rate'] / 2, freq_res_hz)

    # Find frequency indices
    if params['spec_cfg']['freq_min'] is None:
        min_row_idx = 0
    else:
        min_row_idx = np.where(axis_frequencies < params['spec_cfg']['freq_min'])
        if np.size(min_row_idx) == 0:
            min_row_idx = 0
        else:
            min_row_idx = min_row_idx[0][-1] + 1
    if params['spec_cfg']['freq_max'] is None:
        max_row_idx = axis_frequencies.size - 1
    else:
        max_row_idx = np.where(axis_frequencies > params['spec_cfg']['freq_max'])
        if np.size(max_row_idx) == 0:
            max_row_idx = axis_frequencies.size - 1
        else:
            max_row_idx = max_row_idx[0][0]

    nbins = max_row_idx - min_row_idx + 1
    
    #params['spec_cfg']['win_bw_bins'] = int(np.ceil(freq_bw_hz/freq_res_hz)+1)
    params['spec_cfg']['win_bw_bins'] = nbins
    params['spec_cfg']['win_dur_bins'] = int(np.ceil(params['spec_cfg']['duration']/time_res_s))

    # load annotations
    annot_train = pd.read_csv(
        params["train_annot_file"],
        converters={
            "audio_file_name": str,
            "class_ID": str,
            "group_label_ID": str,
            "label_class": str,
            "label_subclass": str,
            "hydrophone_SN": str,
        },
    )
    annot_test = pd.read_csv(
        params["test_annot_file"],
        converters={
            "audio_file_name": str,
            "class_ID": str,
            "group_label_ID": str,
            "label_class": str,
            "label_subclass": str,
            "hydrophone_SN": str,
        },
    )

    # display classes available
    print("Train set classes:", set(annot_train["class_ID"]))
    print("Test set classes:", set(annot_test["class_ID"]))

    print(" ")
    print("---- Reformating annotations ----")

    # reformating file names
    annot_train["sound_file"] = list(
        annot_train["audio_file_name"].apply(lambda x: str(x) + ".wav")
    )
    annot_test["sound_file"] = list(
        annot_test["audio_file_name"].apply(lambda x: str(x) + ".wav")
    )

    # Assigning positive and negative class labels
    print(" ")
    print("Train set:")
    ID_list = []
    for cl_id, cl_labels in enumerate(params["class_labels"]):
        pattern = "|".join(cl_labels)
        annot_train["label"][annot_train.class_ID.str.fullmatch(pattern)] = int(
            cl_id
        )
        ID_list.append(int(cl_id))
        print("- Class ID: ", str(cl_id))
        print("   - Labels: ", cl_labels)
        print(
            "   - Number of annotations: ", sum(annot_train["label"] == int(cl_id))
        )
    tmp = annot_train.loc[~annot_train["label"].isin(ID_list)]
    if len(tmp) > 0:
        print("- Annotations ignored: ", str(len(tmp)))
        print(list(set(tmp["label"])))
        annot_train.loc[tmp.index, "label"] = -1
    else:
        print("- Annotations ignored: 0")

    print(" ")
    print("Test set:")
    ID_list = []
    for cl_id, cl_labels in enumerate(params["class_labels"]):
        pattern = "|".join(cl_labels)
        annot_test["label"][annot_test.class_ID.str.fullmatch(pattern)] = int(
            cl_id
        )
        ID_list.append(int(cl_id))
        print("- Class ID: ", str(cl_id))
        print("   - Labels: ", cl_labels)
        print(
            "   - Number of annotations: ", sum(annot_test["label"] == int(cl_id))
        )
    tmp = annot_test.loc[~annot_test["label"].isin(ID_list)]
    if len(tmp) > 0:
        print("- Annotations ignored: ", str(len(tmp)))
        print(list(set(tmp["label"])))
        annot_test.loc[tmp.index, "label"] = -1
    else:
        print("- Annotations ignored: 0")


    # reformat annotations to Ketos format
    # sl.is_standardized(annot_train, verbose=False)
    map_to_ketos_annot_std = {
        "filename": "sound_file",
        "start": "time_min_offset",
        "end": "time_max_offset",
        "min_freq": "frequency_min",
        "max_freq": "frequency_max",
    }

    # map_to_ketos_annot_std = {
    #     "sound_file": "filename",
    #     "time_min_offset": "start",
    #     "time_max_offset": "end",
    #     "frequency_min": "min_freq",
    #     "frequency_max": "max_freq",
    # }


    std_annot_train = sl.standardize(
        table=annot_train,
        mapper=map_to_ketos_annot_std,
        trim_table=True,
        labels = ID_list,
        start_labels_at_1=False,
    )
    print(" ")
    print(
        "Train set properly formatted: ",
        sl.is_standardized(std_annot_train, verbose=False),
    )

    std_annot_test = sl.standardize(
        table=annot_test,
        mapper=map_to_ketos_annot_std,
        trim_table=True,
        labels = ID_list,
        start_labels_at_1=False,
    )
    print(
        "Test set properly formatted: ",
        sl.is_standardized(std_annot_test, verbose=False),
    )

    print(" ")
    print("---- Segmentation + augmentation ----")

    # Segmentation + Augmentation: Training set
    std_annot_train_aug = sl.select(
        annotations=std_annot_train,
        length=params["classif_window_sec"],
        step=params["aug_win_step_sec"],
        min_overlap=params["aug_min_annot_ovlp"],
        center=True,
        keep_freq=True,
        label=list(range(0, len(params["class_labels"]))),
        # discard_outside-True,
        # files=...
    )
    std_annot_train_aug.index.set_names(
        "annot_id", level=1, inplace=True
    )  # correct index name (bug from ketos)
    print(
        "Augmented train set properly formated: ",
        sl.is_standardized(std_annot_train_aug, verbose=False),
    )
    std_annot_train_aug.to_csv(os.path.join(current_out_dir, "train_ketos.csv"))
    print("-Train set: ", len(std_annot_train_aug))

    # Segmentation + Augmentation: Test set
    std_annot_test_aug = sl.select(
        annotations=std_annot_test,
        length=params["classif_window_sec"],
        step=params["aug_win_step_sec"],
        min_overlap=params["aug_min_annot_ovlp"],
        center=True,
    )
    std_annot_test_aug.index.set_names(
        "annot_id", level=1, inplace=True
    )  # correct index name (bug from ketos)
    print(
        "Augmented test set properly formated: ",
        sl.is_standardized(std_annot_test_aug, verbose=False),
    )
    std_annot_test_aug.to_csv(os.path.join(current_out_dir, "test_ketos.csv"))
    print("-Test set: ", len(std_annot_test_aug))



    # Re-adjust number of samples if defined by user
    print('Adjusting number of samples if requested by user')
    for cl_id, cl_labels in enumerate(params["class_labels"]):

        # adjust train set
        if params["max_samples_per_class_train"][cl_id]:
            if cl_id ==0:
                std_annot_train_aug_final = std_annot_train_aug.query("label == " + str(cl_id)).sample(n=params["max_samples_per_class_train"][cl_id])
            else:
                std_annot_train_aug_final = pd.concat([std_annot_train_aug_final, std_annot_train_aug.query("label == " + str(cl_id)).sample(n=params["max_samples_per_class_train"][cl_id])])
        else:
            if cl_id ==0:
                std_annot_train_aug_final = std_annot_train_aug.query("label == " + str(cl_id))
            else:
                std_annot_train_aug_final = pd.concat([std_annot_train_aug_final, std_annot_train_aug.query("label == " + str(cl_id))])


        # adjust train set
        if params["max_samples_per_class_test"][cl_id]:
            if cl_id ==0:
                std_annot_test_aug_final = std_annot_test_aug.query("label == " + str(cl_id)).sample(n=params["max_samples_per_class_test"][cl_id])
            else:
                std_annot_test_aug_final = pd.concat([std_annot_test_aug_final, std_annot_test_aug.query("label == " + str(cl_id)).sample(n=params["max_samples_per_class_test"][cl_id])])
        else:
            if cl_id ==0:
                std_annot_test_aug_final = std_annot_test_aug.query("label == " + str(cl_id))
            else:
                std_annot_test_aug_final = pd.concat([std_annot_test_aug_final, std_annot_test_aug.query("label == " + str(cl_id))])

    print('Train dataset')
    print(std_annot_train_aug_final.pivot_table(values='min_freq',columns='label',aggfunc='count'))
    print('Test dataset')
    print(std_annot_test_aug_final.pivot_table(values='min_freq',columns='label',aggfunc='count'))


    print("---- Creating database of spectrograms ----")
    db_file = os.path.join(current_out_dir, "database.h5")

    print('Training set ===========================')
    dataset_name = "train"
    data_dir = params["data_train_dir"]
    selections = std_annot_train_aug_final
    write_to_database(db_file,
                      dataset_name,
                      selections,
                      data_dir,
                      params,
                      db_mode="w",
                      )

    print('Test set ===========================')
    dataset_name="test"
    data_dir=params["data_test_dir"]
    selections=std_annot_test_aug_final
    write_to_database(db_file,
                      dataset_name,
                      selections,
                      data_dir,
                      params,
                      db_mode="a",
                      )

    print("---- Saving parameters to log file ----")
    ## writes input parameters as csv file for book keeping purposes
    log_file = open(
        os.path.join(current_out_dir, "ketos_database_parameters.csv"), "w"
    )
    writer = csv.writer(log_file)
    for key, value in params.items():
        writer.writerow([key, value])
    log_file.close()

if __name__ == '__main__':
    run()
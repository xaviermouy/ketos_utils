# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 18:00:17 2022

This script takes the annotation logs for the train and test sets and creates
a database of spectrograms that will be used to train/test the neural net

This script takes the annotation csv already formatted in the Ketos format (after augmentation, etc.) that was created
by the script create_ketos_database_v2-db.py.

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
        file_buffer_sec = params["spec_cfg"]['denoiser'][0]['window_duration_sec']
    except:
        file_buffer_sec =0
    pbar = tqdm(total=selections.shape[0],colour="green")
    idxx=0

    #print('Warning just for testimg... needs to be removed')
    #selections.drop(index=('PAM_20171202_010003_000_CH1.wav', 0),inplace=True)

    for annot_index, annot_file in selections.groupby(level=0): # go through each file
        file = annot_file.index[0][0]
        time_first_annot = min(annot_file.start)
        time_last_annot = max(annot_file.end)
        try:
            # load audio data
            #file = annot_row.name[0]
            sound = Sound(os.path.join(data_dir, file))
            # t1 = annot_row.start.min() - file_buffer_sec
            # t2 = annot_row.end.max() + file_buffer_sec
            t1 = time_first_annot - file_buffer_sec
            t2 = time_last_annot + file_buffer_sec

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
            if len(params["spec_cfg"]['denoiser']) > 0:
                print('Denoise')
                spectro.denoise(params["spec_cfg"]['denoiser'][0]['name'],
                                window_duration=params["spec_cfg"]['denoiser'][0]['window_duration_sec'],
                                use_dask=params["spec_cfg"]['denoiser'][0]['use_dask'],
                                dask_chunks=tuple(params["spec_cfg"]['denoiser'][0]['dask_chunks']),
                                inplace=True)

            # readjust time axis of spectro due to t1
            spectro._axis_times = spectro.axis_times + t1

            for annot_index, annot_row in annot_file.droplevel(0).iterrows():

                #time_idx = np.array(abs(spectro.axis_times - clip_start_sec)).argmin()
                time_idx = np.array(abs(spectro.axis_times - annot_row.start)).argmin()
                spectro_matrix = spectro.spectrogram[:, time_idx:time_idx + params['spec_cfg']['win_dur_bins']].T

                # #for sanity check /debugging only
                # plt.figure()
                # plt.imshow(spectro_matrix.T)
                # plt.gca().invert_yaxis()

                if spectro_matrix.shape == (params['spec_cfg']['win_dur_bins'], params['spec_cfg']['win_bw_bins']):
                    table_item['data'] = spectro_matrix
                    table_item['filename'] = file
                    table_item['id'] = int(idxx)
                    table_item['label'] = int(annot_row['label'])
                    idxx += 1
                    # Insert a new record to table
                    table_item.append()
                    # writes to file
                    table.flush()
                else:
                    print('Wrong spectrogram dimensions - data sample ignored.' + 'Spectro shape=' + str(
                        spectro_matrix.shape) + ', expected shape= ' + str(
                        (params['spec_cfg']['win_dur_bins'], params['spec_cfg']['win_bw_bins'])))
                pbar.update(1)
        except:
            print('something went wrong...')
            continue
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

    # ## MW ----------------------------------------------------------------------------
    # params = dict()
    # params["train_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train.csv"
    # params["test_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test.csv"
    # params["data_train_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\train_data"
    # params["data_test_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\test_data"
    # params["out_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\run_3_MW-only"
    # params["spectro_config_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\spec_config.json"
    #
    # params["class_labels"] = []
    # params["class_labels"].append(["NN", "HB", "HK", "HKP", "NNS", "HKPT"])  # class 0
    # params["class_labels"].append(["MW"])  # class 1
    #
    # #params["max_samples_per_class_train"] = [70000, 71207]
    # #params["max_samples_per_class_test"] = [70000, 16435]
    #
    # params["max_samples_per_class_train"] = [10, 30000]
    # params["max_samples_per_class_test"] = [10, 7800]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 5  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 2  # 0.1 #0.1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 1

    # ## MW NOISE ONLY ----------------------------------------------------------------------------
    # params = dict()
    # params["train_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\train.csv"
    # params["test_annot_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\test.csv"
    # params["data_train_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\train_data"
    # params["data_test_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\test_data"
    # params[
    #     "out_dir"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222"
    # params[
    #     "spectro_config_file"] = r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\spec_config.json"
    #
    # params["class_labels"] = []
    # params["class_labels"].append(["NN"])  # class 0
    # #params["class_labels"].append(["MW"])  # class 1
    #
    # params["max_samples_per_class_train"] = [None, None]
    # params["max_samples_per_class_test"] = [None, None]
    #
    # # segemnt duration in sec for the CNN:
    # params["classif_window_sec"] = 5  # 0.4
    # # step between consecutive windows in sec (0: no augmentation):
    # params["aug_win_step_sec"] = 1  # 0.1 #0.1
    # # windows must contain at least x% of the annotation:
    # params["aug_min_annot_ovlp"] = 0.9

    ## MW ----------------------------------------------------------------------------
    params = dict()
    params["train_annot_file"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\train.csv"
    params["test_annot_file"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\test.csv"
    params["data_train_dir"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\train_data"
    params["data_test_dir"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\test_data"
    params[
        "out_dir"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800"
    params[
        "spectro_config_file"] = r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20241127T185315_final_paper\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800\spec_config.json"

    params["class_labels"] = []
    params["class_labels"].append(["NN", "HB", "HK", "HKP", "NNS", "HKPT"])  # class 0
    params["class_labels"].append(["MW"])  # cla

    params["max_samples_per_class_train"] = [50000, 30000]
    params["max_samples_per_class_test"] = [30000, 10000]

    # segemnt duration in sec for the CNN:
    params["classif_window_sec"] = 5  # 0.4
    # step between consecutive windows in sec (0: no augmentation):
    params["aug_win_step_sec"] = 2  # 0.1 #0.1
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

    # load training annotations
    annot_train = pd.read_csv(params["train_annot_file"])
    std_annot_train = sl.standardize(
        table=annot_train,
        trim_table=True,
    )
    sl.is_standardized(std_annot_train, verbose=False)

    # load test annotations
    annot_test = pd.read_csv(params["test_annot_file"])
    std_annot_test = sl.standardize(
        table=annot_test,
        trim_table=True,
    )
    sl.is_standardized(std_annot_test, verbose=False)


    # Re-adjust number of samples if defined by user
    print('Adjusting number of samples if requested by user')
    for cl_id, cl_labels in enumerate(params["class_labels"]):
        # adjust train set
        if params["max_samples_per_class_train"][cl_id]:
            n_samples = min(len(std_annot_train.query("label == " + str(cl_id))),
                            params["max_samples_per_class_train"][cl_id])
            if cl_id ==0:
                std_annot_train_final = std_annot_train.query("label == " + str(cl_id)).sample(n=n_samples)
            else:
                std_annot_train_final = pd.concat([std_annot_train_final, std_annot_train.query("label == " + str(cl_id)).sample(n=n_samples)])
        else:
            if cl_id ==0:
                std_annot_train_final = std_annot_train.query("label == " + str(cl_id))
            else:
                std_annot_train_final = pd.concat([std_annot_train_final, std_annot_train_aug.query("label == " + str(cl_id))])


        # adjust train set
        if params["max_samples_per_class_test"][cl_id]:
            n_samples = min(len(std_annot_test.query("label == " + str(cl_id))),
                            params["max_samples_per_class_test"][cl_id])
            if cl_id ==0:
                std_annot_test_final = std_annot_test.query("label == " + str(cl_id)).sample(n=n_samples)
            else:
                std_annot_test_final = pd.concat([std_annot_test_final, std_annot_test.query("label == " + str(cl_id)).sample(n=n_samples)])
        else:
            if cl_id ==0:
                std_annot_test_final = std_annot_test.query("label == " + str(cl_id))
            else:
                std_annot_test_final = pd.concat([std_annot_test_final, std_annot_test.query("label == " + str(cl_id))])

    print('Train dataset')
    print(std_annot_train_final.pivot_table(values='start',columns='label',aggfunc='count'))
    print('Test dataset')
    print(std_annot_test_final.pivot_table(values='start',columns='label',aggfunc='count'))


    print("---- Creating database of spectrograms ----")
    db_file = os.path.join(current_out_dir, "database.h5")

    print('Training set ===========================')
    dataset_name = "train"
    data_dir = params["data_train_dir"]
    selections = std_annot_train_final
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
    selections=std_annot_test_final
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
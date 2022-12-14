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


# # Input parameters #################################################################
# params = dict()
# params[
#     "train_annot_file"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\train.csv"
# params[
#     "test_annot_file"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\test.csv"

# params[
#     "data_train_dir"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\train_data"
# params[
#     "data_test_dir"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\test_data"
# params[
#     "out_dir"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\ketos_databases\spectro-20s_fft-0.128_step-0.064_fmin-0_fmax-800"
#
# params['positive_labels'] = ['MW']
# params['negative_labels'] = ['HK', 'NN', 'HKPT', 'HKP', 'NNS', 'HB']


# params["class_labels"] = []
# params["class_labels"].append(
#     ["HK", "NN", "HKPT", "HKP", "NNS", "HB"]
# )  # class 0
# params["class_labels"].append(["MW"])  # class 1

# params["classif_window_sec"] = 20.0  # segemnt duration in sec for the CNN
# params[
#     "aug_win_step_sec"
# ] = 3.0  # step between consecutive windows in sec (0: no augmentation)
# params[
#     "aug_min_annot_ovlp"
# ] = 0.75  # windows must contain at least x% of the annotation
# params[
#     "spectro_config_file"
# ] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\minke\20220401T140308\ketos_databases\spectro-20s_fft-0.128_step-0.064_fmin-0_fmax-800\spec_config.json"


params = dict()
params[
    "train_annot_file"
] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\train.csv"
params[
    "test_annot_file"
] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\test.csv"

params[
    "data_train_dir"
] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\train_data"
params[
    "data_test_dir"
] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\test_data"
params[
    "out_dir"
] = r"D:\Detector\ketos\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700_denoised"
params[
    "spectro_config_file"
] = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700_denoised\spec_config.json"


params["class_labels"] = []
params["class_labels"].append(["NN"])  # class 0
params["class_labels"].append(["FS"])  # class 1

# segemnt duration in sec for the CNN:
params["classif_window_sec"] = 10  # 0.4
# step between consecutive windows in sec (0: no augmentation):
params["aug_win_step_sec"] = 0.05
# windows must contain at least x% of the annotation:
params["aug_min_annot_ovlp"] = 0.75


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
    annot_train["label"][annot_train.class_ID.str.contains(pattern)] = int(
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
    annot_test["label"][annot_test.class_ID.str.contains(pattern)] = int(cl_id)
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
    start_labels_at_1=False,
)
print(" ")
print(
    "Train set properly formated: ",
    sl.is_standardized(std_annot_train, verbose=False),
)

std_annot_test = sl.standardize(
    table=annot_test,
    mapper=map_to_ketos_annot_std,
    trim_table=True,
    start_labels_at_1=False,
)
print(
    "Test set properly formated: ",
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

print("---- Creating database of spectrograms ----")
## creatae spectrogram database
db_file = os.path.join(current_out_dir, "database.h5")
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

dbi.create_database(
    output_file=db_file,
    data_dir=params["data_test_dir"],
    dataset_name="test",
    selections=std_annot_test_aug,
    audio_repres=spec_cfg,
)

# db = dbi.open_file("database.h5", 'r')

print("---- Saving parameters to log file ----")
## writes input parameters as csv file for book keeping purposes
log_file = open(
    os.path.join(current_out_dir, "ketos_database_parameters.csv"), "w"
)
writer = csv.writer(log_file)
for key, value in params.items():
    writer.writerow([key, value])
log_file.close()

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:54:31 2022

@author: xavier.mouy
"""
from ketos.neural_networks.resnet import ResNetInterface
import ketos.neural_networks
from ketos.neural_networks.dev_utils.detection import (
    process,
    save_detections,
    merge_overlapping_detections,
)

# # kt_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\batch-size-64_test\ketos_model.ktpb"
# kt_file = (
#     r"C:\Users\xavier.mouy\Desktop\new_MW_model_from_ckpoint\rezipped_model.kt"
# )
# new_model_folder = (
#     r"C:\Users\xavier.mouy\Desktop\new_MW_model_from_ckpoint\tmp"
# )


kt_file = r"D:\NOAA\2022_BC_fish_detector\ketos\dataset_20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\models\FS-400ms-bs16-ep50-norm\kt_model\fs_model.kt"
new_model_folder = r"D:\NOAA\2022_BC_fish_detector\ketos\dataset_20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\models\FS-400ms-bs16-ep50-norm\kt_model\tmp"


model, audio_repr = ketos.neural_networks.load_model_file(
    kt_file,
    new_model_folder,
    overwrite=True,
    load_audio_repr=True,
    replace_top=False,
    diff_n_classes=None,
)
spec_config = audio_repr[0]["spectrogram"]

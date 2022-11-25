# -*- coding: utf-8 -*-
"""
Created on Mon Nov  7 17:08:16 2022

@author: xavier.mouy
"""
import matplotlib.pyplot as plt
from ketos.audio.spectrogram import MagSpectrogram
from ketos.audio.spectrogram import PowerSpectrogram
from ketos.audio.spectrogram import MelSpectrogram
from ketos.data_handling.parsing import load_audio_representation
from ketos.data_handling.parsing import encode_audio_representation

infile = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\train_data\AMAR173.4.20190916T014248Z.wav"
# infile = r".\test_data\Old_Harry_ete_2018-06-16_202635_0.wav"
cfg_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700_denoised\spec_config.json"
# cfg_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\spec_config.json"

# cfg_file = r"C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\fish_bc\20221028T154731\databases\spectro-0.4s_fft-0.064_step-0.00125_fmin-0_fmax-1700\test.json"


spec_config = {
    "rate": 4000,
    "window": 0.064,
    "step": 0.00125,
    "freq_min": 0,
    "freq_max": 1700,
    "window_func": "blackman",
    "type": MagSpectrogram,
    "duration": 0.4,  # 0.25
    "normalize_wav": True,
    "decibel": True,
    # "transforms": [
    #     # {
    #     #     "name": "reduce_tonal_noise",
    #     #     "method": "RUNNING_MEAN",
    #     #     "time_constant": 5,
    #     # },
    #     # {"name": "crop", "start": 0.0, "end": 0.5},
    #     # # {"name": "enhance_signal", "enhancement": 1},
    #     {"name": "normalize", "mean": 0.0, "std": 1.0},
    # ],
    # "waveform_transforms": [{"name": "add_gaussian_noise", "sigma": 0.005}],
}


# spec_config = load_audio_representation(cfg_file, name="spectrogram")


# 143.2
# 83.3
# 320.8
# spectrogram of a 6-s long segment starting 3 s from the beginning of the audio file
spec1 = MagSpectrogram.from_wav(infile, offset=320.8, **spec_config)

# plot
spec1.plot()
plt.show()

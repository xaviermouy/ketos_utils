# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 08:22:48 2022

@author: xavier.mouy
"""

from ecosound.core.measurement import Measurement
import os

in_dir = r"/net/stellwagen/STAFF/Xavier/ketos_minke_detector/NEFSC_MA-RI_COX01/NEFSC_MA-RI_202107_COX01"
sqlite_file = "detections.sqlite"


# Load dataset
print('Loading detections...')
dataset = Measurement()
dataset.from_sqlite(os.path.join(in_dir, sqlite_file))

# Filter
print('Filtering detections...')
dataset.filter("label_class=='MW'", inplace=True)
dataset.filter("confidence>=0.8", inplace=True)

# Create spectrograms and wav files
print('Extracting detection spectrograms...')
out_dir = os.path.join(in_dir, "extracted_detections")
if os.path.isdir(out_dir) == False:
    os.mkdir(out_dir)

dataset.export_spectrograms(
    out_dir,
    time_buffer_sec=5,
    spectro_unit="sec",
    spetro_nfft=0.128,
    spetro_frame=0.128,
    spetro_inc=0.064,
    freq_min_hz=0,
    freq_max_hz=1000,
    sanpling_rate_hz=2000,
    filter_order=8,
    filter_type="iir",
    fig_size=(15, 10),
    deployment_subfolders=False,
    date_subfolders=True,
    file_name_field="uuid",
    #file_name_field="audio_file_name",
    file_prefix_field="confidence",
    channel=0,
    colormap="viridis",
    save_wav=True,
)

# Create daily agggregate.
print('Creating detections daily aggregates...')
daily_counts = dataset.calc_time_aggregate_1D(
    integration_time="1D", resampler="count"
)

daily_counts.to_csv(os.path.join(out_dir, "daily_counts.csv"))

print('Done!')
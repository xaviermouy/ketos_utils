# -*- coding: utf-8 -*-
"""
Created on Mon Sep  11 02:16:25 2022
@author: xavier.mouy
"""
# from ketos.data_handling.database_interface import AudioWriter
# from ketos.audio.spectrogram import MagSpectrogram
#
# aw = AudioWriter('db.h5') #create an audio writer instance
# spec = MagSpectrogram.from_wav('sound.wav', window=0.2, step=0.01) #load a spectrogram
#
# aw.write(spec) #save the spectrogram to the database (by default, the spectrogram is stored under /audio)
# aw.close() #close the database file

import tables
import yaml

# Read db file
path = r'C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\tests\database.h5'
mode ='r'
file = tables.open_file(path, mode)

print(file)
file.root.train.data.shape
idx=0
file.root.train.data[idx]['data']
file.root.train.data[idx]['data'].shape
file.root.train.data[idx]['filename']
file.root.train.data[idx]['id']
file.root.train.data[idx]['label']



## create new hdf file ----------------------------------------------------------------------------
from ecosound.core.audiotools import Sound
from ecosound.core.spectrogram import Spectrogram
from ecosound.detection.detector_builder import DetectorFactory
from ecosound.measurements.measurer_builder import MeasurerFactory
import ecosound.core.tools

infile= r'G:\NOAA\2022_BC_fish_detector\manual_annotations\ONC_delta-node_2014\JASCOAMARHYDROPHONE742_20140913T115518.797Z.wav'
db_file = r'C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\tests\new_database.h5'
config_file = r'C:\Users\xavier.mouy\Documents\GitHub\ketos_utils\tests\config.yaml'
dur_bin =100

# import config file
yaml_file = open(config_file)
config = yaml.load(yaml_file, Loader=yaml.FullLoader)


# load audio data
sound = Sound(infile)
sound.read(channel=config['AUDIO']['channel'], unit='sec')
# Calculates  spectrogram
print('Spectrogram')
spectro = Spectrogram(config['SPECTROGRAM']['frame_sec'],
                      config['SPECTROGRAM']['window_type'],
                      config['SPECTROGRAM']['nfft_sec'],
                      config['SPECTROGRAM']['step_sec'],
                      sound.waveform_sampling_frequency,
                      unit='sec',
                      )
spectro.compute(sound,
                config['SPECTROGRAM']['dB'],
                config['SPECTROGRAM']['use_dask'],
                config['SPECTROGRAM']['dask_chunks'],
                )
# Crop unused frequencies
spectro.crop(frequency_min=config['SPECTROGRAM']['fmin_hz'],
             frequency_max=config['SPECTROGRAM']['fmax_hz'],
             inplace=True,
             )
# Denoise
print('Denoise')
spectro.denoise(config['DENOISER']['denoiser_name'],
                window_duration=config['DENOISER']['window_duration_sec'],
                use_dask=config['DENOISER']['use_dask'],
                dask_chunks=tuple(config['DENOISER']['dask_chunks']),
                inplace=True)

freq_bins = spectro.spectrogram.shape[0]

class SpectroTable(tables.IsDescription):
    data = tables.Float32Col()
    filename = tables.StringCol(100)
    id = tables.UInt32Col()
    label = tables.UInt8Col()
    offset = tables.Float64Col()    # 32-bit integer

# create db file
h5file = tables.open_file(db_file, mode="w", title="Database")

#create group
group = h5file.create_group("/", 'train', '')
# create table
filters = tables.Filters(complevel=1, fletcher32=True)
SpectroTable.columns['data'] = tables.Float32Col(shape=(freq_bins,dur_bin))
table = h5file.create_table(group, 'data', SpectroTable, '', filters=filters, chunkshape=21)

table_item = table.row
for i in range(10):
    idx=i*100
    table_item['data']  = spectro.spectrogram[:,i:i+dur_bin]
    table_item['filename'] = 'JASCOAMARHYDROPHONE742_20140913T115518.797Z.wav'
    table_item['id'] = 1
    table_item['label'] = int(i)
    # Insert a new record to table
    table_item.append()

#writes to file
table.flush()

# close file
h5file.close()
print('yeah')
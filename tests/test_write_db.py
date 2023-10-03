# -*- coding: utf-8 -*-
"""
Created on Mon Sep  11 02:16:25 2022
@author: xavier.mouy
"""
from ketos.data_handling.database_interface import AudioWriter
from ketos.audio.spectrogram import MagSpectrogram

aw = AudioWriter('db.h5') #create an audio writer instance
spec = MagSpectrogram.from_wav('sound.wav', window=0.2, step=0.01) #load a spectrogram

aw.write(spec) #save the spectrogram to the database (by default, the spectrogram is stored under /audio)
aw.close() #close the database file
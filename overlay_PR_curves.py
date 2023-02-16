# -*- coding: utf-8 -*-
"""
Created on Fri Apr 22 09:43:31 2022

@author: xavier.mouy
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df1 = pd.read_csv(r'D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-2s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\results\run_results\performance.csv')
df2 = pd.read_csv(r'D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\models\5s_bs128_ep50\run_results\performance.csv')
df3 = pd.read_csv(r'D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-10s_fft-0.128_step-0.064_fmin-0_fmax-800_no-norm\results\bs128_ep50\run_results\performance.csv')
df4 = pd.read_csv(r'D:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\results\bs128_ep50\run_results\performance.csv')
#df5 = pd.read_csv(r'C:\Users\xavier.mouy\Documents\GitHub\Ketos\minke\20220401T140308\results_PR_test-set\15s_avg-1s\performance.csv')

## Graphs          
plt.plot(df1['P'],df1['R'],'k',label='2 s (res 1)')
plt.plot(df2['P'],df2['R'],'g',label='5 s (res 1)')
plt.plot(df3['P'],df3['R'],'r',label='10 s (res 1)')
plt.plot(df4['P'],df4['R'],'y',label='5 s (res 2)')
#plt.plot(df5['P'],df5['R'],'b',label='20 s')
plt.xlabel('Precision')
plt.ylabel('Recall')
plt.grid()
plt.xticks(np.arange(0, 1+0.02, 0.02))
plt.yticks(np.arange(0, 1+0.02, 0.02))
plt.ylim([0.8,1])
plt.xlim([0.8,1])
#plt.savefig(os.path.join(detec_dir,'PR_curve.png'))
plt.legend()
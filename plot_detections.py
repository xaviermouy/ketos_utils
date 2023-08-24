# -*- coding: utf-8 -*-
"""
Created on Fri Jun 10 17:34:55 2022

This script creat a duirnal plot of detections or annotations

# TO DO
 - Dates/time are a bit off -> need to correct this


@author: xavier.mouy
"""

#import sys
#sys.path.append(r"C:\Users\xavier.mouy\Documents\GitHub\ecosound") # Adds higher directory to python modules path.

from ecosound.core.annotation import Annotation
import ecosound

## inputs #####################################################################
annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2022_DFO_fish_catalog\RCA_Analysis\Ketos_test_results\RCA_In_Dec3_2018_Jan30_201967391491\detections.sqlite'
#annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_CARIBBEAN_201612_MTQ_run1\detections.nc'
#annot_file = r'C:\Users\xavier.mouy\Documents\Projects\2021_Minke_detector\results\NEFSC_GA_201611_CH6\detections.nc'
#annot_file = r'C:\Users\xavier.mouy\Documents\GitHub\minke-whale-dataset\datasets\FRA-NEFSC-CARIBBEAN-201612-MTQ\Annotations_dataset_FRA-NEFSC-CARIBBEAN-201612-MTQ annotations.nc'

integration_time = '1H'
is_binary = False
norm_max = 150
threshold = 0.9
###############################################################################

# Load detections
detec = Annotation()
detec.from_sqlite(annot_file)

# filter to species

# filter to confidence
detec.filter('confidence >= '+ str(threshold), inplace=True)

# plot
detec.heatmap(norm_value=norm_max)

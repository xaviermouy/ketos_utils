import ketos.data_handling.database_interface as dbi
from ketos.data_handling.database_interface import open_file, open_table
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
matplotlib.use('TkAgg')

class SpectroTable(tables.IsDescription):
    data = tables.Float32Col()
    filename = tables.StringCol(100)
    id = tables.UInt32Col()
    label = tables.UInt8Col()
    offset = tables.Float64Col()    # 32-bit integer


## #####################################################################################################################
db_files = []
# db_files.append(r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\noise_MA-RI\database.h5")
# db_files.append(r"G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20240520T195302\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\noise-StantonBank\database.h5")
# db_files.append(r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\run_1_2000samples\noise_only_db\database.h5')
# db_files.append(r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222\run_3_MW-only\database.h5')

db_files.append(r"H:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_full-dataset_and_more-noise\database.h5")

## #####################################################################################################################
## #####################################################################################################################

h5file = tables.open_file(db_files[0], mode='r')

train_table = h5file.get_node('/train/data')
test_table = h5file.get_node('/train/test')

train_files = train_table[:]['filename']
test_files = test_table[:]['filename']

train_



h5file.close()
#
# # create new h5 file
# new_h5file = tables.open_file(new_db_file, mode='w', title="Database")
# # create table
# group1 = new_h5file.create_group("/", 'train', '')  # create group
# group2 = new_h5file.create_group("/", 'test', '')  # create group
# filters = tables.Filters(complevel=1, fletcher32=True)
# SpectroTable.columns['data'] = tables.Float32Col(shape=(data_size[0], data_size[1],))
# table1 = new_h5file.create_table(group1, 'data', SpectroTable, '', filters=filters, chunkshape=21)
# table2 = new_h5file.create_table(group2, 'data', SpectroTable, '', filters=filters, chunkshape=21)
# table1_item = table1.row
# table2_item = table2.row
#
# idxx_1=0
# idxx_2=0
# for db_file in db_files: # go through each db file
#     print(db_file)
#     # load h5 file to add
#     h5file = tables.open_file(db_file, mode='r')
#     train_table = h5file.get_node('/train/data')
#     test_table = h5file.get_node('/test/data')
#     # populate train table
#     for train in train_table:
#         table1_item['data'] = train['data']
#         table1_item['filename'] = train['filename']
#         table1_item['id'] = int(idxx_1)
#         table1_item['label'] = train['label']
#         idxx_1 += 1
#         table1_item.append() # Insert a new record to table
#         table1.flush() # writes to file
#     # populate test table
#     for test in test_table:
#         table2_item['data'] = test['data']
#         table2_item['filename'] = test['filename']
#         table2_item['id'] = int(idxx_2)
#         table2_item['label'] = test['label']
#         idxx_2 += 1
#         table2_item.append() # Insert a new record to table
#         table2.flush() # writes to file
#     # close file
#     h5file.close()
#
# # close new h5 file
# new_h5file.close()
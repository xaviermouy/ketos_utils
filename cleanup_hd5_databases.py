import ketos.data_handling.database_interface as dbi
from ketos.data_handling.database_interface import open_file, open_table
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tables
from ecosound.core.tools import list_files
matplotlib.use('TkAgg')

class SpectroTable(tables.IsDescription):
    data = tables.Float32Col()
    filename = tables.StringCol(100)
    id = tables.UInt32Col()
    label = tables.UInt8Col()
    offset = tables.Float64Col()    # 32-bit integer


## #####################################################################################################################
db_files = []
db_files.append(r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_with_more_noise\database.h5')
spectro_dir = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_with_more_noise\db_spectrograms'
new_db_file = r'G:\NOAA\2022_Minke_whale_detector\ketos\dataset_20221214T163342\databases\spectro-5s_fft-0.128_step-0.04_fimin-0_fmax-800_medianfilt-60s\CNN-222_with_more_noise\db_manually_cleanedup\database.h5'
## #####################################################################################################################
## #####################################################################################################################

# read first file to get dimensions
h5file = tables.open_file(db_files[0], mode='r')
data_size = h5file.get_node('/train/data')[0]['data'].shape
h5file.close()

# create new h5 file
new_h5file = tables.open_file(new_db_file, mode='w', title="Database")
# create table
group1 = new_h5file.create_group("/", 'train', '')  # create group
group2 = new_h5file.create_group("/", 'test', '')  # create group
filters = tables.Filters(complevel=1, fletcher32=True)
SpectroTable.columns['data'] = tables.Float32Col(shape=(data_size[0], data_size[1],))
table1 = new_h5file.create_table(group1, 'data', SpectroTable, '', filters=filters, chunkshape=21)
table2 = new_h5file.create_table(group2, 'data', SpectroTable, '', filters=filters, chunkshape=21)
table1_item = table1.row
table2_item = table2.row

idxx_1=0
idxx_2=0
for db_file in db_files: # go through each db file
    print(db_file)
    # load h5 file to add
    h5file = tables.open_file(db_file, mode='r')

    ## Train table #####################
    train_table = h5file.get_node('/train/data')

    # define indices to keep for each class
    train_labels = train_table[:]["label"]
    train_class_0_indices = [i for i, elem in enumerate(train_labels) if elem == 0]
    train_class_1_indices =  list_files(spectro_dir,suffix='.png')

    # populate train table for class 0
    for train in train_table:
        table1_item['data'] = train['data']
        table1_item['filename'] = train['filename']
        table1_item['id'] = int(idxx_1)
        table1_item['label'] = train['label']
        idxx_1 += 1
        table1_item.append() # Insert a new record to table
        table1.flush() # writes to file


    ## Test table #####################
    # populate test table
    test_table = h5file.get_node('/test/data')
    for test in test_table:
        table2_item['data'] = test['data']
        table2_item['filename'] = test['filename']
        table2_item['id'] = int(idxx_2)
        table2_item['label'] = test['label']
        idxx_2 += 1
        table2_item.append() # Insert a new record to table
        table2.flush() # writes to file
    # close file
    h5file.close()

# close new h5 file
new_h5file.close()
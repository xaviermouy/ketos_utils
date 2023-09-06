from ecosound.core.measurement import Measurement
import datetime
import os


import sqlite3
import pandas as pd
from ecosound.core.tools import filename_to_datetime


in_dir = r"\\stellwagen.nefsc.noaa.gov\stellwagen\STAFF\Xavier\ketos_minke_detector\JASA-EL_paper\results\spectro-5s_fft-0.256_step-0.03_fmin-0_fmax-800_no-norm\FRA-NEFSC-CARIBBEAN-201612-MTQ\detections"
sqlite_file = "detections.sqlite"

# Load dataset
print("Loading detections...")
dataset = Measurement()
dataset.from_sqlite(os.path.join(in_dir, sqlite_file))

# load files processed
conn = sqlite3.connect(os.path.join(in_dir, sqlite_file))
files_list = pd.read_sql_query("SELECT * FROM " + "files_processed",conn)
conn.close()
dates = filename_to_datetime(list(files_list['File_processed']))


# Filter
print("Filtering detections...")
dataset.filter("label_class=='MW'", inplace=True)
dataset.filter("confidence>="+ str(0.9) , inplace=True)

daily_counts = dataset.calc_time_aggregate_1D(
    integration_time="1D",
    resampler="count",
    #start_date= str(min(dates)),
    #end_date= str(max(dates)),
)


daily_counts2 = dataset.calc_time_aggregate_1D(
    integration_time="1D",
    resampler="count",
    start_date= str(min(dates)),
    end_date= str(max(dates)),
)



print('done')
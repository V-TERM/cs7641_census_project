import os

import pandas as pd

path = './tmp/'

filepaths = [os.path.join(path, f) for f in os.listdir(path) if f.endswith('.csv')]
df = pd.concat([pd.read_csv(f) for f in filepaths])
df.to_csv('./tmp/census_acs_data_3.csv', index=False)

from shutil import copyfile

import pandas as pd

from root import DATA_DIR

df = pd.read_csv(DATA_DIR / 'en/ILDC_multi.csv')

# flip label number to align it with our corpus
df.label = df.label.replace({0: 1, 1: 0})
df.label = df.label.replace({0: 'approval', 1: 'dismissal'})  # replace label ids with strings

# copy labels.json file
copyfile(f'{DATA_DIR}/de/labels.json', f'{DATA_DIR}/en/labels.json')

# create temporal splits
df['year'] = df.name.apply(lambda x: int(x.split('_')[0]))  # extract the year from the filename

# drop unused columns
df = df.drop(columns=['split', 'name'])

# the years range from 1947 to 2020
train_df = df[df.year.isin(list(range(1940, 2014)))]
val_df = df[df.year.isin(list(range(2015, 2016)))]
test_df = df[df.year.isin(list(range(2017, 2020)))]

train_df.to_csv(DATA_DIR / 'en/train.csv', index_label='id')
val_df.to_csv(DATA_DIR / 'en/val.csv', index_label='id')
test_df.to_csv(DATA_DIR / 'en/test.csv', index_label='id')

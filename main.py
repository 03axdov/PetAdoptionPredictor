import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd


csv_file = 'datasets/petfinder-mini/petfinder-mini.csv'

"""dataset_url = 'http://storage.googleapis.com/download.tensorflow.org/data/petfinder-mini.zip'
tf.keras.utils.get_file(
    'petfinder-mini.zip',
    origin=dataset_url,
    extract=True,
    cache_dir=".",
    cache_subdir="datasets"
)"""

dataframe = pd.read_csv(csv_file)
# print(dataframe.head())

dataframe["target"] = np.where(dataframe["AdoptionSpeed"]==4, 0, 1)
dataframe = dataframe.drop(columns=['AdoptionSpeed', 'Description'])

train, val, test = np.split(dataframe.sample(frac=1), [int(0.8*len(dataframe)), int(0.9*len(dataframe))])

# print(len(train))
# print(len(val))
# print(len(test))


def df_to_dataset(dataframe, shuffle=True, batch_size=32):
    df = dataframe.copy()
    labels = df.pop('target')
    df = {key: value[..., tf.newaxis] for key, value in dataframe.items()}
    ds = tf.data.Dataset.from_tensor_slices((dict(df), labels))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(dataframe))
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


BATCH_SIZE = 5
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)

[(train_batch, labels_batch)] = train_ds.take(1)

# print(f"Every feature: {list(train_batch.keys())}")
# print('A batch of ages:', train_batch['Age'])
# print('A batch of targets:', labels_batch)

def get_normalization_layer(name, dataset):
    normalizer = layers.Normalization(axis=None)

    feature_ds = dataset.map(lambda x, y: x[name])

    normalizer.adapt(feature_ds)
    return normalizer


photo_count_col = train_batch["PhotoAmt"]
layer = get_normalization_layer('PhotoAmt', train_ds)
# print(layer(photo_count_col))


def get_category_encoding_layer(name, dataset, dtype, max_tokens=None):
    if dtype == 'string':
        index = layers.StringLookup(max_tokens=max_tokens)
    else:
        index = layers.IntegerLookup(max_tokens=max_tokens)

    feature_ds = dataset.map(lambda x, y: x[name])

    index.adapt(feature_ds)

    encoder = layers.CategoryEncoding(num_tokens=index.vocabulary_size())

    return lambda feature: encoder(index(feature))


test_type_col = train_batch["Type"]
test_type_layer = get_category_encoding_layer(name="Type",
                                              dataset=train_ds,
                                              dtype='string')

print(test_type_layer(test_type_col))

test_age_col = train_batch["Age"]
test_age_layer = get_category_encoding_layer(name="Age",
                                             dataset=train_ds,
                                             dtype="int",
                                             max_tokens=5)

print(test_age_layer(test_age_col))

BATCH_SIZE = 256
train_ds = df_to_dataset(train, batch_size=BATCH_SIZE)
val_ds = df_to_dataset(val, shuffle=False, batch_size=BATCH_SIZE)
test_ds = df_to_dataset(test, shuffle=False, batch_size=BATCH_SIZE)

all_inputs = []
from old.imports import *

# set the parameters for dataset
target_size = (32, 32)
channels = 1
BATCH_SIZE = 64


# Normalization helper
def preprocess(x):
    return tf.image.resize(tf.cast(x, tf.float32) / 127.5 - 1, (32, 32))


def get_datasets():
    # Load the MNIST dataset
    train_ds = tfds.load('coco', split="train")

    # Normalize to [-1, 1], shuffle and batch
    train_ds = train_ds.map(preprocess, tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(5000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

    # Return numpy arrays instead of TF tensors while iterating
    return tfds.as_numpy(train_ds)


dataset = get_datasets()
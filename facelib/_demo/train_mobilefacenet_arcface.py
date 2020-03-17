"""Prepare mobilefacenet model with arcface loss."""
#%% [markdown]
## Import modules
import tensorflow as tf
from facelib.dev.layer import DenseL2
from facelib.dev.loss import ArcFace
from facelib.dev.model import mobile_face_net
import numpy as np
#%% [markdown]
## Prepare model architecture
input_shape = (112,112,3)
num_classes = 500
num_features = 128
model = mobile_face_net(input_shape = input_shape)
x = model.output
x = tf.keras.layers.BatchNormalization(name='features')(x)
x = DenseL2(num_classes)(x)
model = tf.keras.models.Model(model.input, x)
#%% [markdown]
## Compile model
af = ArcFace(m=0.4, scale=30)
losses = [af.loss]
optimizer = tf.keras.optimizers.Adam()
metrics = [
    tf.keras.metrics.CategoricalAccuracy(name='acc'),
]
model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
#%% [markdown]
## Print model architecture
model.summary()
#%% [markdown]
## Dummy training data/label generator (batched)
num_epochs = 2
num_data = 5000
batch_size = 64
steps_per_epoch = num_data // batch_size

def data_gen():
    for i in range(steps_per_epoch):
        shape = (batch_size,) + input_shape
        dummy_img = np.random.rand(*shape).astype('float32')
        yield dummy_img # shape: (64,112,112,3)

def label_gen():
    for i in range(steps_per_epoch):
        dummy_label = np.random.randint(
            0, num_classes,
            (batch_size,)
        ) # shape: (64)
        one_hotted_label = np.eye(num_classes, dtype='float32')[dummy_label]
        yield one_hotted_label # shape: (64,500)
#%% [markdown]
## Convert to tfdata
dt_data = tf.data.Dataset.from_generator(
    data_gen,
    output_types = (tf.float32),
    output_shapes = (64, 112, 112, 3)
)
dt_data = dt_data.repeat(num_epochs)
dt_data = dt_data.prefetch(2)

dt_label = tf.data.Dataset.from_generator(
    label_gen,
    output_types = (tf.float32),
    output_shapes = (64, 500)
)
dt_label = dt_label.repeat(num_epochs)
dt_label = dt_label.prefetch(2)

# shape: ((64,112,112,3), (64,500)
dt_train = tf.data.Dataset.zip((dt_data, dt_label))

#%% [markdown]
## Initiate training
model.fit(
    dt_train,
    steps_per_epoch=steps_per_epoch,
    epochs=num_epochs,
    verbose=1,
)
#%%
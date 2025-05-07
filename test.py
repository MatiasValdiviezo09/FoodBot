import tensorflow as tf
print(tf.__version__)  # Debería mostrar 2.16.1

from tensorflow.keras import layers
print(layers.DepthwiseConv2D)
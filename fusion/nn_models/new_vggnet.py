from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer
from fusion.scheduling.batch_size import get_batch_size

"""
VGGNet-16

Simonyan and Zisserman, 2014
"""
batch_size = get_batch_size()
NN = Network('VGG')

NN.set_input_layer(InputLayer(3, 224, nimg=batch_size))

NN.add('conv_1_1', ConvLayer(3, 64, 224, 3, nimg=batch_size))
NN.add('conv_1_2', ConvLayer(64, 64, 224, 3, nimg=batch_size))
NN.add('maxpool_1', PoolingLayer(64, 113, 2, nimg=batch_size))

NN.add('conv_2_1', ConvLayer(64, 128, 113, 3, nimg=batch_size))
NN.add('conv_2_2', ConvLayer(128, 128, 113, 3, nimg=batch_size))
NN.add('maxpool_2', PoolingLayer(128, 57, 2, nimg=batch_size))

NN.add('conv_3_1', ConvLayer(128, 256, 57, 3, nimg=batch_size))
NN.add('conv_3_2', ConvLayer(256, 256, 57, 3, nimg=batch_size))
NN.add('conv_3_3', ConvLayer(256, 256, 57, 3, nimg=batch_size))
NN.add('maxpool_3', PoolingLayer(256, 29, 2, nimg=batch_size))

NN.add('conv_4_1', ConvLayer(256, 512, 29, 3, nimg=batch_size))
NN.add('conv_4_2', ConvLayer(512, 512, 29, 3, nimg=batch_size))
NN.add('conv_4_3', ConvLayer(512, 512, 29, 3, nimg=batch_size))
NN.add('maxpool_4', PoolingLayer(512, 15, 2, nimg=batch_size))

NN.add('conv_5_1', ConvLayer(512, 512, 15, 3, nimg=batch_size))
NN.add('conv_5_2', ConvLayer(512, 512, 15, 3, nimg=batch_size))
NN.add('conv_5_3', ConvLayer(512, 512, 15, 3, nimg=batch_size))
NN.add('maxpool_5', PoolingLayer(512, 8, 2, nimg=batch_size))

NN.add('connect_1', ConvLayer(512, 4096, 1, 8, nimg=batch_size))
NN.add('connect_2', ConvLayer(4096, 4096, 1, 1, nimg=batch_size))
NN.add('connect_3', ConvLayer(4096, 1000, 1, 1, nimg=batch_size))


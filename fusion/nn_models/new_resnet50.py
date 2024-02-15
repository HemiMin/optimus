from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, EltwiseLayer
from fusion.scheduling.batch_size import get_batch_size

"""
ResNet-50

He, Zhang, Ren, and Sun, 2015
"""
batch_size = get_batch_size()
NN = Network('ResNet50')

NN.set_input_layer(InputLayer(3, 256, nimg=batch_size))

NN.add('conv_0', ConvLayer(3, 64, 254, 3, 1, nimg=batch_size))
NN.add('maxpool_0', PoolingLayer(64, 128, 2, 2, nimg=batch_size))

RES_PREV = 'maxpool_0'

for i in range(3):
    NN.add('conv_1_{}_0'.format(i), ConvLayer(64 if i == 0 else 256, 64, 128, 1, nimg=batch_size))
    NN.add('conv_1_{}_1'.format(i), ConvLayer(64, 64, 128, 3, nimg=batch_size))
    NN.add('conv_1_{}_2'.format(i), ConvLayer(64, 256, 128, 1, nimg=batch_size))

    # With residual shortcut.
    NN.add('res_1_{}'.format(i), EltwiseLayer(256, 128, 1, nimg=batch_size),
           prevs=(RES_PREV, 'conv_1_{}_2'.format(i)))
    RES_PREV = 'res_1_{}'.format(i)

for i in range(4):
    NN.add('conv_2_{}_0'.format(i), ConvLayer(256 if i == 0 else 512, 128, 64, 1, nimg=batch_size))
    NN.add('conv_2_{}_1'.format(i), ConvLayer(128, 128, 64, 3, 2 if i == 0 else 1, nimg=batch_size))
    NN.add('conv_2_{}_2'.format(i), ConvLayer(128, 512, 64, 1, nimg=batch_size))

    # With residual shortcut.
    NN.add('res_2_{}'.format(i), EltwiseLayer(512, 64, 1, nimg=batch_size),
           prevs=(RES_PREV, 'conv_2_{}_2'.format(i)))
    RES_PREV = 'res_2_{}'.format(i)

for i in range(6):
    NN.add('conv_3_{}_0'.format(i), ConvLayer(512, 256, 64 if i == 0 else 32, 1, nimg=batch_size))
    NN.add('conv_3_{}_1'.format(i), ConvLayer(256, 256, 32, 3, 2 if i ==0 else 1, nimg=batch_size))

    # With residual shortcut.
    NN.add('res_3_{}'.format(i), EltwiseLayer(512, 32, 1, nimg=batch_size),
           prevs=(RES_PREV, 'conv_3_{}_1'.format(i)))
    RES_PREV = 'res_3_{}'.format(i)

NN.add('conv_4_0_1'.format(i), ConvLayer(512, 512, 16, 3, 2, nimg=batch_size))
NN.add('res_4_0'.format(i), EltwiseLayer(512, 16, 1, nimg=batch_size),
       prevs=(RES_PREV, 'conv_4_0_1'.format(i)))
RES_PREV = 'res_4_0'.format(i)

for i in range(1,3):
    NN.add('conv_4_{}_0'.format(i), ConvLayer(512, 512, 16, 1, nimg=batch_size))
    NN.add('conv_4_{}_1'.format(i), ConvLayer(512, 512, 16, 3, 2 if i ==0 else 1, nimg=batch_size))

    # With residual shortcut.
    NN.add('res_4_{}'.format(i), EltwiseLayer(512, 16, 1, nimg=batch_size),
           prevs=(RES_PREV, 'conv_4_{}_1'.format(i)))
    RES_PREV = 'res_4_{}'.format(i)

NN.add('avg_pool', PoolingLayer(512, 1, 16, nimg=batch_size))

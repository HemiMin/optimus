from fusion.scheduling import Network
from fusion.scheduling import InputLayer, ConvLayer, FCLayer, PoolingLayer, ConcatLayer
from fusion.scheduling.batch_size import get_batch_size

"""
SqueezeNet

Forrest N. Iandola, Song Han, ect., 2016
"""
batch_size = get_batch_size()

NN = Network('SqueezeNet')

NN.set_input_layer(InputLayer(3, 227, nimg=batch_size))
NN.add('conv_0', ConvLayer(3, 64, 113, 3, 2, nimg=batch_size))
NN.add('maxpool_0', PoolingLayer(64, 58, 3, strd=2, nimg=batch_size))


def add_fire(network, id, sfmap, nfmaps_in, nfmaps_s1, nfmaps_e1,
             nfmaps_e3, nfmaps_c):
    fire_id = str(id)
    ''' Add an inception module to the network. '''
    # squeeze1x1
    network.add('conv_'+fire_id+'_0', ConvLayer(nfmaps_in, nfmaps_s1, sfmap, 1, nimg=batch_size))

    prev = 'conv_'+fire_id+'_0'
    # expand1x1
    network.add('conv_'+fire_id+'_1', ConvLayer(nfmaps_s1, nfmaps_e1, sfmap, 1, nimg=batch_size),
                prevs=prev)
    # expand3x3
    network.add('conv_'+fire_id+'_2', ConvLayer(nfmaps_s1, nfmaps_e3, sfmap, 3, nimg=batch_size),
                prevs=prev)

    # concat
    prevs = ('conv_'+fire_id+'_1', 'conv_'+fire_id+'_2')
    network.add('concat_'+fire_id, ConcatLayer(nfmaps_c, sfmap, nimg=batch_size), prevs=prevs)

add_fire(NN, '1', 58, 64, 16, 64, 64, 128)
add_fire(NN, '2', 58, 128, 16, 64, 64, 128)
NN.add('maxpool_1', PoolingLayer(128, 30, 3, strd=2, nimg=batch_size))
add_fire(NN, '3', 30, 128, 32, 128, 128, 256)
add_fire(NN, '4', 30, 256, 32, 128, 128, 256)
NN.add('maxpool_2', PoolingLayer(256, 16, 3, strd=2, nimg=batch_size))
add_fire(NN, '5', 16, 256, 48, 192, 192, 384)
add_fire(NN, '6', 16, 384, 48, 192, 192, 384)
add_fire(NN, '7', 16, 384, 64, 256, 256, 512)
add_fire(NN, '8', 16, 512, 64, 256, 256, 512)
NN.add('conv_9', ConvLayer(512, 1000, 16, 1, 1, nimg=batch_size))
NN.add('avg_pool', PoolingLayer(1000, 1, 16, nimg=batch_size))

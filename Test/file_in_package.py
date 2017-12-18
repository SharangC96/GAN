from Layers import *
import tensorflow as tf

class GAN:

    def __init__(self,train=True):
        pass

    def generate_image(self):
        x=tf.random_normal([1,100],name='input_noise')
        return self.create_generator(x,'Generator',reuse=True)

    def create_generator(self,x,name='Generator',reuse=False):

        with tf.variable_scope(name_or_scope=name,reuse=reuse):

            x = dense_layer(x,4*4*1024,name='project')
            x = batch_norm(x,name='bnLayer1')
            x = tf.reshape(x,[-1,4,4,512],name='reshape')
            x = deconv_layer(x,256,2,2,name='DeconvLayer1')
            x = batch_norm(x, name='bnLayer2')
            x = deconv_layer(x,128,2,2,name='DeconvLayer2')
            x = deconv_layer(x,3,2,2,name='DeconvLayer3',activation=tf.nn.tanh)

        return x

    def create_dicriminator(self,x,name='Dicriminator',reuse=False):

        with tf.variable_scope(name_or_scope=name,reuse=reuse):

            x = conv_layer(x,128,2,2,name='ConvLayer1',activation=tf.nn.leaky_relu)
            x = conv_layer(x, 256, 2, 2, name='ConvLayer2', activation=tf.nn.leaky_relu)
            x = batch_norm(x,'bnLayer1')
            x = conv_layer(x, 512, 2, 2, name='ConvLayer3', activation=tf.nn.leaky_relu)
            x = batch_norm(x, 'bnLayer2')
            x = tf.reshape(x,[-1,4*4*1024],'reshape')
            x = dense_layer(x,2,lambda x:x)

        return x

    def loss(self):


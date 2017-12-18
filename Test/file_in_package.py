from Layers import *
import tensorflow as tf
import numpy as np

class GAN:

    def __init__(self,batch_size=128,train=True):
        self.batch_size=batch_size

    def real_images(self):
        #some queue will return images
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

    def create_discriminator(self,x,name='Dicriminator',reuse=False):

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

        fake_image = self.generate_image()
        discriminator_out_fake = self.create_discriminator(fake_image,reuse=True)
        discriminator_out_real = self.create_discriminator(,reuse=True)

        pred_real=np.stack(np.array([0.0]*self.batch_size),np.array([1.0]*self.batch_size),axis=1).astype(np.float32)
        pred_fake=1-pred_real

        gen_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred_real, logits = discriminator_out_fake))
        disc_loss_real = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred_real),logits=disc_loss_real)
        disc_loss_fake = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=pred_fake),logits=disc_loss_fake)

        return {'gen_loss':gen_loss,'disc_loss':(disc_loss_fake+disc_loss_real)/2.0}

    def train_step_op(self):
        """
        It returns tensorflow ops in form of a list

        [0] for gen_train_step
        [1] for disc_train_step

        run them in a session to begin training

        """

        loss=self.loss()
        optimizer=tf.train.AdamOptimizer(learning_rate=0.0002,beta1=0.5)
        gen_train_step = optimizer.minimize(loss['gen_loss'])
        disc_train_step = optimizer.minimize(loss['disc_loss'])

        return [gen_train_step,disc_train_step]
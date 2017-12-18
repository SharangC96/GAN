from Layers import *
import tensorflow as tf
import numpy as np
from PIL import Image

class GAN:

    def __init__(self, dataset, batch_size=128):

        self.batch_size = batch_size

        self.real_image_queue = self.queue(dataset,name='real_images',shape=[32,32,3])
        self.noise_queue = self.queue(np.random.randn(100,100),name='noise',shape=[100])

        self.fake_image = self.create_generator(self.noise_queue.dequeue_many(self.batch_size),'Generator')

        self.pred_real =  self.create_discriminator(self.real_image_queue.dequeue_many(self.batch_size),'Discriminator')
        self.pred_fake =  self.create_discriminator(self.fake_image,'Discriminator',reuse = True )

        self.gen_loss, self.disc_loss = self.loss()

        self.gen_train_step, self.disc_train_step = self.train_step_op()

        return


    def queue(self,dataset,name,shape):

        with tf.name_scope(name):
            q1=tf.RandomShuffleQueue(capacity=1000,min_after_dequeue=300,dtypes=tf.float32,
                                  shapes=shape)

            enqueue_op1= q1.enqueue_many(vals=dataset)

            numberOfThreads = 5
            qr1 = tf.train.QueueRunner(q1, [enqueue_op1] * numberOfThreads)
            tf.train.add_queue_runner(qr1)

        return q1


    def create_generator(self, x, name='Generator', reuse=False):

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            x = dense_layer(x, 4 * 4 * 512, name='project')
            x = tf.reshape(x, [-1, 4, 4, 512], name='reshape')
            x = deconv_layer(x, 256, 2, 2, name='DeconvLayer1')
            x = batch_norm(x, name='bnLayer1')
            x = deconv_layer(x, 128, 2, 2, name='DeconvLayer2')
            x = batch_norm(x, name='bnLayer2')
            x = deconv_layer(x, 3, 2, 2, name='DeconvLayer3', activation=tf.nn.tanh)

        return x


    def create_discriminator(self, x, name='Discriminator', reuse=False):

        with tf.variable_scope(name_or_scope=name, reuse=reuse):
            x = conv_layer(x, 128, 2, 2, name='ConvLayer1', activation=tf.nn.leaky_relu)
            x = conv_layer(x, 256, 2, 2, name='ConvLayer2', activation=tf.nn.leaky_relu)
            x = batch_norm(x, 'bnLayer1')
            x = conv_layer(x, 512, 2, 2, name='ConvLayer3', activation=tf.nn.leaky_relu)
            x = batch_norm(x, 'bnLayer2')
            x = tf.reshape(x, [-1, 4 * 4 * 512], 'reshape')
            x = dense_layer(x, 2, activation=lambda y: y)

        return x


    def loss(self):

        with tf.name_scope('Generator_loss'):
            pred_real = np.stack((np.array([0.0] * self.batch_size), np.array([1.0] * self.batch_size)), axis=1)
            pred_real=pred_real.astype(np.float32)


            gen_loss = tf.nn.softmax_cross_entropy_with_logits(labels=pred_real, logits=self.pred_fake)
            gen_loss=tf.reduce_mean(gen_loss)

        with tf.name_scope('Discriminator_loss'):
            pred_fake = 1 - pred_real

            disc_loss_real = tf.nn.softmax_cross_entropy_with_logits(labels=pred_real,logits=self.pred_real)
            disc_loss_real = tf.reduce_mean(disc_loss_real)

            disc_loss_fake = tf.nn.softmax_cross_entropy_with_logits(labels=pred_fake,logits=self.pred_fake)
            disc_loss_fake = tf.reduce_mean(disc_loss_fake)

            disc_loss = (disc_loss_fake + disc_loss_real) / 2.0

        return gen_loss,disc_loss

    def train_step_op(self):

        optimizer = tf.train.AdamOptimizer(learning_rate=0.0002, beta1=0.5)

        disc_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Generator')
        gen_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,'Discriminator')

        gen_train_step = optimizer.minimize(self.gen_loss,var_list = gen_variables)
        disc_train_step = optimizer.minimize(self.disc_loss,var_list = disc_variables)

        return [gen_train_step, disc_train_step]

    def write_graph_to_disk(self,path):

        with tf.Session() as sess:
            train_writer = tf.summary.FileWriter(path, sess.graph)
            train_writer.add_graph(sess.graph)

    def start_queues(self,sess):

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        return coord,threads

    def start_training(self, iterations, path, sess, verbose=False, save=True, load=True):

        saver = tf.train.Saver()

        if(load):
            saver.restore(sess,path)

        for i in range(iterations):

            if verbose:
                _,_,gloss,dloss=sess.run([self.disc_train_step,self.gen_train_step,
                                          self.disc_loss,self.gen_loss])

                if(i%100==0):

                    print('After Iteration:',i,'\nGenerator loss is:',gloss,'\nDiscriminator loss is:',dloss,'\n')

            else:
                sess.run([self.disc_train_step, self.gen_train_step])

        if(save):
            saver.save(sess,path)

        return

    def save_sample_png(self,sess,path,name='samples'):

        fake_images=sess.run(self.fake_image)
        fake_images = fake_images*127.5 + 127.5
        fake_images = fake_images.astype(np.uint8)

        new_im = Image.new('RGB', (32*16, 32*8))

        index = 0
        for i in range(0, 32*16, 32):
            for j in range(0, 32*8, 32):

                img = Image.fromarray(fake_images[index], 'RGB')
                new_im.paste(img, (i, j))
                index += 1

        new_im.save(path+name+'.png')

        return
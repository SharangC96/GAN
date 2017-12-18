from Models.GAN import GAN
import tensorflow as tf
# import pandas as pd
import numpy as np

# dataset=pd.read_csv('/home/sharang/Documents/ML/GAN/train.csv').values
# dataset.reshape(-1,32,32,3)

dataset = np.random.uniform(low=-1, high=+1, size=(10000, 32, 32, 3)).astype(np.float32)

G1 = GAN(dataset=dataset, batch_size=128)

G1.write_graph_to_disk('/home/sharang/Documents/ML/GAN/summaries')

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    coord, threads = G1.start_queues(sess)

    G1.start_training(5, path='/home/sharang/Documents/ML/GAN/trained_params', verbose=True, save=True,
                      load=False, sess=sess)

    G1.save_sample_png(sess, '/home/sharang/Documents/ML/GAN/')

    coord.request_stop()
    coord.join(threads)

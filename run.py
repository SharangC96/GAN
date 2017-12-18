from Models.GAN import GAN
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

dataset = unpickle('/home/sharang/Documents/ML/GAN/CIFAR-10/cifar-10-batches-py/data_batch_1')[str.encode('data')].reshape(-1,3,1024)
dataset=np.transpose(dataset,axes=(0,2,1))
dataset=dataset.reshape(-1,32,32,3).astype(np.float32)



G1 = GAN(dataset=dataset, batch_size=128)

# G1.write_graph_to_disk('/home/sharang/Documents/ML/GAN/summaries')

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())
    coord, threads = G1.start_queues(sess)

    G1.start_training(101, path='/home/sharang/Documents/ML/GAN/trained_params/', verbose=True, save=True,
                      load=False,sess=sess)

    plt.imshow((sess.run(G1.fake_image)[0]*127.5+127.5).astype(np.float32))
    plt.show()

    G1.save_sample_png(sess, '/home/sharang/Documents/ML/GAN/')

    coord.request_stop()
    coord.join(threads)

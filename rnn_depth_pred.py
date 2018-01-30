

import tensorflow as tf
from nets_optflow_depth import *
import glob
import numpy as np
import PIL.Image as pil
import matplotlib.pyplot as plt
from PIL import Image
#import cv2

import scipy.misc


flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("output_dir", "", "Dataset directory")
flags.DEFINE_integer("image_height", 480, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 640, "The size of of a sample batch")

FLAGS = flags.FLAGS
FLAGS.resizedheight = 192
FLAGS.resizedwidth = 256
FLAGS.checkpoint_dir="/home/wrlife/project/deeplearning/depth_prediction_rnn/checkpoints/"

def main(_):

    imlist = sorted(glob.glob(FLAGS.dataset_dir+"/*.jpg"))
    #import pdb;pdb.set_trace()

    #==============
    #Build RNN model
    #==============

    with tf.variable_scope("model_rnndepth") as scope:


        inputdata = tf.placeholder(shape=[1, FLAGS.resizedheight, FLAGS.resizedwidth, 7], dtype=tf.float32)

        pred_depth, pred_pose, _ = rnn_depth_net(inputdata,is_training=False)

        saver_pair = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="model_rnndepth"))
        checkpoint_pair = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        

        with tf.Session() as sess:

            sess.run(tf.global_variables_initializer())
            # load weights
            saver_pair.restore(sess, checkpoint_pair)
     
            #The first view is target view so -1
            init_state = np.zeros([FLAGS.resizedheight, FLAGS.resizedwidth,1])


            f, axarr = plt.subplots(1,2)

            for i in range(len(imlist)-1):

                
                I = pil.open(imlist[i])
                I = np.array(I.resize((FLAGS.resizedwidth, FLAGS.resizedheight),pil.ANTIALIAS))/255.0-0.5

                I1 = pil.open(imlist[i+1])
                I1 = np.array(I1.resize((FLAGS.resizedwidth, FLAGS.resizedheight),pil.ANTIALIAS))/255.0-0.5



                _inputdata = np.concatenate([I,I1,init_state],axis=2)

                pred = sess.run(pred_depth,feed_dict={inputdata: _inputdata[np.newaxis,:]})
                init_state = pred[0][0,:,:,:]
                #import pdb;pdb.set_trace()
                output = np.concatenate([np.repeat(pred[0][0,20:FLAGS.resizedheight-20,30:FLAGS.resizedwidth-30,:]*255,3,axis=2), ((I1+0.5)*255)[20:FLAGS.resizedheight-20,30:FLAGS.resizedwidth-30,:].astype(np.uint8)],axis=1)

                scipy.misc.imsave("./output/"+'result%05d' % i+".jpg", output)
                #import pdb;pdb.set_trace()
                # axarr[1].imshow(((I1+0.5)*255).astype(np.uint8))
                # axarr[0].imshow(pred[0][0,20:FLAGS.resizedheight-20,30:FLAGS.resizedwidth-30,0].squeeze(),  cmap='Greys')
                # plt.draw()
                # plt.pause(0.01)

                print(imlist[i])



if __name__ == '__main__':
   tf.app.run()


import tensorflow as tf
from nets_optflow_depth import *

from utils_lr import *
from tfutils import *

from my_losses import *

from tensorflow.contrib.slim.python.slim.learning import train_step



#def rnn_depth_train(dataset):

    #Preprocess data
    # num_views = dataset['num_views']
    # img_target = dataset['img_target']
    # img_sources = dataset['img_sources']
    # gt_depth = dataset['gt_depth']
    # init_state = tf.placeholder(tf.float32,tf.shape(gt_depth))


    # #Define model
    # global_step = tf.Variable(0, 
    #                                name='global_step', 
    #                                trainable=False)
    # incr_global_step = tf.assign(global_step, 
    #                                   global_step+1)

    # state_series = []
    # current_state - init_state
    # for img_source in img_sources:
    #     current_input = tf.concat([img_target,img_source,init_state],axis = 3)
    #     next_state = rnn_depth_net(current_input,is_training=True)
    #     state_series.append(next_state)
    #     current_state = next_state

    # #Compute loss
 #    depth_loss,smooth_loss,loss_depth_sig = compute_loss(,label,global_step_single,FLAGS)
 #    total_loss = depth_loss+smooth_loss+loss_depth_sig

    #Tensorboard plots

    #Session

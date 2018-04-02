from __future__ import division
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python.layers import utils
import numpy as np
import BasicConvLSTMCell
from layer_def import *

# Range of disparity/inverse depth values
DISP_SCALING = 4
MIN_DISP = 0

def resize_like(inputs, ref):
    iH, iW = inputs.get_shape()[1], inputs.get_shape()[2]
    rH, rW = ref.get_shape()[1], ref.get_shape()[2]
    if iH == rH and iW == rW:
        return inputs
    return tf.image.resize_nearest_neighbor(inputs, [rH.value, rW.value])

def pose_exp_net(tgt_image, src_image_stack, do_exp=True, is_training=True):
    inputs = tf.concat([tgt_image, src_image_stack], axis=3)
    batch_norm_params = {'is_training': is_training}
    H = inputs.get_shape()[1].value
    W = inputs.get_shape()[2].value
    num_source = int(src_image_stack.get_shape()[3].value//3)
    with tf.variable_scope('pose_exp_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            normalizer_params=batch_norm_params,
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            # cnv1 to cnv5b are shared between pose and explainability prediction
            cnv1  = slim.conv2d(inputs,16,  [7, 7], stride=2, scope='cnv1')
            cnv2  = slim.conv2d(cnv1, 32,  [5, 5], stride=2, scope='cnv2')
            cnv3  = slim.conv2d(cnv2, 64,  [3, 3], stride=2, scope='cnv3')
            cnv4  = slim.conv2d(cnv3, 128, [3, 3], stride=2, scope='cnv4')
            cnv5  = slim.conv2d(cnv4, 256, [3, 3], stride=2, scope='cnv5')
            # Pose specific layers
            with tf.variable_scope('pose'):
                cnv6  = slim.conv2d(cnv5, 256, [3, 3], stride=2, scope='cnv6')
                cnv7  = slim.conv2d(cnv6, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = 0.01 * tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers
            if do_exp:
                with tf.variable_scope('exp'):
                    upcnv5 = slim.conv2d_transpose(cnv5, 256, [3, 3], stride=2, scope='upcnv5')

                    upcnv4 = slim.conv2d_transpose(upcnv5, 128, [3, 3], stride=2, scope='upcnv4')
                    mask4 = slim.conv2d(upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv3 = slim.conv2d_transpose(upcnv4, 64,  [3, 3], stride=2, scope='upcnv3')
                    mask3 = slim.conv2d(upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                        normalizer_fn=None, activation_fn=None)
                    
                    upcnv2 = slim.conv2d_transpose(upcnv3, 32,  [5, 5], stride=2, scope='upcnv2')
                    mask2 = slim.conv2d(upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                        normalizer_fn=None, activation_fn=None)

                    upcnv1 = slim.conv2d_transpose(upcnv2, 16,  [7, 7], stride=2, scope='upcnv1')
                    mask1 = slim.conv2d(upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                        normalizer_fn=None, activation_fn=None)
            else:
                mask1 = None
                mask2 = None
                mask3 = None
                mask4 = None
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return pose_final, [mask1, mask2, mask3, mask4], end_points

def disp_net(tgt_image, is_training=True):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    with tf.variable_scope('depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')
            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')
            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp2')# + MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp1')# + MIN_DISP
            
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points



def depth_net(tgt_image, is_training=True):
    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = tgt_image.get_shape()[1].value
    W = tgt_image.get_shape()[2].value
    num_source=1
    with tf.variable_scope('depth_cam_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(tgt_image, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
                #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
                pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
            # Exp mask specific layers

            with tf.variable_scope('exp'):
                exp_upcnv5 = slim.conv2d_transpose(cnv5b, 256, [3, 3], stride=2, scope='exp_upcnv5')

                exp_upcnv4 = slim.conv2d_transpose(exp_upcnv5, 128, [3, 3], stride=2, scope='exp_upcnv4')
                mask4 = slim.conv2d(exp_upcnv4, num_source * 2, [3, 3], stride=1, scope='mask4', 
                    normalizer_fn=None, activation_fn=None)

                exp_upcnv3 = slim.conv2d_transpose(exp_upcnv4, 64,  [3, 3], stride=2, scope='exp_upcnv3')
                mask3 = slim.conv2d(exp_upcnv3, num_source * 2, [3, 3], stride=1, scope='mask3', 
                    normalizer_fn=None, activation_fn=None)
                
                # exp_upcnv2 = slim.conv2d_transpose(exp_upcnv3, 32,  [5, 5], stride=2, scope='exp_upcnv2')
                # mask2 = slim.conv2d(exp_upcnv2, num_source * 2, [5, 5], stride=1, scope='mask2', 
                #     normalizer_fn=None, activation_fn=None)

                # exp_upcnv1 = slim.conv2d_transpose(exp_upcnv2, 16,  [7, 7], stride=2, scope='exp_upcnv1')
                # mask1 = slim.conv2d(exp_upcnv1, num_source * 2, [7, 7], stride=1, scope='mask1', 
                #     normalizer_fn=None, activation_fn=None)
            #end_points = utils.convert_collection_to_dict(end_points_collection)
            



            # if is_training:
            #     cnv6b_drop = slim.dropout(cnv6b, 0.5, scope='dropout6')
            # else:
            cnv6b_drop = cnv6b
            cnv7  = slim.conv2d(cnv6b_drop, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            # if is_training:
            #     cnv7b_drop = slim.dropout(cnv7b, 0.5, scope='dropout7')
            # else:
            cnv7b_drop = cnv7b

            



            upcnv7 = slim.conv2d_transpose(cnv7b_drop, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')

            #import pdb;pdb.set_trace()

            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            # disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            # upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            # i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            # icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            # disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
            #      activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            # disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            # upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            # i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            # icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            # disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
            #     activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP
            

            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp3, disp4],pose_final, [mask3,mask4], end_points


def upconvolution_net(resnet_out, is_training=True):
    batch_norm_params = {'is_training': is_training}

    with tf.variable_scope('upconvnet_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.05),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):


            


            upcnv5_i = slim.conv2d(resnet_out[0], 512, [1, 1], stride=1, scope='upcnv5')
            upcnv5 = resize_like(upcnv5_i, resnet_out[1])
            i5_in  = tf.add(upcnv5, resnet_out[1])
            #icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
            # disp5  = DISP_SCALING * slim.conv2d(i5_in, 1,   [3, 3], stride=1, 
            #     activation_fn=tf.sigmoid, normalizer_fn=None, scope='disp5') + MIN_DISP
            
                    
            upcnv4_i = slim.conv2d(i5_in, 256, [1, 1], stride=1, scope='upcnv4')
            upcnv4 = resize_like(upcnv4_i, resnet_out[2])
            i4_in  = tf.add(upcnv4, resnet_out[2])
            #icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
            disp4  = slim.conv2d(i4_in, 1,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp4')

            upcnv3_i = slim.conv2d(i4_in, 64, [1, 1], stride=1, scope='upcnv3')
            upcnv3 = resize_like(upcnv3_i, resnet_out[3])
            i3_in  = tf.add(upcnv3, resnet_out[3])
            i3_in = tf.image.resize_bilinear(i3_in, [np.int(i3_in.get_shape()[1]+1), np.int(i3_in.get_shape()[2]+1)])
            #icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
            disp3  =  slim.conv2d(i3_in, 1,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp3') 

            upcnv2_i = slim.conv2d(i3_in, 64, [1, 1], stride=1, scope='upcnv2')
            upcnv2 = resize_like(upcnv2_i, resnet_out[4])
            i2_in  = tf.add(upcnv2, resnet_out[4])
            #icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')
            disp2  = slim.conv2d(i2_in, 1,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp2')


            upcnv1_i = slim.conv2d(i2_in, 32, [1, 1], stride=1, scope='upcnv1')
            disp1_up = tf.image.resize_bilinear(upcnv1_i, [np.int(disp2.get_shape()[1]*2), np.int(disp2.get_shape()[2]*2)])
            disp1  = slim.conv2d(disp1_up, 1,   [3, 3], stride=1, 
                activation_fn=None, normalizer_fn=None, scope='disp1') 
            
            #import pdb;pdb.set_trace()
            end_points = utils.convert_collection_to_dict(end_points_collection)
            return [disp1, disp2, disp3, disp4], end_points

def rnn_depth_net(current_input,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1
    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            # with tf.variable_scope('pose'):
            #     cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
            #     pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
                

            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')


            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP


            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4],[icnv3,icnv2,icnv1], end_points




def rnn_depth_net_hidst(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1
    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            # with tf.variable_scope('pose'):
            #     cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
            #     pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
                

            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')


            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])


            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([hidden_state[2],upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([hidden_state[1],upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([hidden_state[0], upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(i3_in, 256, [3, 3], stride=2, scope='cam_cnv7')
                cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cam_cnv8')
                pose_pred = slim.conv2d(cam_cnv8, 6, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 1, 6])



            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4],[icnv1,icnv2,icnv3],pose_final, end_points


def rnn_depth_net_hidst_mh(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1
    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            # with tf.variable_scope('pose'):
            #     cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
            #     pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
                

            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([hidden_state[6],upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([hidden_state[5],upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([hidden_state[4],upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')

            i4_in  = tf.concat([hidden_state[3],upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])


            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([hidden_state[2],upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([hidden_state[1],upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([hidden_state[0], upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(i3_in, 256, [3, 3], stride=2, scope='cam_cnv7')
                cam_cnv7_in  = tf.concat([cam_cnv7,i4_in], axis=3)
                cam_cnv8  = slim.conv2d(cam_cnv7_in, 256, [3, 3], stride=2, scope='cam_cnv8')
                cam_cnv8_in  = tf.concat([cam_cnv8,i5_in], axis=3)
                pose_pred = slim.conv2d(cam_cnv8, 6, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 1, 6])



            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4],[icnv1,icnv2,icnv3,icnv4,icnv5,icnv6,icnv7],pose_final, end_points


def rnn_depth_net_hidst_flow(current_input,hidden_state,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1
    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [3, 3], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [3, 3], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [3, 3], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [3, 3], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')
                
            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')


            i4_in  = tf.concat([hidden_state[3],upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP

            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])


            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([hidden_state[2],upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_f  = DISP_SCALING * slim.conv2d(icnv3, 2,   [3, 3], stride=1, 
                 activation_fn=None,normalizer_fn=None, scope='flow3')

            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([hidden_state[1],upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_f  = DISP_SCALING * slim.conv2d(icnv2, 2,   [3, 3], stride=1, 
                 activation_fn=None,normalizer_fn=None, scope='flow2')
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([hidden_state[0], upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP
            disp1_f  = DISP_SCALING * slim.conv2d(icnv1, 2,   [3, 3], stride=1, 
                 activation_fn=None,normalizer_fn=None, scope='flow1')

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(i1_in, 32, [3, 3], stride=2, scope='cam_cnv7')
                cam_cnv8  = slim.conv2d(cam_cnv7, 64, [3, 3], stride=2, scope='cam_cnv8')
                cam_cnv9  = slim.conv2d(cam_cnv8, 128, [3, 3], stride=2, scope='cam_cnv9')
                pose_pred = slim.conv2d(cam_cnv9, 6, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 1, 6])


            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4],[disp1_f, disp2_f, disp3_f],[icnv1,icnv2,icnv3,icnv4],pose_final, end_points


def rnn_depth_net_stack(current_input,is_training=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1
    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            # with tf.variable_scope('pose'):
            #     cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
            #     pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
                

            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')


            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])


            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')
            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            icnv3  = slim.conv2d(i3_in, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            icnv2  = slim.conv2d(i2_in, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([upcnv1, disp2_up], axis=3)
            icnv1  = slim.conv2d(i1_in, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(i3_in, 256, [3, 3], stride=2, scope='cam_cnv7')
                cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cam_cnv8')
                pose_pred = slim.conv2d(cam_cnv8, 6, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 1, 6])



            end_points = utils.convert_collection_to_dict(end_points_collection)

            return [disp1, disp2, disp3, disp4],pose_final, end_points




def rnn_depth_net_LSTM(current_input,hidden,is_training=True, lstm=True):

    batch_norm_params = {'is_training': is_training,'decay':0.99}
    H = current_input.get_shape()[1].value
    W = current_input.get_shape()[2].value
    num_source=1

    if hidden is None:
        hidden3 = None
        hidden2 = None
        hidden1 = None
    else:
        hidden3 = hidden[2]
        hidden2 = hidden[1]
        hidden1 = hidden[0]

    with tf.variable_scope('rnn_depth_net') as sc:
        end_points_collection = sc.original_name_scope + '_end_points'
        with slim.arg_scope([slim.conv2d, slim.conv2d_transpose],
                            normalizer_fn=slim.batch_norm,
                            normalizer_params=batch_norm_params,
                            weights_regularizer=slim.l2_regularizer(0.0004),
                            activation_fn=tf.nn.relu,
                            outputs_collections=end_points_collection):
            cnv1  = slim.conv2d(current_input, 32,  [7, 7], stride=2, scope='cnv1')
            cnv1b = slim.conv2d(cnv1,  32,  [7, 7], stride=1, scope='cnv1b')
            cnv2  = slim.conv2d(cnv1b, 64,  [5, 5], stride=2, scope='cnv2')
            cnv2b = slim.conv2d(cnv2,  64,  [5, 5], stride=1, scope='cnv2b')

            cnv3  = slim.conv2d(cnv2b, 128, [3, 3], stride=2, scope='cnv3')
            cnv3b = slim.conv2d(cnv3,  128, [3, 3], stride=1, scope='cnv3b')
            cnv4  = slim.conv2d(cnv3b, 256, [3, 3], stride=2, scope='cnv4')
            cnv4b = slim.conv2d(cnv4,  256, [3, 3], stride=1, scope='cnv4b')
            cnv5  = slim.conv2d(cnv4b, 512, [3, 3], stride=2, scope='cnv5')
            cnv5b = slim.conv2d(cnv5,  512, [3, 3], stride=1, scope='cnv5b')
            cnv6  = slim.conv2d(cnv5b, 512, [3, 3], stride=2, scope='cnv6')
            cnv6b = slim.conv2d(cnv6,  512, [3, 3], stride=1, scope='cnv6b')

            # with tf.variable_scope('pose'):
            #     cam_cnv7  = slim.conv2d(cnv6b, 256, [3, 3], stride=2, scope='cam_cnv7')
            #     #cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cnv7')
            #     pose_pred = slim.conv2d(cam_cnv7, 6*num_source, [1, 1], scope='pred', 
            #         stride=1, normalizer_fn=None, activation_fn=None)
            #     pose_avg = tf.reduce_mean(pose_pred, [1, 2])
            #     # Empirically we found that scaling by a small constant 
            #     # facilitates training.
            #     pose_final = tf.reshape(pose_avg, [-1, num_source, 6])
                

            cnv7  = slim.conv2d(cnv6b, 512, [3, 3], stride=2, scope='cnv7')
            cnv7b = slim.conv2d(cnv7,  512, [3, 3], stride=1, scope='cnv7b')
            

            upcnv7 = slim.conv2d_transpose(cnv7b, 512, [3, 3], stride=2, scope='upcnv7')
            # There might be dimension mismatch due to uneven down/up-sampling
            upcnv7 = resize_like(upcnv7, cnv6b)
            i7_in  = tf.concat([upcnv7, cnv6b], axis=3)
            icnv7  = slim.conv2d(i7_in, 512, [3, 3], stride=1, scope='icnv7')

            upcnv6 = slim.conv2d_transpose(icnv7, 512, [3, 3], stride=2, scope='upcnv6')
            upcnv6 = resize_like(upcnv6, cnv5b)
            i6_in  = tf.concat([upcnv6, cnv5b], axis=3)
            icnv6  = slim.conv2d(i6_in, 512, [3, 3], stride=1, scope='icnv6')

            upcnv5 = slim.conv2d_transpose(icnv6, 256, [3, 3], stride=2, scope='upcnv5')
            upcnv5 = resize_like(upcnv5, cnv4b)
            i5_in  = tf.concat([upcnv5, cnv4b], axis=3)
            icnv5  = slim.conv2d(i5_in, 256, [3, 3], stride=1, scope='icnv5')

            upcnv4 = slim.conv2d_transpose(icnv5, 128, [3, 3], stride=2, scope='upcnv4')

            i4_in  = tf.concat([upcnv4, cnv3b], axis=3)
            icnv4  = slim.conv2d(i4_in, 128, [3, 3], stride=1, scope='icnv4')
            disp4  = DISP_SCALING * slim.conv2d(icnv4, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp4')# + MIN_DISP
            disp4_up = tf.image.resize_bilinear(disp4, [np.int(H/4), np.int(W/4)])

            #import pdb; pdb.set_trace()



            upcnv3 = slim.conv2d_transpose(icnv4, 64,  [3, 3], stride=2, scope='upcnv3')

            i3_in  = tf.concat([upcnv3, cnv2b, disp4_up], axis=3)
            
            #import pdb;pdb.set_trace()
            if lstm:
                # conv lstm cell 
                with tf.variable_scope('conv_lstm3', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                    cell = BasicConvLSTMCell.BasicConvLSTMCell([upcnv3.get_shape()[1], upcnv3.get_shape()[2]], [3,3], 64)
                    if hidden3 is None:
                        hidden3 = cell.zero_state(5, tf.float32) 

                    y_3, hidden3 = cell(i3_in, hidden3)
            else:
                y_3 = i3_in

            icnv3  = slim.conv2d(y_3, 64,  [3, 3], stride=1, scope='icnv3')
            disp3  = DISP_SCALING * slim.conv2d(icnv3, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp3')# + MIN_DISP
            disp3_up = tf.image.resize_bilinear(disp3, [np.int(H/2), np.int(W/2)])

            upcnv2 = slim.conv2d_transpose(icnv3, 32,  [3, 3], stride=2, scope='upcnv2')
            i2_in  = tf.concat([upcnv2, cnv1b, disp3_up], axis=3)
            
            
            if lstm:
                # conv lstm cell 
                with tf.variable_scope('conv_lstm2', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                    cell = BasicConvLSTMCell.BasicConvLSTMCell([upcnv2.get_shape()[1], upcnv2.get_shape()[2]], [3,3], 32)
                    if hidden2 is None:
                        hidden2 = cell.zero_state(5, tf.float32) 
                    y_2, hidden2 = cell(i2_in, hidden2)

            else:
                y_2 = i2_in

            icnv2  = slim.conv2d(y_2, 32,  [3, 3], stride=1, scope='icnv2')
            disp2  = DISP_SCALING * slim.conv2d(icnv2, 1,   [3, 3], stride=1, 
                 activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp2') #+ MIN_DISP
            disp2_up = tf.image.resize_bilinear(disp2, [H, W])

            upcnv1 = slim.conv2d_transpose(icnv2, 16,  [3, 3], stride=2, scope='upcnv1')
            i1_in  = tf.concat([ upcnv1, disp2_up], axis=3)
            
            if lstm:
                # conv lstm cell 
                with tf.variable_scope('conv_lstm1', initializer = tf.random_uniform_initializer(-.01, 0.1)):
                    cell = BasicConvLSTMCell.BasicConvLSTMCell([upcnv1.get_shape()[1], upcnv1.get_shape()[2]], [3,3], 16)
                    if hidden1 is None:
                        hidden1 = cell.zero_state(5, tf.float32) 
                    y_1, hidden1 = cell(i1_in, hidden1)

            else:
                y_1 = i1_in

            icnv1  = slim.conv2d(y_1, 16,  [3, 3], stride=1, scope='icnv1')
            disp1  = DISP_SCALING * slim.conv2d(icnv1, 1,   [3, 3], stride=1, 
                activation_fn=tf.sigmoid,normalizer_fn=None, scope='disp1') #+ MIN_DISP

            with tf.variable_scope('pose'):
                cam_cnv7  = slim.conv2d(y_3, 256, [3, 3], stride=2, scope='cam_cnv7')
                cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cam_cnv8')
                pose_pred = slim.conv2d(cam_cnv8, 6, [1, 1], scope='pred', 
                    stride=1, normalizer_fn=None, activation_fn=None)
                pose_avg = tf.reduce_mean(pose_pred, [1, 2])
                # Empirically we found that scaling by a small constant 
                # facilitates training.
                pose_final = tf.reshape(pose_avg, [-1, 1, 6])



            end_points = utils.convert_collection_to_dict(end_points_collection)
            if lstm:
                return [disp1, disp2, disp3, disp4],[hidden1,hidden2,hidden3],pose_final, end_points
            else:
                return [disp1, disp2, disp3, disp4],pose_final, end_points






def residual_u_network(inputs, hiddens=None, start_filter_size=32, nr_downsamples=3, nr_residual_per_downsample=2, nonlinearity="concat_elu"):

    
    # set filter size (after each down sample the filter size is doubled)
    filter_size = start_filter_size

    # set nonlinearity
    nonlinearity = set_nonlinearity(nonlinearity)

    # make list of hiddens if None
    if hiddens is None:
        hiddens = (2*nr_downsamples -1)*[None]

    # store for u network connections and new hiddens
    a = []
    hidden_out = []
    xs = []

    # encoding piece
    x_i = inputs#tf.cast(inputs,tf.float32)

    for i in range(nr_downsamples):
        x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, stride=2, name="res_encode_" + str(i) + "_block_0", begin_nonlinearity=False)
        for j in range(nr_residual_per_downsample):
          x_i = res_block(x_i, filter_size=filter_size, nonlinearity=nonlinearity, name="res_encode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
          if i!=nr_downsamples-1:
            a.append(x_i)
        x_i, hidden_new = res_block_lstm(x_i, hiddens[i], name="res_encode_lstm_" + str(i))
        
        hidden_out.append(hidden_new)
        filter_size = filter_size * 2

    # pop off last element to a.
    #a.pop()
    filter_size = int(filter_size / 2)

    with tf.variable_scope('pose'):
        cam_cnv7  = slim.conv2d(x_i, 256, [3, 3], stride=2, scope='cam_cnv7')
        cam_cnv8  = slim.conv2d(cam_cnv7, 256, [3, 3], stride=2, scope='cam_cnv8')
        pose_pred = slim.conv2d(cam_cnv8, 6, [1, 1], scope='pred', 
            stride=1, normalizer_fn=None, activation_fn=None)
        pose_avg = tf.reduce_mean(pose_pred, [1, 2])
        # Empirically we found that scaling by a small constant 
        # facilitates training.
        pose_final = tf.reshape(pose_avg, [-1, 1, 6])

    
    # decoding piece
    for i in range(nr_downsamples - 1):
        filter_size = int(filter_size / 2)
        x_i = transpose_conv_layer(x_i, 4, 2, int(filter_size), "up_conv_" + str(i))

        for j in range(nr_residual_per_downsample):
          x_i = res_block(x_i, a=a.pop(), filter_size=filter_size, nonlinearity=nonlinearity, name="res_decode_" + str(i) + "_block_" + str(j+1), begin_nonlinearity=True)
        x_i, hidden_new = res_block_lstm(x_i, hiddens[i + nr_downsamples], name="res_decode_lstm_" + str(i))
        hidden_out.append(hidden_new)
        #xs.append(x_i)
    #import pdb;pdb.set_trace()
    x_i = transpose_conv_layer(x_i, 4, 2, 1, "up_conv_" + str(nr_downsamples-1))
    #xs.append(x_i)

    return x_i,hidden_out,pose_final#[tf.expand_dims(x_i[:,:,:,0],-1),x_i[:,:,:,1:]], hidden_out,pose_final
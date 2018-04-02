from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np

#from Demon_Data_loader import *
#import .demon.lmbspecialops.python.lmbspecialops as sops
#from .demon.python.depthmotionnet.v2.losses import *
#from tensorflow.contrib.slim.python.slim.learning import train_step
import os
from utils_lr import *
#from .tfutils.python.tfutils import *

import sys
sys.path.append('/data/rui_wang/demon/lmbspecialops/python')
sys.path.append('/data/rui_wang/demon/python/')
sys.path.append('/data/rui_wang/tfutils/python')

from tfutils import *
import lmbspecialops as sops
from depthmotionnet.v2.losses import *


def get_reference_explain_mask( downscaling, FLAGS):
    opt = FLAGS
    tmp = np.array([0,1])
    ref_exp_mask = np.tile(tmp, 
                           (opt.batch_size, 
                            int(opt.resizedheight/(2**downscaling)), 
                            int(opt.resizedwidth/(2**downscaling)), 
                            1))
    ref_exp_mask = tf.constant(ref_exp_mask, dtype=tf.float32)
    return ref_exp_mask



def compute_smooth_loss(pred_disp,):
    def gradient(pred):
        D_dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        D_dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        return D_dx, D_dy
    dx, dy = gradient(pred_disp)
    dx2, dxdy = gradient(dx)
    dydx, dy2 = gradient(dy)
    smoothout = (tf.reduce_mean(tf.abs(dx2)) + tf.reduce_mean(tf.abs(dxdy)) + tf.reduce_mean(tf.abs(dydx)) + tf.reduce_mean(tf.abs(dy2)))
    return smoothout


def compute_exp_reg_loss( pred, ref):
    l = tf.nn.softmax_cross_entropy_with_logits(
        labels=tf.reshape(ref, [-1, 2]),
        logits=tf.reshape(pred, [-1, 2]))
    return tf.reduce_mean(l)


def compute_loss_single_depth(pred_depth,label,global_step,FLAGS):

    #=======
    #Depth loss
    #=======
    depth_loss = 0
    smooth_loss = 0
    loss_depth_sig=0
    epsilon = 0.000001

    global_stepf = tf.to_float(global_step)
    depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))



    for s in range(FLAGS.num_scales):

        #=======
        #Smooth loss
        #=======
        # smooth_loss += FLAGS.smooth_weight/(2**s) * \
        #     compute_smooth_loss(1.0/pred_depth[s])





        curr_label = tf.image.resize_area(label, 
            [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

        if s==0:
            sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
        else:
            sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}
        #import pdb;pdb.set_trace()
        pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)
        gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)
        loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**s)
        

        # diff = sops.replace_nonfinite(curr_label - pred_depth[s])
        # curr_depth_error = tf.abs(diff)
        # depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)
        depth_loss+=pointwise_l2_loss(pred_depth[s],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s))






    return depth_loss,smooth_loss,loss_depth_sig




def compute_loss_pairwise_depth(image_left, image_right,
                                pred_depth_left, pred_poses_right, pred_exp_logits_left,
                                pred_depth_right, pred_poses_left, pred_exp_logits_right,
                                gt_right_cam,
                                intrinsics,
                                label, FLAGS,
                                global_step):

    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):
        depth_loss = 0
        pixel_loss = 0
        smooth_loss = 0
        exp_loss = 0
        consist_loss = 0
        cam_loss = 0
        loss_depth_sig = 0

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []
        optflow_x_all = []
        optflow_y_all = []
        exp_mask_all = []


        #Adaptively changing weights
        #import pdb;pdb.set_trace()
        GT_proj_l2r = pose_vec2mat(gt_right_cam,'eular')
        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        #data_weight = ease_out_quad(global_stepf, 0, FLAGS.data_weight, float(FLAGS.max_steps//3))
        # depth_weight_consist = ease_out_quad_zero(global_stepf, 0, FLAGS.depth_weight_consist, float(FLAGS.max_steps//3))                      


        # proj_l2r = tf.cond(depth_weight_consist > 0, lambda: pose_vec2mat(pred_poses_right[:,0,:],'angleaxis'), lambda: GT_proj_l2r)
        # proj_r2l = tf.cond(depth_weight_consist > 0, lambda: pose_vec2mat(pred_poses_left[:,0,:],'angleaxis'), lambda: tf.matrix_inverse(GT_proj_l2r))
        
        #import pdb;pdb.set_trace()
        proj_l2r = pose_vec2mat(pred_poses_right[:,0,:],'eular')
        #proj_r2l = pose_vec2mat(pred_poses_left[:,0,:],'eular')
 
        # proj_l2r_loss = pose_vec2mat(pred_poses_right[:,0,:],'angleaxis')
        # proj_r2l_loss = pose_vec2mat(pred_poses_left[:,0,:],'angleaxis')






        #=============
        #Compute camera loss
        #=============
        # cam_loss += tf.reduce_mean((gt_right_cam[:,0:3]-pred_poses_right[:,0,:][:,0:3])**2)*FLAGS.cam_weight_tran
        # cam_loss += tf.reduce_mean((gt_right_cam[:,3:]-pred_poses_right[:,0,:][:,3:])**2)*FLAGS.cam_weight_rot
        #import pdb;pdb.set_trace()
        cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,0:3]-proj_l2r[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
        #cam_loss  += tf.reduce_mean((tf.matrix_inverse(GT_proj_l2r)[:,0:3,0:3]-proj_r2l[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
        cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,3]-proj_l2r[:,0:3,3])**2)*FLAGS.cam_weight_tran
        #cam_loss  += tf.reduce_mean((tf.matrix_inverse(GT_proj_l2r)[:,0:3,3]-proj_r2l[:,0:3,3])**2)*FLAGS.cam_weight_tran



        for s in range(2,FLAGS.num_scales):


            curr_label = tf.image.resize_area(label, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
            curr_image_left = tf.image.resize_area(image_left, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 
            curr_image_right = tf.image.resize_area(image_right, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 


            #=======
            #sig depth loss
            #=======
            if s == 2:
                sig_params = {'deltas':[1,2,4], 'weights':[1,1,1], 'epsilon': 0.001}
            else:
                sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}

            pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth_left[s-2], perm=[0,3,1,2]), **sig_params)

            gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

            loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s-2))


            #import pdb;pdb.set_trace()
            #=======
            #Depth loss
            #=======
            # diff = sops.replace_nonfinite(curr_label - pred_depth_left[s-2])
            # curr_depth_error = tf.abs(diff)
            # depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**(s))
            depth_loss+=pointwise_l2_loss(pred_depth_left[s-2],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s-2))



            #=======
            #Pixel loss
            #=======

                        
            curr_proj_image_left, src_pixel_coords_right,wmask_left, warp_depth_right,_= projective_inverse_warp(
                curr_image_right, 
                tf.squeeze(1.0/(curr_label), axis=3),
                proj_l2r,
                intrinsics[:,s,:,:],
                format='matrix'
                )


            # curr_proj_image_right, src_pixel_coords_left,wmask_right, warp_depth_left, _ = projective_inverse_warp(
            #     curr_image_left, 
            #     tf.squeeze(1.0/(pred_depth_right[s-2]), axis=3),
            #     tf.matrix_inverse(GT_proj_l2r),
            #     intrinsics[:,s,:,:],
            #     format='matrix'
            #     )

            # curr_proj_error_left = tf.abs(curr_proj_image_left - curr_image_left)
            # curr_proj_error_right = tf.abs(curr_proj_image_right - curr_image_right)


                                       

            # #===============
            # #exp mask
            # #===============

            # ref_exp_mask = get_reference_explain_mask(s,FLAGS)
            
            # if FLAGS.explain_reg_weight > 0:
            #     curr_exp_logits_left = tf.slice(pred_exp_logits_left[s-2], 
            #                                [0, 0, 0, 0], 
            #                                [-1, -1, -1, 2])
            #     exp_loss += FLAGS.explain_reg_weight * \
            #         compute_exp_reg_loss(curr_exp_logits_left,
            #                                   ref_exp_mask)
            #     curr_exp_left = tf.nn.softmax(curr_exp_logits_left)
            # # Photo-consistency loss weighted by explainability
            # if FLAGS.explain_reg_weight > 0:
            #     pixel_loss += tf.reduce_mean(curr_proj_error_left * \
            #         tf.expand_dims(curr_exp_left[:,:,:,1], -1))*FLAGS.data_weight/(2**(s))

            # exp_mask = tf.expand_dims(curr_exp_left[:,:,:,1], -1)                    
            # exp_mask_all.append(exp_mask)


            
            # if FLAGS.explain_reg_weight > 0:
            #     curr_exp_logits_right = tf.slice(pred_exp_logits_right[s-2], 
            #                                [0, 0, 0, 0], 
            #                                [-1, -1, -1, 2])
            #     exp_loss += FLAGS.explain_reg_weight * \
            #         compute_exp_reg_loss(curr_exp_logits_right,
            #                                   ref_exp_mask)
            #     curr_exp_right = tf.nn.softmax(curr_exp_logits_right)
            # # Photo-consistency loss weighted by explainability
            # if FLAGS.explain_reg_weight > 0:
            #     pixel_loss += tf.reduce_mean(curr_proj_error_right * \
            #         tf.expand_dims(curr_exp_right[:,:,:,1], -1))*FLAGS.data_weight/(2**(s))


            # if not depth_weight_consist is None:
            #     #=======
            #     #left right depth Consistent loss
            #     #=======
            #     right_depth_proj_error=consistent_depth_loss(1.0/(pred_depth_right[s-2]),warp_depth_right, src_pixel_coords_right)
            #     left_depth_proj_error=consistent_depth_loss(1.0/(pred_depth_left[s-2]),warp_depth_left, src_pixel_coords_left)

            #     consist_loss += tf.reduce_mean(right_depth_proj_error*tf.expand_dims(curr_exp_left[:,:,:,1], -1))*depth_weight_consist
            #     consist_loss += tf.reduce_mean(left_depth_proj_error*tf.expand_dims(curr_exp_right[:,:,:,1], -1))*depth_weight_consist



            #========
            #For tensorboard visualize
            #========    
            left_image_all.append(curr_image_left)
            right_image_all.append(curr_image_right)


            proj_image_left_all.append(curr_proj_image_left)
            # proj_image_right_all.append(curr_proj_image_right)

            # proj_error_stack_all.append(curr_proj_error_right)




    return depth_loss, cam_loss, pixel_loss, consist_loss, loss_depth_sig, exp_loss, left_image_all, right_image_all, proj_image_left_all#,proj_image_right_all,proj_error_stack_all





def compute_loss_rnn(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']

        #Adaptively changing weights
        #import pdb;pdb.set_trace()

        GT_tgt_motion = dataset['tgt_motion']

        src_motions = dataset['src_motions']

        label = dataset['tgt_depth']
        src_images = dataset['src_images']
        height = dataset['height']
        width = dataset['width']
        intrinsics = dataset['intrinsics']       
        GT_src_motion =   tf.slice(src_motions,
                              [0,0,0],
                              [-1,-1,4])



        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.concat([GT_tgt_motion, filler], axis=1)
        GT_src_motion = tf.concat([GT_src_motion, filler], axis=1)

        GT_proj_l2r = tf.matmul(GT_src_motion,tf.matrix_inverse(GT_tgt_motion));

        curr_image_right =  tf.slice(src_images,
                                  [0, 0, 0, 0], 
                                  [-1, -1, int(width), -1])

        #import pdb;pdb.set_trace()
        curr_image_right.set_shape([batch_size,height,width,3])
        label.set_shape([batch_size,height,width,1])
        curr_proj_image_left, _,_, _,_= projective_inverse_warp(
            curr_image_right, 
            tf.squeeze(1.0/(label), axis=3),
            GT_proj_l2r,
            intrinsics[:,:,:],
            format='matrix'
            )

        test_gtcam = tf.concat([dataset['tgt_image'],curr_proj_image_left],axis = 2)


        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))

        #label = dataset['tgt_depth']

        src_depths = dataset['src_depths']

        # image_left = dataset['tgt_image']

        # src_images = dataset['src_images']

        # width = dataset['width']





        height = dataset['height']
        width = dataset['width']
        for i in range(num_views-1):

            label =  tf.slice(src_depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0

            pred_depth = state_series[i]

            #import pdb;pdb.set_trace()

            # proj_l2r = pose_vec2mat(pred_pose[:,0,:],'eular')

            # #=============
            # #Compute camera loss
            # #=============
            # cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,0:3]-proj_l2r[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
            # cam_loss  += tf.reduce_mean((GT_proj_l2r[:,0:3,3]-proj_l2r[:,0:3,3])**2)*FLAGS.cam_weight_tran
            # cam_loss_total.append(cam_loss)





            for s in range(FLAGS.num_scales):


                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
                # curr_image_left = tf.image.resize_area(image_left, 
                #     [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))]) 


                #=======
                #sig depth loss
                #=======
                if s==0:
                    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
                else:
                    sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}

                pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

                gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

                loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))

                

                #=======
                #Depth loss
                #=======
                #depth_loss+=pointwise_l2_loss(pred_depth[s],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s-2))
                
                diff = sops.replace_nonfinite(curr_label - pred_depth[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)

            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)




        #rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['proj_left'] = test_gtcam;

    return rnn_loss

    # return depth_loss, cam_loss, pixel_loss, consist_loss, loss_depth_sig, exp_loss, left_image_all, right_image_all, proj_image_left_all#,proj_image_right_all,proj_error_stack_all


def compute_loss_rnn_stack(dataset,pred_depth,pred_pose,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']

        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])



        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)

        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        threeD_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))
        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']


        height = dataset['height']
        width = dataset['width']

        depth_loss = 0
        cam_loss = 0
        loss_depth_sig = 0
        threeD_loss = 0.0


        #Convert label to 3D
        label =  tf.slice(depths,
                              [0, 0, width*(num_views-1), 0], 
                              [-1, -1, int(width), -1])
        image =  tf.slice(images,
                              [0, 0, width*(num_views-1), 0], 
                              [-1, -1, int(width), -1])


        


        proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')
        #=============
        #Compute camera loss
        #=============
        GT_src_motion =  tf.slice(motions,
                              [0,0,4*(num_views-1)],
                              [-1,-1,4])
        GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
        GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)

        # cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,0:3]-proj_r2l[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
        # cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,3]-proj_r2l[:,0:3,3])**2)*FLAGS.cam_weight_tran
        GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])
        
        cam_loss += tf.reduce_mean((GT_angle-pred_pose[:,0,3:])**2)*FLAGS.cam_weight_rot
        cam_loss += tf.reduce_mean((GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3])**2)*FLAGS.cam_weight_tran

        cam_loss_total.append(cam_loss)



        pred_poses.append(pred_pose[:,0,:]);
        gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))


        
        for s in range(FLAGS.num_scales):


            curr_label = tf.image.resize_area(label, 
                [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

            if s==0:
                sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
            else:
                sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}

            pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

            gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

            loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))


            diff = sops.replace_nonfinite(curr_label - pred_depth[s])
            curr_depth_error = tf.abs(diff)
            depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)


        depth_loss_total.append(depth_loss)
        loss_depth_sig_total.append(loss_depth_sig)
        loss_threeD_total.append(threeD_loss)
        #loss_normal_total.append(normal_loss)



        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        #rnn_loss['proj_left'] = test_gtcam;


    return rnn_loss,pred_poses,gt_poses


def compute_loss_rnn_hs(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']



        #Adaptively changing weights
        #import pdb;pdb.set_trace()

        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])

        # src_motions = dataset['src_motions']

        # label = dataset['tgt_depth']
        # src_images = dataset['src_images']
        # height = dataset['height']
        # width = dataset['width']
        # intrinsics = dataset['intrinsics']       
        # GT_src_motion =   tf.slice(src_motions,
        #                       [0,0,0],
        #                       [-1,-1,4])



        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)
        # GT_src_motion = tf.concat([GT_src_motion, filler], axis=1)

        # GT_proj_l2r = tf.matmul(GT_src_motion,tf.matrix_inverse(GT_tgt_motion));

        # curr_image_right =  tf.slice(src_images,
        #                           [0, 0, 0, 0], 
        #                           [-1, -1, int(width), -1])

        # #import pdb;pdb.set_trace()
        # curr_image_right.set_shape([batch_size,height,width,3])
        # label.set_shape([batch_size,height,width,1])
        # curr_proj_image_left, _,_, _,_= projective_inverse_warp(
        #     curr_image_right, 
        #     tf.squeeze(1.0/(label), axis=3),
        #     GT_proj_l2r,
        #     intrinsics[:,:,:],
        #     format='matrix'
        #     )

        # test_gtcam = tf.concat([dataset['tgt_image'],curr_proj_image_left],axis = 2)


        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        threeD_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))
        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']

        # image_left = dataset['tgt_image']

        # src_images = dataset['src_images']

        # width = dataset['width']





        height = dataset['height']
        width = dataset['width']
        for i in range(num_views):

            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0
            threeD_loss = 0.0


            #Convert label to 3D
            label =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            pred_depth,pred_pose = state_series[i]


            proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')

            

            # proj_l2r = pose_vec2mat(pred_pose[:,0,:],'eular')


            #=============
            #Compute camera loss
            #=============
            GT_src_motion =  tf.slice(motions,
                                  [0,0,4*i],
                                  [-1,-1,4])
            GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
            GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)

            # cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,0:3]-proj_r2l[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
            # cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,3]-proj_r2l[:,0:3,3])**2)*FLAGS.cam_weight_tran
            GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])
            
            cam_loss += (FLAGS.cam_weight_rot/batch_size)*l1_loss(GT_angle-pred_pose[:,0,3:],epsilon=epsilon)
            cam_loss += (FLAGS.cam_weight_tran/batch_size)*l1_loss(GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3],epsilon=epsilon)

            cam_loss_total.append(cam_loss)



            pred_poses.append(pred_pose[:,0,:]);
            gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))


            
            for s in range(FLAGS.num_scales):


                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

          #       pixel_coords = meshgrid(int(batch_size), int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s)))

          #       #import pdb;pdb.set_trace()
          #       # Generate ground truth 3D
          #       curr_label.set_shape([batch_size,int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s)),1])
          #       cam_coords = pixel2cam(tf.squeeze((1.0/curr_label), axis=3), pixel_coords, intrinsics/(2**s))

          #       #condition = tf.equal(cam_coords, 0)

          #       cam_coords = tf.reshape(cam_coords, [batch_size, 4, -1])
          #       cam_coords = tf.matmul(GT_proj_r2l, cam_coords)
          #       cam_coords = tf.reshape(cam_coords, [batch_size,-1,int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
          #       cam_coords = tf.transpose(cam_coords, perm=[0, 2, 3, 1])[:,:,:,0:3]
          #       condition = tf.equal(cam_coords, 0)
		        # # Generate pred 3D




                #=======
                #sig depth loss
                #=======

                #=======
                #Smooth loss
                #=======
                #threeD_loss += FLAGS.smooth_weight/(2**s) * \
                #    compute_smooth_loss(1.0/pred_depth[s])


                if s==0:
                    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
                else:
                    sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}

                pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

                gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

                loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))

                

                #=======
                #3D loss
                #=======
                # diff = sops.replace_nonfinite(cam_coords - pred_3D)
                # diff = tf.where(condition, cam_coords, diff)
                # curr_threeD_error = tf.abs(diff)
                # threeD_loss += tf.reduce_mean(curr_threeD_error)*threeD_weight/(2**s)


                #=======
                #depth loss
                #=======
                #depth_loss+=pointwise_l2_loss(pred_depth[s],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s-2))
                #import pdb;pdb.set_trace()
                diff = sops.replace_nonfinite(curr_label - pred_depth[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)



                #==========
                #Normal loss
                #==========

                # normal = sops.depth_to_normals(gt_depth, intrinsics_homo/(2**s), inverse_depth=False)
                # pred_normal = sops.depth_to_normals(pred_depth, intrinsics_homo/(2**s), inverse_depth=False)
                # diff = sops.replace_nonfinite(normal - pred_normal)
                # curr_normal_error = tf.abs(diff)
                # normal_loss += tf.reduce_mean(curr_normal_error)*FLAGS.normal_weight/(2**s)



                # if s==0:
                #     pixel_coords = meshgrid(int(batch_size), int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s)))
                #     cur_pred = pred_depth[s]
                #     cur_pred.set_shape([batch_size,int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s)),1])
                #     pred_3D = pixel2cam(tf.squeeze((1.0/cur_pred), axis=3), pixel_coords, intrinsics/(2**s))
                #     pred_3D = tf.reshape(pred_3D, [batch_size, 4, -1]) 
                #     pred_3D = tf.matmul(proj_r2l, pred_3D)
                #     pred_3D = tf.reshape(pred_3D, [batch_size,-1,int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])
                #     pred_3D = tf.transpose(pred_3D, perm=[0, 2, 3, 1])[:,:,:,0:3]
                #     test.append(pred_3D[0,:,:,:])
                #     testimg.append(image[0,:,:,:])

            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)
            loss_threeD_total.append(threeD_loss)
            #loss_normal_total.append(normal_loss)



        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        #rnn_loss['proj_left'] = test_gtcam;


    return rnn_loss,pred_poses,gt_poses


def compute_loss_rnn_hs_increa(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']

        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])


        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)


        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        threeD_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))


        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']



        height = dataset['height']
        width = dataset['width']
        for i in range(num_views):

            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0
            threeD_loss = 0.0


            #Convert label to 3D
            label =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            pred_depth,pred_pose = state_series[i]


            proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')

        
            #=============
            #Compute camera loss
            #=============
            GT_src_motion =  tf.slice(motions,
                                  [0,0,4*i],
                                  [-1,-1,4])
            GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
            GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)

            GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])
            
            cam_loss += (FLAGS.cam_weight_rot/batch_size)*l1_loss(GT_angle-pred_pose[:,0,3:],epsilon=epsilon)/(4)*(2**(dataset['scale_factor'][0]+dataset['scale_factor'][1]))
            cam_loss += (FLAGS.cam_weight_tran/batch_size)*l1_loss(GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3],epsilon=epsilon)/(4)*(2**(dataset['scale_factor'][0]+dataset['scale_factor'][1]))

            cam_loss_total.append(cam_loss)

            pred_poses.append(pred_pose[:,0,:]);
            gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))


            
            for s in range(FLAGS.num_scales):


                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])


                if s==0:
                    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
                else:
                    sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}

                pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

                gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

                loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))*dataset['scale_factor'][s]

                #=======
                #depth loss
                #=======
                #depth_loss+=pointwise_l2_loss(pred_depth[s],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s-2))
                #import pdb;pdb.set_trace()
                diff = sops.replace_nonfinite(curr_label - pred_depth[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)*dataset['scale_factor'][s]



            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)
            loss_threeD_total.append(threeD_loss)
            #loss_normal_total.append(normal_loss)



        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        #rnn_loss['proj_left'] = test_gtcam;


    return rnn_loss,pred_poses,gt_poses


def compute_loss_rnn_hs_flow(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []
        loss_normal_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']


        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])


        height = dataset['height']
        width = dataset['width']

        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)


        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        flow_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))
        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']



        norm_factor = np.ones((batch_size,4),dtype = np.float32);


        #import pdb;pdb.set_trace()
        norm_intrinsics = tf.concat([tf.expand_dims(intrinsics_homo[:,0,0]/width,-1),
                                    tf.expand_dims(intrinsics_homo[:,1,1]/height,-1),
                                    tf.expand_dims(intrinsics_homo[:,0,2]/width,-1),
                                    tf.expand_dims(intrinsics_homo[:,1,2]/height,-1)],axis=1)


        height = dataset['height']
        width = dataset['width']
        for i in range(num_views):

            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0
            loss_flow_sig = 0
            threeD_loss = 0.0
            normal_loss = 0.0


            #Convert label to 3D
            label =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            pred_depth,pred_pose,pred_normal = state_series[i]


            proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')


            #=============
            #Compute camera loss
            #=============
            GT_src_motion =  tf.slice(motions,
                                  [0,0,4*i],
                                  [-1,-1,4])
            GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
            GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)


            GT_tgt_motion = GT_src_motion

            GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])

            cam_loss += (FLAGS.cam_weight_rot/batch_size)*l1_loss(GT_angle-pred_pose[:,0,3:],epsilon=epsilon)
            cam_loss += (FLAGS.cam_weight_tran/batch_size)*l1_loss(GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3],epsilon=epsilon)

            cam_loss_total.append(cam_loss)

            pred_poses.append(pred_pose[:,0,:])
            gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))
            
            for s in range(FLAGS.num_scales):


                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

                normal = sops.depth_to_flow(tf.transpose(curr_label, perm=[0,3,1,2]), norm_intrinsics, GT_proj_r2l[:,0:3,0:3], GT_proj_r2l[:,0:3,3],rotation_format='matrix', inverse_depth=True, normalize_flow=True, name='DepthToFlow0')
                #normal = sops.depth_to_normals(tf.transpose(curr_label, perm=[0,3,1,2]), norm_intrinsics, inverse_depth=True)
                normal = tf.transpose(normal, perm=[0,2,3,1])

                #import pdb;pdb.set_trace()
                pad = tf.expand_dims(tf.zeros_like(normal[:,:,:,0]),axis=-1)
                normal_pad = tf.concat([normal,pad],axis=3)

                if i==0 and s==0:
                    gt_normals = normal_pad
                elif s==0:
                    gt_normals = tf.concat([gt_normals,normal_pad],axis = 2)

                if i==0 and s==2:
                    get_normals2 = normal_pad
                elif s==2:
                    get_normals2 = tf.concat([get_normals2,normal_pad],axis = 2) 
                                      
                if s==0:
                    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
                else:
                    sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}


                pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

                gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

                loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))

                

                diff = sops.replace_nonfinite(curr_label - pred_depth[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)



                pre_flow_sig = scale_invariant_gradient(tf.transpose(pred_normal[s], perm=[0,3,1,2]), **sig_params)

                gt_flow_sig = scale_invariant_gradient(tf.transpose(normal, perm=[0,3,1,2]), **sig_params)

                loss_flow_sig += flow_sig_weight* pointwise_l2_loss(pre_flow_sig, gt_flow_sig, epsilon=epsilon)/(2**(s))

            
                diff = sops.replace_nonfinite(normal - pred_normal[s])
                curr_normal_error = tf.abs(diff)
                normal_loss += tf.reduce_mean(curr_normal_error)*FLAGS.depth_weight/(2**s)/2              

            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)
            loss_threeD_total.append(loss_flow_sig)
            loss_normal_total.append(normal_loss)



        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        rnn_loss['normal_loss'] = tf.reduce_mean(loss_normal_total)
        #rnn_loss['proj_left'] = test_gtcam;


    return rnn_loss,gt_normals,get_normals2,pred_poses,gt_poses


def compute_loss_rnn_hs_overall(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []
        loss_normal_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']


        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])


        height = dataset['height']
        width = dataset['width']

        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])

        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)


        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        flow_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))
        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']



        norm_factor = np.ones((batch_size,4),dtype = np.float32);


        #import pdb;pdb.set_trace()
        norm_intrinsics = tf.concat([tf.expand_dims(intrinsics_homo[:,0,0]/width,-1),
                                    tf.expand_dims(intrinsics_homo[:,1,1]/height,-1),
                                    tf.expand_dims(intrinsics_homo[:,0,2]/width,-1),
                                    tf.expand_dims(intrinsics_homo[:,1,2]/height,-1)],axis=1)


        height = dataset['height']
        width = dataset['width']
        for i in range(num_views):

            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0
            loss_flow_sig = 0
            threeD_loss = 0.0
            normal_loss = 0.0


            #Convert label to 3D
            label =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            pred_depth,pred_pose,pred_normal = state_series[i]


            proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')


            #=============
            #Compute camera loss
            #=============
            GT_src_motion =  tf.slice(motions,
                                  [0,0,4*i],
                                  [-1,-1,4])
            GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
            GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)

            GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])

            cam_loss += (FLAGS.cam_weight_rot/batch_size)*l1_loss(GT_angle-pred_pose[:,0,3:],epsilon=epsilon)#*tf.reduce_prod(dataset['scale_factor'])
            cam_loss += (FLAGS.cam_weight_tran/batch_size)*l1_loss(GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3],epsilon=epsilon)#*tf.reduce_prod(dataset['scale_factor'])

            cam_loss_total.append(cam_loss)

            pred_poses.append(pred_pose[:,0,:])
            gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))
            
            for s in range(FLAGS.num_scales):


                curr_label = tf.image.resize_area(label, 
                    [int(FLAGS.resizedheight/(2**s)), int(FLAGS.resizedwidth/(2**s))])

                normal = sops.depth_to_flow(tf.transpose(curr_label, perm=[0,3,1,2]), norm_intrinsics, GT_proj_r2l[:,0:3,0:3], GT_proj_r2l[:,0:3,3],rotation_format='matrix', inverse_depth=True, normalize_flow=True, name='DepthToFlow0')
                #normal = sops.depth_to_normals(tf.transpose(curr_label, perm=[0,3,1,2]), norm_intrinsics, inverse_depth=True)
                normal = tf.transpose(normal, perm=[0,2,3,1])

                #import pdb;pdb.set_trace()
                pad = tf.expand_dims(tf.zeros_like(normal[:,:,:,0]),axis=-1)
                normal_pad = tf.concat([normal,pad],axis=3)

                if i==0 and s==0:
                    gt_normals = normal_pad
                elif s==0:
                    gt_normals = tf.concat([gt_normals,normal_pad],axis = 2)

                if i==0 and s==2:
                    get_normals2 = normal_pad
                elif s==2:
                    get_normals2 = tf.concat([get_normals2,normal_pad],axis = 2) 
                                      
                if s==0:
                    sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}
                else:
                    sig_params = {'deltas':[1,2], 'weights':[1,1], 'epsilon': 0.001}


                pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth[s], perm=[0,3,1,2]), **sig_params)

                gt_depth_sig = scale_invariant_gradient(tf.transpose(curr_label, perm=[0,3,1,2]), **sig_params)

                loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)/(2**(s))#*dataset['scale_factor'][s]

                

                diff = sops.replace_nonfinite(curr_label - pred_depth[s])
                curr_depth_error = tf.abs(diff)
                depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight/(2**s)#*dataset['scale_factor'][s]



                pre_flow_sig = scale_invariant_gradient(tf.transpose(pred_normal[s], perm=[0,3,1,2]), **sig_params)

                gt_flow_sig = scale_invariant_gradient(tf.transpose(normal, perm=[0,3,1,2]), **sig_params)

                loss_flow_sig += flow_sig_weight* pointwise_l2_loss(pre_flow_sig, gt_flow_sig, epsilon=epsilon)/(2**(s))#*dataset['scale_factor'][s]

            
                diff = sops.replace_nonfinite(normal - pred_normal[s])
                curr_normal_error = tf.abs(diff)
                normal_loss += tf.reduce_mean(curr_normal_error)*FLAGS.depth_weight/(2**s)/2#*dataset['scale_factor'][s]              

            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)
            loss_threeD_total.append(loss_flow_sig)
            loss_normal_total.append(normal_loss)



        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        rnn_loss['normal_loss'] = tf.reduce_mean(loss_normal_total)
        #rnn_loss['proj_left'] = test_gtcam;


    return rnn_loss,gt_normals,get_normals2,pred_poses,gt_poses



def compute_loss_rnn_lstm(dataset,state_series,global_step,FLAGS):


    #============================================   
    #Specify the loss function:
    #============================================
    with tf.name_scope("compute_loss"):

        rnn_loss = {}


        cam_loss_total = []
        depth_loss_total = []
        loss_depth_sig_total = []
        loss_threeD_total = []

        epsilon = 0.000001

        left_image_all = []
        right_image_all = []

        proj_image_left_all = []
        proj_image_right_all = []

        proj_error_stack_all = []

        test = []
        testimg = []
        testgt = []

        pred_poses = []
        gt_poses = []


        num_views = dataset['num_views']
        batch_size = dataset['batch_size']
        intrinsics = dataset['intrinsics']
        motions = dataset['motions']



        #Adaptively changing weights
        #import pdb;pdb.set_trace()

        GT_tgt_motion = tf.slice(motions,
                                  [0,0,0],
                                  [-1,-1,4])


        filler = tf.constant([0.0, 0.0, 0.0, 1.0], shape=[1, 1, 4])
        filler = tf.tile(filler, [batch_size, 1, 1])


        GT_tgt_motion = tf.matrix_inverse(tf.concat([GT_tgt_motion, filler], axis=1))


        intrinsics_homo = tf.concat([intrinsics, tf.zeros([int(batch_size), 3, 1])], axis=2)
        intrinsics_homo = tf.concat([intrinsics_homo, filler], axis=1)

        global_stepf = tf.to_float(global_step)
        depth_sig_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_sig_weight, float(FLAGS.max_steps//3))
        threeD_weight = ease_out_quad(global_stepf, 0, FLAGS.depth_weight, float(FLAGS.max_steps//3))
        #label = dataset['tgt_depth']

        depths = dataset['depths']
        images = dataset['images']


        height = dataset['height']
        width = dataset['width']
        for i in range(num_views):

            depth_loss = 0
            cam_loss = 0
            loss_depth_sig = 0
            threeD_loss = 0.0


            #Convert label to 3D
            label =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])


            pred_depth,pred_pose = state_series[i]


            proj_r2l = pose_vec2mat(pred_pose[:,0,:],'eular')

            

            #=============
            #Compute camera loss
            #=============
            GT_src_motion =  tf.slice(motions,
                                  [0,0,4*i],
                                  [-1,-1,4])

            GT_src_motion = tf.matrix_inverse(tf.concat([GT_src_motion, filler], axis=1))
            GT_proj_r2l = tf.matmul(tf.matrix_inverse(GT_tgt_motion),GT_src_motion)

            #Get eularangle
            GT_angle = rotationMatrixToEulerAngles(GT_proj_r2l[:,0:3,0:3])

            #cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,0:3]-proj_r2l[:,0:3,0:3])**2)*FLAGS.cam_weight_rot
            #cam_loss  += tf.reduce_mean((GT_proj_r2l[:,0:3,3]-proj_r2l[:,0:3,3])**2)*FLAGS.cam_weight_tran

            cam_loss += tf.reduce_mean((GT_angle-pred_pose[:,0,3:])**2)*FLAGS.cam_weight_rot
            cam_loss += tf.reduce_mean((GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3])**2)*FLAGS.cam_weight_tran

            # cam_loss += (FLAGS.cam_weight_rot/batch_size)*l1_loss(GT_angle-pred_pose[:,0,3:],epsilon=epsilon)
            # cam_loss += (FLAGS.cam_weight_tran/batch_size)*l1_loss(GT_proj_r2l[:,0:3,3]-pred_pose[:,0,0:3],epsilon=epsilon)


            cam_loss_total.append(cam_loss)

            pred_poses.append(pred_pose[:,0,:]);
            gt_poses.append(tf.concat([GT_proj_r2l[:,0:3,3],GT_angle],axis=1))

            #import pdb;pdb.set_trace()
            sig_params = {'deltas':[1,2,4,8,16], 'weights':[1,1,1,1,1], 'epsilon': 0.001}

            pre_depth_sig = scale_invariant_gradient(tf.transpose(pred_depth, perm=[0,3,1,2]), **sig_params)

            gt_depth_sig = scale_invariant_gradient(tf.transpose(label, perm=[0,3,1,2]), **sig_params)

            loss_depth_sig += depth_sig_weight* pointwise_l2_loss(pre_depth_sig, gt_depth_sig, epsilon=epsilon)

            

                #=======
                #3D loss
                #=======
                # diff = sops.replace_nonfinite(cam_coords - pred_3D)
                # diff = tf.where(condition, cam_coords, diff)
                # curr_threeD_error = tf.abs(diff)
                # threeD_loss += tf.reduce_mean(curr_threeD_error)*threeD_weight/(2**s)


                #=======
                #depth loss
                #=======
                #depth_loss+=pointwise_l2_loss(pred_depth[s],curr_label ,epsilon=epsilon)*FLAGS.depth_weight/(2**(s-2))
                #import pdb;pdb.set_trace()
            diff = sops.replace_nonfinite(label - pred_depth)
            curr_depth_error = tf.abs(diff)
            depth_loss += tf.reduce_mean(curr_depth_error)*FLAGS.depth_weight

            #depth_loss += FLAGS.depth_weight* pointwise_l2_loss(tf.transpose(label, perm=[0,3,1,2]), tf.transpose(pred_depth, perm=[0,3,1,2]), epsilon=epsilon)


                
            # pixel_coords = meshgrid(int(batch_size), int(FLAGS.resizedheight), int(FLAGS.resizedwidth))
            # cur_pred = pred_depth
            # cur_pred.set_shape([batch_size,int(FLAGS.resizedheight), int(FLAGS.resizedwidth),1])
            # pred_3D = pixel2cam(tf.squeeze((1.0/cur_pred), axis=3), pixel_coords, intrinsics)
            # pred_3D = tf.reshape(pred_3D, [batch_size, 4, -1]) 
            # pred_3D = tf.matmul(proj_r2l, pred_3D)
            # pred_3D = tf.reshape(pred_3D, [batch_size,-1,int(FLAGS.resizedheight), int(FLAGS.resizedwidth)])
            # pred_3D = tf.transpose(pred_3D, perm=[0, 2, 3, 1])[:,:,:,0:3]
            # test.append(pred_3D[0,:,:,:])
            # testimg.append(image[0,:,:,:])

            
            # # Generate ground truth 3D
            # label.set_shape([batch_size,int(FLAGS.resizedheight), int(FLAGS.resizedwidth),1])
            # cam_coords = pixel2cam(tf.squeeze((1.0/label), axis=3), pixel_coords, intrinsics)
            # cam_coords = tf.reshape(cam_coords, [batch_size, 4, -1])
            # cam_coords = tf.matmul(GT_proj_r2l, cam_coords)
            # cam_coords = tf.reshape(cam_coords, [batch_size,-1,int(FLAGS.resizedheight), int(FLAGS.resizedwidth)])
            # cam_coords = tf.transpose(cam_coords, perm=[0, 2, 3, 1])[:,:,:,0:3]
            # testgt.append(cam_coords[0,:,:,:])
            

            depth_loss_total.append(depth_loss)
            loss_depth_sig_total.append(loss_depth_sig)
            loss_threeD_total.append(threeD_loss)




        rnn_loss['cam_loss'] = tf.reduce_mean(cam_loss_total)
        rnn_loss['depth_loss'] = tf.reduce_mean(depth_loss_total)
        rnn_loss['loss_depth_sig'] = tf.reduce_mean(loss_depth_sig_total)
        rnn_loss['threeD_loss'] = tf.reduce_mean(loss_threeD_total)
        #rnn_loss['proj_left'] = test_gtcam;
        test = depth_sig_weight

    return rnn_loss,test,pred_poses,gt_poses

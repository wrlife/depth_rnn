from __future__ import division


import pprint
import random
import numpy as np

from imageselect_Dataloader_optflow_dim11_seq import DataLoader
#from Demon_Data_loader import *

from rnn_depth_train_seq import *
import os


import PIL.Image as pil
from PIL import Image

flags = tf.app.flags
flags.DEFINE_string("dataset_dir", "", "Dataset directory")
flags.DEFINE_string("validate_dir", "./validation", "Dataset directory")
flags.DEFINE_string("checkpoint_dir", "./checkpoints/", "Directory name to save the checkpoints")
flags.DEFINE_string("checkpoint_dir_single", "./checkpoints_single/", "Directory name to save the checkpoints")
flags.DEFINE_integer("image_height", 192, "The size of of a sample batch")
flags.DEFINE_integer("image_width", 256, "The size of of a sample batch")
flags.DEFINE_float("learning_rate", 0.0002, "Learning rate of for adam")
flags.DEFINE_float("beta1", 0.9, "Momentum term of adam")
flags.DEFINE_integer("batch_size", 5, "The size of of a sample batch")
flags.DEFINE_string("pretrain_weight_dir", "./pretrained", "Directory name to pretrained weights")
flags.DEFINE_integer("validation_check", 100, "Directory name to pretrained weights")
flags.DEFINE_integer("num_sources", 2, "number of sources")


flags.DEFINE_boolean("continue_train", False, "Continue training from previous checkpoint")
flags.DEFINE_boolean("continue_train_single", False, "Continue training from previous checkpoint")
flags.DEFINE_integer("save_latest_freq", 5000, \
    "Save the latest model every save_latest_freq iterations (overwrites the previous latest model)")

flags.DEFINE_integer("max_steps_single", 450001, "Maximum number of training iterations")
flags.DEFINE_integer("max_steps", 300001, "Maximum number of training iterations")

flags.DEFINE_integer("summary_freq", 100, "Logging every log_freq iterations")
flags.DEFINE_string("init_checkpoint_file", None, "Specific checkpoint file to initialize from")



FLAGS = flags.FLAGS

FLAGS.num_scales = 4
FLAGS.smooth_weight = 50
FLAGS.data_weight = 0

FLAGS.optflow_weight = 0
FLAGS.depth_weight = 500
FLAGS.depth_weight_consist = 10
FLAGS.depth_sig_weight = 1000
FLAGS.explain_reg_weight = 1
FLAGS.cam_weight_rot = 160
FLAGS.cam_weight_tran = 10


FLAGS.resizedheight = 192
FLAGS.resizedwidth = 256




def deprocess_image(image,FLAGS):

    image = image*255.0
    return tf.image.convert_image_dtype(image, dtype=tf.uint8)


####################
#Validate
####################
def sculpt_validation(valleft,valright):

    #Load image and label

    with tf.variable_scope("model_pairdepth") as scope:

        scope.reuse_variables()
        inputdata = tf.concat([valright,valleft], axis=3)

        pred_valid, pred_poses_right, pred_exp_logits_left, depth_net_endpoints_left = depth_net(inputdata,                                                    
                                                                                            is_training=False)
    return pred_valid





def single_depth_training(label,FLAGS,image_left,pred_depth_left,saver_pair,checkpoint_pair):

    #============================================
    #First train a depth and camera estimation from image pair
    #============================================


    with tf.variable_scope("model_singledepth") as scope:

        
        #estimate depth and optical flow from both left and right image
        batch, height, width, _ = image_left.get_shape().as_list()
        
        global_step_single = tf.Variable(0, 
                                       name='global_step_single', 
                                       trainable=False)
        incr_global_step = tf.assign(global_step_single, 
                                          global_step_single+1)
        #upsample
        #import pdb;pdb.set_trace()
        pred_depth_left_up = tf.image.resize_nearest_neighbor(pred_depth_left, [height, width])

        inputdata = tf.concat([pred_depth_left_up,image_left], axis=3)
        pred_depth_single_left,depth_endpoints = disp_net(inputdata,                                                 
                                              is_training=True)


        depth_loss,smooth_loss,loss_depth_sig = compute_loss_single_depth(pred_depth_single_left,label,global_step_single,FLAGS)
        total_loss = depth_loss+smooth_loss+loss_depth_sig

            
            


    with tf.name_scope("train_op"):
        tf.summary.scalar('losses/total_loss', total_loss)
        tf.summary.scalar('losses/smooth_loss', smooth_loss)
        tf.summary.scalar('losses/depth_loss', depth_loss)
        tf.summary.scalar('losses/loss_depth_sig', loss_depth_sig)


        tf.summary.histogram("scale_depth", sops.replace_nonfinite(label))
        tf.summary.histogram('scale%d_pred_depth_single_left' % 0,
            pred_depth_single_left[0])

        tf.summary.image('left_image' , \
                         image_left)
               
        tf.summary.image('gt_depth' , \
                         label)
        tf.summary.image('left_image' , \
                         pred_depth_single_left[0])
        tf.summary.image('left_image_pair' , \
                         pred_depth_left)        



        # Specify the optimization scheme:
        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step_single,
                                           10000, 0.9, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1)

        # create_train_op that ensures that when we evaluate it to get the loss,
        # the update_ops are done and the gradient updates are computed.
        train_op = slim.learning.create_train_op(total_loss, optimizer)
        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,scope="model_singledepth"))


        with tf.Session() as sess:
            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir_single + '/sum',
                                                  sess.graph)
            tf.initialize_all_variables().run()
            tf.initialize_local_variables().run()
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            
            if FLAGS.continue_train_single:
                if FLAGS.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir_single)
                else:
                    checkpoint = FLAGS.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                saver.restore(sess, checkpoint)


            #restore pairwise depth parameters
            
            saver_pair.restore(sess, checkpoint_pair)

            for step in range(1, FLAGS.max_steps_single):
                #print("steps %d" % (step))
                fetches = {
                    "train": train_op,
                    "global_step": global_step_single,
                    "incr_global_step": incr_global_step
                }

                if step % FLAGS.summary_freq == 0:
                    fetches["loss"] = total_loss
                    fetches["summary"] = merged
                    fetches["learn_rate"] = learning_rate


                results = sess.run(fetches)
                gs = results["global_step_single"]

                if step % FLAGS.summary_freq == 0:
                    train_writer.add_summary(results["summary"], gs)

                    print("steps: %d === loss: %.3f" \
                            % (gs,
                                results["loss"]))
                    print("learning rate %f" % (results["learn_rate"]))



                if step % FLAGS.save_latest_freq == 0:
                    saver.save(sess, FLAGS.checkpoint_dir_single+'/model', global_step=gs)

            coord.request_stop()
            coord.join(threads)



def pairwise_depth_train(image_left,image_right,label,gt_right_cam,intrinsics,FLAGS):

    #============================================
    #Then train a cam pose estimator and refined depth from pair
    #============================================
    with tf.variable_scope("model_pairdepth") as scope:
        #===========================
        #Using left right to predict
        #===========================
        inputdata = tf.concat([image_left, image_right], axis=3)

        global_step = tf.Variable(0,
                                       name='global_step',
                                       trainable=False)
        incr_global_step = tf.assign(global_step,
                                          global_step+1)

        pred_depth_left, pred_poses_right, pred_exp_logits_left, depth_net_endpoints_left = depth_net(inputdata,                                                     
                                                                                            is_training=True)  

    #===============
    #Estimate depth
    #===============
    depth_loss, cam_loss, pixel_loss, consist_loss, loss_depth_sig, exp_loss, left_image_all, right_image_all, proj_image_left_all = compute_loss_pairwise_depth(
                                                                                                                                                                    image_left, image_right,
                                                                                                                                                                    pred_depth_left, pred_poses_right, pred_exp_logits_left,
                                                                                                                                                                    gt_right_cam,
                                                                                                                                                                    intrinsics,
                                                                                                                                                                    label, FLAGS,
                                                                                                                                                                    global_step)        

    total_loss = depth_loss +cam_loss + pixel_loss+ consist_loss+loss_depth_sig +exp_loss


    ####################
    #Validate
    ####################
    valleft = tf.placeholder(shape=[1, FLAGS.resizedheight, FLAGS.resizedwidth, 3], dtype=tf.float32)
    valright = tf.placeholder(shape=[1, FLAGS.resizedheight, FLAGS.resizedwidth, 3], dtype=tf.float32)
    pred_valid = sculpt_validation(valright,valleft)



    #============================================
    #Start training
    #============================================
    with tf.name_scope("train_op"):
        tf.summary.scalar('losses/total_loss', total_loss)
        #tf.summary.scalar('losses/smooth_loss', smooth_loss)
        tf.summary.scalar('losses/depth_loss', depth_loss)
        tf.summary.scalar('losses/pixel_loss', pixel_loss)
        tf.summary.scalar('losses/cam_loss', cam_loss)
        tf.summary.scalar('losses/loss_depth_sig', loss_depth_sig)
        tf.summary.scalar('losses/consist_loss', consist_loss)
        tf.summary.scalar('losses/exp_loss', exp_loss)

        tf.summary.histogram("scale_depth", sops.replace_nonfinite(label))
        tf.summary.histogram('scale%d_pred_depth_single_left' % 0,
            pred_depth_left[0])

        tf.summary.image('validate_pred',
            1.0/pred_valid[0])
        tf.summary.image('validate_image',
            valright*255+0.5)

        for s in range(0,FLAGS.num_scales-2):
            
            tf.summary.image('scale%d_left_image' % s, \
                             left_image_all[s])
            tf.summary.image('scale%d_right_image' % s, \
                             right_image_all[s])
            tf.summary.image('scale%d_projected_image_left' % s, \
                             proj_image_left_all[s])
               

            tf.summary.image('scale%d_pred_depth_left' % s,
                pred_depth_left[s])


        learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                           10000, 0.96, staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1)

        train_op = slim.learning.create_train_op(total_loss, optimizer)


        saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="model_pairdepth"))

        with tf.Session() as sess:

            merged = tf.summary.merge_all()
            train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/sum',
                                                  sess.graph)

            tf.initialize_all_variables().run()
            tf.initialize_local_variables().run()

            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=sess,coord=coord)
            
            if FLAGS.continue_train:
                if FLAGS.init_checkpoint_file is None:
                    checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
                else:
                    checkpoint = FLAGS.init_checkpoint_file
                print("Resume training from previous checkpoint: %s" % checkpoint)
                saver.restore(sess, checkpoint)

            for step in range(1, FLAGS.max_steps):
                #print("steps %d" % (step))
                fetches = {
                    "train": train_op,
                    "global_step": global_step,
                    "incr_global_step": incr_global_step
                }

                if step % FLAGS.summary_freq == 0:
                    fetches["loss"] = total_loss
                    fetches["summary"] = merged
                    fetches["GT_cam"] = gt_right_cam
                    fetches["est_cam"] = pred_poses_right
                    fetches["learn_rate"] = learning_rate

                feed_dict = {}

                if step % FLAGS.summary_freq == 0:

                    ####################
                    #Validate
                    ####################
                    I = Image.open("/home/wrlife/project/deeplearning/depth_prediction/data/sculpture1.png")
                    I1 = Image.open("/home/wrlife/project/deeplearning/depth_prediction/data/sculpture2.png")
                    I = I.resize((FLAGS.resizedwidth, FLAGS.resizedheight),pil.ANTIALIAS)
                    I1 = I1.resize((FLAGS.resizedwidth, FLAGS.resizedheight),pil.ANTIALIAS)
                    I = np.array(I).astype(np.float32)/255.0 -0.5
                    I1 = np.array(I1).astype(np.float32)/255.0 -0.5

                    fetches["validate"] = pred_valid

                    feed_dict={valleft: I[np.newaxis,:],
                             valright: I1[np.newaxis,:],
                             }
                    results = sess.run(fetches,feed_dict={valleft: I[np.newaxis,:],valright: I1[np.newaxis,:]})
                else:
                    results = sess.run(fetches)
                gs = results["global_step"]

                if step % FLAGS.summary_freq == 0:
                    train_writer.add_summary(results["summary"], gs)

                    print("steps: %d === loss: %.3f" \
                            % (gs,
                                results["loss"]))
                    translation_rotation = results["GT_cam"]
                    print(translation_rotation[0])
                    print(results["est_cam"][0])
                    print("learning rate %f" % (results["learn_rate"]))



                if step % FLAGS.save_latest_freq == 0:
                    saver.save(sess, FLAGS.checkpoint_dir+'/model', global_step=gs)


            coord.request_stop()
            coord.join(threads)   





def main(_):

    if not tf.gfile.Exists(FLAGS.checkpoint_dir):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir)
    if not tf.gfile.Exists(FLAGS.checkpoint_dir_single):
      tf.gfile.MakeDirs(FLAGS.checkpoint_dir_single)


    with tf.Graph().as_default():
        

        #============================================
        #Load image and labels
        #============================================
        with tf.name_scope("data_loading"):
            imageloader = DataLoader(FLAGS.dataset_dir,
                                     FLAGS.batch_size,
                                     FLAGS.image_height, 
                                     FLAGS.image_width,
                                     'train',
                                     FLAGS.num_scales)

            dataset = imageloader.load_train_batch2()


        

            #image_left, image_right, label, intrinsics, gt_right_cam = imageloader.load_train_batch()
            # label2 = tf.image.resize_area(label, 
            #     [int(FLAGS.resizedheight/(2**2)), int(FLAGS.resizedwidth/(2**2))])
            # data_dict,ground_truth,intrinsics = Demon_Dataloader()
            # image_left, image_right = tf.split(value=data_dict['IMAGE_PAIR'], num_or_size_splits=2, axis=3)
            # gt_right_cam = tf.concat([ground_truth['translation'],ground_truth['rotation']],axis=1)
            # label = ground_truth['depth0']
            # label2 = ground_truth['depth2']


        #============================================
        #Run RNN depth training
        #============================================
        rnn_depth_train(dataset,FLAGS)



        # pairwise_depth_train(image_left,image_right,label2,gt_right_cam,intrinsics,FLAGS)


        # with tf.variable_scope("model_pairdepth") as scope:
            
        #     inputdata = tf.concat([image_left, image_right], axis=3)
        #     scope.reuse_variables()
        #     pred_depth_left, pred_poses_right, pred_exp_logits_left, depth_net_endpoints_left = depth_net(inputdata,                                                    
        #                                                                                         is_training=False)
        #     saver_pair = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,scope="model_pairdepth"))
        #     checkpoint_pair = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)

        # # FLAGS.depth_weight = 10
        # # FLAGS.depth_sig_weight = 20
        
        # single_depth_training(label,FLAGS,image_left,pred_depth_left[0],saver_pair,checkpoint_pair)







 








             






if __name__ == '__main__':
   tf.app.run()



import tensorflow as tf
from nets_optflow_depth import *

import PIL.Image as pil
from PIL import Image

from utils_lr import *
#from tfutils import *

from my_losses_seq import *

from tensorflow.contrib.slim.python.slim.learning import train_step
from util import *


import os
os.environ["CUDA_VISIBLE_DEVICES"]="2"

####################
#Validate
####################
def sculpt_validation(valleft,valright,init_state):

    #Load image and label

    with tf.variable_scope("model_rnndepth") as scope:

        scope.reuse_variables()
        inputdata = tf.concat([valright,valleft,init_state], axis=3)

        pred_valid, _, _ = rnn_depth_net(inputdata,is_training=False)


    return pred_valid


def rnn_depth_train(dataset,dataset_valid,FLAGS):

    #==============
    #Build RNN model
    #==============
    with tf.variable_scope("model_rnndepth") as scope:

        #Preprocess data
        #dataset['num_views'] = 2;
        num_views = dataset['num_views']
        images = dataset['images']
        depths = dataset['depths']
        motions = dataset['motions']
        height = dataset['height']
        width = dataset['width']
        batch_size = dataset['batch_size']

        init_state1 = tf.placeholder(tf.float32,[batch_size,height,width,16])
        init_state2 = tf.placeholder(tf.float32,[batch_size,height/2,width/2,32])
        init_state3 = tf.placeholder(tf.float32,[batch_size,height/4,width/4,64])
        init_state4 = tf.placeholder(tf.float32,[batch_size,height/8,width/8,128])

        #Define model
        global_step = tf.Variable(0, 
                                  name='global_step', 
                                  trainable=False)
        incr_global_step = tf.assign(global_step, 
                                     global_step+1)

        state_series = []
        hidden_state = [init_state1,init_state2,init_state3,init_state4]#None
     
        #The first view is target view so -1
        for i in range(num_views):
            #import pdb;pdb.set_trace()
            image =  tf.slice(images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            depth =  tf.slice(depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])

            pred_depth, hidden_state,pred_pose, _ = rnn_depth_net_hidst(image,hidden_state,is_training=True)
            image.set_shape([batch_size,height,width,3])
            #image.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 3])
            #pred_depth, hidden_state, pred_pose = residual_u_network(image,hidden_state)
            scope.reuse_variables()
            state_series.append([pred_depth,pred_pose])
        
            if i==0:
                est_depths = pred_depth[0]
                gt_depths = depth
            else:
                est_depths = tf.concat([est_depths,pred_depth[0]],axis = 2)
                gt_depths = tf.concat([gt_depths,depth],axis = 2)



        ####################
        #Validate
        ####################
        #import pdb;pdb.set_trace()
        if(dataset_valid is not None):
            num_views_val = dataset_valid['num_views']
            images_val = dataset_valid['images']
            depths_val = dataset_valid['depths']
            motions_val = dataset_valid['motions']

            state_series_val = []
            hidden_state = [init_state1,init_state2,init_state3,init_state4]#
         
            #The first view is target view so -1
            for i in range(num_views_val):
                #import pdb;pdb.set_trace()
                image_val =  tf.slice(images_val,
                                      [0, 0, width*i, 0], 
                                      [-1, -1, int(width), -1])
                depth_val =  tf.slice(depths_val,
                                      [0, 0, width*i, 0], 
                                      [-1, -1, int(width), -1])

                image_val.set_shape([batch_size,height,width,3])
                scope.reuse_variables()
                pred_depth_val, hidden_state,pred_pose_val, _ = rnn_depth_net_hidst(image_val,hidden_state,is_training=False)
                #pred_depth_val, hidden_state,pred_pose_val = residual_u_network(image_val,hidden_state)
                #scope.reuse_variables()
                #state_series_val.append([pred_depth_val,pred_pose_val])

                if i==0:
                    est_depths_val = pred_depth_val[0]
                    gt_depths_val = depth_val
                else:
                    est_depths_val = tf.concat([est_depths_val,pred_depth_val[0]],axis = 2)
                    gt_depths_val = tf.concat([gt_depths_val,depth_val],axis = 2)

            #import pdb;pdb.set_trace()
            # abs_depth = sops.replace_nonfinite(gt_depths_val - est_depths_val)
            # abs_depth = tf.abs(abs_depth)
            # abs_depth = tf.reduce_mean(abs_depth) 
            d = sops.replace_nonfinite(tf.log(1.0/gt_depths_val) - tf.log(1.0/est_depths_val))
            abs_depth = tf.sqrt(tf.reduce_mean(d**2)-tf.reduce_mean(d)**2)

    # valleft = tf.placeholder(shape=[1, FLAGS.resizedheight, FLAGS.resizedwidth, 3], dtype=tf.float32)
    # valright = tf.placeholder(shape=[1, FLAGS.resizedheight, FLAGS.resizedwidth, 3], dtype=tf.float32)
    # valid_init_state = tf.placeholder(tf.float32,[1,height,width,1])
    # pred_valid = sculpt_validation(valright,valleft,valid_init_state)



    #==============
    #Compute Loss and define bp
    #==============
    #rnn_loss,test,test_img,testgt = compute_loss_rnn_lstm(dataset,state_series,global_step,FLAGS)
    rnn_loss,predpose,gtpose = compute_loss_rnn_hs(dataset,state_series,global_step,FLAGS)
    total_loss = rnn_loss['depth_loss']+rnn_loss['loss_depth_sig'] + rnn_loss['cam_loss']+ rnn_loss['threeD_loss']

    # Specify the optimization scheme:
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                       10000, 0.9, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    variables = slim.get_variables_to_restore()
    #saver_restore = tf.train.Saver(variables)

    # not_restore = ['model_rnndepth/rnn_depth_net/cnv1/weights:0', 
    #                 'model_rnndepth/rnn_depth_net/cnv1/weights/Adam:0', 
    #                 'model_rnndepth/rnn_depth_net/cnv1/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/icnv1/weights:0',
    #                 'model_rnndepth/rnn_depth_net/icnv1/weights/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/icnv1/weights/Adam_0:0',
    #                 'model_rnndepth/rnn_depth_net/icnv1/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/icnv3/weights:0',
    #                 'model_rnndepth/rnn_depth_net/icnv3/weights/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/icnv3/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/icnv2/weights:0',
    #                 'model_rnndepth/rnn_depth_net/icnv2/weights/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/icnv2/weights/Adam_1:0',

    #                 'model_rnndepth/rnn_depth_net/pose/pred/weights:0',
    #                 'model_rnndepth/rnn_depth_net/pose/pred/biases/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/pred/biases/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/pose/pred/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/pred/weights/Adam:0',

    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/weights:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/BatchNorm/beta/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/BatchNorm/beta/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/weights/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/BatchNorm/beta:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/BatchNorm/moving_mean:0' ,
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv8/BatchNorm/moving_variance:0',

    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/weights:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/BatchNorm/beta/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/BatchNorm/beta/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/weights/Adam_1:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/weights/Adam:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/BatchNorm/beta:0',
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/BatchNorm/moving_mean:0' ,
    #                 'model_rnndepth/rnn_depth_net/pose/cam_cnv7/BatchNorm/moving_variance:0',

    #                 # 'model_rnndepth/rnn_depth_net/disp1/biases:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp1/biases/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp1/biases/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp1/weights/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp1/weights/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp1/weights:0',

    #                 # 'model_rnndepth/rnn_depth_net/disp2/biases:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp2/biases/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp2/biases/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp2/weights/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp2/weights/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp2/weights:0',

    #                 # 'model_rnndepth/rnn_depth_net/disp3/biases:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp3/biases/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp3/biases/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp3/weights/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp3/weights/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp3/weights:0',


    #                 # 'model_rnndepth/rnn_depth_net/disp4/biases:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp4/biases/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp4/biases/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp4/weights/Adam_1:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp4/weights/Adam:0',
    #                 # 'model_rnndepth/rnn_depth_net/disp4/weights:0',

    #                 ]


    # restore_var = [v for v in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'model_rnndepth') if v.name not in not_restore]

    #import pdb;pdb.set_trace()
    #restore_saver = tf.train.Saver(restore_var)
    
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'model_rnndepth'))#[var for var in tf.global_variables()])     #tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'model_rnndepth'),max_to_keep=10   

    #==============
    #Tensorboard plot
    #==============
    #image_seq = tf.concat([tgt_image, src_images],axis = 2)
    tf.summary.scalar('losses/total_loss', total_loss)
    tf.summary.scalar('losses/threeD_loss', rnn_loss['threeD_loss'])
    tf.summary.scalar('losses/cam_loss', rnn_loss['cam_loss'])
    tf.summary.scalar('losses/depth_loss', rnn_loss['depth_loss'])
    tf.summary.scalar('losses/loss_depth_sig', rnn_loss['loss_depth_sig'])
    # f.summary.scalar('losses/normal_loss', rnn_loss['normal_loss'])


    

    tf.summary.histogram("gt_depth", sops.replace_nonfinite(gt_depths))
    tf.summary.histogram('pred_depth',
        est_depths)


    tf.summary.image('image_seq' , \
                     images)

    #import pdb;pdb.set_trace()
    condition = tf.less_equal(est_depths, 0.1)

    th = tf.ones_like(est_depths,dtype = tf.float32)*0.1

    est_depths = tf.where(condition, th, est_depths)
           
    tf.summary.image('est_depth' , \
                     1.0/est_depths)     

    tf.summary.image('gt_depth' , \
                     1.0/gt_depths)


    est_depth1 =  tf.slice(est_depths,
                          [0, 0, 0, 0], 
                          [-1, -1, int(width), -1])
    est_depth2 =  tf.slice(est_depths,
                          [0, 0, width*8, 0], 
                          [-1, -1, int(width), -1])
    tf.summary.image('difference' , \
                     est_depth1-est_depth2)

    if(dataset_valid is not None):

        condition = tf.less_equal(est_depths_val, 0.1)

        th = tf.ones_like(est_depths_val,dtype = tf.float32)*0.1

        est_depths_val = tf.where(condition, th, est_depths_val)

        tf.summary.scalar('valid/abs_depth', abs_depth)
        tf.summary.image('valid_est_depth' , \
                         1.0/est_depths_val)     

        tf.summary.image('valid_gt_depth' , \
                         1.0/gt_depths_val)
        tf.summary.image('valid_image_seq' , \
                         images_val)        

        # tf.summary.image('validate_pred',
        #     pred_valid[0])
        # tf.summary.image('validate_image',
        #     valright*255+0.5)

    #Session
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    with tf.Session(config=config) as sess:

        merged = tf.summary.merge_all()
        train_writer = tf.summary.FileWriter(FLAGS.checkpoint_dir + '/sum',
                                              sess.graph)

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess,coord=coord)
        
        if FLAGS.continue_train:
            if FLAGS.init_checkpoint_file is None:
                checkpoint = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
            else:
                checkpoint = FLAGS.init_checkpoint_file
            print("Resume training from previous checkpoint: %s" % checkpoint)
            saver.restore(sess, checkpoint)


        step = 1
        for epoch in range(1, FLAGS.max_steps):
            print("epoch %d" % (epoch))
            sess.run(dataset['iter'].initializer)

            while True:
                try:
                    fetches = {
                        "train": train_op,
                        "global_step": global_step,
                        "incr_global_step": incr_global_step
                        }


                    _init_state = [np.zeros([dataset['batch_size'],dataset['height'],dataset['width'],16]),
                                    np.zeros([dataset['batch_size'],int(dataset['height']/2),int(dataset['width']/2),32]),
                                    np.zeros([dataset['batch_size'],int(dataset['height']/4),int(dataset['width']/4),64]),
                                    np.zeros([dataset['batch_size'],int(dataset['height']/8),int(dataset['width']/8),128])]

                    if step % FLAGS.summary_freq == 0:

                        fetches["loss"] = total_loss
                        fetches["summary"] = merged
                        fetches["learn_rate"] = learning_rate
                        #fetches["pred_3d"] = test
                        #fetches["image"] = test_img
                        #fetches["gt_3d"] = testgt
                        fetches["predpose"] = predpose
                        fetches["gtpose"] = gtpose
                        fetches["angle"] = dataset['angle']

                    if(dataset_valid is not None):
                        
                        fetches["validate"] = abs_depth

                    results= sess.run(fetches,feed_dict={init_state1: _init_state[0],init_state2: _init_state[1],init_state3: _init_state[2],init_state4: _init_state[3]}) #,valid_init_state:_valid_state


                    if step % FLAGS.summary_freq == 0:

                        gs = results["global_step"]
                        train_writer.add_summary(results["summary"], gs)

                        print("steps: %d / %d === loss: %.3f, valid sc-inv: %.3f" \
                                % (gs,epoch,
                                    results["loss"], results["validate"]))

                        print("learning rate %f" % (results["learn_rate"]))

                        # m_test = results["pred_3d"]
                        # m_testimg = results["image"]
                        # m_testgt = results["gt_3d"]
                        # m_angle = results["angle"]
                        m_predpose =  results["predpose"]
                        m_gtpose = results["gtpose"] 

                        if step == 1:
                            print("sig weight: %f"%(m_test))

                        print("pred cam"  )
                        firstfew = 0
                        for precam in m_predpose:
                            
                            if firstfew%2==0:
                                print(precam[0])
                            firstfew = firstfew+1
                        
                        print("GT cam ")
                        firstfew = 0
                        for gtcam in m_gtpose:
                            
                            if firstfew%2==0:
                                print(gtcam[0])
                            firstfew = firstfew+1


                    if step % FLAGS.save_latest_freq == 0:
                        saver.save(sess, FLAGS.checkpoint_dir+'/model', global_step=gs)

                    step = step+1
                except tf.errors.OutOfRangeError:
                    print("End epoch %d"%(epoch))
                    break


 

        coord.request_stop()
        coord.join(threads)

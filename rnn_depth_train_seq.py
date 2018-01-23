

import tensorflow as tf
from nets_optflow_depth import *

from utils_lr import *
from tfutils import *

from my_losses_seq import *

from tensorflow.contrib.slim.python.slim.learning import train_step



def rnn_depth_train(dataset,FLAGS):

    #==============
    #Build RNN model
    #==============
    with tf.variable_scope("model_rnndepth") as scope:

        #Preprocess data
        #dataset['num_views'] = 2;
        num_views = dataset['num_views']

        tgt_image = dataset['tgt_image']
        src_images = dataset['src_images']
        tgt_depth = dataset['tgt_depth']
        src_depths = dataset['src_depths'] 
        tgt_motion = dataset['tgt_motion']
        src_motions = dataset['src_motions']
        height = dataset['height']
        width = dataset['width']
        batch_size = dataset['batch_size']

        
        init_state = tf.placeholder(tf.float32,[batch_size,height,width,1])

        #Define model
        global_step = tf.Variable(0, 
                                  name='global_step', 
                                  trainable=False)
        incr_global_step = tf.assign(global_step, 
                                     global_step+1)

        state_series = []
        current_est = init_state
     
        #The first view is target view so -1
        for i in range(num_views-1):
            #import pdb;pdb.set_trace()
            src_image =  tf.slice(src_images,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            src_depth =  tf.slice(src_depths,
                                  [0, 0, width*i, 0], 
                                  [-1, -1, int(width), -1])
            current_input = tf.concat([tgt_image,src_image,current_est],axis = 3)
            pred_depth, pred_pose, _ = rnn_depth_net(current_input,is_training=True)
            scope.reuse_variables()
            state_series.append([pred_depth,pred_pose])
            current_est = pred_depth[0]
            if i==0:
                est_depths = pred_depth[0]
                gt_depths = src_depth
            else:
                est_depths = tf.concat([est_depths,pred_depth[0]],axis = 2)
                gt_depths = tf.concat([gt_depths,src_depth],axis = 2)

            tgt_image = src_image


    #==============
    #Compute Loss and define bp
    #==============
    rnn_loss = compute_loss_rnn(dataset,state_series,global_step,FLAGS)
    
    total_loss = rnn_loss['depth_loss']+rnn_loss['loss_depth_sig'] #rnn_loss['cam_loss']+

    # Specify the optimization scheme:
    learning_rate = tf.train.exponential_decay(FLAGS.learning_rate, global_step,
                                       10000, 0.9, staircase=True)

    optimizer = tf.train.AdamOptimizer(learning_rate,FLAGS.beta1)

    # create_train_op that ensures that when we evaluate it to get the loss,
    # the update_ops are done and the gradient updates are computed.
    train_op = slim.learning.create_train_op(total_loss, optimizer)
    saver = tf.train.Saver(tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, 'model_rnndepth'),max_to_keep=10)        

    #==============
    #Tensorboard plot
    #==============
    image_seq = tf.concat([tgt_image, src_images],axis = 2)
    tf.summary.scalar('losses/total_loss', total_loss)
    #tf.summary.scalar('losses/cam_loss', rnn_loss['cam_loss'])
    tf.summary.scalar('losses/depth_loss', rnn_loss['depth_loss'])
    tf.summary.scalar('losses/loss_depth_sig', rnn_loss['loss_depth_sig'])

    tf.summary.histogram("gt_depth", sops.replace_nonfinite(gt_depths))
    tf.summary.histogram('pred_depth',
        est_depths)


    tf.summary.image('image_seq' , \
                     image_seq)
           
    tf.summary.image('est_depth' , \
                     est_depths)     

    tf.summary.image('gt_depth' , \
                     1.0/gt_depths)


    est_depth1 =  tf.slice(est_depths,
                          [0, 0, 0, 0], 
                          [-1, -1, int(width), -1])
    est_depth2 =  tf.slice(est_depths,
                          [0, 0, width*8, 0], 
                          [-1, -1, int(width), -1])
    tf.summary.image('difference' , \
                     1.0/est_depth1-1.0/est_depth2)
    #Session
    with tf.Session() as sess:

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


        for step in range(1, FLAGS.max_steps):
            #print("steps %d" % (step))
            fetches = {
                "train": train_op,
                "global_step": global_step,
                "incr_global_step": incr_global_step
            }

           
            _current_state = np.zeros([dataset['batch_size'],dataset['height'],dataset['width'],1])


            if step % FLAGS.summary_freq == 0:
                fetches["loss"] = total_loss
                fetches["summary"] = merged
                fetches["learn_rate"] = learning_rate


            results = sess.run(fetches,feed_dict={init_state: _current_state})

            if step % FLAGS.summary_freq == 0:
                
                gs = results["global_step"]
                train_writer.add_summary(results["summary"], gs)

                print("steps: %d === loss: %.3f" \
                        % (gs,
                            results["loss"]))

                print("learning rate %f" % (results["learn_rate"]))



            if step % FLAGS.save_latest_freq == 0:
                saver.save(sess, FLAGS.checkpoint_dir+'/model', global_step=gs)

        coord.request_stop()
        coord.join(threads)

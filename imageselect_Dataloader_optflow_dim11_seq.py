from __future__ import division
import tensorflow as tf
import pprint
import random
import numpy as np
import os
import glob
from random import shuffle


class DataLoader(object):
    def __init__(self,
                 dataset_dir,
                 valid_dir,
                 batch_size,
                 image_height,
                 image_width,
                 split,
                 num_scales
                 ):
        self.dataset_dir=dataset_dir
        self.valid_dir=valid_dir
        self.batch_size=batch_size
        self.image_height=image_height
        self.image_width=image_width
        self.split=split
        self.num_scales = num_scales
        self.resizedheight = 192
        self.resizedwidth = 256
        self.num_views = 10
        self.depth_dir = '/home/wrlife/project/Unsupervised_Depth_Estimation/scripts/data/goodimages/'



    def load_train_batch(self):
        
        seed = random.randint(0, 2**31 - 1)

        # Reads pfathes of images together with their labels
        tfrecords = glob.glob(self.dataset_dir+"/*.tfrecords")

        #import pdb;pdb.set_trace()

        image_paths_queue = tf.convert_to_tensor(tfrecords, dtype=tf.string)

        filename_queue = tf.train.string_input_producer(image_paths_queue, shuffle=True)


        #tg_image,src_images, tgt_depth,tgt_motion, src_motions
        #,tgt2scr_projs,m_scale
        dataset = self.read_labeled_tfrecord_list(filename_queue)

        # Form training batches
        # dataset  = \
        #         tf.train.batch(dataset, 
        #                        batch_size=self.batch_size)
        #[tg_image, src_images, tgt_depth,tgt_motion, src_motions]
        #tgt_image_stack,src_images_stack, tgt_depth_stack, tgt_motion_stack, src_motions_stack
        dataset = tf.train.shuffle_batch( dataset,
                                                     batch_size=self.batch_size,
                                                     capacity=1000,
                                                     num_threads=16,
                                                     min_after_dequeue=20)

        # intrinsics = self.get_multi_scale_intrinsics(
        #     intrinsics, self.num_scales,tf.cast(self.resizedwidth,tf.float32)/self.image_width,tf.cast(self.resizedheight,tf.float32)/self.image_height)
        
        dataset['num_views'] = self.num_views
        dataset['height'] = self.resizedheight
        dataset['width'] = self.resizedwidth
        dataset['batch_size'] = self.batch_size

        #import pdb;pdb.set_trace()
        return dataset
    


    def load_train_batch2(self):
        # Creates a dataset that reads all of the examples from filenames.
        tfrecords = shuffle(glob.glob(self.dataset_dir+"/*.tfrecords"))
        dataset = tf.contrib.data.TFRecordDataset(tfrecords)

        # example proto decode
        def _parse_function(example_proto):
            keys_to_features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'max_views_num': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
                'motion_raw': tf.FixedLenFeature([], tf.string),
                }
            features = tf.parse_single_example(example_proto, keys_to_features)

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            depth = tf.decode_raw(features['depth'], tf.float32)
            motion = tf.decode_raw(features['motion_raw'], tf.float32)


            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            num_views = tf.cast(features['max_views_num'], tf.int32)



            image = tf.to_float(tf.image.resize_images(tf.reshape(image, [height, width*num_views, 3]),[self.resizedheight,self.resizedwidth*self.num_views]))/255.0-0.5
            tg_image,src_images = self.unpack_image_sequence(image,self.resizedheight,self.resizedwidth,self.num_views)
            #image.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 3])


            depth = 1.0/tf.image.resize_images(tf.reshape(depth, [height, width*num_views, 1]),[self.resizedheight,self.resizedwidth*self.num_views])

            tgt_depth = tf.slice(depth, 
                                 [0, 0, 0], 
                                 [-1, self.resizedwidth, -1])
            tgt_depth.set_shape([self.resizedheight,self.resizedwidth, 1])
            
            src_depths = tf.slice(depth, 
                                [0, self.resizedwidth, 0], 
                                [-1, self.resizedwidth * (self.num_views-1), -1])

            src_depths.set_shape([self.resizedheight, self.resizedwidth*(self.num_views-1), 1])   
            
            #depth.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 1])

            motion = tf.reshape(motion,[3,4*self.num_views])
            tgt_motion = tf.slice(motion,
                                  [0,0],
                                  [-1,4])
            tgt_motion.set_shape([3,4])

            src_motions = tf.slice(motion,
                                  [0,4],
                                  [-1,4*(self.num_views-1)])
            src_motions.set_shape([3,4*(self.num_views-1)])

            x_resize_ratio = self.resizedwidth/self.image_width
            y_resize_ratio = self.resizedheight/self.image_height

            intrinsics = np.array([(570.3422*x_resize_ratio ,0 ,320*x_resize_ratio), (0 ,570.3422*y_resize_ratio ,240*y_resize_ratio), (0 ,0 ,1)],dtype=np.float32);


            # images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
            #                                              batch_size=2,
            #                                              capacity=30,
            #                                              num_threads=2,
            #                                              min_after_dequeue=10)

            #normalize motion and depth

            dataset = {}
            dataset['tgt_image'] = tg_image
            dataset['src_images'] = src_images
            dataset['tgt_depth'] = tgt_depth
            dataset['src_depths'] = src_depths
            dataset['tgt_motion'] = tgt_motion
            dataset['src_motions'] = src_motions
            dataset['intrinsics'] = intrinsics

            return dataset       


        # Parse the record into tensors.
        dataset = dataset.map(_parse_function,output_buffer_size = 600, num_parallel_calls = 30)  

        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=500)

        # Repeat the input indefinitly
        dataset = dataset.repeat()  

        # Generate batches
        dataset = dataset.batch(self.batch_size)

        # Create a one-shot iterator
        iterator = dataset.make_one_shot_iterator()

        # Get batch X and y
        m_batch = iterator.get_next()

        m_batch['num_views'] = self.num_views
        m_batch['height'] = self.resizedheight
        m_batch['width'] = self.resizedwidth
        m_batch['batch_size'] = self.batch_size

        #import pdb;pdb.set_trace()
        return m_batch

    def load_train_batch_hs(self):
        # Creates a dataset that reads all of the examples from filenames.
        tfrecords = glob.glob(self.dataset_dir+"/*.tfrecords")
        dataset = tf.contrib.data.TFRecordDataset(tfrecords)

        # example proto decode
        def _parse_function(example_proto):
            keys_to_features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'max_views_num': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
                'motion_raw': tf.FixedLenFeature([], tf.string),
                }
            features = tf.parse_single_example(example_proto, keys_to_features)

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            depth = tf.decode_raw(features['depth'], tf.float32)
            motion = tf.decode_raw(features['motion_raw'], tf.float32)


            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            num_views = tf.cast(features['max_views_num'], tf.int32)


            #image = tf.to_float(tf.image.resize_images(tf.reshape(image, [height, width*num_views, 3]),[self.resizedheight,self.resizedwidth*self.num_views]))/255.0-0.5
            image = tf.reshape(image,[height,width*num_views,3])
            image = tf.image.resize_images(image,[self.resizedheight,self.resizedwidth*self.num_views])

            image = tf.image.random_brightness(image,0.3)
            image = tf.image.random_contrast(image,0.2,1.8)
            image = tf.image.random_hue(image,0.2)
            #image = tf.image.random_satuation(image,)

            image = tf.to_float(image)/255.0-0.5
            image.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 3])


            depth = 1.0/tf.image.resize_images(tf.reshape(depth, [height, width*num_views, 1]),[self.resizedheight,self.resizedwidth*self.num_views])
            depth.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 1])


            motion = tf.reshape(motion,[3,4*self.num_views])
            motion.set_shape([3,4*self.num_views])

            x_resize_ratio = self.resizedwidth/self.image_width
            y_resize_ratio = self.resizedheight/self.image_height

            intrinsics = np.array([(570.3422*x_resize_ratio ,0 ,320*x_resize_ratio), (0 ,570.3422*y_resize_ratio ,240*y_resize_ratio), (0 ,0 ,1)],dtype=np.float32);



            dataset = {}
            dataset['images'] = image
            dataset['depths'] = depth
            dataset['motions'] = motion
            dataset['intrinsics'] = intrinsics

            return dataset       

        # Parse the record into tensors.
        dataset = dataset.map(_parse_function,output_buffer_size = 600, num_parallel_calls = 30)  

        # Shuffle the dataset
        dataset = dataset.shuffle(buffer_size=500)

        dataset = dataset.repeat()  
       
        # Generate batches
        dataset = dataset.batch(self.batch_size)

        # Create a one-shot iterator
        iterator = dataset.make_one_shot_iterator()

        # Get batch X and y
        m_batch = iterator.get_next()

        m_batch['num_views'] = self.num_views
        m_batch['height'] = self.resizedheight
        m_batch['width'] = self.resizedwidth
        m_batch['batch_size'] = self.batch_size

        #import pdb;pdb.set_trace()
        return m_batch

    def load_valid_batch_hs(self):
        # Creates a dataset that reads all of the examples from filenames.

        if(self.valid_dir == "none"):
            return None

        tfrecords = glob.glob(self.valid_dir+"/*.tfrecords")
        dataset = tf.contrib.data.TFRecordDataset(tfrecords)

        # example proto decode
        def _parse_function(example_proto):
            keys_to_features = {
                'height': tf.FixedLenFeature([], tf.int64),
                'width': tf.FixedLenFeature([], tf.int64),
                'max_views_num': tf.FixedLenFeature([], tf.int64),
                'image_raw': tf.FixedLenFeature([], tf.string),
                'depth': tf.FixedLenFeature([], tf.string),
                'motion_raw': tf.FixedLenFeature([], tf.string)
                }
            features = tf.parse_single_example(example_proto, keys_to_features)

            image = tf.decode_raw(features['image_raw'], tf.uint8)
            depth = tf.decode_raw(features['depth'], tf.float32)
            motion = tf.decode_raw(features['motion_raw'], tf.float32)
            

            height = tf.cast(features['height'], tf.int32)
            width = tf.cast(features['width'], tf.int32)
            num_views = tf.cast(features['max_views_num'], tf.int32)


            image = tf.reshape(image,[height,width*num_views,3])
            image = tf.image.resize_images(image,[self.resizedheight,self.resizedwidth*self.num_views])
            # image = tf.image.random_brightness(image,0.3)
            # image = tf.image.random_contrast(image,0.2,1.8)
            # image = tf.image.random_hue(image,0.2)
            #image = tf.image.random_satuation(image,)

            image = tf.to_float(image)/255.0-0.5

            image.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 3])


            depth = depth = 1.0/tf.image.resize_images(tf.reshape(depth, [height, width*num_views, 1]),[self.resizedheight,self.resizedwidth*self.num_views])
            depth.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 1])

            motion = tf.reshape(motion,[3,4*self.num_views])
            motion.set_shape([3,4*self.num_views])

            x_resize_ratio = self.resizedwidth/self.image_width
            y_resize_ratio = self.resizedheight/self.image_height

            intrinsics = np.array([(570.3422*x_resize_ratio ,0 ,320*x_resize_ratio), (0 ,570.3422*y_resize_ratio ,240*y_resize_ratio), (0 ,0 ,1)],dtype=np.float32)


            dataset = {}
            dataset['images'] = image
            dataset['depths'] = depth
            dataset['motions'] = motion
            dataset['intrinsics'] = intrinsics

            return dataset       

        # Repeat the input indefinitly

        dataset = dataset.repeat() 

        # Parse the record into tensors.
        dataset = dataset.map(_parse_function)  

        # Generate batches
        dataset = dataset.batch(self.batch_size)

        # Create a one-shot iterator
        iterator = dataset.make_one_shot_iterator()

        # Get batch X and y
        m_batch = iterator.get_next()

        m_batch['num_views'] = self.num_views
        m_batch['height'] = self.resizedheight
        m_batch['width'] = self.resizedwidth
        m_batch['batch_size'] = self.batch_size

        #import pdb;pdb.set_trace()
        return m_batch


    def read_labeled_tfrecord_list(self,filename_queue):
        reader = tf.TFRecordReader()

        _, serialized_example = reader.read(filename_queue)

        features = tf.parse_single_example(
          serialized_example,
          # Defaults are not specified since both keys are required.
          features={
            'height': tf.FixedLenFeature([], tf.int64),
            'width': tf.FixedLenFeature([], tf.int64),
            'max_views_num': tf.FixedLenFeature([], tf.int64),
            'image_raw': tf.FixedLenFeature([], tf.string),
            'depth': tf.FixedLenFeature([], tf.string),
            'motion_raw': tf.FixedLenFeature([], tf.string),
            })

        # Convert from a scalar string tensor (whose single string has
        # length mnist.IMAGE_PIXELS) to a uint8 tensor with shape
        # [mnist.IMAGE_PIXELS].
        image = tf.decode_raw(features['image_raw'], tf.uint8)
        depth = tf.decode_raw(features['depth'], tf.float32)
        motion = tf.decode_raw(features['motion_raw'], tf.float32)


        height = tf.cast(features['height'], tf.int32)
        width = tf.cast(features['width'], tf.int32)
        num_views = tf.cast(features['max_views_num'], tf.int32)


        image = tf.to_float(tf.image.resize_images(tf.reshape(image, [height, width*num_views, 3]),[self.resizedheight,self.resizedwidth*self.num_views]))
        tg_image,src_images = self.unpack_image_sequence(image,self.resizedheight,self.resizedwidth,self.num_views)
        #image.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 3])


        depth = 1.0/tf.image.resize_images(tf.reshape(depth, [height, width*num_views, 1]),[self.resizedheight,self.resizedwidth*self.num_views])

        tgt_depth = tf.slice(depth, 
                             [0, 0, 0], 
                             [-1, self.resizedwidth, -1])
        tgt_depth.set_shape([self.resizedheight,self.resizedwidth, 1])
        #depth.set_shape([self.resizedheight,self.resizedwidth*self.num_views, 1])

        motion = tf.reshape(motion,[3,4*self.num_views])
        tgt_motion = tf.slice(motion,
                              [0,0],
                              [-1,4])
        tgt_motion.set_shape([3,4])

        src_motions = tf.slice(motion,
                              [0,4],
                              [-1,4*(self.num_views-1)])
        src_motions.set_shape([3,4*(self.num_views-1)])

        intrinsics = np.array([(570.3422/2 ,0 ,320/2), (0 ,570.3422/2 ,240/2), (0 ,0 ,1)],dtype=np.float32);


        # images, annotations = tf.train.shuffle_batch( [resized_image, resized_annotation],
        #                                              batch_size=2,
        #                                              capacity=30,
        #                                              num_threads=2,
        #                                              min_after_dequeue=10)

        #normalize motion and depth

        dataset = {}
        dataset['tgt_image'] = tg_image
        dataset['src_images'] = src_images
        dataset['tgt_depth'] = tgt_depth
        dataset['tgt_motion'] = tgt_motion
        dataset['src_motions'] = src_motions
        dataset['intrinsics'] = intrinsics




        return dataset         




    def unpack_image_sequence(self, image_seq, img_height, img_width, num_source):
        # Assuming the center image is the target frame

        tgt_image = tf.slice(image_seq, 
                             [0, 0, 0], 
                             [-1, img_width, -1])

        # Source frames before the target frame
        src_image_stack = tf.slice(image_seq, [0, img_width, 0], [-1, img_width * (num_source-1), -1])

        src_image_stack.set_shape([img_height, 
                                   img_width*(num_source-1), 
                                   3])
        tgt_image.set_shape([img_height, img_width, 3])
        return tgt_image, src_image_stack



    def make_intrinsics_matrix(self, fx, fy, cx, cy):
        # Assumes batch input
        batch_size = fx.get_shape().as_list()[0]
        zeros = tf.zeros_like(fx)
        r1 = tf.stack([fx, zeros, cx], axis=1)
        r2 = tf.stack([zeros, fy, cy], axis=1)
        r3 = tf.constant([0.,0.,1.], shape=[1, 3])
        r3 = tf.tile(r3, [batch_size, 1])
        intrinsics = tf.stack([r1, r2, r3], axis=1)
        return intrinsics


    def get_multi_scale_intrinsics(self,intrinsics, num_scales,x_resize_ratio,y_resize_ratio):

        intrinsics_mscale = []


        # Scale the intrinsics accordingly for each scale
        for s in range(num_scales):
            fx = intrinsics[:,0,0]/(2 ** s)*x_resize_ratio
            fy = intrinsics[:,1,1]/(2 ** s)*y_resize_ratio
            cx = intrinsics[:,0,2]/(2 ** s)*x_resize_ratio
            cy = intrinsics[:,1,2]/(2 ** s)*y_resize_ratio
            intrinsics_mscale.append(
                self.make_intrinsics_matrix(fx, fy, cx, cy))
        intrinsics_mscale = tf.stack(intrinsics_mscale, axis=1)
        return intrinsics_mscale

#!/usr/bin/env python3s
import os.path
import tensorflow as tf
import helper
import warnings
from distutils.version import LooseVersion
import project_tests as tests
from tqdm import tqdm
import numpy as np

KEEP_PROB = 0.8 #lower value will help generalize more (but with fewer epochs, higher keep_prob creates more clearer segmentations)
LEARNING_RATE = 0.0009 #high learning rate will cause overshooting and huge oscillations in loss. (i.e. even 0.009 - 10 times higher will completely ruin the training)
IMAGE_SHAPE = (160, 576) #higher resolution will help segmenting in a more detailed fashion
EPOCHS = 50
BATCH_SIZE = 5 #with batch_size smaller, lower memory will be used as less number of images need to be loaded into memory, the training will go on in SGD fashion, and even with 1 epoch, the small batch size and SGD will make the training look like many epochs training if the trianing sets are somewhat similar (i.e. all roads and we're doing only 2 classes)
NUM_CLASSES = 2 #the smaller the classes, the easier it is to segment using lower number of epochs and batch_size

USE_L2_LOSS = False
L2_LOSS_WEIGHT = 0.01

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer.  You are using {}'.format(tf.__version__)
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))

def load_vgg(sess, vgg_path):
    """
    Load Pretrained VGG Model into TensorFlow.
    :param sess: TensorFlow Session
    :param vgg_path: Path to vgg folder, containing "variables/" and "saved_model.pb"
    :return: Tuple of Tensors from VGG model (image_input, keep_prob, layer3_out, layer4_out, layer7_out)
    """
    # TODO: Implement function
    #   Use tf.saved_model.loader.load to load the model and weights

    
    vgg_tag = 'vgg16'
    vgg_input_tensor_name = 'image_input:0'
    vgg_keep_prob_tensor_name = 'keep_prob:0'
    vgg_layer3_out_tensor_name = 'layer3_out:0'
    vgg_layer4_out_tensor_name = 'layer4_out:0'
    vgg_layer7_out_tensor_name = 'layer7_out:0'
    
    #load the vgg model located at data/vgg16/vgg, this path is defined by vgg_path later
    tf.saved_model.loader.load(sess, [vgg_tag], vgg_path)
    #make sure to load the default graph from the loaded model before pulling tensors by name into storage variables
    graph = tf.get_default_graph()
    
    input_image = graph.get_tensor_by_name(vgg_input_tensor_name)
    keep_prob = graph.get_tensor_by_name(vgg_keep_prob_tensor_name)
    layer3_out = graph.get_tensor_by_name(vgg_layer3_out_tensor_name)
    layer4_out = graph.get_tensor_by_name(vgg_layer4_out_tensor_name)
    layer7_out = graph.get_tensor_by_name(vgg_layer7_out_tensor_name)
    
#     a = tf.Print(layer7_out, [tf.shape(layer7_out)])
#     with tf.Session() as sess:
#         sess.run(a)
    
    return input_image, keep_prob, layer3_out, layer4_out, layer7_out
tests.test_load_vgg(load_vgg, tf)

def layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes, sess=None, vgg_input=None, keep_prob=None):
    """
    Create the layers for a fully convolutional network.  Build skip-layers using the vgg layers.
    :param vgg_layer3_out: TF Tensor for VGG Layer 3 output
    :param vgg_layer4_out: TF Tensor for VGG Layer 4 output
    :param vgg_layer7_out: TF Tensor for VGG Layer 7 output
    :param num_classes: Number of classes to classify
    :return: The Tensor for the last layer of output
    """
    # TODO: Implement function
    #constructing FCN-8 architecture
    
    #reduce the number of outputs to match the num_classes in the training set (in this case 2, roads vs not roads) by using 1x1 convolution
    conv_1x1 = tf.layers.conv2d(vgg_layer7_out, num_classes, 1, padding="same", 
                                kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
        
    #upsample or deconv (from 1x1 to 2v2 just like in the FCN paper: https://people.eecs.berkeley.edu/~jonlong/long_shelhamer_fcn.pdf)
    layer4a_in1 = tf.layers.conv2d_transpose(conv_1x1, num_classes, 4, 2, padding="same", kernel_regularizer=tf.contrib.layers.l2_regularizer(1e-3))
    
    # make sure the shapes are the same!
    # 1x1 convolution of vgg layer 4
    layer4a_in2 = tf.layers.conv2d(vgg_layer4_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # skip connection (element-wise addition)
    layer4a_out = tf.add(layer4a_in1, layer4a_in2)
    # upsample
    layer3a_in1 = tf.layers.conv2d_transpose(layer4a_out, num_classes, 4,  
                                             strides= (2, 2), 
                                             padding= 'same', 
                                             kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                             kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    # 1x1 convolution of vgg layer 3
    layer3a_in2 = tf.layers.conv2d(vgg_layer3_out, num_classes, 1, 
                                   padding= 'same', 
                                   kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                   kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    # skip connection (element-wise addition)
    layer3a_out = tf.add(layer3a_in1, layer3a_in2)
    # upsample
    nn_last_layer = tf.layers.conv2d_transpose(layer3a_out, num_classes, 16,  
                                               strides= (8, 8), 
                                               padding= 'same', 
                                               kernel_initializer= tf.random_normal_initializer(stddev=0.01), 
                                               kernel_regularizer= tf.contrib.layers.l2_regularizer(1e-3))
    
    #following are used for printing shapes of layers of interest - very useful.
#     if sess is not None:
#         img = np.random.rand(1, 160, 576, 3)
#         prints = [
#             tf.Print(conv_1x1, [tf.shape(conv_1x1), " -------------------1x1conv  before deconv starts  -------------------"],
#                      summarize=4)]
#         sess.run(tf.global_variables_initializer())
#         sess.run(prints, feed_dict={vgg_input: img, keep_prob: 1.0})
        
#     if sess is not None:
#         img2 = np.random.rand(1, 160, 576, 3)
#         prints = [
#             tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out), " ------------------- vgg_layer7_out -------------------"],
#                      summarize=4)]
#         sess.run(tf.global_variables_initializer())
#         sess.run(prints, feed_dict={vgg_input: img2, keep_prob: 1.0})
    
    return nn_last_layer
tests.test_layers(layers)

def optimize(nn_last_layer, correct_label, learning_rate, num_classes):
    """
    Build the TensorFLow loss and optimizer operations.
    :param nn_last_layer: TF Tensor of the last layer in the neural network
    :param correct_label: TF Placeholder for the correct label image
    :param learning_rate: TF Placeholder for the learning rate
    :param num_classes: Number of classes to classify
    :return: Tuple of (logits, train_op, cross_entropy_loss)
    """
    # make logits a 2D tensor where each row represents a pixel and each column a class
    logits = tf.reshape(nn_last_layer, (-1, num_classes))
    correct_label = tf.reshape(correct_label, (-1,num_classes))
    # define loss function
    cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits= logits, labels= correct_label))
    
    if USE_L2_LOSS:
        #adding L2 losses to apply to loss
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES) #collect all losses from every layer
        reg_constant = L2_LOSS_WEIGHT  # Choose an appropriate one.
        final_loss = cross_entropy_loss + reg_constant * sum(reg_losses)
    
    # define training operation
    optimizer = tf.train.AdamOptimizer(learning_rate= learning_rate)
    if USE_L2_LOSS:
        train_op = optimizer.minimize(final_loss)
    else:
        train_op = optimizer.minimize(cross_entropy_loss)

    return logits, train_op, cross_entropy_loss
tests.test_optimize(optimize)

def train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate):
    """
    Train neural network and print out the loss during training.
    :param sess: TF Session
    :param epochs: Number of epochs
    :param batch_size: Batch size
    :param get_batches_fn: Function to get batches of training data.  Call using get_batches_fn(batch_size)
    :param train_op: TF Operation to train the neural network
    :param cross_entropy_loss: TF Tensor for the amount of loss
    :param input_image: TF Placeholder for input images
    :param correct_label: TF Placeholder for label images
    :param keep_prob: TF Placeholder for dropout keep probability
    :param learning_rate: TF Placeholder for learning rate
    """
    # TODO: Implement function
    sess.run(tf.global_variables_initializer())
    
    print("Training...")
    print()
    for i in range(epochs):
        print("EPOCH {} ...".format(i+1))
        for image, label in get_batches_fn(batch_size):
            _, loss = sess.run([train_op, cross_entropy_loss], 
                               feed_dict={input_image: image, correct_label: label, keep_prob: KEEP_PROB, learning_rate:LEARNING_RATE})
            print("Loss: = {:.3f}".format(loss))
        print()
tests.test_train_nn(train_nn)



def run():
    num_classes = NUM_CLASSES
    image_shape = IMAGE_SHAPE
    data_dir = './data'
    runs_dir = './runs'
    tests.test_for_kitti_dataset(data_dir)

    # Download pretrained vgg model
    helper.maybe_download_pretrained_vgg(data_dir)

    # OPTIONAL: Train and Inference on the cityscapes dataset instead of the Kitti dataset.
    # You'll need a GPU with at least 10 teraFLOPS to train on.
    #  https://www.cityscapes-dataset.com/

    with tf.Session() as sess:
        # Path to vgg model
        vgg_path = os.path.join(data_dir, 'vgg16/vgg')
        # Create function to get batches
        get_batches_fn = helper.gen_batch_function(os.path.join(data_dir, 'data_road/training'), image_shape)

        # OPTIONAL: Augment Images for better results
        #  https://datascience.stackexchange.com/questions/5224/how-to-prepare-augment-images-for-neural-network

        # TODO: Build NN using load_vgg, layers, and optimize function

        # TODO: Train NN using the train_nn function

        # TODO: Save inference data using helper.save_inference_samples
        #  helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)

        # OPTIONAL: Apply the trained model to a video


        epochs = EPOCHS
        batch_size = BATCH_SIZE

        # TF placeholders
        correct_label = tf.placeholder(tf.int32, [None, None, None, num_classes], name='correct_label')
        learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        input_image, keep_prob, vgg_layer3_out, vgg_layer4_out, vgg_layer7_out = load_vgg(sess, vgg_path)
        #sess.run(tf.Print(vgg_layer7_out, [tf.shape(vgg_layer7_out)]))

        #nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes)
        nn_last_layer = layers(vgg_layer3_out, vgg_layer4_out, vgg_layer7_out, num_classes,
                                   sess=sess, vgg_input=input_image, keep_prob=keep_prob)

        logits, train_op, cross_entropy_loss = optimize(nn_last_layer, correct_label, learning_rate, num_classes)

        # TODO: Train NN using the train_nn function

        train_nn(sess, epochs, batch_size, get_batches_fn, train_op, cross_entropy_loss, input_image,
             correct_label, keep_prob, learning_rate)

        # TODO: Save inference data using helper.save_inference_samples
        helper.save_inference_samples(runs_dir, data_dir, sess, image_shape, logits, keep_prob, input_image)
        print("running on images - done")

        # OPTIONAL: Apply the trained model to a video


if __name__ == '__main__':
    run()

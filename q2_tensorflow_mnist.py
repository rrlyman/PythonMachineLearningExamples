"""# ==========================================================================

# Copyright 2015 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

This sample program is a modified version of the Google mnist convolutional 
network tutorial example.  See the mnist tutorial in www.tensorflow.org 

The tutorial version of the program is modified in order to send some
features directly to the fully connected layer, thus bypassing the
convolution layer.

Images go through convolution.  Everything else bypasses.

see tensor_flow_graph.png
"""# ==============================================================================
import ocr_utils
import datetime
from collections import namedtuple
import numpy as np
 
   
def train_a_font(input_filters_dict,output_feature_list, nEpochs=5000):
 
    ds = ocr_utils.read_data(input_filters_dict = input_filters_dict, 
                                output_feature_list=output_feature_list,
                                test_size = .1,
                                engine_type='tensorflow')

        
    """# ==============================================================================
    
    Start TensorFlow Interactive Session
    
    """# ==============================================================================
    
    import tensorflow as tf
    sess = tf.InteractiveSession()
    
    """# ==============================================================================
    
    Placeholders
    
    Compute the size of various layers 
    
    Create a tensorflow Placeholder for each feature of data returned from the
    dataset
    
    """# ==============================================================================

    

    lst = []
    extra_features_width = 0 # width of extra features
    
    for i,nm in enumerate(output_feature_list):
        
        # features[0], is the target, 'm_label_one_hot' 
        # the second features[1] is the 'image' that is passed to the convolution layers 
        # Any additional features bypass the convolution layers and go directly 
        # into the fully connected layer.  
        
        # The width of the extra features is calculated in order to allocate 
        # the correct widths of weights,  # and inputs 
        # names are assigned to make the look pretty on the tensorboard graph.
        
        if i == 0:
            nm = 'y_'+nm
        else:
            nm = 'x_'+nm
        if i>1:
            extra_features_width += ds.train.feature_width[i]
        lst.append(tf.placeholder(tf.float32, shape=[None, ds.train.feature_width[i]], name=nm))
        
    # ph is a named tuple with key names like 'image', 'm_label', and values that
    # are tensors.  The display name on the Chrome graph are 'y_m_label', 'x_image, 
    # x_upper_case etc.
    Place_Holders = namedtuple('Place_Holders', output_feature_list)   
    ph = Place_Holders(*lst) # unpack placeholders into named Tuple
        
    nRows = ds.train.num_rows #image height
    nCols = ds.train.num_columns #image width    
    nFc = 1024      # size of fully connected layer
    nConv1 = 32     # size of first convolution layer
    nConv2 = 64     # size of second convolution layer
    nTarget = ds.train.feature_width[0]  # the number of one_hot features in the target, 'm_label'    
    n_h_pool2_outputs = int(nRows/4) * int(nCols/4) * nConv2 # second pooling layer 
    n_h_pool2_outputsx = n_h_pool2_outputs + extra_features_width # fully connected
        
    """# ==============================================================================
    
    Build a Multilayer Convolutional Network
    
    Weight Initialization
    
    """# ==============================================================================
    
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    """# ==============================================================================
    
    Convolution and Pooling
    
    keep our code cleaner, let's also abstract those operations into functions.
    
    """# ==============================================================================
    
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                            strides=[1, 2, 2, 1], padding='SAME')
    
    """# ==============================================================================
    
    First Convolutional Layer
    
    """# ==============================================================================
    with tf.name_scope("w_conv1") as scope:
        W_conv1 = weight_variable([5, 5, 1, nConv1])
        b_conv1 = bias_variable([nConv1])    
    
    with tf.name_scope("reshape_x_image") as scope:
        x_image = tf.reshape(ph.image, [-1,nCols,nRows,1])
    
    image_summ = tf.image_summary("x_image", x_image)
    
    """# ==============================================================================
    
    We then convolve x_image with the weight tensor, add the bias, apply the ReLU function,
     and finally max pool.
    
    """# ==============================================================================
    
    with tf.name_scope("convolve_1") as scope:
        h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
        
    with tf.name_scope("pool_1") as scope:    
        h_pool1 = max_pool_2x2(h_conv1)
    
    """# ==============================================================================
    
    Second Convolutional Layer
    
    In order to build a deep network, we stack several layers of this type. The second 
    layer will have 64 features for each 5x5 patch.
    
    """# ==============================================================================
    
    with tf.name_scope("convolve_2") as scope:
        W_conv2 = weight_variable([5, 5, nConv1, nConv2])
        b_conv2 = bias_variable([64])
        h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  
         
    with tf.name_scope("pool_2") as scope:
        h_pool2 = max_pool_2x2(h_conv2)
    
    """# ==============================================================================
    
    Densely Connected Layer
    
    Now that the image size has been reduced to 7x7, we add a fully-connected layer 
    with neurons to allow processing on the entire image. We reshape the tensor 
    from the pooling layer into a batch of vectors, multiply by a weight matrix, add 
    a bias, and apply a ReLU.
    
    """# ==============================================================================
    
    with tf.name_scope("W_fc1_b") as scope:
        W_fc1 = weight_variable([n_h_pool2_outputsx, nFc])
        b_fc1 = bias_variable([nFc])
            
        h_pool2_flat = tf.reshape(h_pool2, [-1, n_h_pool2_outputs])
        
        # append the features, the 2nd on, that go directly to the fully connected layer
        for i in range(2,ds.train.num_features ):
            h_pool2_flat = tf.concat(1, [h_pool2_flat, ph[i]])  
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    """# ==============================================================================
    
    Dropout
    
    """# ==============================================================================
    keep_prob = tf.placeholder(tf.float32,name='keep_prob')
    
    with tf.name_scope("drop") as scope:
        h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    """# ==============================================================================
    
    Readout Layer
    
    """# ==============================================================================
    with tf.name_scope("softmax") as scope:
        W_fc2 = weight_variable([nFc, nTarget])
        b_fc2 = bias_variable([nTarget])    
        y_conv=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    
    """# ==============================================================================
    
    Train and Evaluate the Model
    
    """# ==============================================================================
    
    with tf.name_scope("xent") as scope:
        # 1e-8 added to eliminate the crash of training when taking log of 0
        cross_entropy = -tf.reduce_sum(ph.m_label_one_hot*tf.log(y_conv+1e-8))
        ce_summ = tf.scalar_summary("cross entropy", cross_entropy)
            
    with tf.name_scope("train") as scope:
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
        
    with tf.name_scope("test") as scope:        
        correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(ph.m_label_one_hot,1))
    
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        accuracy_summary = tf.scalar_summary("accuracy", accuracy)    
    
    merged = tf.merge_all_summaries()
    tm = ""
    tp = datetime.datetime.now().timetuple()
    for i in range(4):
        tm += str(tp[i])+'-'
    tm += str(tp[4])    
    writer = tf.train.SummaryWriter("/tmp/ds_logs/"+ tm, sess.graph_def)
    
    # To see the results in Chrome, 
    # Run the following in terminal to activate server.
    # tensorboard --logdir '/tmp/ds_logs/'
    # See results on localhost:6006 
    
    sess.run(tf.initialize_all_variables())
    
    perfect_count=10
    for i in range(nEpochs):
    
        batch = ds.train.next_batch(100)
        # assign feature data to each placeholder
        # the batch list is returned in the same order as the features requested
        feed = {keep_prob: 0.5}
        for j in range(ds.train.num_features):
            feed[ph[j]] = batch[j]  
            
        if i%100 == 0:
            # sh=h_pool2_flat.get_shape()
            feed[keep_prob] = 1.0
            result = sess.run([merged, accuracy ], feed_dict=feed)    
            summary_str = result[0]
            #acc = result[1]       
            writer.add_summary(summary_str, i)
            train_accuracy = accuracy.eval(feed)    
            if train_accuracy != 1:
                perfect_count=10;
            else:
                perfect_count -= 1
                if perfect_count==0:
                    break;  
                
            print ("step %d, training accuracy %g"%(i, train_accuracy),flush=True)
        train_step.run(feed_dict=feed)
    
    def computeSize(s,tens):
        sumC = 1
        tShape = tens.get_shape()
        nDims = len(tShape)
        for i in range(nDims):
            sumC *= tShape[i].value
        print ('\t{}\t{}'.format(s,sumC),flush=True)
        return sumC
            
    print ('network size:',flush=True)
    total = computeSize("W_fc1",W_fc1)+ \
    computeSize ("b_fc1",b_fc1) + \
    computeSize ("W_conv1",W_conv1) + \
    computeSize ("b_conv1",b_conv1) + \
    computeSize ("W_conv2",W_conv2) + \
    computeSize ("b_conv2",b_conv2) + \
    computeSize ("W_fc2",W_fc2) + \
    computeSize ("b_fc2",b_fc2)
    print('\ttotal\t{}'.format(total),flush=True)
    
    feed={keep_prob: 1.0}
    # assign feature data to each placeholder
    error_images = np.empty((0,nRows,nCols))
        
    test_accuracy=0
    m=0
    for n in range(0,ds.test.features[0].shape[0],100 ):   
        for i in range(ds.train.num_features ):  
            feed[ph[i]] = ds.test.features[i] [n:n+100]
        result = sess.run([accuracy, x_image, W_conv1, correct_prediction], feed_dict=feed)    
        test_accuracy += result[0]
        error_images = np.append(error_images, result[1][:,:,:,0][result[3]==False],axis=0)
        m += 1
                         
    print ("test accuracy {} for font: {}".format(test_accuracy/m, input_filters_dict['font']),flush=True)       
    ocr_utils.montage(error_images,title='TensorFlow {} Error Images'.format(input_filters_dict['font']))
    
    tf.reset_default_graph() # only necessary when iterating through fonts
    sess.close()

    
if True:
    # single font train
    
    # esamples
    # select only images from 'OCRB'  scanned font
    # input_filters_dict = {'font': ('OCRA',)}
    
    # select only images from 'HANDPRINT'  font
    #input_filters_dict = {'font': ('HANDPRINT',)}
    
    # select only images from 'OCRA' and 'OCRB' fonts with the 'scanned" fontVariant
    # input_filters_dict = {'font': ('OCRA','OCRB'), 'fontVariant':('scanned',)}
    
    # select everything; all fonts , font variants, etc.
    # input_filters_dict = {}
    
    # select the digits 0 through 9 in the E13B font
    # input_filters_dict = {'m_label': range(48,58), 'font': 'E13B'}
    
    # select the digits 0 and 2in the E13B font
    # input_filters_dict = {'m_label': (48,50), 'font': 'E13B'}
    
    # output the character label, image, italic flag, aspect_ratio and upper_case flag
    # output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case']    
    
    # output only the character label and the image
    # output_feature_list = ['m_label_one_hot','image'] 

    # train the digits 0-9 for all fonts
    input_filters_dict = {'m_label': range(48,58)}
    output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case']    
    train_a_font(input_filters_dict,  output_feature_list, nEpochs = 5000)    
    
else:
    # loop through all the fonts and train individually

    # pick up the entire list of fonts and font variants. Train each one.
    lst = ocr_utils.get_list(input_filters_dict={'font': ()})      
    
    import pprint as pprint
    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(lst)
   
    output_feature_list = ['m_label_one_hot','image','italic','aspect_ratio','upper_case']
    
    # Change nEpochs to 5000 for better results
    for l in lst:
        input_filters_dict= {'font': (l[0],)}       
        train_a_font(input_filters_dict,output_feature_list, nEpochs = 1000) 
    
    
print ('\n########################### No Errors ####################################')


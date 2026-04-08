import numpy as np
import skimage.transform
from skimage import io
import tensorflow as tf
from skimage.measure import regionprops
from tensorflow.keras.preprocessing import image
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer, Input, Concatenate, Dense, Flatten, Lambda, Reshape, Multiply, Dot
from tensorflow.keras.layers import Conv2D, MaxPool2D, GlobalAveragePooling2D, AveragePooling2D, GlobalMaxPooling2D, ZeroPadding2D

from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler, CSVLogger, EarlyStopping
# from tensorflow.keras.applications.vgg16 import VGG16
# from tensorflow.keras.applications.resnet50 import ResNet50
# from tensorflow.keras.applications.inception_v3 import InceptionV3
# from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
# from tensorflow.keras.applications.xception import Xception
# from tensorflow.keras.applications.densenet import DenseNet201

from skimage.segmentation import slic
import collections
import pickle
from sklearn.preprocessing import MinMaxScaler
from math import ceil
    
_EPSILON = 10e-8

class Softmax4D(Layer):
    '''Apply softmax fully convolutionally'''

    def __init__(self, axis=-1,**kwargs):
        self.axis=axis
        super(Softmax4D, self).__init__(**kwargs)

    def build(self,input_shape):
        pass

    def call(self, x,mask=None):
        e = K.exp(x - K.max(x, axis=self.axis, keepdims=True))
        s = K.sum(e, axis=self.axis, keepdims=True)
        return e / s

    def get_output_shape(self, input_shape):
        axis_index = self.axis % len(input_shape)
        return input_shape
                      #if i != axis_index ])

def load_base_model( model_name ):
    '''Load pre-trained model.

    Parameters:
    model_name - one of ResNet50, VGG16, InceptionV3, InceptionResNetV2, Xception, DenseNet201

    Returns:
    TF model with pre-trained weights and no softmax layer
    '''

    max_dim = None
    max_channel = 38  #change dimension here diabetes 38/brain tumor 18/breast cancer 52
    input_tensor = Input(shape=(max_dim,max_dim,max_channel)) # change here
    if model_name.lower() == 'resnet50':
        from tensorflow.keras.applications.resnet50 import ResNet50
        from tensorflow.keras.applications.resnet50 import preprocess_input
        base_model = ResNet50(input_shape=(max_dim,max_dim,max_channel),include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'vgg16':
        from tensorflow.keras.applications.vgg16 import VGG16
        from tensorflow.keras.applications.vgg16 import preprocess_input
        base_model = VGG16(input_shape=(max_dim,max_dim,max_channel),include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'inceptionv3':
        from tensorflow.keras.applications.inception_v3 import InceptionV3
        from tensorflow.keras.applications.inception_v3 import preprocess_input
        base_model = InceptionV3(input_tensor=input_tensor,include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'inceptionresnetv2':
        from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
        from tensorflow.keras.applications.inception_resnet_v2 import preprocess_input
        base_model = InceptionResNetV2(input_tensor=input_tensor,include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'xception':
        from tensorflow.keras.applications.xception import Xception
        from tensorflow.keras.applications.xception import preprocess_input
        base_model = Xception(input_tensor=input_tensor,include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'densenet201':
        from tensorflow.keras.applications.densenet import DenseNet201
        from tensorflow.keras.applications.densenet import preprocess_input
        base_model = DenseNet201(input_tensor=input_tensor,include_top=False,weights=None) #'imagenet'
    elif model_name.lower() == 'customize':
        from tensorflow.keras.applications.densenet import preprocess_input
        input_layer = tf.keras.layers.Input(shape=(max_dim, max_dim, max_channel))

        conv1 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, padding="same", activation=tf.nn.relu)(input_layer)
        pool1 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

        conv2 = tf.keras.layers.Conv2D(filters=1024, kernel_size=3, padding="same", activation=tf.nn.relu)(pool1)
        pool2 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv2)

        conv3 = tf.keras.layers.Conv2D(filters=2048, kernel_size=3, padding="same", activation=tf.nn.relu)(pool2)
        pool3 = tf.keras.layers.MaxPooling2D(pool_size=2, strides=2)(conv3)

        dense1 = tf.keras.layers.Dense(units=512, activation=tf.keras.activations.relu)(pool3)
        dense2 = tf.keras.layers.Dense(units=128, activation=None)(dense1)

        base_model = tf.keras.models.Model(inputs=input_layer, outputs=dense2)

    else:
        print('Error: unsupported model')
        sys.exit(1)
    # print(base_model.summary())
    return base_model,preprocess_input

def add_mi_layer( orig_model, classes, mi_type, quantiles=16, use_mask=False ):
    '''Add MI layer to existing model.

    Parameters:
    orig_model - TF model (typically pre-trained)
    classes - list of classes and labels, e.g., [('class1',[0,1]),('class2',[0,1,2])]
    mi_type - type of MI aggregation: None (default, mean pool features), mean, quantile
    quantiles - number of quantiles to use if mi_type is 'quantile' (default 16)
    use_mask - whether to apply mask to image when pooling
    '''

    top_layer = orig_model.output
    
    if use_mask:
        # downsize mask to match image downsize operations
        
        shape = orig_model.input_shape
        mask_input = Input(shape=(shape[1],shape[2],1))
        xmask = mask_input

        done_layers = []
        for layer in orig_model.layers:
            config = layer.get_config()
            if 'padding' in config:
                padding = config['padding']
            else:
                padding = 'valid'
            if 'strides' in config:
                strides = config['strides']
            else:
                strides = None
            if 'pool_size' in config:
                pool_size = config['pool_size']
            elif 'kernel_size' in config:
                pool_size = config['kernel_size']
            else:
                pool_size = 1
            if type(layer.input) is list:
                for l in layer.input:
                    if l.name in done_layers:
                        continue
                else:
                    done_layers.append(l.name)
            else:
                if layer.input.name in done_layers:
                    continue
                else:
                    done_layers.append(layer.input.name)
            if pool_size == 1 and ( strides is None or strides == (1,1) ) and padding in ['valid','same']:
                continue
            if strides == (1,1) and padding in ['valid','same']:
                continue
            if type(padding) is not str:
                xmask = ZeroPadding2D(padding)(xmask)
                padding = 'valid'
            xmask = AveragePooling2D(pool_size,strides,padding)(xmask)

    if use_mask:
        # normalized to sum to one
        xmask_norm = Lambda(lambda z: z / (K.sum(z, axis=(1,2), keepdims=True)+_EPSILON), output_shape=lambda input_shape:input_shape, name='mask_norm')(xmask)
        
    # loop through classes
    outputs = []
    for c,cl in classes:
        if mi_type is None:
            x = top_layer
        else:#if mi_type == 'mean':
            x = Conv2D(len(cl),(1,1),name='softmaxfc_'+c)(top_layer)
            x = Softmax4D(axis=1,name='softmax_'+c)(x)
            
        if not use_mask:
            x = GlobalAveragePooling2D()(x)
        else:
            x = Multiply(name='multiply_'+c)([x,xmask_norm])
            x = Lambda( lambda z: K.sum(z, axis=(1,2), keepdims=False), output_shape=lambda input_shape:(input_shape[0],input_shape[3]), name='sum_'+c)(x)

        if mi_type is None:
            x = Dense(len(cl), activation='softmax', name='softmax_'+c)(x)

        outputs.append(x)

    if use_mask:
        model = Model(inputs=[orig_model.input,mask_input], outputs=outputs)
    else:
        model = Model(inputs=orig_model.input, outputs=outputs)

    return model

def categorical_crossentropy_missing(target, output):
    """Loss function that ignores samples with a missing label (all 0s)."""

    target = K.cast(target,'float32')
    output = K.cast(output,'float32')
    # scale preds so that the class probas of each sample sum to 1
    output /= (K.sum(output,axis=1, keepdims=True)+_EPSILON)
    # avoid numerical instability with _EPSILON clipping
    output = K.clip(output, _EPSILON, 1.0 - _EPSILON)
    select = K.cast( K.greater_equal(K.max(target,axis=1),0.5), 'float32' )
    ce = -K.sum(target * K.log(output), axis=1)
    return K.sum( ce * select ) / (K.sum(select)+_EPSILON)

def categorical_accuracy_missing(y_true, y_pred):
    """Metric to calculate accuracy while ignoring samples with a missing label."""
    
    select = K.cast( K.greater_equal(K.max(y_true,axis=1),0.5), 'float32' )
    return K.sum(K.cast(K.equal(K.argmax(y_true, axis=1),
                                 K.argmax(y_pred, axis=1)),'float32')*select) / (K.sum(select)+_EPSILON)


class ImageSequence(Sequence):
    '''Generate image,label pairs for training.

    Parameters:
    image_dir - directory where images are stored
    image_list - list of lists of image files; each list is for a different sample
    labels - numpy array of labels for each sample
    classes - list of classes and labels, e.g., [('class1',[0,1]),('class2',[0,1,2])]
    crop - size of image to randomly crop
    batch_size - batch size for training
    preprocess_input - function for preprocessing images
    sample_instances - max number of instances to use from each sample
    mask_list - same as image_list but for mask files (if needed)
    random - whether to sample randomly
    balance - draw samples so that class labels are balanced
    '''

    def __init__(self, image_dir, image_list, labels, classes, crop, batch_size, preprocess_input, sample_instances=1, mask_list=None, random=True, balance=False, test_crop=False, select_top_left=None, seg_pixel_num=None):
        self.image_dir = image_dir
        self.image_list = image_list
        self.labels = labels
        self.classes = classes
        self.crop = crop
        self.batch_size = batch_size
        self.preprocess_input = preprocess_input
        self.sample_instances = sample_instances
        self.mask_list = mask_list
        self.random = random
        self.balance = balance
        self.test_crop = test_crop
        self.select_top_left = select_top_left
        self.seg_pixel_num = seg_pixel_num
        self.slic_max_crop = 0

    def __len__(self):
        return int(np.ceil(np.sum([min(len(im),self.sample_instances) for im in self.image_list]) / float(self.batch_size)))
    
    def get_max_crop(self, in_max_crop):
        self.slic_max_crop = in_max_crop

    def __getitem__(self, idx):

        x_img = []
        x_mask = []
        y = []
        get_coordinate = []
        rot_list = []
        max_height, max_width = 0, 0
        max_square = 0
        segments_dict = {}
        img_dict = {}

        for i in range(idx*self.batch_size,(idx+1)*self.batch_size):
            if i >= len(self.image_list):
                break
            if self.balance:
                # choose image to balance classes
                i = np.random.choice(np.where(self.labels[:,c]==l)[0])
            ninst = len(self.image_list[i])
            rand_inst = np.random.choice(np.arange(ninst),min(self.sample_instances,ninst),replace=False)
            for j in rand_inst:
                img_fn = self.image_list[i][j]


                # read image file and get max crop size for superpixel cropping
                #diabetes                
                img = io.imread(self.image_dir + 'ims/' + img_fn ) 
                img = np.arcsinh(1. / 5 * img)

                
                # brain
                # img = io.imread(self.image_dir + img_fn ) 
                # img = np.transpose(img, (2, 1, 0))
                # img = np.arcsinh(1. / 5 * img)
                # img = img[:,:,[i for i in range(17)] + [18]]

                # breast cancer
                # img = io.imread(self.image_dir + img_fn ) 
                # img = np.transpose(img, (2, 1, 0))
                # img = np.arcsinh(1. / 5 * img)


                img_dict[img_fn] = img

                numSegments = 100 # change here for number of superpixels
                segments = slic(np.array(img), n_segments = numSegments, sigma = 5)
                segments_dict[img_fn] = segments

                regions = regionprops(segments)

                for region in regions:
                    minr, minc, maxr, maxc = region.bbox  # Get bounding box coordinates
                    height = maxr - minr
                    width = maxc - minc

                    max_height = max(max_height, height)
                    max_width = max(max_width, width)
                
                if self.slic_max_crop == 0:
                    self.slic_max_crop = ceil((max(max_height, max_width) + (numSegments/2))/10)* 10
                else:
                    if max(max_height, max_width) > self.slic_max_crop:
                        self.slic_max_crop = ceil((max(max_height, max_width) + (numSegments/2))/10)* 10

        # print("get max crop", max(max_height, max_width), self.slic_max_crop)
        for c in range(self.labels.shape[1]):
            y += [[]]
        for i in range(idx*self.batch_size,(idx+1)*self.batch_size):
            if i >= len(self.image_list):
                break

            if self.balance:
                # choose image to balance classes
                c = np.random.randint(0,len(self.classes))
                l = np.random.choice(self.classes[c][1])
                i = np.random.choice(np.where(self.labels[:,c]==l)[0])
            
            # randomly choose image from set
            ninst = len(self.image_list[i])
            rand_inst = np.random.choice(np.arange(ninst),min(self.sample_instances,ninst),replace=False)
            for j in rand_inst:
                img_fn = self.image_list[i][j]
                if self.mask_list is not None:
                    mask_fn = self.mask_list[i][j]

                
                img = img_dict[img_fn]
                img = np.array(img)
                # print("file in", img_fn, img.shape)
                
                #sub set storage
                zero_collect = []
                seg_x_img = []
                seg_x_mask = []
                sub_y = []

                # read mask
                segments = segments_dict[img_fn]



                # TODO: pad images with zero to fit largest size
                if self.crop[0] > img.shape[0] or self.crop[1] > img.shape[1]:
                    pad = ( (self.crop[0]-img.shape[0])//2, (self.crop[1]-img.shape[1])//2 )
                    img = skimage.util.pad( img, pad_width=pad, mode='constant' )

                if self.test_crop: 
                    start_crop_times = self.seg_pixel_num if self.seg_pixel_num <= np.max(segments) else np.max(segments)
                    random_crop_times = start_crop_times + 1 # fix code here for define mask for priticular superpixel mask
                    # print("check here select crop:", start_crop_times)
                else: 
                    start_crop_times = 1
                    random_crop_times = np.max(segments)+1

                for seg_num in range(start_crop_times, random_crop_times ,1): # np.max(segments)+1
                    mask = [ [ 1 if segments[i][j] == seg_num else 0 for j in range(len(segments[0]))] for i in range(len(segments))]
                    mask = np.array(mask)
                    mask_idx = np.where(mask == 1)
                    
                    if self.select_top_left!= None:
                        image_cor = self.select_top_left[i][j]
                        top = image_cor[0]
                        left = image_cor[1]
                        crop_size = self.crop[0]
                        # print("test test test",crop_size)
                    else:
                        crop_size = self.slic_max_crop # max(max_height, max_width)
                        top = min(np.min(mask_idx[0]),img.shape[0]-crop_size)
                        left = min(np.min(mask_idx[1]),img.shape[1]-crop_size)
                    
                    img_crop = img[top:top+crop_size, left:left+crop_size, : ]
                    
                    get_coordinate.append((top, top+crop_size, left, left+crop_size)) # check coor
                    # print("img crop area", top, top+self.crop[0], left, left+self.crop[1])
                    if self.mask_list is not None:
                        mask_crop = mask[top:top+crop_size,left:left+crop_size]
                        # shape1 = mask.shape
                        mask_crop = np.expand_dims( mask_crop, axis=0 )
                        mask_crop = np.expand_dims( mask_crop, axis=3 )

                    zero_collect_mask = mask[np.min(mask_idx[0]):np.max(mask_idx[0])+1,np.min(mask_idx[1]):np.max(mask_idx[1])+1]
                    counter_collect = collections.Counter(zero_collect_mask.flatten())
                    zero_collect.append(counter_collect[0])
                    img_crop = np.expand_dims( img_crop, axis=0 )

                    seg_x_img.append(img_crop) # add image

                    if self.mask_list is not None:
                        seg_x_mask.append(mask_crop) # add mask

                    for c in range(len(y)):
                        # y[c].append(self.labels[i,c])
                        sub_y.append(self.labels[i,c])
                if self.select_top_left!= None or self.test_crop:
                    zero_rank_index = [pick_index for pick_index in range(len(zero_collect))]
                else:
                    # zero_rank_index = random.sample(range(len(zero_collect)), 15)
                    zero_rank_index = [zero_collect.index(max_zero_count) for max_zero_count in sorted(zero_collect, reverse=True)[:30]]
                for rank_index in zero_rank_index:
                    img_crop = seg_x_img[rank_index]
                    img_crop = self.preprocess_input( img_crop )
                    x_img.append(img_crop)
                    if self.mask_list is not None:
                        x_mask.append(seg_x_mask[rank_index])
                    y[c].append(sub_y[rank_index])

        x_img = np.concatenate(x_img,axis=0)

        # convert labels to categorical
        y_cat = []
        for yi,cl in zip(y,self.classes):
            yi = np.array(yi,dtype='float16')
            yi2 = to_categorical(yi,len(cl[1]))
            yi2[yi==-1,:] = 0
            y_cat.append(yi2)

        if self.mask_list is not None:
            x_mask = np.concatenate(x_mask,axis=0)
            # print("check shape: ", x_img.shape, x_mask.shape, y_cat[0].shape)
            return (x_img,x_mask),y_cat
        return x_img,y_cat

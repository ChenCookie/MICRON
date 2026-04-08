import os
import sys
import argparse
import numpy as np
import skimage.io
import sklearn.metrics

import tensorflow as tf
from tensorflow.keras.optimizers import Adam, Nadam
from skimage import io

import util
import cnn

from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.utils import shuffle
import pickle
import random
import pandas as pd

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc


class PrintLayerOutput(tf.keras.callbacks.Callback):
    def __init__(self, layer_name):
        super().__init__()
        self.layer_name = layer_name
        self.layer_output = None

    def on_epoch_end(self, epoch, logs=None):
        print(logs)
        print(f'Output of layer "{self.layer_name}" after epoch {epoch + 1}')

if __name__ == "__main__":

    parser = argparse.ArgumentParser( description='Train CNN with MI learning.' )
    parser.add_argument('--in_dir', '-i', required=True, help='input directory' )
    parser.add_argument('--out_dir', '-o', required=True, help='output directory' )
    parser.add_argument('--in_model', help='Filename of model to initialize with' )
    parser.add_argument('--out_model', help='Filename of model to save' )
    parser.add_argument('--save_results', help='Save predictions on test set to file' )
    parser.add_argument('--test_only', action='store_true', help='Test only' )
    #parser.add_argument('--prior', action='store_true', help='Use prior probabilities from training set' )
    parser.add_argument('--fold', '-f', required=True, help='Cross validation fold #' )
    parser.add_argument('--cat', help='label categories to train (comma separated); default: all' )
    parser.add_argument('--model', '-m', required=True, help='CNN model' )
    parser.add_argument('--crop', '-c', help='Crop size' )
    parser.add_argument('--test_crop', help='Test crop size (default: 3000)' )
    parser.add_argument('--mi', help='MI aggregation type (mean, quantile)' )
    parser.add_argument('--quantiles', '-q', help='Number of quantiles; default: 16' )
    parser.add_argument('--rate', '-r', help='Learning rate; if cyclic "r0,r1"' )
    parser.add_argument('--lr_range', help='Learning rate test; "start,stop,steps"' )
    parser.add_argument('--batch_size', '-b', help='Batch size' )
    parser.add_argument('--epochs', '-e', help='Epochs' )
    parser.add_argument('--init_epoch', help='Initial epoch' )
    parser.add_argument('--mask', action='store_true', help='use mask' )
    parser.add_argument('--freeze', action='store_true', help='freezer lower layers' )
    parser.add_argument('--balance', action='store_true', help='balance training samples by class labels' )
    parser.add_argument('--n_jobs', help='number of parallel threads' )
    parser.add_argument('--gpu', '-g', help='selected GPU' )
    args = parser.parse_args()


    src_dir = args.in_dir
    if len(src_dir) > 1 and src_dir[-1] != '/':
        src_dir += '/'
    out_dir = args.out_dir
    if len(out_dir) > 1 and out_dir[-1] != '/':
        out_dir += '/'
    in_model = args.in_model
    out_model = args.out_model
    save_results = args.save_results
    test_only = args.test_only
    fold = args.fold
    categories = args.cat
    model_name = args.model
    crop = args.crop
    if crop is not None:
        crop = crop.split(',')
        if len(crop) == 1:
            crop = (int(crop[0]),int(crop[0]))
        else:
            crop = (int(crop[0]),int(crop[1]))
    test_crop = args.test_crop
    if test_crop is not None:
        test_crop = test_crop.split(',')
        if len(test_crop) == 1:
            test_crop = (int(test_crop[0]),int(test_crop[0]))
        else:
            test_crop = (int(test_crop[0]),int(test_crop[1]))
    mi_type = args.mi
    quantiles = args.quantiles
    lr = args.rate
    if lr is not None:
        lr = lr.split(',')
        lr = [float(r) for r in lr]
    lr_range = args.lr_range
    if lr_range is not None:
        lr_range = args.lr_range.split(',')
        if len(lr_range) == 1:
            lr_range = float(lr_range[0])
        else:
            lr_range = [float(r) for r in lr_range]
    batch_size = args.batch_size
    if batch_size is not None:
        batch_size = int(batch_size)
    epochs = args.epochs
    if epochs is not None:
        epochs = int(epochs)
    init_epoch = int(args.init_epoch) if args.init_epoch is not None else 0
    use_mask = args.mask
    freeze = args.freeze
    balance = args.balance
    n_jobs = int(args.n_jobs) if args.n_jobs is not None else 1
    gpu = args.gpu

    print("out_dir =", out_dir)
    print("out_model =", out_model)
    print("final path =", out_model)

    # load filenames and labels
    sample_images = util.load_sample_images( out_dir )
    samples,cats,labels = util.load_labels( out_dir )

    max_inst = max([len(si) for si in sample_images.values()])
    # print('max instances',max_inst)

    # load filenames and labels
    image_list = util.load_image_list( out_dir )
    if use_mask:
        mask_list = util.load_mask_list( out_dir )
        sample_masks = util.load_sample_masks( out_dir )
    else:
        mask_list = [None]*len(image_list)
        sample_masks = {}

        
    if categories is None:
        categories = cats
    else:
        categories = categories.split(',')
        
    # get labels for list of categories
    label_names = []
    new_labels = np.zeros((labels.shape[0],len(categories)),dtype='int')
    for i,cat in enumerate(categories):
        c = np.where(cats==cat)[0][0]
        ln = np.unique([l[c] for l in labels])
        ln.sort()
        ln = list(ln)
        if '' in ln:
            del ln[ln.index('')]
        label_names.append( ln )
        new_labels[:,i] = np.array([ ln.index(l) if l in ln else -1 for l in labels[:,c] ])
    labels = new_labels
    cats = categories

    # create list of class names for each category
    classes = []
    for c in range(len(cats)):
        cl = np.unique(labels[:,c])
        np.sort(cl)
        if cl[0] == -1:
            cl = cl[1:]
        classes.append((cats[c],cl))

    # split into train/test sets
    if fold is not None:
        print("see fold num", str(fold))
        idx_train_val_test = util.load_cv_files( out_dir, samples, 'fold'+str(fold)+'.csv' )[0]
        idx_train  = idx_train_val_test[0]
        idx_val = idx_train_val_test[1]
        idx_test = idx_train_val_test[2]
    else:
        idx_train = np.arange(len(samples))
        idx_val = np.arange(0)
        idx_test = np.arange(0)


    # drop samples with missing label for all categories
    idx_train = np.array( [ i for i in idx_train if (labels[i,:]!=-1).sum()>0 ] )
    idx_val = np.array( [ i for i in idx_val if (labels[i,:]!=-1).sum()>0 ] )

    if gpu is not None:

        
        tf.keras.backend.set_floatx('float16')
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu

    base_model,preprocess_input = cnn.load_base_model( model_name )
    model = cnn.add_mi_layer( base_model, classes, mi_type, use_mask=use_mask )



    if in_model is not None:
        print('Initializing with '+in_model)
        model.load_weights(out_dir+in_model)

    if not test_only:
        if freeze:
            for layer in model.layers:
                if 'softmax' not in layer.name:
                    layer.trainable = False

        cat_loss = cnn.categorical_crossentropy_missing
        cat_acc = cnn.categorical_accuracy_missing

        # train and val generators
        gen_train = cnn.ImageSequence( src_dir, [sample_images[samples[s]] for s in idx_train], labels[idx_train,:], classes, crop, batch_size, preprocess_input, mask_list=[sample_masks[samples[s]] for s in idx_train] if use_mask else None, random=True, balance=balance, test_crop=False)
        gen_val = cnn.ImageSequence( src_dir, [sample_images[samples[s]] for s in idx_val], labels[idx_val,:], classes, crop, batch_size, preprocess_input, mask_list=[sample_masks[samples[s]] for s in idx_val] if use_mask else None, random=False, test_crop=False)#, sample_instances=max_inst )

        model.compile( optimizer=Nadam(lr=lr[0]), loss=[cat_loss]*len(model.outputs), metrics=[cat_acc] )
        print("check after model compile")

        callbacks = []
        if lr_range is not None:
            # lr range test
            from keras_one_cycle_clr.keras_one_cycle_clr.lr_range_test import LrRangeTest
            lrrt_cb = LrRangeTest( lr_range=(lr_range[0],lr_range[1]), wd_list=[0], steps=lr_range[2], batches_per_step=100//batch_size, validation_data=gen_val, batches_per_val=50, verbose=True, custom_objects={'categorical_crossentropy_missing':cat_loss,'categorical_accuracy_missing':cat_acc} )
            n_epochs = lrrt_cb.find_n_epoch(gen_train)
            model.fit( gen_train, epochs=n_epochs, max_queue_size=5, workers=n_jobs, callbacks=[lrrt_cb] ) # fit_generator
            lrrt_cb.plot()
            sys.exit(0)

        if len(lr) > 1 :
            # cyclic lr
            from keras_one_cycle_clr.keras_one_cycle_clr.cyclic_lr import CLR
            from keras_one_cycle_clr.keras_one_cycle_clr.utils import plot_from_history
            clr_cb = CLR( cyc=lr[2], lr_range=(lr[0],lr[1]), momentum_range=(0.95, 0.85), verbose=True, amplitude_fn=lambda x: np.power(1.0/3, x) )
            clr_hist = model.fit( gen_train, epochs=epochs, validation_data=gen_val, max_queue_size=5, workers=n_jobs, shuffle=True ) # fit_generator
            plot_from_history(clr_hist)
        else:
            model.fit( gen_train, epochs=epochs, verbose=2, validation_data=gen_val, max_queue_size=5, workers=n_jobs, shuffle=True, initial_epoch=init_epoch) # fit_generator


        # save model
        if out_model is not None:
            print("out_dir =", out_dir)
            print("out_model =", out_model)
            print("final path =", out_dir + out_model)
            model.save( out_model )
    
        
    # predict on test data
    if save_results is not None or test_only:
        # put all crop into list
        test_images = [ sample_images[samples[s]] for s in idx_test ]
        if use_mask:
            test_masks = [ sample_masks[samples[s]] for s in idx_test ]
        test_labels = labels[idx_test,:]
        test_labels = np.array([ l for images,l in zip(test_images,test_labels) for s in images ])
        test_inst2sample = np.array([ s for s in range(len(test_images)) for j in test_images[s] ])
        test_images = [ [inst] for im in test_images for inst in im ]
        if use_mask:
            test_masks = [ [inst] for im in test_masks for inst in im ]
        test_labels = np.array(test_labels)

        test_img_embedding = {}
        select_crop_times = 110 # change each crop size here if needed 

        for j in range(0, select_crop_times, 1):


            gen_test = cnn.ImageSequence( src_dir, test_images, test_labels, classes, test_crop, 1, preprocess_input, mask_list=test_masks if use_mask else None, random=False , test_crop=True, seg_pixel_num=j+1)
            p = model.predict( gen_test, steps=len(test_images), max_queue_size=5, workers=n_jobs, use_multiprocessing=True ) # .predict_generator


            input3_output_model = tf.keras.models.Model(
                inputs=model.input,
                outputs=model.get_layer('conv5_block3_out').output
            )

            input3_output = input3_output_model.predict(
                gen_test,
            )

            for x in gen_test:
                print("GEN SHAPE =", x[0].shape)
                break

            if isinstance(input3_output, tf.RaggedTensor):
                input3_output = input3_output.to_tensor() 

            if tf.is_tensor(input3_output):
                input3_output = input3_output.numpy()
            else:
                input3_output = np.asarray(input3_output)



            n_instances = input3_output.shape[0] 
            train_input_data = input3_output.reshape(n_instances, -1)

            for get_filename in range(len(test_images)):
                if test_images[get_filename][0] not in test_img_embedding:
                    test_img_embedding.setdefault(test_images[get_filename][0], np.empty((0,train_input_data.shape[1]), float))
                    # test_img_embedding.setdefault(test_images[get_filename][0] + " probability", np.empty((0,2), float))
                    # test_img_embedding.setdefault(test_images[get_filename][0] + " coordinate", np.empty((0,2), float))
                temp_embedding = test_img_embedding[test_images[get_filename][0]]
                temp_embedding  = np.append(temp_embedding, [train_input_data[get_filename]], axis=0)
                test_img_embedding[test_images[get_filename][0]] = temp_embedding

                # temp_p = test_img_embedding[test_images[get_filename][0] + " probability"]
                # temp_p  = np.append(temp_p, [p[get_filename]], axis=0)
                # test_img_embedding[test_images[get_filename][0] + " probability"] = temp_p

        

        # save the pickle file name imc_brain_testimg_dict_seg70_sel30_white_cleanchannellocfixcrop2_fold' + str(fold) + '_
        f = open('test.pkl',"wb")
        pickle.dump(test_img_embedding,f)
        f.close()


    

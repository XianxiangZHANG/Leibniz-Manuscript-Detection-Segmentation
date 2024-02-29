
# coding: utf-8

import sys, getopt

import argparse 

import sys
import os 
import time 
from datetime import datetime 

import numpy as np 

import pandas as pd 
from sklearn.cluster import KMeans 
from PIL import Image, ImageChops, ImageDraw 
import PIL.ImageOps

import matplotlib as mlp
mlp.use('Agg') 

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg 
import matplotlib.pyplot  as pyplot
import gzip 

import theano 
import theano.tensor as T 

import lasagne
#import nolearn
import csv
import gzip, pickle


sys.path.append('../DataBase/')
import ast


def SpatialSoftmax(x):
    exp_x = T.exp(x - x.max(axis=1, keepdims=True))
    exp_x /= exp_x.sum(axis=1, keepdims=True)
    return exp_x

# This code defines a deep neural network model
def build_map_size(input_var=None, sizeX=1024, sizeY=1024):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, sizeX, sizeY), input_var=input_var)
    print(lasagne.layers.get_output_shape(l_in))

    l_ind2 = lasagne.layers.Pool2DLayer(l_in, pool_size=(2,2), mode='max' )
    l_ind2 = lasagne.layers.Upscale2DLayer(l_ind2, scale_factor=2, mode='repeat')
    l_ind4 = lasagne.layers.Pool2DLayer(l_in, pool_size=(4,4), mode='max' )
    l_ind4 = lasagne.layers.Upscale2DLayer(l_ind4, scale_factor=4, mode='repeat')
    l_ind8 = lasagne.layers.Pool2DLayer(l_in, pool_size=(8,8), mode='max' )
    l_ind8 = lasagne.layers.Upscale2DLayer(l_ind8, scale_factor=8, mode='repeat')
    l_ind16 = lasagne.layers.Pool2DLayer(l_in, pool_size=(16,16), mode='max' )
    l_ind16 = lasagne.layers.Upscale2DLayer(l_ind16, scale_factor=16, mode='repeat')

    l_pyr = lasagne.layers.ConcatLayer([l_in, l_ind2, l_ind4, l_ind8, l_ind16], axis=1)
    # l_pyr = lasagne.layers.ConcatLayer([l_in, l_ind2, l_ind4], axis=1)
    print(lasagne.layers.get_output_shape(l_pyr))

    l_net = lasagne.layers.Conv2DLayer(l_pyr, 
                                        num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify, W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )

    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=64, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=128, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    l_net = lasagne.layers.MaxPool2DLayer(l_net, pool_size=(2,2) )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Conv2DLayer( l_net, num_filters=256, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), pad='same' )
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 128, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 64, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    l_net = lasagne.layers.Upscale2DLayer(l_net, scale_factor=2, mode='repeat')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 32, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    l_net = lasagne.layers.Upscale2DLayer(l_net, scale_factor=2, mode='repeat')
    print(lasagne.layers.get_output_shape(l_net))

    l_net = lasagne.layers.Deconv2DLayer( l_net, 16, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    #l_net = lasagne.layers.Deconv2DLayer( l_net, 16, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.elu , W=lasagne.init.GlorotUniform(), crop='same')
    #print(lasagne.layers.get_output_shape(l_net))
    l_net = lasagne.layers.Deconv2DLayer( l_net, 8, filter_size=(5,5), nonlinearity=lasagne.nonlinearities.rectify , W=lasagne.init.GlorotUniform(), crop='same')
    print(lasagne.layers.get_output_shape(l_net))

    l_out = lasagne.layers.Conv2DLayer( l_net, num_filters=3, filter_size=(5,5), nonlinearity=SpatialSoftmax  , W=lasagne.init.GlorotUniform(), pad='same')
    print(lasagne.layers.get_output_shape(l_out))
    l_out = lasagne.layers.Upscale2DLayer(l_out, scale_factor=4, mode='repeat')
    print(lasagne.layers.get_output_shape(l_out))
    return l_out


def calculate_bounding_rectangle(points_list,w,h):
    points = ast.literal_eval(points_list)
    x_coords = [point[0] for point in points]
    y_coords = [point[1] for point in points]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    return int(min_x*1024/w), int(min_y*1024/h), int(max_x*1024/w), int(max_y*1024/h)



def readImage(image_path, mask_path):
   
    thisImg =Image.open(image_path)
    csvData = pd.read_csv(mask_path, encoding='ISO-8859-1')
    csvData.head()
    width, height = thisImg.size
    print(width,height)
    thisGray = thisImg.convert('L')
    resized_image = thisGray.resize((1024, 1024), Image.ANTIALIAS)
    print(resized_image.size)

    mask_structures= np.zeros((5,1024,1024))
    prob_structures= np.zeros((5,1024,1024))

    mask_structures[0,:,:] = 1
    mask_structures[1,:,:] = 0
    mask_structures[2,:,:] = 0
    mask_structures[3,:,:] = 0
    mask_structures[4,:,:] = 0
    prob_structures[0,:,:] = 1
    
    for index, row in csvData.iterrows():
        bounding_rectangle= calculate_bounding_rectangle(row[2], width, height)
        
        print(row['Label'])
        if 'quation' in row['Label']:
            mask_structures[0, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0
            mask_structures[1, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 1
            mask_structures[2, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 1
            mask_structures[3, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0
            mask_structures[4, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0

        elif 'exte' in row['Label']:
            mask_structures[0, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0
            mask_structures[1, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0
            mask_structures[2, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 0
            mask_structures[3, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 1
            mask_structures[4, bounding_rectangle[1]:bounding_rectangle[3],bounding_rectangle[0]:bounding_rectangle[2]] = 1

        else:
            print("Nothing！！！！！！！")

    nb_pixBg =  (np.sum(mask_structures[0,:,:]))
    nb_pixEq =  (np.sum(mask_structures[1,:,:])) + (np.sum(mask_structures[2,:,:]))
    nb_pixTe =  (np.sum(mask_structures[3,:,:])) + (np.sum(mask_structures[4,:,:]))

    p_pixBg = (1 + nb_pixBg) / (1 + nb_pixEq + nb_pixTe + nb_pixBg)
    p_pixEq = (1 + nb_pixEq) / (1 + nb_pixEq + nb_pixTe + nb_pixBg)
    p_pixTe = (1 + nb_pixTe) / (1 + nb_pixEq + nb_pixTe + nb_pixBg)
    

    prob_structures[0,:,:] = np.where(mask_structures[0,:,:] == 1, p_pixBg, 1-p_pixBg)
    prob_structures[1,:,:] = np.where(mask_structures[1,:,:] == 1, p_pixEq, 1-p_pixEq)
    prob_structures[2,:,:] = np.where(mask_structures[1,:,:] == 1, p_pixEq, 1-p_pixEq)
    prob_structures[3,:,:] = np.where(mask_structures[2,:,:] == 1, p_pixTe, 1-p_pixTe)
    prob_structures[4,:,:] = np.where(mask_structures[2,:,:] == 1, p_pixTe, 1-p_pixTe)
    
    input = np.array(resized_image, dtype=np.uint8)
    return input, mask_structures, prob_structures

def main(args):
    artidoc_denx = 150
    artidoc_deny = 100
    ### Set Network ###
    print("Set network")
    # Define the symbolic variable input_var for the model input, the variable target_var for the target label, 
    # the probability variable prob_var, the categorical variable class_var, the segmentation variable seg_var and the learning rate variable lr_var.
    # sizeInput is the size of the image, obtained from the command line parameter args.imgsize.
    # nn is a neural network model created by calling the build_map_size function.
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    prob_var = T.tensor4('probabilities')
    class_var = T.ivector('classes')
    seg_var = T.tensor4('segmentations')
    lr_var = T.scalar('learningrate')
    sizeInput = args.imgsize
    nn = build_map_size(input_var, sizeX=sizeInput[0], sizeY=sizeInput[1])

    # Responsible for initializing the parameters of the neural network.
    # First, it checks if pretrained model parameters are provided (whether args.preweights is true).
    # If pre-trained parameters are provided, it will load these parameters, 
    # otherwise the network will be initialized with default parameters.
    if args.preweights:
        modelnn = args.preweights
        with np.load(modelnn) as f:
            last_param_nn = [f['arr_%d' % i] for i in range(len(f.files))]
        pnn_init = last_param_nn
        lasagne.layers.set_all_param_values(nn, pnn_init)
    else:
        pnn_init = lasagne.layers.get_all_param_values(nn)
    lasagne.layers.set_all_param_values(nn, pnn_init)

    ### Function Definition ###
    print("Function Definition")

    # Train and evaluate neural network models to track losses during training and evaluate model performance on validation data.
    out_nn = lasagne.layers.get_output(nn)  # Output of neural network model nn.
    out_nn_softmax = T.exp(out_nn - out_nn.max(axis=1, keepdims=True)) 
    out_nn_softmax /= out_nn_softmax.sum(axis=1, keepdims=True)

    target_bb_bin = T.switch(target_var > 0, 1, 0)  
    t_nbPix = T.switch(target_var > -1, 1, 0)
    out_bb_bin = T.switch(out_nn > 0, 1, 0) 
    epsilon=0.0001 
    mse_bb = T.sum( prob_var * T.power( out_nn - target_var, 2 ), dtype=theano.config.floatX)  
    ce = -T.sum( prob_var * target_var * T.log(out_nn+epsilon)  ) 

    obj_nn = ce 

    params_nn = lasagne.layers.get_all_params(nn, trainable=True) 
    updates_nn = lasagne.updates.adam(obj_nn, params_nn)  

 
    train_fn_nn = theano.function([input_var, target_var, prob_var], obj_nn, updates=updates_nn)

    
    est_nn = lasagne.layers.get_output(nn, deterministic=True)
    
    est_nn_softmax = T.exp(est_nn - est_nn.max(axis=1, keepdims=True))
    est_nn_softmax /= est_nn_softmax.sum(axis=1, keepdims=True)

    
    est_mse_bb = T.sum( prob_var * T.power( est_nn- target_var, 2) , dtype=theano.config.floatX )
    
    est_ce = -T.sum( prob_var * target_var * T.log(est_nn+epsilon)  ) #/ T.sum(t_nbPix) 
    est_obj = est_ce

    # Create a Theano function eval_nn, which accepts input data (input_var), target data (target_var) and pixel probability data (prob_var),
    # Then calculate the loss function est_obj, which is used to evaluate model performance.
    eval_nn = theano.function([input_var, target_var, prob_var], est_obj )
    
    # Gets the output of a neural network in evaluation mode. 
    # This output is used to generate an image that represents the model's prediction of the input image
    out_img = lasagne.layers.get_output(nn, deterministic=True)
    out_img_softmax = T.exp(out_img - out_img.max(axis=1, keepdims=True))
    out_img_softmax /= out_img_softmax.sum(axis=1, keepdims=True)
    
    eval_img = theano.function([input_var], out_img)

    ### Training Preparation ###
    # Initialized variables used to track the training and validation process.
    # These variables include training loss, validation loss, some indicators of the validation results, and some iteration counters.
    # best_loss is initialized to a larger value to track the best model.
    res_train_loss = np.array([])
    res_valid_loss = np.array([])
    res_valid_db_loss = np.array([])
    res_valid_tp = np.array([])
    res_valid_tn = np.array([])
    res_valid_fp = np.array([])
    res_valid_fn = np.array([])
    res_valid_sen = np.array([])
    res_valid_spe = np.array([])
    res_valid_acc = np.array([])
    res_valid_pre = np.array([])
    res_valid_f1 = np.array([])
    res_valid_mcc = np.array([])
    
    seq_nb_img = np.array([])
    train_batches = 0
    val_db_loss = 0
    best_loss = 10000000000
    
    # The purpose of this code is to load previously saved results and states from previous training 
    # before continuing training so that loss values can continue to be recorded and compared during subsequent training.
    # If there is no previous result file, the training loss and validation loss will be initialized.
    if args.preresults:
        f = np.load(args.preresults)
        res_train_loss = f['train_loss']
        res_valid_loss = f['valid_loss']
        res_valid_db_loss = f['valid_db_loss']
        seq_nb_img = f['nb_img']
        train_batches = np.max(seq_nb_img)
        val_db_loss = f['valid_db_loss'][-1]
        best_loss = np.min(res_valid_db_loss)
    train_loss = 0
    val_loss = 0
    
    train_batches_sen, train_batches_mcc, batches_sen, batches_mcc = 0,0,0,0
    batches_sen
    SEN, SPE, ACC, PRE, F1, MCC = 0,0,0,0,0,0
    sen, spe, acc, pre, f1, mcc = 0,0,0,0,0,0
    TP, TN, FP, FN = 0,0,0,0
    batches = 0
    nbOutput = 3
    start_time = time.time()

    count = 300
    for epoch in np.arange( args.nepoch ):
        n = 0.005   
        batch = epoch 
        
        #Initialize the variables used to store training data:
        inputs = np.array([]).reshape( (0,1,sizeInput[0], sizeInput[1]) )
        probmaps = np.array([]).reshape( (0,5,sizeInput[0], sizeInput[1]) ) 
        maskmaps = np.array([]).reshape( (0,5,sizeInput[0], sizeInput[1]) ) 
        bckgds = np.array([]).reshape( (0,1,sizeInput[0], sizeInput[1]) ) 


        # You need to change the data path
        inpt, maskm, probm = readImage('../DataBase/training1024V3/image323'+str(count)+'.jpg',
                                         '../DataBase/training1024V3/image323'+str(count)+'.csv')
       
        inputs = np.concatenate( (inputs, inpt.reshape( (1, 1, sizeInput[0], sizeInput[1]) )) , axis=0 )
        maskmaps = np.concatenate( ( maskmaps, maskm.reshape( (1, 5, sizeInput[0], sizeInput[1])) ) , axis=0)
        probmaps = np.concatenate( ( probmaps, probm.reshape( (1, 5, sizeInput[0], sizeInput[1])) ) , axis=0)

        
        targets = np.array( maskmaps, dtype=theano.config.floatX)
        inputs = np.array( inputs, dtype=theano.config.floatX )
        probs = np.array( probmaps>0, dtype=theano.config.floatX )
        
        loss = train_fn_nn( inputs, targets[:,[0,1,3],:,:], probs[:,[0,1,3],:,:] )
        train_loss += loss
        train_batches += args.batchsize
       

        vals = np.array([]).reshape( (0,1,sizeInput[0], sizeInput[1]) )
        valmaskmaps = np.array([]).reshape( (0,5,sizeInput[0], sizeInput[1]) )
        valprobmaps = np.array([]).reshape( (0,5,sizeInput[0], sizeInput[1]) )

        # You need to change the data path
        val, valmaskm, valprobm = readImage('../DataBase/training1024V3/image323'+str(count)+'.jpg',
                                         '../DataBase/training1024V3/image323'+str(count)+'.csv')
        
        vals = np.concatenate( (vals, val.reshape( (1, 1, sizeInput[0], sizeInput[1]) )) , axis=0 )
        valmaskmaps = np.concatenate( (valmaskmaps, valmaskm.reshape( (1, 5, sizeInput[0], sizeInput[1])) ) , axis=0)
        valprobmaps = np.concatenate( (valprobmaps, valprobm.reshape( (1, 5, sizeInput[0], sizeInput[1])) ) , axis=0)

       
        valtargets = np.array( valmaskmaps, dtype=theano.config.floatX)
        vals = np.array(vals, dtype=theano.config.floatX )
        valprobs = np.array( valprobmaps>0, dtype=theano.config.floatX )
        valloss = eval_nn( vals, valtargets[:,[0,1,3],:,:], valprobs[:,[0,1,3],:,:] )
        val_loss += valloss
        out_img = eval_img(vals)

        batches += 1

        if train_batches%1==0:
            (Image.fromarray(vals[0,0,:,:]      ).convert('RGB')).save(args.outpath+'inval'+str(count)+'.bmp')
            (Image.fromarray(255*valmaskmaps[0,0,:,:]).convert('RGB')).save(args.outpath+'mabgval'+str(count)+'.bmp')
            (Image.fromarray(255*valmaskmaps[0,1,:,:]).convert('RGB')).save(args.outpath+'manuval'+str(count)+'.bmp')
            (Image.fromarray(255*valmaskmaps[0,2,:,:]).convert('RGB')).save(args.outpath+'mawoval'+str(count)+'.bmp')
            (Image.fromarray(1/valprobmaps[0,0,:,:]).convert('RGB')).save(args.outpath+'prbgval'+str(count)+'.bmp')
            (Image.fromarray(1/valprobmaps[0,1,:,:]).convert('RGB')).save(args.outpath+'prnuval'+str(count)+'.bmp')
            (Image.fromarray(1/valprobmaps[0,2,:,:]).convert('RGB')).save(args.outpath+'prwoval'+str(count)+'.bmp')
            

            Image.fromarray(255*out_img[0,0,:,:]).convert('RGB').save(args.outpath+'outbgval'+str(count)+'.bmp')
            Image.fromarray(255*out_img[0,1,:,:]).convert('RGB').save(args.outpath+'outnuval'+str(count)+'.bmp')
            Image.fromarray(255*out_img[0,2,:,:]).convert('RGB').save(args.outpath+'outwoval'+str(count)+'.bmp')
 
        if train_batches%1==0:
            t = time.time() - start_time
            hours, minutes, seconds = t//3600, (t - 3600*(t//3600))//60, (t - 3600*(t//3600)) - (60*((t - 3600*(t//3600))//60))
            now = datetime.now()
            print("Actual Date/Time:", "\t%i/%i/%i\t%dh%dm%ds" %(now.day, now.month, now.year, now.hour, now.minute, now.second) )
            print("Total Time:", "\t\t%dh%dm%ds" %(hours,minutes,seconds) )
            print("Batch:\t\t ",train_batches)
            print("train loss:\t\t{:.4f}".format(train_loss / batches))
            print("valid loss:\t\t{:.4f}".format(val_loss / batches) )
            print("valid db loss:\t\t{:.4f}".format(val_db_loss) )
            print("-")

            res_train_loss = np.append( res_train_loss, train_loss / batches )
            res_valid_loss = np.append( res_valid_loss, val_loss / batches )
            res_valid_db_loss = np.append( res_valid_db_loss, val_db_loss )
            seq_nb_img = np.append( seq_nb_img, train_batches )

            resultsnn=args.outpath + args.results
            np.savez(resultsnn,
                        train_loss=res_train_loss, 
                        valid_loss=res_valid_loss, 
                        valid_db_loss=res_valid_db_loss, 

                        nb_img = seq_nb_img)
            
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.plot(seq_nb_img, res_train_loss, label='Train')
            ax.plot(seq_nb_img, res_valid_loss, label='Valid')

            ax.legend()
            fig.savefig(args.outpath + args.lossplot)
            plt.close(fig)
        modelnn=args.outpath + args.weights
        np.savez(modelnn, *lasagne.layers.get_all_param_values(nn))
        if val_db_loss < best_loss:
            modelnn=args.outpath + args.bestweights
            np.savez(modelnn, *lasagne.layers.get_all_param_values(nn))
            best_loss = val_db_loss
            print("New best model saved with loss {:.4f}".format(best_loss))

        train_loss = 0
        val_loss = 0

        batches = 0
        count+=1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Build and Learn a CNN model for structure spotting using pre-trained model.")
    
    parser.add_argument("-o", "--output-path",
                        dest="outpath",
                        type=str,
                        default="./output15/",
                        help="output path where to save results",)
    parser.add_argument("-e", "--best-weights",
                        dest="bestweights",
                        default="nn-best-weights.npz",
                        type=str,
                        help="output file to save best weights",)
    parser.add_argument("-w", "--weights-learned",
                        dest="weights",
                        default="nn-weights.npz",
                        type=str,
                        help="output file to save last weights",)
    parser.add_argument("-l", "--loss-plot",
                        dest="lossplot",
                        default="a-nn-loss-plot-o15.jpg",
                        type=str,
                        help="output file to save the loss plot",)
    parser.add_argument("-r", "--results-saves", 
                        dest="results",
                        default="nn-results.npz", 
                        type=str,
                        help="output file to save results (npz numpy pickle)",)
    parser.add_argument("-R", "--previous-results-saves",
                        dest="preresults",
                        type=str,
                        help="previous results file to continue the training",)
    parser.add_argument("-p", "--previous-weigths-learned",
                        dest="preweights",
                        default="nn-weight_structure-spotting.npz",
                        type=str,
                        help="previous weights file to continue the training",)
    parser.add_argument("-B", "--batch-size",
                        dest="batchsize",
                        type=int, 
                        default=1,
                        help="number of ARTIDOC image created by batch",)
    parser.add_argument("-n", "--number-epoch",
                        dest="nepoch",
                        type=int, 
                        default=300,
                        help="number of batch for the training",)
    parser.add_argument("-g", "--generation-parameters",
                        dest="genparam",
                        type=int, 
                        nargs=3,
                        default=(200, 100, 1),
                        help="parameters for the artidoc generation. (MinWidthOfStructure MinHeightOfStructure BlankDensity)")
    parser.add_argument("-s", "--input-size",
                        dest="imgsize",
                        type=int, 
                        nargs=2,
                        default=(1024, 1024),
                        help="Size of the ARTIDOC image genertates for the learning.",)
    args = parser.parse_args()
    main(args)

# coding: utf-8

import sys, getopt

import argparse

import sys
import os
import time

import numpy as np

import pandas as pd
from sklearn.cluster import KMeans
from PIL import Image, ImageEnhance, ImageChops, ImageDraw, ImageFilter

import PIL.ImageOps

import scipy.ndimage
import skimage.morphology
import skimage.measure

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.pyplot  as pyplot
import gzip

import theano
import theano.tensor as T

import lasagne
#import nolearn
import csv
import json
import gzip, pickle
# import cv2

from scipy.ndimage import filters

import scipy.misc
import ast

def SpatialSoftmax(x):
    exp_x = T.exp(x - x.max(axis=1, keepdims=True))
    exp_x /= exp_x.sum(axis=1, keepdims=True)
    return exp_x


# A function named build_map_size is defined for building a deep convolutional neural network (CNN) model. The model takes an input image and processes the image through a series of convolution, pooling and deconvolution layers to finally generate an output.
# The main function of this function is to build a neural network containing convolution, pooling and deconvolution layers. The network can be used for image processing tasks and can be configured according to the size of the input image.
# Use the SpatialSoftmax nonlinear activation function in the last layer of the network, which is often used to generate spatial probability distributions.
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



def classificationStats(out, tar,csvname, alpha=0.5):
    mask_out = out
    mask_tar = tar

    TP = np.sum(np.logical_and(mask_out == 0, mask_tar == 0)) + np.sum(np.logical_and(mask_out == 1, mask_tar == 1)) + np.sum(np.logical_and(mask_out == 2, mask_tar == 2))
    # True Negative (TN): we predict a label of 0 (negative), and
    # the true label is 0.
    TN = np.sum(np.logical_and(mask_out != 0, mask_tar != 0)) + np.sum(np.logical_and(mask_out != 1, mask_tar != 1)) + np.sum(np.logical_and(mask_out != 2, mask_tar != 2))
    # False Positive (FP): we predict a label of 1
    # (positive), but the true label is 0.
    FP = np.sum(np.logical_and(mask_out == 0, mask_tar != 0)) + np.sum(np.logical_and(mask_out == 1, mask_tar != 1)) + np.sum(np.logical_and(mask_out == 2, mask_tar != 2))
    # False Negative (FN): we predict a label of 0
    # (negative), but the true label is 1.
    FN = np.sum(np.logical_and(mask_out != 0, mask_tar == 0)) + np.sum(np.logical_and(mask_out != 1, mask_tar == 1)) + np.sum(np.logical_and(mask_out != 2, mask_tar == 2))
   
    
    TP /= 100
    TN /= 100
    FP /= 100
    FN /= 100

    sen = TP/(TP+FN)
    spe = TN/(TN+FP)
    acc = (TP+TN)/(TP+TN+FN+FP)
    pre = TP/(TP+FP)
    f1 = 2*TP/(2*TP+FP+FN)
    mcc =((TP*TN)-(FP*FN)) / np.sqrt((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))
    current_working_directory = os.getcwd()

    with open(args.outpath+csvname+".txt", 'w') as file:
        file.write("Sensitivity: {}\n".format(sen))
        file.write("Specificity: {}\n".format(spe))
        file.write("Accuracy: {}\n".format(acc))
        file.write("Precision: {}\n".format(pre))
        file.write("F1 Score: {}\n".format(f1))
        file.write("MCC: {}\n".format(mcc))
        file.write("True Positive: {}\n".format(TP))
        file.write("True Negative: {}\n".format(TN))
        file.write("False Positive: {}\n".format(FP))
        file.write("False Negative: {}\n".format(FN))
        
    return sen, spe, acc, pre, f1, mcc, TP, TN, FP, FN


# This function is used to center the input image in a new image and pad it to fit the new dimensions.
def imagePadding(img, newsize=(1024,1024)):
    img_pad =Image.new('L', newsize)
    prow, pcol = np.int(np.floor((newsize[0]-img.size[0])/2)), np.int((np.floor(newsize[1]-img.size[1])/2))
    img_pad.paste(img, (prow, pcol))
    return img_pad, prow, pcol
    
def readImageMask(image_path, mask_path):
   
    thisImg =Image.open(image_path)
    csvData = pd.read_csv(mask_path, encoding='ISO-8859-1')
    csvData.head()
    width, height = thisImg.size
    print(width,height)
    thisGray = thisImg.convert('L')

    mask_structures= np.zeros((height, width))

    mask_structures[:,:] = 0
    
    for index, row in csvData.iterrows():
        bounding_box = ast.literal_eval(row[2])
        #print(row['Label'])
        if 'quation' in row['Label']:
            for point in bounding_box:
                x, y = point
                mask_structures[int(y), int(x)] = 1
        elif 'exte' in row['Label']:
            for point in bounding_box:
                x, y = point
                mask_structures[int(y), int(x)] = 2
        else:
            print("nothing！！！！！！！")
    
    return mask_structures



def main(args):
    jpg_path = args.idimg+".jpg"
    csv_path = args.idimg+".csv"
    
    with Image.open(jpg_path) as img:
        print(f"JPG file: {jpg_path}, Size: {img.size}")
    
    with open(csv_path, 'r') as csvfile:
        print(f"CSV file: {csv_path}")

    
    ## Set Network ###
    print("Set network")
    input_var = T.tensor4('inputs')
    target_var = T.tensor4('targets')
    class_var = T.ivector('classes')
    seg_var = T.tensor4('segmentations')
    nn = build_map_size(input_var, sizeX=args.insize[0], sizeY=args.insize[1])

    bloadweights = True
    modelnn = args.weights 
    with np.load(modelnn) as f:
        last_param_nn = [f['arr_%d' % i] for i in range(len(f.files))]
    pnn_init = last_param_nn
    # This line of code uses the set_all_param_values function in the Lasagne framework to 
    #     set the initial values of all parameters of the neural network model nn to pnn_init.
    # By executing this line of code, you can load the pre-trained model parameters 
    #     into the neural network model for further training or use.
    # This is useful in cases such as transfer learning and fine-tuning 
    #     to speed up training and improve model performance.
    lasagne.layers.set_all_param_values(nn, pnn_init)

    lasagne.layers.set_all_param_values(nn, pnn_init)
    ### Function Definition ###
    print("Fonction Definition")

    out_nn = lasagne.layers.get_output(nn)
    eval_img = theano.function([input_var], out_nn )

    idimg = os.path.basename(jpg_path)

    idcard = idimg 
    print(idimg, idcard)
    inputs_orig = Image.open( jpg_path ).convert('L')
    inputs = PIL.ImageOps.invert(inputs_orig)

    oldsize = inputs.size 
    print(oldsize)
    bfit = False
    if inputs.size[0] > args.insize[0]: 
        fsx = args.insize[0]
        bfit = True
    else:
        fsx = inputs.size[0]
    if inputs.size[1] > args.insize[1]: 
        fsy = args.insize[1]
        bfit = True
    else:
        fsy = inputs.size[1]
    fitsize = (fsx, fsy)
    if bfit:
        inputs_fit = inputs.resize( fitsize, Image.ANTIALIAS )
    else:
        inputs_fit = inputs

    newsize = ( int(np.floor(fitsize[0]*args.inzoom)), int( np.floor( args.inzoom*fitsize[1] ) ) )
    #newsize = ( int(np.floor(fitsize[0])), int( np.floor( fitsize[1] ) ) )
    inputs_resized = inputs_fit.resize( newsize , Image.ANTIALIAS )

    inputs_pad, prow, pcol = imagePadding(inputs_resized, newsize=(args.insize[0], args.insize[1]) )
    print(inputs_pad.size)
    

    img = np.array( inputs_pad, dtype=np.float32 )

    restensor = eval_img(img.reshape( (1, 1, args.insize[0], args.insize[1]) ))
    mask_out = np.argmax(restensor[0], axis=0)

    mask_out = mask_out[  pcol:pcol+inputs_resized.size[1] , prow:prow+inputs_resized.size[0]] 
    #print(mask_out)
    img = img[pcol:pcol+inputs_resized.size[1], prow:prow+inputs_resized.size[0] ] 

    ###Evaluation###
    maskm = readImageMask(jpg_path, csv_path)

    classificationStats(mask_out, maskm,idimg[:-4], alpha=0.5)
    


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="'Launch the Structure-Spotting' using pre-trained weights 'nn-weight_structure-spotting.npz'.")
    
    parser.add_argument("-w", "--weights",
            dest="weights",
            type=str,
            default='./',
            help="parameters (weights) of the pre-trained network",)
    parser.add_argument("-r", "--results-path",
            dest="outpath",
            type=str,
            default='./',
            help="output path to save results (npz numpy pickle)",)
    parser.add_argument("-i", "--id-image",
            dest="idimg",
            type=str, 
            help="filename of input image",)
    parser.add_argument("-z", "--zoom-image",
            dest="inzoom",
            type=float, 
            default=1,
            help="zoom to applied to image.",)
    parser.add_argument("-s", "--input-size",
            dest="insize",
            type=int, 
            nargs=2,
            default=(5120, 5120),
            help="size of area in where the input image. Because CNN input need to be square",)
    # parser.add_argument("-d", "--directory",
    #         dest="directory",
    #         type=str,
    #         default='./test/',
    #         help="Path to the directory containing .jpg and .csv files")
    args = parser.parse_args()
main(args)


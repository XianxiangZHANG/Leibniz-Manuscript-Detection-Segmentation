# Leibniz-Manuscript-Detection-Segmentation

This repository contains the python code to perform structures spotting in handwritten documents. 

This work is based on the paper «Transfer Learning for Structures Spotting in Unlabeled Handwritten Documents using Randomly Generated Documents», International Conference on Pattern Recognition Applications and Methods, 2018. (https://hal.archives-ouvertes.fr/hal-01681114)

In this work, we focus on the localization of word-level structures, distinguishing between structure *mathematical expression* and structure *text* in unlabeled handwritten documents. We can build a coherent map segmentation of mathematical expressions/text/background structures on real documents by using a CNN trained on a large number of synthetic documents (randomly generated) as a pre-training template and then importing Leibniz manuscripts to continue training.

This work is related to the ANR project CIRESFI
- see: http://cethefi.org/ciresfi/doku.php?id=en:projet
- related to : https://github.com/GeoTrouvetout/CIRESFI

## Dependencies
- numpy
- pandas
- sklearn
- scipy
- skimage
- matplotlib
- theano
- lasagne


## Result.py
"Launch the Structure-Spotting" using pre-trained weights "nn-weight_best.npz". 

Usage:
```
python3 ./Result.py -w nn-weight_best.npz -i image.jpg
```

## Training.py
"Build and Learn a CNN model for structure spotting using pretrained model and Leibniz manuscript images. "
Usage:
```
python3 ./Training.py
```


## Evaluation.py
"Launch the Evaluation" using pre-trained weights "nn-weight_best.npz". 

Usage:
```
# image means image.jpg file and image.csv file name
python3 ./Result.py -w nn-weight_best.npz -i image
```

## nn-weight_best.npz
Based on the pre-trained model, import the Leibniz manuscript images to continue training the model.
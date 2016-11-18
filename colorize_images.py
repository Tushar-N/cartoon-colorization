import numpy as np
import skimage.transform
import skimage.color as color
import scipy.ndimage.interpolation as sni

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = (12, 6)

import sys
import os
sys.path.append('%s/caffe-colorization/python'%os.getcwd())
sys.path.append('%s/resources/'%os.getcwd())
import caffe
caffe.set_mode_gpu()
caffe.set_device(0)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prototxt', default='models/deploy/colorization_deploy_v2.prototxt')
parser.add_argument('--caffemodel', default='models/pretrained/colorization_release_v2.caffemodel')
parser.add_argument('--sketch', default='data/raw/sketch/11_246.png')
parser.add_argument('--reference', default='data/raw/frames/11_246.png')
parser.add_argument('--save', default='output/colorized.png')
opt = parser.parse_args()



net = caffe.Net(opt.prototxt, opt.caffemodel, caffe.TEST)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

pts_in_hull = np.load('resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel

# Get sketch information
sketch_rgb = caffe.io.load_image(opt.sketch)
sketch_lab = color.rgb2lab(sketch_rgb)
sketch_l = sketch_lab[:,:,0,np.newaxis] # pull out L channel
(H_orig,W_orig) = sketch_rgb.shape[:2] # original image size

sketch_rs = caffe.io.resize_image(sketch_rgb,(H_in,W_in))
sketch_lab_rs = color.rgb2lab(sketch_rs)
sketch_l_rs = img_lab_rs[:,:,0]
sketch_l_rs = sketch_l_rs - 50
net.blobs['data_l'].data[0,0,:,:] = sketch_l_rs

try: # only push the reference if it exists
	ref_rgb = caffe.io.load_image(opt.reference)
	ref_rs = caffe.io.resize_image(ref_rgb,(H_in,W_in))
	ref_lab = color.rgb2lab(ref_rs)	
	net.blobs['ref_lab'].data[0,:,:,:] = ref_lab
except:
	print 'cannot find layer'

net.forward() # run network

# retrieve output and upsample
ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0))
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1))

# concatenate with original image L, and convert to RGB
img_lab_out = np.concatenate((sketch_l,ab_dec_us),axis=2) 
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) 

plt.imshow(img_rgb_out);
plt.axis('off');
plt.savefig(opt.save)

raw_input('Success')


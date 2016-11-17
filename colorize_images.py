import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import skimage.transform

import sys
import os
sys.path.append('%s/caffe-colorization/python'%os.getcwd())
sys.path.append('%s/resources/'%os.getcwd())

import caffe
import os
import skimage.color as color
import scipy.ndimage.interpolation as sni

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--prototxt', default='models/deploy/colorization_deploy_v2.prototxt')
parser.add_argument('--caffemodel', default='models/pretrained/colorization_release_v2.caffemodel')
parser.add_argument('--sketch', default='data/raw/sketch/11_246.png')
parser.add_argument('--reference', default='data/raw/frames/11_246.png')
parser.add_argument('--save', default='output/colorized.png')
opt = parser.parse_args()

# returns the L channel of the sketch
def get_sketch(H_in, W_in):
	# load the original image
	img_rgb = caffe.io.load_image(opt.sketch)
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size

	img_lab = color.rgb2lab(img_rgb) # convert image to lab color space
	img_l = img_lab[:,:,0] # pull out L channel
	(H_orig,W_orig) = img_rgb.shape[:2] # original image size

	# resize image to network input size
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in))
	img_lab_rs = color.rgb2lab(img_rs)
	img_l_rs = img_lab_rs[:,:,0]
	img_l_rs = img_l_rs-50 #subtract mean

	return img_l_rs, H_orig, W_orig, img_l

# returns the 3channel reference image
def get_reference(H_in, W_in):

	# load the original image
	img_rgb = caffe.io.load_image(opt.reference)
	img_rs = caffe.io.resize_image(img_rgb,(H_in,W_in)) # resize image to network input size
	img_rs = np.transpose(img_rs, (2,0,1)) # right dims
	img_rs = img_rs[::-1,:,:] #RGB to BGR

	return img_rs


#%matplotlib inline
plt.rcParams['figure.figsize'] = (12, 6)

caffe.set_mode_gpu()
caffe.set_device(0)


net = caffe.Net(opt.prototxt, opt.caffemodel, caffe.TEST)

(H_in,W_in) = net.blobs['data_l'].data.shape[2:] # get input shape
(H_out,W_out) = net.blobs['class8_ab'].data.shape[2:] # get output shape

pts_in_hull = np.load('resources/pts_in_hull.npy') # load cluster centers
net.params['class8_ab'][0].data[:,:,0,0] = pts_in_hull.transpose((1,0)) # populate cluster centers as 1x1 convolution kernel

img_l_rs, H_orig, W_orig, img_l= get_sketch(H_in, W_in)
net.blobs['data_l'].data[0,0,:,:] = img_l_rs
try: # only push the reference if it exists
	img_ref=get_reference(H_in, W_in)
	net.blobs['col_reference_data'].data[0,:,:,:] = img_ref
except:
	print 'cannot find layer'

net.forward() # run network

ab_dec = net.blobs['class8_ab'].data[0,:,:,:].transpose((1,2,0)) # this is our result
ab_dec_us = sni.zoom(ab_dec,(1.*H_orig/H_out,1.*W_orig/W_out,1)) # upsample to match size of original image L

l_ch=img_l[:,:,np.newaxis]
img_lab_out = np.concatenate((l_ch,ab_dec_us),axis=2) # concatenate with original image L
img_rgb_out = np.clip(color.lab2rgb(img_lab_out),0,1) # convert back to rgb

plt.imshow(img_rgb_out);
plt.axis('off');
plt.savefig(opt.save)

raw_input('Success')


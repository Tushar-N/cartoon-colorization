import numpy as np
import skimage.transform
import skimage.color as color
import scipy.ndimage.interpolation as sni
import os

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
parser.add_argument('--prototxt', default='models/colorization_deploy_v5_grayscale.prototxt')
parser.add_argument('--caffemodel', default='cv/v5b_grayscale/initial_8.caffemodel')
parser.add_argument('--sketch', default='Testing/cartoon_frames/0.6332_2_232.png')
parser.add_argument('--reference', default='Testing/live_frames')
parser.add_argument('--save', default='Testing/evalPerSketch')
opt = parser.parse_args()

sketch_name = opt.sketch.split('/')[-1]
if not os.path.exists(opt.save): #check if directory exists
    os.makedirs(opt.save)

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
sketch_l_rs = sketch_lab_rs[:,:,0]
sketch_l_rs = sketch_l_rs - 50
net.blobs['data_l'].data[0,0,:,:] = sketch_l_rs

for file in os.listdir(opt.reference):
	if file.endswith(".png"):
		reference = os.path.join(opt.reference,file)

		try: # only push the reference if it exists
			ref_rgb = caffe.io.load_image(reference)
			ref_rs = caffe.io.resize_image(ref_rgb,(H_in,W_in))
			ref_lab = color.rgb2lab(ref_rs)	
			net.blobs['ref_lab'].data[0,:,:,:] = ref_lab.transpose((2,1,0))
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
		plt.savefig(os.path.join(opt.save, sketch_name +file))

raw_input('Success')

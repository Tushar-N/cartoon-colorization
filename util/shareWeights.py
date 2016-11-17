import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import caffe

net = caffe.Net('./models/colorization_v4.prototxt', 
				'./models/colorization_release_v2.caffemodel', 
				caffe.TEST)

params = ['bw_conv1_1', 'conv1_2', 'conv1_2norm',
		'conv2_1','conv2_2','conv2_2norm',
		'conv3_1','conv3_2','conv3_3','conv3_3norm',
		'conv4_1','conv4_2','conv4_3','conv4_3norm',
		'conv5_1','conv5_2','conv5_3','conv5_3norm',
		'conv6_1','conv6_2','conv6_3','conv6_3norm',
		'conv7_1','conv7_2','conv7_3','conv7_3norm',
		'conv8_1','conv8_2','conv8_3']

# fc_params = {name: (weights, biases)}
fc_params = {pr: (net.params[pr][0].data, net.params[pr][1].data) for pr in params}
for fc in params:
	print '{} weights are {} dimensional and biases are {} dimensional'.format(fc, fc_params[fc][0].shape, fc_params[fc][1].shape)

net_full_conv = caffe.Net('./models/colorization_v4.prototxt', 
						  './models/colorization_release_v2.caffemodel',
						  caffe.TEST)

params_full_conv = ['ref_bw_conv1_1', 'ref_conv1_2','conv1_2norm',
		'ref_conv2_1','ref_conv2_2','ref_conv2_2norm',
		'ref_conv3_1','ref_conv3_2','ref_conv3_3','ref_conv3_3norm',
		'ref_conv4_1','ref_conv4_2','ref_conv4_3','ref_conv4_3norm',
		'ref_conv5_1','ref_conv5_2','ref_conv5_3','ref_conv5_3norm',
		'ref_conv6_1','ref_conv6_2','ref_conv6_3','ref_conv6_3norm',
		'ref_conv7_1','ref_conv7_2','ref_conv7_3','ref_conv7_3norm',
		'ref_conv8_1','ref_conv8_2','ref_conv8_3',]

conv_params = {pr: [net_full_conv.params[pr][0].data, net_full_conv.params[pr][1].data] for pr in params_full_conv}

for conv in params_full_conv:
	print '{} weights are {} dimensional and biases are {} dimensional'.format(conv, conv_params[conv][0].shape, conv_params[conv][1].shape)


print conv_params['ref_conv7_1'][0].sum()
print fc_params['conv7_1'][0].sum()

for pr, pr_conv in zip(params, params_full_conv):
	conv_params[pr_conv][0].flat = fc_params[pr][0].flat
	conv_params[pr_conv][1][...] = fc_params[pr][1]
	

print net.params['conv7_1'][0].data.sum()
print net_full_conv.params['ref_conv7_1'][0].data.sum()


net_full_conv.save('output/bvlc_caffenet_full_conv.caffemodel')
raw_input('done')
import numpy as np
import skimage.transform
import skimage.color as color
import scipy.ndimage.interpolation as sni
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import heapq as heap
from tqdm import tqdm
from shutil import copyfile
import argparse

import sys
import os

parser = argparse.ArgumentParser()
parser.add_argument('--log')
parser.add_argument('--save', default='Testing/output')
parser.add_argument('--fpath', default = 'Testing/evalPerReference_3_v5')
opt = parser.parse_args()

if not os.path.exists(opt.save): #check if directory exists
    os.makedirs(opt.save)

pDir = opt.save
cDir =''
with open(opt.log, 'r') as f:
	for line in f:
		if('Euclidean' in line):
			pDir = os.path.join(opt.save,'Euclidean')
			if not os.path.exists(pDir):
				os.makedirs(pDir)
			continue
		elif('Softmax' in line):
			pDir = os.path.join(opt.save,'Softmax')
			if not os.path.exists(pDir):
				os.makedirs(pDir)
			continue
			
		if('Largest' in line):
			cDir = 'Largest'
			if not os.path.exists(os.path.join(pDir, cDir)):
				os.makedirs(os.path.join(pDir, cDir))
			continue
		elif('Smallest' in line):
			cDir = 'Smallest'
			if not os.path.exists(os.path.join(pDir, cDir)):
				os.makedirs(os.path.join(pDir, cDir))
			continue	

		try:
			line = line.rstrip('\n')
			line = os.path.join(opt.fpath, line)
			filepath = os.path.join(pDir,cDir, line.split('/')[-1])
			copyfile(line, filepath)

		except:
			continue


### 
#
# Weighted TV-L1 image flattening using proximal algorithms
# ProxTV library is used
#
#

import prox_tv as ptv
import time
import skimage as ski
from skimage import io, color
import numpy as np

# Load color image (3 dimensions: length, width and color)
X = io.imread('neuron.png')
X_gray = color.rgb2gray(X)
im = ski.img_as_float(X)

row, col = X_gray.shape
im_lab = color.rgb2lab(im)

# Hyperparameters
iterations = 1 # default 2 or 4
exp_factor = 11 # Degree of flattening (changes according to image type)

h = 1 # neighbourhood (best 1)
start = time.time()
iterations_ = iterations

# Mem alocate
grady_l = np.zeros((row, col, h))
grady_a = np.zeros((row, col, h))
grady_b = np.zeros((row, col, h))
gradx_l = np.zeros((row, col, h))
gradx_a = np.zeros((row, col, h))
gradx_b = np.zeros((row, col, h))
F = np.zeros(X.shape)

while iterations > 0:
	#for i in range(h):
	i = 0 # only immediate neighbourhood
	grady_l[:,:,i] =  im_lab[:,:,0] - np.roll(im_lab[:,:,0].reshape(X_gray.shape), i+1, axis=0) 
	grady_a[:,:,i] =  im_lab[:,:,1] - np.roll(im_lab[:,:,1].reshape(X_gray.shape), i+1, axis=0) 
	grady_b[:,:,i] =  im_lab[:,:,2] - np.roll(im_lab[:,:,2].reshape(X_gray.shape), i+1, axis=0) 

	gradx_l[:,:,i] = im_lab[:,:,0] - np.roll(im_lab[:,:,0].reshape(X_gray.shape), i+1, axis=1) 
	gradx_a[:,:,i] = im_lab[:,:,1] - np.roll(im_lab[:,:,1].reshape(X_gray.shape), i+1, axis=1) 
	gradx_b[:,:,i] = im_lab[:,:,2] - np.roll(im_lab[:,:,2].reshape(X_gray.shape), i+1, axis=1) 

	weight_y = 10**exp_factor * np.exp( -1 * (np.sum(grady_l**2, axis=2) + np.sum(grady_b**2, axis=2) + np.sum(grady_a**2, axis=2) ) /2.0 ) 
	weight_x = 10**exp_factor * np.exp( -1 * (np.sum(gradx_l**2, axis=2) + np.sum(gradx_b**2, axis=2) + np.sum(gradx_a**2, axis=2) ) /2.0 ) 
	#weight_y = 10 ** -4 *  ( 1 - (np.sum(grady_l**2, axis=2) + np.sum(grady_b**2, axis=2) + np.sum(grady_a**2, axis=2) ) ) **2
	#weight_x = 10 ** -4 *  ( 1 - (np.sum(gradx_l**2, axis=2) + np.sum(gradx_b**2, axis=2) + np.sum(gradx_a**2, axis=2) ) ) **2

	#print np.mean(weight_y.ravel())
	#print np.mean(weight_x.ravel())
	print('TV optimization: Iterration ' + str(iterations_ - iterations +1))

	#F = ptv.tvgen(X,      [weight_y, weight_x],   [1,2],               np.array([1,1]));
	#              Image | Penalty in each dimension |  Dimensions to penalize  | Norms to use 

	F[:,:,0] = ptv.tv1w_2d(im[:,:,0].reshape(X_gray.shape), weight_y[1:,:], weight_x[:,1:])
	F[:,:,1] = ptv.tv1w_2d(im[:,:,1].reshape(X_gray.shape), weight_y[1:,:], weight_x[:,1:])
	F[:,:,2] = ptv.tv1w_2d(im[:,:,2].reshape(X_gray.shape), weight_y[1:,:], weight_x[:,1:])

	im = F
	im_lab = color.rgb2lab(F)
	iterations -= 1

#F = (F- np.min(F.ravel()) ) / ( np.max(F.ravel()) - np.min(F.ravel()))
print np.min(F.ravel())
print np.max(F.ravel())

io.imsave('result_'+str(iterations_)+'.png', F); 
#io.imsave('result_'+str(iterations_)+'.png', F.clip(0.0, 1.0)); 

end = time.time()
print('Elapsed time ' + str(end-start))

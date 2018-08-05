
###
#
# Weighted TV-L1 image flattening using proximal algorithms
# ProxTV library is used
# Elras S. Bolkar
#

def flatten_color(im, output_folder = None, iterations=4, exp_factor=10):

	import time
	import numpy as np
	import prox_tv as ptv
	from skimage import color
    	import json

	im = np.asfarray(im)
	row, col, ch = im.shape
	shape_2d = (row, col)
	im_lab = np.asfarray(color.rgb2lab(im))

	# Hyperparameters
	#iterations = 1 # default 2 or 4
	#exp_factor = 11 # Degree of flattening (changes according to image type)

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
	F = np.zeros((row, col, ch))

	while iterations > 0:
		#for i in range(h):
		i = 0 # only immediate neighbourhood
		grady_l[:,:,i] =  im_lab[:,:,0] - np.roll(im_lab[:,:,0].reshape(shape_2d), i+1, axis=0)
		grady_a[:,:,i] =  im_lab[:,:,1] - np.roll(im_lab[:,:,1].reshape(shape_2d), i+1, axis=0)
		grady_b[:,:,i] =  im_lab[:,:,2] - np.roll(im_lab[:,:,2].reshape(shape_2d), i+1, axis=0)

		gradx_l[:,:,i] = im_lab[:,:,0] - np.roll(im_lab[:,:,0].reshape(shape_2d), i+1, axis=1)
		gradx_a[:,:,i] = im_lab[:,:,1] - np.roll(im_lab[:,:,1].reshape(shape_2d), i+1, axis=1)
		gradx_b[:,:,i] = im_lab[:,:,2] - np.roll(im_lab[:,:,2].reshape(shape_2d), i+1, axis=1)

		weight_y = 10**exp_factor * np.exp( -1 * (np.sum(grady_l**2, axis=2) + np.sum(grady_b**2, axis=2) + np.sum(grady_a**2, axis=2) ) /2.0 )
		weight_x = 10**exp_factor * np.exp( -1 * (np.sum(gradx_l**2, axis=2) + np.sum(gradx_b**2, axis=2) + np.sum(gradx_a**2, axis=2) ) /2.0 )
		#weight_y = 10 ** -4 *  ( 1 - (np.sum(grady_l**2, axis=2) + np.sum(grady_b**2, axis=2) + np.sum(grady_a**2, axis=2) ) ) **2
		#weight_x = 10 ** -4 *  ( 1 - (np.sum(gradx_l**2, axis=2) + np.sum(gradx_b**2, axis=2) + np.sum(gradx_a**2, axis=2) ) ) **2

		#print np.mean(weight_y.ravel())
		#print np.mean(weight_x.ravel())
		print('TV-l1 flattening by proximal algo: Iterration ' + str(iterations_ - iterations +1))

		#F = ptv.tvgen(X,      [weight_y, weight_x],   [1,2],               np.array([1,1]));
		#              Image | Penalty in each dimension |  Dimensions to penalize  | Norms to use

		F[:,:,0] = ptv.tv1w_2d(im[:,:,0].reshape(shape_2d), weight_y[1:,:], weight_x[:,1:])
		F[:,:,1] = ptv.tv1w_2d(im[:,:,1].reshape(shape_2d), weight_y[1:,:], weight_x[:,1:])
		F[:,:,2] = ptv.tv1w_2d(im[:,:,2].reshape(shape_2d), weight_y[1:,:], weight_x[:,1:])

		# update
		im = F
		im_lab = color.rgb2lab(F)
		iterations -= 1

	print np.min(F.ravel())
	print np.max(F.ravel())

	# Notify if exp_factor too much
    	if np.max(F.ravel())>= 1.2:
        	with open(output_folder+'error_log.json', 'w') as outfile:
            		json.dump('Overflattening: Decrease exp_factor hyperparameter!', outfile)

        	assert np.max(F.ravel())<1.2, 'Overflattening: Decrease exp_factor hyperparameter!'

	#io.imsave('result_'+str(iterations_)+'.png', F);
	#io.imsave('result_'+str(iterations_)+'.png', F.clip(0.0, 1.0));

	end = time.time()
	print('Elapsed time ' + str(end-start))

	return F
	# -------------------------------


def contrast_stretch(img, strength=5, lower_percentile=2, higher_percentile=98):
	# simple linear contrast streching (perc. 2 and 98 is used to increase robustness)
	import numpy as np
	img = np.asfarray(img)
	img_strectched = np.zeros(img.shape)

	print 'contrast stretching .. '
	row, col, ch = img.shape

	p2 = np.percentile(img, lower_percentile) # apply global not per-channel
	p98 = np.percentile(img, higher_percentile)
	for k in range(ch):
		img_strectched[:,:,k] = (((img[:,:,k]-p2)*(1.0/(p98-p2)))*0.12*strength)

	return img_strectched
	# -------------------------------


if __name__ == '__main__':

	import skimage as ski
	from skimage import io
        name = 'neuron'
        #im = io.imread('test_images/'+ name +'.jpg')
        im = io.imread('test_images/'+ name +'.png')
	im = ski.img_as_float(im)
	im_flat = flatten_color(im, None, 4, 11) # level changes depending on image edge density, chaotic image > small ('1.jpg':10, '2':3)
	#im_flat = (im_flat- np.min(im_flat.ravel()) ) / ( np.max(im_flat.ravel()) - np.min(im_flat.ravel()))
	#io.imsave('results/result_flat.png', im_flat);
	io.imsave('test_results/'+name+'_flat.png', im_flat.clip(0.0, 1.0));

	im_flat_s = contrast_stretch(im_flat, 4, 2, 98)
	io.imsave('test_results/'+name+'_flat_streched.png', im_flat_s.clip(0.0, 1.0));

	# -------------------------------

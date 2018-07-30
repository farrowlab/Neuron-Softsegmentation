#
# Soft Segmentation of Viral Labeled Neurons
# adapted from Tan et al., (2015): https://github.com/CraGL/Decompose-Single-Image-Into-Layers
#
# Briefly: Least-squares optimization with Tikhonov (ridge) and l2-TV regularizations
#           + l1 color difference weighting on layers in a*b*
# ----------------------------------------------------------------------------------------------
# Note: Create a conda envirnemnt and install dependencies [recommended]
#       Apply BaSIC ImajeJ background substruction before [Jar plugin is already included]
#       Check misc_code/ for used libs in soft segmentation
#       Check preprocess/ for preprocessing code called here
# 

# Fixed: 
# - Flattening and contrast stretching
# - Uint16/uint8 reading and uint8 saving using single tifffile.py
# - Computing color vertices (only number of colors is input)
# - White color detection adn deletion (noninformative)
# - Making sure black backround color is always 0th row in color patrix
# - Data fidelity term weighting [it is slow, make flag FAST=0]
# - Final weighting

#
# To-do:
# - TV-l1 for inpainting, since TV-l2 cannot do that (is not possble with LBFGSB, non-smooth :/)
# - Find vertices in HSV and process there to get more robust soft color segmentation (hsv is an angle based space :/)
#
#

from numpy import *
from itertools import izip as zip
from skimage import color     
import scipy.optimize
import time
import os
import json
import sys

'''
def E_overlap( Y, C, P, scratches = {} ):
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )

    return -1.0 * sum(prod(Y, axis=1))
    #return sum(prod((1.0-Y), axis=1)) * 100

def grad_E_overlap(Y, C, P, out, scratches = {} ): # CHANGE THIS ONE
    ind1 = 0
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    temp = zeros(Y.shape)
    for k in range(Y.shape[1]):
        temp[:,k] = prod(delete(Y,k,1), axis=1) * -1.0
        #temp[:,k] = prod(delete((1.0-Y),k,1), axis=1) * 100

    out = temp.flatten() 
'''
def E_ridge( Y, C, P, scratches = {} ):

    '''
    # Weight opacity (1 -Y) by laplacian/gaussian in a*b* color space [L* ignored]
    P_temp = color.rgb2lab(transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) )[:,:,1:]/100.0 #norm
    C_temp = color.rgb2lab(tile(C[1:,:], (P.shape[0], 1, 1)))[:,:,1:]/100.0

    #P_temp = transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) # ignore black [0,0,0]
    #C_temp = tile(C[1:,:], (P.shape[0], 1, 1))

    sigma = 2.0  # increasing it makes robust

    # Laplacian
    we_ridge =  exp(mean(sqrt((P_temp - C_temp) * (P_temp - C_temp)), axis= 2) /(-2.0*sigma)).ravel() # TUNING
    '''
    #return -dot( Y, Y*(1.0 - we_ridge) ) 
    return -dot( Y, Y ) 

def grad_E_ridge( Y, C, P, out, scratches = {} ):

    '''
    # Weight opacity (1 -Y) by laplacian/gaussian in a*b* color space [L* ignored]
    P_temp = color.rgb2lab(transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) )[:,:,1:]/100.0 #norm
    C_temp = color.rgb2lab(tile(C[1:,:], (P.shape[0], 1, 1)))[:,:,1:]/100.0

    #P_temp = transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) # ignore black [0,0,0]
    #C_temp = tile(C[1:,:], (P.shape[0], 1, 1))

    sigma = 2.0  # increasing it makes robust

    # Laplacian
    we_ridge =  exp(mean(sqrt((P_temp - C_temp) * (P_temp - C_temp)), axis= 2) /(-2.0*sigma)).ravel() # TUNING

    '''
    #out = (1.0 - we_ridge)  * Y * -2
    multiply( -2, Y, out ) 

'''
def E_spatial_static( Y, Ytarget, scratches = {} ):
    if 'Y' not in scratches: scratches['Y'] = Y.copy()
    scratch = scratches['Y']
    
    subtract( Y, Ytarget, scratch )
    return dot( scratch, scratch )

def grad_E_spatial_static( Y, Ytarget, out, scratches = {} ):
    subtract( Y, Ytarget, out )
    out *= 2
'''

def E_tvl2( Y, LTL, scratches = {} ):
    ## I don't see how to specify the output memory
    return dot( Y, LTL.dot( Y ) )

def grad_E_tvl2( Y, LTL, out, scratches = {} ):
    ## I don't see how to specify the output memory
    out[:] = LTL.dot( Y )
    out *= 2

def E_fidelity_lsl2_pieces( Y, C, P, scratches = {} ):
    '''
    Y is a #pix-by-#layers flattened array
    C is a (#layers+1)-by-#channels not-flattened array (the 0-th layer is the background color)
    P is a #pix-by-#channels not-flattened array
    '''
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )

    if RUN_FAST == 0:
    	# Weight opacity (1 -Y) by laplacian/gaussian in a*b* color space [L* ignored]
    	P_temp = color.rgb2lab(transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) )[:,:,1:]/100.0 #norm
    	C_temp = color.rgb2lab(tile(C[1:,:], (P.shape[0], 1, 1)))[:,:,1:]/100.0
    	#P_temp = transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) # ignore black [0,0,0]
    	#C_temp = tile(C[1:,:], (P.shape[0], 1, 1))

        sigma = 2.0  # increasing it makes robust

        # Laplacian
        Y= 1.0 - ((1.0-Y) * exp(mean(sqrt((P_temp - C_temp) * (P_temp - C_temp)), axis= 2) /(-2.0*sigma))  ) # TUNING
		    
        # Gaussian
        #Y= 1.0 - ((1.0-Y) * exp(mean((P_temp - C_temp) * (P_temp - C_temp), axis= 2) /(-2.0*sigma))  ) # TUNING

    # _OLD -------------------------
    # signum ***
    #temp = mean((P_temp - C_temp) * (P_temp - C_temp), axis= 2)
    #Y= 1.0 - ((1.0-Y) * piecewise(temp, [temp>0.08, temp<=0.08], [1, 0]) ) # TUNING
 	#Y= 1.0 - ((1.0-Y) * piecewise(temp, [temp>0.2, temp<=0.2], [1, 0]) ) * exp(mean((P_temp - C_temp) * (P_temp - C_temp), axis= 2) /(-2.0*sigma))   # TUNING
    
    # alternative signum [worthless]
    #Y= 1.0 -  piecewise((1.0 - Y), [(1.0 - Y) > 0.15, (1.0 - Y) <= 0.15], [0, 1]) 

    # linear
    #temp = mean((P_temp - C_temp) * (P_temp - C_temp), axis= 2)
    #Y= 1.0 - ((1.0-Y) * (1.0 - temp)) # TUNING

    # quadratic 
    #temp = mean((P_temp - C_temp) * (P_temp - C_temp), axis= 2)
    #Y= 1.0 - ((1.0-Y) * (1.0 - temp)**2) # TUNING

    # Lorentzian ***
    #gamma =  1.0
    #print mean(gamma/(sum((moveaxis(P_temp, 0,1) - C_temp) * (moveaxis(P_temp, 0,1) - C_temp), axis= 2)**2 + (gamma)**2 )   )
    #Y= 1.0 - ((1.0-Y) * gamma/(sum((moveaxis(P_temp, 0,1) - C_temp) * (moveaxis(P_temp, 0,1) - C_temp), axis= 2)**2 + (gamma)**2 )    ) # TUNING

    ## Allocate scratch space
    if 'F' not in scratches:
        scratches['F'] = empty( P.shape, dtype = Y.dtype )
    F = scratches['F'] 
    
    if 'M' not in scratches:
        ## We want the non-flattened Y's shape.
        assert len( Y.shape ) > 1
        scratches['M'] = empty( Y.shape, dtype = Y.dtype )
    M = scratches['M']
    
    if 'D' not in scratches:
        scratches['D'] = empty( ( C.shape[0]-1, C.shape[1] ), dtype = Y.dtype )
    D = scratches['D']
    
    if 'DM' not in scratches:
        scratches['DM'] = empty( ( P.shape[0], D.shape[0], D.shape[1] ), dtype = Y.dtype )
    DM = scratches['DM']
    
    if 'energy_presquared' not in scratches:
        scratches['energy_presquared'] = empty( F.shape, dtype = Y.dtype )
    energy_presquared = scratches['energy_presquared']
    
    ## Compute F
    subtract( C[newaxis,-1,:], P, F )
    
    ## Compute M
    cumprod( Y[:,::-1], axis = 1, out = M )
    M = M[:,::-1]
    
    ## Compute D
    subtract( C[:-1,:], C[1:,:], D )
    
    ## Finish the computation
    multiply( D[newaxis,...], M[...,newaxis], DM )
    DM.sum( 1, out = energy_presquared )
    energy_presquared += F

def E_fidelity_lsl2( Y, C, P, scratches = {} ):
    E_fidelity_lsl2_pieces( Y, C, P, scratches )
    
    energy_presquared = scratches['energy_presquared']
    
    square( energy_presquared, energy_presquared )
    #print energy_presquared.shape
    return energy_presquared.sum()

def gradY_E_fidelity_lsl2( Y, C, P, out, scratches = {} ):
    E_fidelity_lsl2_pieces( Y, C, P, scratches )
    
    ### Reshape Y the way we want it.
    Y = Y.reshape( ( P.shape[0], C.shape[0]-1 ) )
    
    energy_presquared = scratches['energy_presquared']
    D = scratches['D']
    M = scratches['M']
    DM = scratches['DM']
    
    if 'Mi' not in scratches:
        scratches['Mi'] = empty( DM.shape, dtype = Y.dtype )
    Mi = scratches['Mi']
    assert Mi.shape[1] == Y.shape[1]
    
    if 'Yli' not in scratches:
        scratches['Yli'] = empty( Y.shape[0], dtype = Y.dtype )
    Yli = scratches['Yli']
    
    for li in range( Y.shape[1] ):
        Yli[:] = Y[:,li]
        Y[:,li] = 1.
        ## UPDATE: I cannot use cumprod() when aliasing
        ## the input and output parameters and one is the reverse of the other.
        cumprod( Y[:,::-1], axis = 1, out = M )
        Y[:,li] = Yli
        Mr = M[:,::-1]
        Mr[:,li+1:] = 0.
        
        multiply( D[newaxis,...], Mr[...,newaxis], DM )
        DM.sum( 1, out = Mi[:,li,:] )
    
    multiply( energy_presquared[:,newaxis,:], Mi, Mi )
    out.shape = Y.shape
    Mi.sum( 2, out = out )
    out *= 2.
    out.shape = ( prod( Y.shape ), )

def gen_energy_and_gradient( img, layer_colors, weights, img_spatial_static_target = None, scratches = None ):
    '''
    Given a rows-by-cols-by-#channels 'img', where channels are the 3 color channels,
    and (#layers+1)-by-#channels 'layer_colors' (the 0-th color is the background color),
    and a dictionary of floating-point or None weights { w_spatial, w_opacity },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    returns a tuple of functions:
        ( e, g )
        where e( Y ) computes the scalar energy of a flattened rows-by-cols-by-#layers array of (1-alpha) values,
        and g( Y ) computes the gradient of e.
    '''
    
    img = asfarray( img )
    layer_colors = asfarray( layer_colors )
    
    assert len( img.shape ) == 3 
    assert len( layer_colors.shape ) == 2
    assert img.shape[2] == layer_colors.shape[1]
    
    #from pprint import pprint
    # pprint( weights )
    assert set( weights.keys() ).issubset( set([ 'w_fidelity_lsl2', 'w_ridge', 'w_spatial_static', 'w_tvl2' ]) )

    C = layer_colors
    P = img.reshape( -1, img.shape[2] )
    
    num_layers = C.shape[0]-1
    Ylen = P.shape[0] * num_layers
    
    if 'w_spatial_static' in weights:
        assert img_spatial_static_target is not None
        Yspatial_static_target = img_spatial_static_target.ravel()
    
    if 'w_tvl2' in weights:
        # print 'Preparing a Laplacian matrix for E_tvl2...'
        import fast_energy_laplacian
        import scipy.sparse
        # print '    Generating L...'
        LTL = fast_energy_laplacian.gen_grid_laplacian( img.shape[0], img.shape[1] )
        # print '    Computing L.T*L...'
        # LTL = LTL.T * LTL
        # print '    Replicating L.T*L for all layers...'
        ## Now repeat LTL #layers times.
        ## Because the layer values are the innermost dimension,
        ## every entry (i,j, val) in LTL should be repeated
        ## (i*#layers + k, j*#layers + k, val) for k in range(#layers).
        LTL = LTL.tocoo()
        ## Store the shape. It's a good habit, because there may not be a nonzero
        ## element in the last row and column.
        shape = LTL.shape
        
        ## There is a "fastest" version below.
        '''
        rows = zeros( LTL.nnz * num_layers, dtype = int )
        cols = zeros( LTL.nnz * num_layers, dtype = int )
        vals = zeros( LTL.nnz * num_layers )
        count = 0
        ks = arange( num_layers )
        for r, c, val in zip( LTL.row, LTL.col, LTL.data ):
            ## Slow
            #for k in range( num_layers ):
            #    rows.append( r*num_layers + k )
            #    cols.append( c*num_layers + k )
            #    vals.append( val )
            
            ## Faster
            rows[ count : count + num_layers ] = r*num_layers + ks
            cols[ count : count + num_layers ] = c*num_layers + ks
            vals[ count : count + num_layers ] = val
            count += num_layers
            
        assert count == LTL.nnz * num_layers
        '''
        
        ## Fastest
        ks = arange( num_layers )
        rows = ( repeat( asarray( LTL.row ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        cols = ( repeat( asarray( LTL.col ).reshape( LTL.nnz, 1 ) * num_layers, num_layers, 1 ) + ks ).ravel()
        vals = ( repeat( asarray( LTL.data ).reshape( LTL.nnz, 1 ), num_layers, 1 ) ).ravel()
        
        LTL = scipy.sparse.coo_matrix( ( vals, ( rows, cols ) ), shape = ( shape[0]*num_layers, shape[1]*num_layers ) ).tocsr()
        # print '...Finished.'
    
    if scratches is None:
        scratches = {}
    
    def e( Y ):
        e = 0.
        
        if 'w_fidelity_lsl2' in weights:
            e += weights['w_fidelity_lsl2'] * E_fidelity_lsl2( Y, C, P, scratches )
        
        if 'w_ridge' in weights:
            e += weights['w_ridge'] * E_ridge( Y, C, P, scratches )
        
        if 'w_spatial_static' in weights:
            e += weights['w_spatial_static'] * E_spatial_static( Y, Yspatial_static_target, scratches )
        
        if 'w_tvl2' in weights:
            e += weights['w_tvl2'] * E_tvl2( Y, LTL, scratches )

        # Extra energy term to penalize overlapping opacities (8 colors)
        #e += E_overlap(Y,C,P,scratches)
        #print E_overlap(Y,C,P,scratches)
        return e
    
    ## Preallocate this memory
    gradient_space = [ zeros( Ylen ), zeros( Ylen ) ]
    # total_gradient = zeros( Ylen )
    # gradient_term = zeros( Ylen )
    
    def g( Y ):
        total_gradient = gradient_space[0]
        gradient_term = gradient_space[1]
        
        total_gradient[:] = 0.
        
        if 'w_fidelity_lsl2' in weights:
            gradY_E_fidelity_lsl2( Y, C, P, gradient_term, scratches )
            gradient_term *= weights['w_fidelity_lsl2']
            total_gradient += gradient_term
        
        if 'w_ridge' in weights:
            grad_E_ridge( Y, C, P, gradient_term, scratches )
            gradient_term *= weights['w_ridge']
            total_gradient += gradient_term
        
        if 'w_spatial_static' in weights:
            grad_E_spatial_static( Y, Yspatial_static_target, gradient_term, scratches )
            gradient_term *= weights['w_spatial_static']
            total_gradient += gradient_term
        
        if 'w_tvl2' in weights:
            grad_E_tvl2( Y, LTL, gradient_term, scratches )
            gradient_term *= weights['w_tvl2']
            total_gradient += gradient_term
        
        #grad_E_overlap( Y, C, P, gradient_term, scratches )
        #total_gradient += gradient_term

        # print 'Y:', Y
        # print 'total_gradient:', total_gradient
        return total_gradient

    return e, g


def composite_layers( layers ):
    layers = asfarray( layers )
    
    ## Start with ridge white.
    out = 255*ones( layers[0].shape )[:,:,:3]
    for layer in layers:
        out += layer[:,:,3:]/255.*( layer[:,:,:3] - out )
    
    return out
      
def optimize( arr, colors, Y0, weights, img_spatial_static_target = None, scratches = None, saver = None ):
    '''
    Given a rows-by-cols-by-#channels array 'arr', where channels are the 3 color channels,
    and (#layers+1)-by-#channels 'colors' (the 0-th color is the background color),
    and rows-by-cols-by-#layers array 'Y0' of initial (1-alpha) values for each pixel (flattened or not),
    and a dictionary of floating-point or None weights { w_fidelity_lsl2, w_opacity, w_tvl2, w_spatial_static },
    and an optional parameter 'img_spatial_static_target' which are the target values for 'w_spatial_static' (if not flattened, it will be),
    and an optional parameter 'scratches' which should be a dictionary that will be used to store scratch space between calls to this function (use only *if* arguments are the same size),
    and an optional parameter 'saver' which will be called after every iteration with the current state of Y.
    returns a rows-by-cols-#layers array of optimized Y values, which are (1-alpha).
    '''
    
    start = time.clock()

    Y0 = Y0.ravel()
    
    Ylen = len( Y0 )

    e, g = gen_energy_and_gradient( arr, colors, weights, img_spatial_static_target = img_spatial_static_target, scratches = scratches )
    
    bounds = zeros( ( Ylen, 2 ) )
    bounds[:,1] = 1.
    
    ## Save the result-in-progress in case the users presses control-C.
    ## [number of iterations, last Y]
    Ysofar = [0,None]
    def callback( xk ):
        Ysofar[0] += 1
        ## Make a copy
        xk = array( xk )
        Ysofar[1] = xk
        
        if saver is not None: saver( xk )
    
    # print 'Optimizing...'
    # start = time.clock()
    
    try:
        ## WOW! TNC does a really bad job on our problem.
        # opt_result = scipy.optimize.minimize( e, Y0, method = 'TNC', jac = g, bounds = bounds )
        ## I did an experiment with the 'tol' parameter.
        ## I checked in the callback for a max/total absolute difference less than 1./255.
        ## Passing tol directly doesn't work, because the solver we are using (L-BFGS-B)
        ## normalizes it by the maximum function value, whereas we want an
        ## absolute stopping criteria.
        ## Max difference led to stopping with visible artifacts.
        ## Total absolute difference terminated on the very iteration that L-BFGS-B did
        ## anyways.

        opt_result = scipy.optimize.minimize( e, Y0, jac = g, bounds = bounds, callback = callback
          ,method='L-BFGS-B'
        
          ,options={'ftol': 1e-4, 'gtol': 1e-4}
         # ,options={'gtol': 1e-4} # Not optimum but faster CHANGED
        
         )

        #opt_result=scipy.optimize.least_squares(e, Y0, jac = g, bounds = (0, 1), loss='huber',ftol= 1e-5, gtol= 1e-5) # Huber is a soft-L1 norm
        
    
    except KeyboardInterrupt:
        ## If the user 
        print 'KeyboardInterrupt after %d iterations!' % Ysofar[0]
        Y = Ysofar[1]
        ## Y will be None if we didn't make it through 1 iteration before a KeyboardInterrupt.
        if Y is None:
            Y = -31337*ones( ( arr.shape[0], arr.shape[1], len( colors )-1 ) )
    
    else:
        # print opt_result
        Y = opt_result.x
    
    # duration = time.clock() - start
    # print '...Finished optimizing in %.3f seconds.' % duration
    
    end = time.clock()
    print 'Optimize an image of size ', Y.shape, ' took ', (end-start), ' seconds.'

    Y = Y.reshape( arr.shape[0], arr.shape[1], len( colors )-1 )
    return Y

def run_one(Y0, arr, json_data, outprefix, color_vertices ,save_every = None, solve_smaller_factor = None, too_small = None):
    #print imgpath
    '''
    Given a path `imgpath` to an image,
    a path `colorpath` to a JSON file containing an array of RGB triplets of layer colors (the 0-th color is the background color),
    a prefix `outprefix` to use for saving files,
    an optional path `weightspath` to a JSON file containing a dictionary of weight values,
    an optional positive number `save_every` which specifies how often to save progress,
    an optional positive integer `solve_smaller_factor` which, if specified,
    will first solve on a smaller image whose dimensions are `1/solve_smaller_factor` the full size image,
    and an optional positive integer `too_small` which, if specified, determines
    the limit of the `solve_smaller_factor` recursion as the minimum image size (width or height),
    runs optimize() on it and saves the output to e.g. `outprefix + "-layer01.png"`.
    '''
    
    #from PIL import Image

#    with open(param_path) as json_file:
#        json_data = json.load(json_file)


    input_image=json_data["stack_path"]
    order = range(color_vertices.shape[0]) # no order necessary for our purposes
    w_fidelity_lsl2=json_data["w_fidelity_lsl2"]
    w_ridge=json_data["w_ridge"]
    w_tvl2=json_data["w_tvl2"]
    threshold_opacity = json_data["threshold_opacity"]
    weights = {'w_fidelity_lsl2':w_fidelity_lsl2, 'w_ridge':w_ridge, 'w_tvl2':w_tvl2}
    #order=json_data["vertex_order"]
    #colorpath = json_data["color_path"]
    
    #arr = asfarray(imgpath)
    arr_backup=arr.copy()
    #if is_uint8 ==1:
    #    arr = arr/255.0
    #else:
    #    arr = arr/65535.0
    

    #colors = asfarray(json.load(open(colorpath))['vs'])
    colors =  asfarray(color_vertices.reshape(color_vertices.shape[0],3))
    colors_backup=colors.copy()
    colors=colors[order,:]/255.0
    
    assert solve_smaller_factor is None or int( solve_smaller_factor ) == solve_smaller_factor
    
    if save_every is None:
        save_every = 100.
    
    if solve_smaller_factor is None:
        solve_smaller_factor = 2
    
    if too_small is None:
        too_small = 5
    
    # arr = arr[:1,:1,:]
    # colors = colors[:3]
    
    kSaveEverySeconds = save_every
    ## [ number of iterations, time of last save, arr.shape ]
    last_save = [ None, None, None ]
    def reset_saver( arr_shape ):
        last_save[0] = 0
        last_save[1] = time.clock()
        last_save[2] = arr_shape
    def saver( xk ):
        arr_shape = last_save[2]
        
        last_save[0] += 1
        now = time.clock()
        ## Save every 10 seconds!
        if now - last_save[1] > kSaveEverySeconds:
            print 'Iteration', last_save[0]
            save_results( xk, colors, arr, arr_shape, outprefix, order, threshold_opacity ) # MIGHT CAUSE TROUBLE when saving smaller image [arr is input nwo]
            ## Get the time again instead of using 'now', because that doesn't take into
            ## account the time to actually save the images, which is a lot for large images.
            last_save[1] = time.clock()
    
    Ylen = arr.shape[0]*arr.shape[1]*( len(colors) - 1 )
    
    # Y0 = random.random( Ylen )
    # Y0 = zeros( Ylen ) + 0.0001
    #Y0 = .5*ones( Ylen )
    # Y0 = ones( Ylen )
    
    static = None
   # if weightspath is not None:
    #    weights = json.load( open( weightspath ) )
    #else:
    #    weights = { 'w_fidelity_lsl2': 375, 'w_ridge': 1., 'w_tvl2': 100. }
        # weights = { 'w_fidelity_lsl2': 1., 'w_ridge': 100. }
        # weights = { 'w_ridge': 100. }
        # weights = { 'w_spatial_static': 100. }
        # static = 0.75 * ones( Ylen )
        # weights = { 'w_tvl2': 100. }
        # weights = { 'w_tvl2': 100., 'w_ridge': 100. }

    num_layers=len(colors)-1
    ### adjust the weights:
    if 'w_fidelity_lsl2' in weights:
        # weights['w_fidelity_lsl2'] *= 50000.0 #### old one is 255*255
        weights['w_fidelity_lsl2'] /= arr.shape[2]
    
    if 'w_ridge' in weights:
        weights['w_ridge'] /= num_layers
    
    if 'w_spatial_static' in weights:
        weights['w_spatial_static'] /= num_layers
    
    if 'w_tvl2' in weights:
        weights['w_tvl2'] /= num_layers

    reset_saver( arr.shape )
    Y = optimize( arr, colors, Y0, weights, img_spatial_static_target = static, saver = saver )
    
    composite_img=save_results( Y, colors, arr, arr.shape, outprefix, order, threshold_opacity )
    img_diff=composite_img-arr_backup
    RMSE=sqrt(square(img_diff).sum()/(composite_img.shape[0]*composite_img.shape[1]))
    
    print 'img_shape is: ', img_diff.shape
    #print 'max dist: ', sqrt(square(img_diff).sum(axis=2)).max()
    #print 'median dist', median(sqrt(square(img_diff).sum(axis=2)))
    #print 'RMSE: ', RMSE
    return Y

    # **************************
def run_initial(arr, json_data, outprefix, color_vertices, save_every = None, solve_smaller_factor = None, too_small = None):
    
    #from PIL import Image

 #   with open(param_path) as json_file:
 #       json_data = json.load(json_file)

    input_image=json_data["stack_path"]
    order = range(color_vertices.shape[0])
    w_fidelity_lsl2=json_data["w_fidelity_lsl2"]
    w_ridge=json_data["w_ridge"]
    w_tvl2=json_data["w_tvl2"]
    threshold_opacity = json_data["threshold_opacity"]
    weights = {'w_fidelity_lsl2':w_fidelity_lsl2, 'w_ridge':w_ridge, 'w_tvl2':w_tvl2}
    #order=json_data["vertex_order"]
    #colorpath = json_data["color_path"]

    #print max(arr.flatten())
    #arr = asfarray(imgpath)

    arr_backup=arr.copy()
    #if is_uint8 ==1:
    #    arr = arr/255.0
    #else:
    #    arr = arr/65535.0
    
    #colors = asfarray(json.load(open(colorpath))['vs'])
    colors =  asfarray(color_vertices.reshape(color_vertices.shape[0],3))
    colors_backup=colors.copy()
    colors=colors[order,:]/255.0
    #print colors.shape
    
    assert solve_smaller_factor is None or int( solve_smaller_factor ) == solve_smaller_factor
    
    if save_every is None:
        save_every = 100.
    
    if solve_smaller_factor is None:
        solve_smaller_factor = 2
    
    if too_small is None:
        too_small = 5
    
    # arr = arr[:1,:1,:]
    # colors = colors[:3]
    
    kSaveEverySeconds = save_every
    ## [ number of iterations, time of last save, arr.shape ]
    last_save = [ None, None, None ]
    def reset_saver( arr_shape ):
        last_save[0] = 0
        last_save[1] = time.clock()
        last_save[2] = arr_shape
    def saver( xk ):
        arr_shape = last_save[2]
        
        last_save[0] += 1
        now = time.clock()
        ## Save every 10 seconds!
        if now - last_save[1] > kSaveEverySeconds:
            print 'Iteration', last_save[0]
            save_results( xk, colors, arr_shape, outprefix, order, threshold_opacity )
            ## Get the time again instead of using 'now', because that doesn't take into
            ## account the time to actually save the images, which is a lot for large images.
            last_save[1] = time.clock()
    
    Ylen = arr.shape[0]*arr.shape[1]*( len(colors) - 1 )
    
    # Y0 = random.random( Ylen )
    # Y0 = zeros( Ylen ) + 0.0001
    Y0 = .5*ones( Ylen )
    # Y0 = ones( Ylen )
    
    static = None
    #if weightspath is not None:
    #    weights = json.load( open( weightspath ) )
    #else:
    #    weights = { 'w_fidelity_lsl2': 375, 'w_ridge': 1., 'w_tvl2': 100. }
        # weights = { 'w_fidelity_lsl2': 1., 'w_ridge': 100. }
        # weights = { 'w_ridge': 100. }
        # weights = { 'w_spatial_static': 100. }
        # static = 0.75 * ones( Ylen )
        # weights = { 'w_tvl2': 100. }
        # weights = { 'w_tvl2': 100., 'w_ridge': 100. }

    num_layers=len(colors)-1
    ### adjust the weights:
    if 'w_fidelity_lsl2' in weights:
        # weights['w_fidelity_lsl2'] *= 50000.0 #### old one is 255*255
        weights['w_fidelity_lsl2'] /= arr.shape[2]
    
    if 'w_ridge' in weights:
        weights['w_ridge'] /= num_layers
    
    if 'w_spatial_static' in weights:
        weights['w_spatial_static'] /= num_layers
    
    if 'w_tvl2' in weights:
        weights['w_tvl2'] /= num_layers

    
    #if solve_smaller_factor != 1:
    #    assert solve_smaller_factor > 1
        def optimize_smaller( solve_smaller_factor, large_arr, large_Y0, large_img_spatial_static_target ):
            ## Terminate recursion if the image is too small.
            if large_arr.shape[0]//solve_smaller_factor < too_small or large_arr.shape[1]//solve_smaller_factor < too_small:
                return large_Y0
            
            ## small_arr = downsample( large_arr )
            small_arr = large_arr[::solve_smaller_factor,::solve_smaller_factor]
            ## small_Y0 = downsample( large_Y0 )
            small_Y0 = large_Y0.reshape( large_arr.shape[0], large_arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            ## small_img_spatial_static_target = downsample( large_img_spatial_static_target )
            small_img_spatial_static_target = None
            if large_img_spatial_static_target is not None:
                small_img_spatial_static_target = large_img_spatial_static_target.reshape( arr.shape[0], arr.shape[1], -1 )[::solve_smaller_factor,::solve_smaller_factor].ravel()
            
            ## get an improved Y by recursively shrinking
            small_Y1 = optimize_smaller( solve_smaller_factor, small_arr, small_Y0, small_img_spatial_static_target )
            
            ## solve on the downsampled problem
            print '==> Optimizing on a smaller image:', small_arr.shape, 'instead of', large_arr.shape
            reset_saver( small_arr.shape )
            small_Y = optimize( small_arr, colors, small_Y1, weights, img_spatial_static_target = small_img_spatial_static_target, saver = saver )
            
            ## save the intermediate solution.
            saver( small_Y )
            
            ## large_Y1 = upsample( small_Y )
            ### 1 Make a copy
            large_Y1 = array( large_Y0 ).reshape( large_arr.shape[0], large_arr.shape[1], -1 )
            ### 2 Fill in as much as will fit using numpy.repeat()
            small_Y = small_Y.reshape( small_arr.shape[0], small_arr.shape[1], -1 )
            small_Y_upsampled = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            large_Y1[:,:] = small_Y_upsampled[ :large_Y1.shape[0], :large_Y1.shape[1] ]
            # large_Y1[ :small_Y.shape[0]*solve_smaller_factor, :small_Y.shape[1]*solve_smaller_factor ] = repeat( repeat( small_Y, solve_smaller_factor, 0 ), solve_smaller_factor, 1 )
            ### 3 The right and bottom edges may have been missed due to rounding
            # large_Y1[ small_Y.shape[0]*solve_smaller_factor:, : ] = large_Y1[ small_Y.shape[0]*solve_smaller_factor - 1 : small_Y.shape[0]*solve_smaller_factor, : ]
            # large_Y1[ :, small_Y.shape[1]*solve_smaller_factor: ] = large_Y1[ :, small_Y.shape[1]*solve_smaller_factor - 1 : small_Y.shape[1]*solve_smaller_factor ]
            
            return large_Y1.ravel()
        
        Y0_initial = optimize_smaller( solve_smaller_factor, arr, Y0, static )
    
    reset_saver( arr.shape )
    return Y0_initial


def save_results( Y, colors, img, img_shape, outprefix, order=[] , threshold_opacity = 0): # saving to layer folder commented out 
    #from PIL import Image
    import tifffile as tifffile

    # Hard thresholding
    Y[Y>(1-float(threshold_opacity)/255.0)] = 1.0
    P = img.reshape(-1, img.shape[2] ).copy()
    # Last weighting
    from skimage import color     
    P_temp = color.rgb2lab(transpose(tile(P, (colors[1:,:].shape[0], 1, 1)), (1,0,2)) )[:,:,1:]/100.0 #norm
    C_temp = color.rgb2lab(tile(colors[1:,:], (P.shape[0], 1, 1)))[:,:,1:]/100.0
    #P_temp = transpose(tile(P, (C[1:,:].shape[0], 1, 1)), (1,0,2)) # ignore black [0,0,0]
    #C_temp = tile(C[1:,:], (P.shape[0], 1, 1))

    # 
    #final_weight_linear = (1.0 - mean(sqrt((P_temp - C_temp) * (P_temp - C_temp)), axis= 2) ) # LINEAR
    final_weight_q1 = (1.0 - mean(absolute(P_temp - C_temp), axis= 2) )**2 # QUADRATIC
    #final_weight_q2 = 1.0 - (mean(sqrt((P_temp - C_temp) * (P_temp - C_temp)), axis= 2) )**2 # QUADRATIC2

    Y = Y.reshape( img_shape[0], img_shape[1], -1 )
    Y= 1.0 - ((1.0-Y) * final_weight_q1.reshape( Y.shape ) )

    alphas = 1. - Y
    layers = []
    lay_folder = 0
    for li, color in enumerate( colors ):  ### colors are now in range[0.0,1.0] not [0,255]
        layer = ones( ( img_shape[0], img_shape[1], 4 ), dtype = uint8 )
        layer[:,:,:3] = asfarray(color*255.0).round().clip( 0,255 ).astype( uint8 )
        layer[:,:,3] = 255 if ( li == 0 ) else (alphas[:,:,li-1]*255.).round().clip( 0,255 ).astype( uint8 )
        layers.append( layer )
        outpath = output_folder+str(order[lay_folder]) +'/'+ outprefix + '-layer%02d.png' % li
        outpath_tiff = output_folder+'layer%02d.tif' % li
        #Image.fromarray( layer ).save( outpath )
        if ( li != 0 ): tifffile.imsave(outpath_tiff, layer[:,:,3] , append='True')  # save alphas as tiff stacks	(except 0th beackground)		
        if ( li != 0 ) and SAVE_COLOR==1: tifffile.imsave(outpath_tiff+'_colored', layer, append='True')  # save alphas as tiff stacks	(except 0th beackground)		

        #print 'Saved layer:', outpath
        lay_folder = lay_folder +1
    
    composited = composite_layers( layers )
    composited = composited.round().clip( 0, 255 ).astype( uint8 )
    #outpath2 = output_folder+'composite/' + outprefix + '-composite.png'
    #Image.fromarray( composited ).save( outpath2 )
    #print 'Saved composite:', outpath
    return composited


# ----------------------------------------------------------------------------------------
if __name__ == '__main__':

    sys.path.insert(0, 'misc_code/')
    sys.path.insert(0, 'preprocess/flattening/')

    param_path = 'params.json'

    with open(param_path) as json_file:
        json_data = json.load(json_file)

    input_image=json_data["stack_path"]

    level_flattening =json_data["level_flattening"]
    level_contrast_enhancement =json_data["level_contrast_enhancement"]
    iterations_flattening =json_data["iterations_flattening"]

    #print level_flattening, level_contrast_enhancement

    N = json_data["number_soft_segments"]
    output_folder = json_data["output_path"]
    global RUN_FAST # global seemed easy at this moment
    RUN_FAST = json_data["FAST"]

    global SAVE_COLOR 
    SAVE_COLOR = json_data["SAVE_COLOR"]

    save_every = 1000000 # Intermediate saving step: Save [every] second [Not needed]
    solve_smaller_factor = None
    too_small = None       

    start=time.clock()
    
    print '-----------------------------------------------'
    print "Running from stack ..."
    if RUN_FAST == 0:
    	print '- Weighting in data fidelty term activated [Slower]'

    #from skimage import io
    #import numpy as np
    #im_stack = io.imread(input_image)
    from tifffile import imread, imshow
    im_stack = imread(input_image)
    try:
    	n_image, row, col, ch = im_stack.shape
    except ValueError:
    	row, col, ch = im_stack.shape
    	im_stack = im_stack.reshape([1, row, col, ch])
    	n_image, row, col, ch = im_stack.shape
    	print '- It is a single image plane, not a stack'
    #print im_stack.shape

    print '- input stack path: ' + input_image
    print '- output path: ' + output_folder

    if ch > 3: # when reading 16 bit color stacks saved from ImageJ
    	print '- Warn: # of color ch. =? ' + str(ch)
    	print '- Swapping channels ..'
    	im_stack = transpose(im_stack,(0,2,3,1))
    	n_image, row, col, ch = im_stack.shape
    	print im_stack.shape

    # Exception if the image not uint8 or uint16 
    assert im_stack.dtype in ['uint8', 'uint16']        
    if im_stack.dtype == 'uint16':
        print 'uint16 stack reading ...'
        im_stack = asfarray(im_stack)/65535.0
        print ''
        im_stack =  (im_stack / percentile(im_stack, 99.3)).clip(0.0, 1.0) # allow little saturation in uint16
        max_projection = (255.0 * im_stack).astype('uint8').max(axis=0)
    else:
        print 'uint8 stack reading ...'
        im_stack = asfarray(im_stack)/255.0
        max_projection = (255.0 * im_stack).astype('uint8').max(axis=0)

    #print max_projection.shape
    from compute_color import compute_color
    color_vertices = compute_color(max_projection, N)
    

    #for k in [0]:
    for k in range(n_image):

        print "-----------"
        print  "Running "+ str(k+1) + "/"+str(n_image)  

        ## Piecewise image recovery ---
        from flatten import flatten, contrast_stretch
        if level_contrast_enhancement !=0:
            im = flatten(im_stack[k,:,:,:], iterations_flattening, level_flattening) # 2 iterations, level 11 

        # Contrast enhance 1
        if level_contrast_enhancement != 0:
            im = contrast_stretch(im, level_contrast_enhancement, 2, 98)

        if level_contrast_enhancement != 0 and level_contrast_enhancement !=0:
            im = im_stack[k,:,:,:]
      
        ## plane show
        #from matplotlib import pyplot as plt
        #imshow(im_stack[k,:,:,:])
        #imshow(im)
        #plt.show()

        # use image pyramid to initialize for the first one then propogate initilization
        if k==0:
            Y0 = run_initial(im, json_data, str(k), color_vertices, save_every = save_every, solve_smaller_factor = solve_smaller_factor, too_small = too_small)
        else:
            Y0=Y_prev.copy()    
        Y_prev = run_one( Y0, im, json_data, str(k), color_vertices ,save_every = save_every, solve_smaller_factor = solve_smaller_factor, too_small = too_small)
    end=time.clock()
    print 'time: ', end - start

## EOF

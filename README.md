## Soft Segmentation of Viral Labeled Neurons

The script designed to segment spectral (pseudo-color) viral labeled dendritic trees. It avoids hard borders and creates soft segments for each distinct pseudo-color. The package contains automatic palette calculation and post-processing (in Matlab) scripts to obtain denoised binary masks from soft segments. We developed an image flattering algorithm based on weighted L1-norm total variation to increase diffusion of viral vectors, and implemented it using Prox-TV library. We also employed BaSIC ImageJ background subtraction plugin (freely available online, also included here), if the image is corrupted by heavy background noise. As an example, recovery of green and orange neurons can be seen below:

<div align="center">
  <img src="docs/sparse.png" width="900"><br>
</div>


### Install dependencies

Anaconda environments are recommended. After installing it, create an environment with Python 2.7
    
    $ conda create -n softsegment python=2.7

Then install the following dependencies: Scipy, scikit-image, cvxopt, cython, cffi, pathlib and enum34 by:

    $ conda install -c anaconda -n softsegment <package_name>
    
Make sure to activate the environment, then install Prox-TV library by:

	$ source activate softsegment
    $ pip install prox-tv

### How to use

When you open terminal, first activate your conda environment by:

    $ source activate softsegment

Then simply run following, we already have a sample image at input/ and softsegments with corresponding pseudo-color tags should appear at output/:

    $ python softsegment.py


Note that **params.json** has all parameters. No need to modify the code if you are lazy, directly change from the file.


| Parameters |  Notes |
| ------ | ------ |
| stack_path | directory containing input stack |
| output_path | directory to save results |
| automatic_color_vertices | set 1 to compute pseudo-color palette, 0 to manually indicate |
| manual_vertices | if automatic computation set 0, indicate pseudo-color vertices of the hull. e.g., [[1,0,0], [0,1,0], [0,0,1]]|
| number_soft_segments | number of output segments (has to be same as the number of vertices, do not count background black): e.g., 6 |
| drop_color | ignore one of the vertices when computing soft segments (useful if two segments are highly overlapping, just drop one) e.g., 2 or put -1 not to drop any |
| FAST | disable weighting term in the optimization function (much faster, but little suboptimal): 1 or 0 |
| SAVE_COLOR | also saves colored versions of soft segments: 1 or 0 |
| start_plane| compute segments from substack starting from e.g., 25 or set -1 to disable|
| end_plane | compute segments from substack until: e.g., 125 or set -1 to disable|

The following parameters are used to control weighting of soft segmentation:

|  |   |
| ------ | ------ |
| w_fidelity_lsl2 | weight of L2-norm least-squares data fidelity term |
| w_ridge | weight of ridge (Tikhonov) regularization (prevents over-fitting)|
| w_tvl2 | weight of L2-norm total variation regularization (smoothing term)|
| threshold_opacity| clips opacity values below the threshold to 0 before saving |

The following parameters are used to control degree of flattening and contrast enhancement:

|  |   |
| ------ | ------ |
| level_contrast_enhancement | within {1, 10}, choose -1 not to enhance contrast|
| level_flattening | choose > 1, you may get assertation error if too high, just decrease it |
| iterations_flattening | flatten several times |

Soft segments (opacity layers) will be saved as tiff files, you may want to use ImageJ to easily load and see the results. Below, example processing results of our fast peicewise image recovery: 

<div align="center">
  <img src="docs/github2.png" width="600"><br>
</div>


### Citing
S. Bolkar, “Soft Segmentation of Viral Labeled Neurons,” MSc Thesis, COSI Erasmus Mundus Joint Master Degree, Norwegian University of Science and Technology & KU Leuven, Leuven, 2018.

### Acknowledgements

Our implementation is adapted from image manipulation algorithm developed by Tan et. al. (2015), please check their great work here: https://github.com/CraGL/Decompose-Single-Image-Into-Layers \
Paper: Tan, J., Lien, J. M., & Gingold, Y. (2017). Decomposing images into layers via RGB-space geometry. ACM Transactions on Graphics (TOG), 36(1), 7.

We use Prox-Tv library for our weighted L1-norm image flattening implementation, fast and effective: https://github.com/albarji/proxTV \
Paper: Modular proximal optimization for multidimensional total-variation regularization. Álvaro Barbero, Suvrit Sra. http://arxiv.org/abs/1411.0589

We recommend ImageJ plugin of BaSIC background subtraction: https://github.com/QSCD/BaSiC \
Paper: Peng, T., Thorn, K., Schroeder, T., Wang, L., Theis, F. J., Marr, C., & Navab, N. 2017. A basic tool for background and shading correction of optical microscopy images. Nature Communications, 8, 14836.



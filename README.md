
## Soft Segmentation of Viral Labeled Neurons

The script designed to segment spectral (pseudo-color) viral labeled dendritic trees. It avoids hard borders and creats soft segments for each distinct pseudo-color. The package contains automatic palette computation and post processing scripts to obtain denopised binary masks from soft segments. We also developed a image flatteinmg algorthm based on weighted L1-norm TV, and implemete it using Prox-TV package. Additionaly, we also recommend using BaSIC ImageJ background subruction plugin (freely avaliable, also included here) if image are corrupted by heavy background noise. As an example (recovery of green and orange neuron):

<div align="center">
  <img src="docs/sparse.png" width="900"><br>
</div>

### Install dependencies

Anaconda environemnts are recommended. After installing it, create a enivronment wtih Python 2.7
    
    conda create -n softsegment python=2.7

Then istall following dependecies: Scipy, scikit-image, cvxopt, cython, cffi, pathlib and enum34 by:

    conda install -c anaconda -n softsegment <package_name>
    
Then install Prox-tv library by:

    pip install prox-tv

### How to use

First activate you conda evironemnt by:

	source activate softsegment

Then simply run following, we alreay have a sample image at input/ and softsegments with corresponding pseudo-color tags should appear at output/:

	python softsegment,py

Note that params.json has the all parameters, no need to modify scripts. 

| Parameter |  |
| ------ | ------ |
| stack_path | directory of the input stack |
| output_path | directory to save results |
| automatic_color_vertices | 1 to compute pseudo-color palette, 0 to manually indicate |
| manual_vertices | if automatic computation set 0, indicate pseudo-color vertices of the hull. e.g., [[1,0,0], [0,1,0], [0,0,1]]|
| number_soft_segments | number of output segments (has to be same as number of vertices, do not count background black): e.g., 6|
| drop_color | ignore one of the vertices when computing soft segments (usefull if two segments are highly overlapping, just drop one) e.g., 2 or put -1 not to drop any |
| FAST | disable weighting term in the monomization function (much faster, but little suboptimal): 1 or 0 |
| SAVE_COLOR | also saves colored versions of soft segments: 1 or 0 |
| start_plane| compute segments from substack starting from e.g., 25|
| end_plane | compute segments from substack until: e.g., 125|

The fallowing paramters are used to control weighting of soft segmentation:

| w_fidelity_lsl2 | weight of L2-norm least-squares data fidelity term |
| w_ridge | weight of ridge (Tikhonov) regularizaiton (prevents over-fitting)|
| w_tvl2 | weight of L2-norm total variation regulization (smoother)|
| threshold_opacity| hard tresholds opacity layers before saving |
| end_plane | compute segments from substack until: e.g., 125|

The folllowing paramters are used to control degree of flattening and contrast enhancement:

| level_contrast_enhancement | within {1, 10}, choose -1 not to enhance contrast|
| level_flattening | choose > 1, you may get assertation error if too high, just decrease it |
| iterations_flattening | flaten several times |

Soft segmenets (opcaity layers) will be saved as tiff files, you may want ouse ImageJ to easliy load and se the stacks.


### Acknowledgements

Our implementation is adapted from the paper "Decomposing Images into Layers via RGB-space Geometry" by Tan et. al. (2015), please check their great work here: https://github.com/CraGL/Decompose-Single-Image-Into-Layers

We use Prox-Tv library for weighted L1-norm image flatttening, fast and effective: https://github.com/albarji/proxTV 

We use ImgeJ plugin of BaSIC background subtruction: https://github.com/QSCD/BaSiC

### Referances




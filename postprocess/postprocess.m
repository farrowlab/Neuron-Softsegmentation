%
% simple softsegment to hardsegment conversion for sparse images
%  
% Alg: median3d (optional) + dilate and connect close ccs + erosion + keep leargest N  
%                + over-dilate and get close ccs to create a mask + get the largest cc + use it as mask on
%                  filtered version
%

%cd /home/elrasbolkar/Desktop/brainbow_elras/A_final/compare/b_qualitative/sparse_katja_code
clear, clc;
fname =  '../sparse_orange/layer03.tif';
info = imfinfo(fname);
num_images = numel(info);

% read image
for k = 1:num_images % get info from alpha
     k
     GFP(:,:,k) = imread(fname, k);
end
[ny, nx, nz] = size(GFP);

% 3d median filter to remove speckle
figure; imshow(max(GFP, [],3));

%stack = medfilt3(GFP, [3,3,3]);
stack =GFP; % if you dont need median filtering
%figure; imshow(max(stack, [],3));

% parameters
n_biggest_cc          = 200; % Robust, other size based cc may remove neuron parts 
distance_to_merge1    = 3;
distance_to_merge2    = 9;
erode_radius          = 3;

threshold             = 0.5;            
dilationRadius        = 1;              
dilationBase          = dilationRadius; 
dilationHeight        = 1;              
%sizeThreshold         = 200;             
%conservativeThreshold = 0.6;            

% make sure stack is between 0 and 1
stack = double(stack);
stack = stack - min(stack(:)); stack = stack/max(stack(:));

% binarize stack at threshold (a high value ensures noise is removed)
% dilate binarized stack - to ensure neuron remains connected
[xx, yy, zz] = meshgrid(-dilationBase:dilationBase,-dilationBase:dilationBase,-dilationHeight:dilationHeight);
dilationKernel = sqrt(xx.^2+yy.^2+zz.^2)<=dilationRadius;
dilatedStack = imdilate(stack>threshold, dilationKernel);
figure; imshow(max(dilatedStack, [],3));

% connect object with distances smaller than k
dilatedStack2 = bwdist(dilatedStack) < distance_to_merge1;
%dilatedStack2 = bwdist(dilatedStack, 'chessboard') < distance_to_merge1;
%dilatedStack2 = bwdist(dilatedStack, 'cityblock') < distance_to_merge1;
dilatedStack2 = imerode(dilatedStack2, strel('disk',erode_radius));
figure; imshow(max(dilatedStack2, [],3));

% keep largest n connected components to relax algo
result = zeros(size(dilatedStack2));
conn = bwconncomp(dilatedStack2);
A = regionprops(conn, 'Area');
[sorted, sorted_indeces] = sort(cell2mat(struct2cell(A)));
largest_ccs_inds= sorted_indeces((end-n_biggest_cc+1):end);

for k = 1:length(largest_ccs_inds)
    result(cell2mat(conn.PixelIdxList(largest_ccs_inds(k)))) =1;
end
figure; imshow(max(result, [],3));

% connect closer objects further
result2 = bwdist(result) < distance_to_merge2;
conn2 = bwconncomp(result2);
L = labelmatrix(conn2);
rgb = label2rgb(max(L, [],3), 'jet', [.7 .7 .7], 'shuffle');
figure; imshow(rgb)

% get the biggest connected compoenent a
last_mask = zeros(size(dilatedStack2));
B = regionprops(L, 'Area');
[val, ind] = max(cell2mat(struct2cell(B)));
last_mask(cell2mat(conn2.PixelIdxList(ind))) =1;
%figure; imshow(max(last_mask, [],3));

% use biggest connected component as mask
GFP_masked = stack;
GFP_masked_dil = last_mask;
GFP_masked_dil(~dilatedStack) = 0;
GFP_masked(~last_mask) = 0; GFP_masked(GFP_masked>0) = 1;
figure; imshow(max(GFP_masked_dil, [],3));
figure; imshow(max(GFP_masked, [],3), []);

max_z = max(GFP_masked, [],3);
max_x = max(GFP_masked, [],2); max_x = reshape(max_x, [ny, nz]);
max_y = max(GFP_masked, [],1); max_y = reshape(max_y, [nx, nz])';
%figure; imshow(max_y);
%figure; imshow(max_x);

% save projections and result
width = 50;
im_prof = ones(ny+nz+width, nx+nz+width);
im_prof(1:ny, 1:nx) = max_z;
im_prof(ny+width+1:end, 1:nx) = max_y;
im_prof(1:ny, nx+width+1:end) = max_x;
figure; imshow(im_prof);

%
imwrite(max(GFP, [],3) ,'original_proj.png');
imwrite(im_prof ,'masked_proj.png');
save('neuron_masked.mat', 'GFP', 'GFP_masked', 'GFP_masked_dil');

%EOF
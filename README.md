# Zooming-image-using-Nearest-Neighbor-Interpolation
Digital Image Processing magnifying image using Nearest Neighbor Interpolation.
As its name suggest , in this method , we just replicate the neighboring pixels. As we have already discussed in the tutorial of Sampling , that zooming is nothing but increase amount of sample or pixels. This algorithm works on the same principle.

WORKING:
In this method we create new pixels form the already given pixels. Each pixel is replicated in this method n times row wise and column wise and you got a zoomed image. Its as simple as that.

Example: (open in edit or raw mode) : 
initial image:
1 2
3 4
step1:
1 1 2 2
3 3 4 4
step2:
1 1 2 2
1 1 2 2
3 3 4 4
3 3 4 4

and we have the 2x magnified image.

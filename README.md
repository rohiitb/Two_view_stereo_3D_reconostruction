# Two_view_stereo_3D_reconostruction

This project was done as the part of the course CIS580 : Machine Perception. In this project, 3D reconstruction from multiple images was done using two different algorithms :- Two-view stereo and Multi-view stereo algorithms.
<br>
In two-view stereo algorithm, firstly, the images were rectified. Then the disparity map was created by matching the patches in the images using three different distance metric kernels - ssd, sad and zncc. Then, from the disparity map, we obtained the depth map which in turn gave us the pointcloud.<br>
In the multiview stereo algorithm, we sweep imaginary depth planes (fronto-parallel) across a range of candidate depths and then we project the neighboring views onto an imaginary depth plane and back onto the reference view using homography. Then we produce a cost/similarity map between the reference view and warped neighboring view using patches similar to the previous algorithm. Then we construct a cost volume by stacking the depth  cost maps and then take the argmin of the costs across the depths at each pixel to output the depth map. 

# Results

Below is the reconstruction using two-view algorithm:

<img src="./Results/two_view1.gif">

Below is the reconstruction using multi-view algorithm:

<img src="./Results/plane1.gif">

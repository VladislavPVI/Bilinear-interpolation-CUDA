# Bilinear-interpolation-CUDA
In mathematics, bilinear interpolation is an extension of linear interpolation for interpolating functions of two variables (e.g., x and y) on a rectilinear 2D grid.
Bilinear interpolation is performed using linear interpolation first in one direction, and then again in the other direction. Although each step is linear in the sampled values and in the position, the interpolation as a whole is not linear but rather quadratic in the sample location.
Bilinear interpolation is one of the basic resampling techniques in computer vision and image processing, where it is also called bilinear filtering or bilinear texture mapping.

*Microsoft visual studio 19 +  CUDA Toolkit 11*

Build and Run
-------------

1. Install Microsoft Visual Studio.
2. Install CUDA Toolkit (Nvidea GPU with CUDA-support required).
3. Make new CUDA-project.
4. Enjoy.

## System configuration

| Name  | Values  |
|-------|---------|
| CPU  | Intel® Pentium® G860 |
| RAM  | 6 GB DDR3 |
| GPU  | GeForce GTX 750 Ti 2GB (overlock) |
| OS   | Windows 10 64-bit  |

## Results

<img src="https://github.com/VladislavPVI/Bilinear-interpolation-CUDA/blob/master/DOC/angelina4.jpg" /> |
------------ |
Input 240 x 150

<img src="https://github.com/VladislavPVI/Bilinear-interpolation-CUDA/blob/master/DOC/GPUoutAngelina.jpg" /> | <img src="https://github.com/VladislavPVI/Bilinear-interpolation-CUDA/blob/master/DOC/CPUoutAngelina.jpg" />
------------ | ------------- 
Output GPU 480 x 300 | Output CPU 480 x 300

Average results after 100 times of runs.

|    Input size  |   Output size |          CPU        |         GPU       | Acceleration |
|-------------|-|--------------------|-------------------|--------------|
| 480x300   | 960x600 |3 ms               | 1.5 ms            |    2      |
| 1920x1200   | 3840x2400 |72 ms               | 20 ms            |    3.6      |
| 3840x2400   | 7680x4800 |207 ms              | 96.64 ms             |    2.14      |

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

<img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/NoiseAngelina.jpg" width="480" height="300" /> | <img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/CPUoutAngelina.jpg" width="480" height="300" />
------------ | ------------- 
Distorted image (noise 8%) | Filtered image (CPU)

<img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/NoiseAngelina.jpg" width="480" height="300" /> | <img src="https://github.com/VladislavPVI/Salt-and-Pepper-Filter-CUDA/blob/master/DOC/GPUoutAngelina.jpg" width="480" height="300" />
------------ | ------------- 
Distorted image (noise 8%) | Filtered image (GPU)

Average results after 100 times of runs.

|    Input size  |   Output size |          CPU        |         GPU       | Acceleration |
|-------------|-|--------------------|-------------------|--------------|
| 240 x 150   | |9 ms               | 0.1 ms            |    90      |
| 480 x 300   | |34 ms               | 0.37 ms            |    91.89      |
| 960 x 600   | |140 ms              | 1.47 ms             |    95.23      |
| 1920x1200 | |452 ms   | 5.66 ms            |    79.85      |
| 3840x2400 | |2608 ms | 20.73 ms |    125.8      |

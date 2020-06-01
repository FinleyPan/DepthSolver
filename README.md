# Introduction

DepthSolver is a depth map estimator for both monocular and binocular cases.

# dependencies
- CMake
- Eigen3
- OpenCV
- OpenCL

# Results for [TUM Dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download#freiburg2_desk)
## solve depth map in realtime:
![depth_result](https://github.com/FinleyPan/DepthSolver/blob/master/depth_res.gif)
## recover point cloud by voxel merging
![point_cloud_res](https://github.com/FinleyPan/DepthSolver/blob/master/point_cloud_res.gif)
## matters need attentation
Before running the app, download the dataset and a python script for data alignment firstly:
```shell
$ cd /path/to/save/your/data
$ wget https://vision.in.tum.de/rgbd/dataset/freiburg2/rgbd_dataset_freiburg2_desk.tgz
$ tar -xvf rgbd_dataset_freiburg2_desk.tgz
$ cd rgbd_dataset_freiburg2_desk
$ wget https://svncvpr.in.tum.de/cvpr-ros-pkg/trunk/rgbd_benchmark/rgbd_benchmark_tools/src/rgbd_benchmark_tools/associate.py
```
Run above python script to get associate file:
```
$ python associate.py rgb.txt groundtruth.txt > associate.txt
```
Now, you can run the app by following commands:
```
./pc_maker /path/to/save/your/data/associate.txt
```


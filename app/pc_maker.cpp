#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <chrono>

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "depth_solver.h"
#include "utils.h"

using namespace utils;

#define FX 525.0
#define FY 525.0
#define CX 319.5
#define CY 239.5

#define WIDTH 640
#define HEIGHT 480

int main(int argc, char const* const* argv)
{
    if(argc != 2){
        std::cerr<<"usage: ./pc_maker <path_to_assoc_file>"<<std::endl;
        return -1;
    }
    SetCrashHandler();
    Eigen::Matrix3f K = Eigen::Matrix3f::Identity();    
    K(0, 0) = FX;
    K(1, 1) = FY;
    K(0, 2) = CX;
    K(1, 2) = CY;
    int width = WIDTH;
    int height = HEIGHT;
    float scale = 0.5f;

    std::string path_assoc(argv[1]),line;
    std::string root = path_assoc.substr(0, path_assoc.find_last_of("\\/"));
    std::ifstream fin(argv[1]);
    if(!fin.is_open()){
        std::cerr<<"ERROR: assoc file doesn't exist!"<<std::endl;
        return -1;
    }

    ds::CameraParams cam0(K(0,0), K(1,1), K(0,2), K(1,2), nullptr);
    ds::Configuration config(width, height, &cam0, nullptr, scale, ds::SGM_OCL);
    ds::DepthSolver solver(&config);
    ds::Frame::set_mode(config.mode);

    PointCloud point_cloud;
    std::ofstream pc_dumper("pc.obj");

    int id = 0;
    while(getline(fin, line)){
        std::istringstream iss(line);
        std::string t_img, t_pose, img_rel_path;
        Eigen::Vector3f t;
        Eigen::Quaternionf q;
        iss>>t_img>>img_rel_path>>t_pose>>
             t.x()>>t.y()>>t.z()>>
             q.x()>>q.y()>>q.z()>>q.w();
        Eigen::Matrix4f T_wc = Eigen::Matrix4f::Identity();
        T_wc.block<3,3>(0,0) = Eigen::Matrix3f(q);
        T_wc.block<3,1>(0,3) = t;

        cv::Mat img = cv::imread(root + "/" + img_rel_path, CV_LOAD_IMAGE_GRAYSCALE);
        if(img.empty())
            continue;

        cv::resize(img, img, cv::Size(scale*width, scale*height));
        ds::Frame frame(id++, scale * width, scale * height, img.data, T_wc.data());
        cv::Mat depth(scale * height, scale * width, CV_16UC1, frame.depth());
        cv::Mat depth_clone;
        solver.Execute(frame);

        point_cloud.Update(frame, K * scale);

        cv::normalize(depth, depth_clone, 0, 255, cv::NORM_MINMAX, CV_8U);
        cv::Mat res;
        cv::hconcat(img, depth_clone, res);
        cv::imshow("disp", res);
        if(cv::waitKey(1) == 27)
            break;
    }

    //save point cloud
    pc_dumper << point_cloud;
    fin.close();
    pc_dumper.close();
    return 0;
}

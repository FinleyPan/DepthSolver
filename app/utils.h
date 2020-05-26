#ifndef POINTCLOUD_H
#define POINTCLOUD_H

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>
#include <forward_list>
#include <unordered_map>

#include <Eigen/Dense>
#include "depth_solver.h"

namespace utils{

typedef ds::Frame Frame;
typedef ds::Float Float;
typedef ds::Byte Byte;
typedef ds::Mode Mode;
typedef void(*HandlerType)(int sig);

struct Voxel{
    std::string key;
    Eigen::Vector3f position;
    Eigen::Vector3f normal;
    Eigen::Vector3f color;
    int track_length;
};

class PointCloud
{
public:
    typedef std::forward_list<Voxel>::iterator LIterType;
    PointCloud() : total_voxels_(0), added_voxels_(0){}
    void Update(const Frame& frame, const Eigen::Matrix3f& K,
                Mode mode = Mode::MOTION_STEREO);
    inline uint32_t size() const {return total_voxels_;}
    inline const std::forward_list<Voxel>& voxel_list() const
    {return voxel_list_;}

private:
    //return number of trimmed voxels
    uint32_t TrimVoxelList();

    uint32_t total_voxels_, added_voxels_;
    LIterType trim_end_;
    std::forward_list<Voxel> voxel_list_;
    //key: describe the stringified voxel.
    //value: iterator to the real voxel.
    std::unordered_map<std::string, LIterType> voxel_map_iter_;
};

std::ostream& operator<<(std::ostream& os, const PointCloud& pc);

//some helper functions
void DefaultHandler(int sig);
void SetCrashHandler(HandlerType handler = DefaultHandler);

}

#endif // POINTCLOUD_H

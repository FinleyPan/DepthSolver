#include "utils.h"

#include <execinfo.h>
#include <signal.h>

#include <opencv2/core/core.hpp>
#include <Eigen/Geometry>

namespace utils{

static constexpr float VOXEL_RESOLUTION = 100.0f;
static constexpr float MAX_SQUARED_REPROJ_ERROR = 2.0f;
static constexpr float MAX_DEPTH_ERROR = 0.001f;
static constexpr int MIN_TRACK_LENGTH = 7;
static constexpr uint32_t NUM_ADDED_VOXELS_PER_POLL = 4e5;

void PointCloud::Update(const Frame &frame, const Eigen::Matrix3f &K, Mode mode){
    if(added_voxels_ >= NUM_ADDED_VOXELS_PER_POLL){
        uint32_t trimmed_voxels = TrimVoxelList();
        added_voxels_ = 0;
        total_voxels_ -= trimmed_voxels;
    }

    if(mode == Mode::MOTION_STEREO){
        int width = frame.width();
        int height = frame.height();
        int fid = frame.get_id();
        Byte* img_data = const_cast<Byte*>(frame.image0());
        uint16_t* depth_data = const_cast<uint16_t*>(frame.depth());
        Float* pose_data = const_cast<Float*>(frame.pose0());

        const cv::Mat image(height, width, CV_8UC1, img_data);
        const cv::Mat depth(height, width, CV_16UC1, depth_data);
        const Eigen::Matrix4f T_wc(pose_data);
        const Eigen::Matrix3f R = T_wc.block<3,3>(0, 0);
        const Eigen::Vector3f t = T_wc.block<3,1>(0, 3);
        const Eigen::Matrix3f RT = R.transpose();
        const Eigen::Vector3f t_inv = -RT * t;
        const Float fx = K(0,0);
        const Float fy = K(1,1);
        const Float cx = K(0,2);
        const Float cy = K(1,2);

        for(int y=0; y<height; y++){
            const Byte* head_I = image.ptr<Byte>(y);
            const uint16_t* head_D = depth.ptr<uint16_t>(y);
            for(int x=0; x<width; x++){
                Float d = (Float)head_D[x] / 1000.0f;
                if(std::fabs(d) < 1e-6)
                    continue;

                float gray = head_I[x];
                Eigen::Vector3f pw = R * Eigen::Vector3f(d * (x - cx) / fx,
                                                         d * (y - cy) / fy,
                                                         d) + t;
                int xi = pw.x() * VOXEL_RESOLUTION;
                int yi = pw.y() * VOXEL_RESOLUTION;
                int zi = pw.z() * VOXEL_RESOLUTION;
                std::string key;
                key.append(std::to_string(xi));
                key.append(std::to_string(yi));
                key.append(std::to_string(zi));
                auto it = voxel_map_iter_.find(key);
                if( it == voxel_map_iter_.end()){
                    Voxel voxel;
                    voxel.position = pw;
                    voxel.color = Eigen::Vector3f(gray, gray, gray);
                    voxel.track_length = 1;
                    voxel.key = key;
                    voxel_list_.emplace_front(voxel);
                    voxel_map_iter_[key] = voxel_list_.begin();
                    ++total_voxels_;
                    ++added_voxels_;
                }else{
                    Voxel& voxel = *(it->second);
                    const Eigen::Vector3f pc = RT * voxel.position + t_inv;

                    int px = std::round(fx * pc[0] / pc[2] + cx);
                    int py = std::round(fy * pc[1] / pc[2] + cy);
                    float diff_x = px - x;
                    float diff_y = py - y;
                    float squared_reproj_error = diff_x * diff_x + diff_y * diff_y;
                    if(squared_reproj_error > MAX_SQUARED_REPROJ_ERROR)
                        continue;

                    float depth_error = (pc[2] - d) / d;
                    if(std::fabs(depth_error) > MAX_DEPTH_ERROR)
                        continue;

                    float n = voxel.track_length;
                    voxel.position = (voxel.position * n + pw) / (n + 1);
                    voxel.color = (voxel.color * n + Eigen::Vector3f(
                                       gray, gray, gray)) / (n + 1);
                    ++voxel.track_length;
                }
            }
        }
    }else{
        ///TODO
    }
}

uint32_t PointCloud::TrimVoxelList(){
    uint32_t num_rm = 0;
    LIterType prev = voxel_list_.before_begin();
    LIterType curr = voxel_list_.begin();
    LIterType end = (total_voxels_ == added_voxels_)?
                     voxel_list_.end() : trim_end_;

    while(curr != end){
        if(curr->track_length< MIN_TRACK_LENGTH){
            voxel_map_iter_.erase(curr->key);
            curr = voxel_list_.erase_after(prev);
            ++num_rm;
        }else{
            prev = curr;
            ++curr;
        }
    }

    trim_end_ = voxel_list_.begin();
    return num_rm;
}

std::ostream& operator<<(std::ostream& os, const PointCloud& pc){
    for(const Voxel& voxel : pc.voxel_list()){
        if(voxel.track_length < MIN_TRACK_LENGTH)
            continue;
        const Eigen::Vector3f& pos = voxel.position;
        const Eigen::Vector3f& col = voxel.color;
        os<<"v "<<pos.x()<<" "<<pos.y()<<" "<<pos.z()<<" "
         <<col.x()<<" "<<col.y()<<" "<<col.z()<<" 1.0"<<std::endl;
    }
    return os;
}

void DefaultHandler(int sig){
    void *array[10];
    size_t size;

    // get void*'s for all entries on the stack
    size = backtrace(array, 10);

    // print out all the frames to stderr
    fprintf(stderr, "Error: signal %d:\n", sig);
    backtrace_symbols_fd(array, size, STDERR_FILENO);
    exit(1);
}

void SetCrashHandler(HandlerType handler){
    signal(SIGSEGV, handler);
}

}

#include "depth_solver.h"
#include "sgm_cl.h"

#include <deque>
#include <vector>
#include <opencv2/core/core.hpp>

namespace ds{

static int MAX_BUFF_SIZE = 60;

int Frame::cv_depth_ = CV_8UC1;
Mode Frame::mode_ = MOTION_STEREO;

Frame::Frame(unsigned int _id, int _width, int _height,
             const Byte* _image0, const Float* _pose0,
             const Byte* _image1, const Float* _pose1):
    id_(_id), refcount_(new int{1}), image0_(nullptr),
    pose0_(nullptr), width_(_width), height_(_height),
    image1_(nullptr), pose1_(nullptr),depth_(nullptr){
    size_t num_pixels = width_ * height_;
    if(mode_ == MOTION_STEREO){
        if(_image0 == nullptr || _pose0 == nullptr){
            empty_ = true;
        }else{
            image0_ = new Byte[num_pixels];
            pose0_ = new Float[16];
            depth_ = new uint16_t[num_pixels];
            memcpy(image0_, _image0, num_pixels);
            memcpy(pose0_, _pose0, 16 * sizeof(Float));
            memset(depth_, 0, sizeof(uint16_t) * num_pixels);
            empty_ = false;
        }
    }else{
        if(_image0 == nullptr || _image1 == nullptr){
            empty_ = true;
        }else{
            image0_ = new Byte[num_pixels];
            image1_ = new Byte[num_pixels];
            depth_ = new uint16_t[num_pixels];
            memcpy(image0_, _image0, num_pixels);
            memcpy(image1_, _image1, num_pixels);
            memset(depth_, 0, sizeof(uint16_t) * num_pixels);
            empty_ = false;
        }
    }
}

Frame::Frame(const Frame& rhs): id_(rhs.id_),
    image0_(rhs.image0_), pose0_(rhs.pose0_),
    image1_(rhs.image1_), pose1_(rhs.pose1_),
    width_(rhs.width_), height_(rhs.height_),
    refcount_(rhs.refcount_),
    empty_(rhs.empty_), depth_(rhs.depth_){
    ++(*refcount_);
}

Frame& Frame::operator=(const Frame &rhs){
    if(this != &rhs){
        Destruct();
        id_ = rhs.id_;
        depth_ = rhs.depth_;
        width_ = rhs.width_;
        height_ = rhs.height_;
        empty_ = rhs.empty_;
        image0_ = rhs.image0_;
        pose0_ = rhs.pose0_;
        image1_ = rhs.image1_;
        pose1_ = rhs.pose1_;
        ++(*refcount_);
    }

    return *this;
}

Frame::~Frame() noexcept{
    Destruct();
}

void Frame::Destruct() noexcept{
    --(*refcount_);
    if(*refcount_ == 0){
        delete refcount_;
        if(!empty_){
            if(mode_ == MOTION_STEREO){
                delete[] image0_;
                delete[] pose0_;
            }else{
                delete[] image0_;
                delete[] image1_;
            }
            delete[] depth_;
        }
    }
}

CameraParams::CameraParams():fx(0), fy(0),
    cx(0), cy(0), extrinsics(nullptr){
}

CameraParams::CameraParams(Float _fx, Float _fy,Float _cx, Float _cy,
                           const Float* _extrin):fx(_fx), fy(_fy),
                           cx(_cx), cy(_cy), extrinsics(nullptr){
    if(_extrin != nullptr){
        extrinsics = new Float[16];
        memcpy(extrinsics, _extrin, 16 * sizeof(Float));
    }
}

CameraParams::CameraParams(const CameraParams& rhs):
                          fx(rhs.fx), fy(rhs.fy),
                          cx(rhs.cx), cy(rhs.cy),
                          extrinsics(nullptr){
    if(rhs.extrinsics != nullptr){
        extrinsics = new Float[16];
        memcpy(extrinsics, rhs.extrinsics, 16 *sizeof(Float));
    }
}

CameraParams& CameraParams::operator=(const CameraParams& rhs){
    if(this != &rhs){
        fx = rhs.fx;
        fy = rhs.fy;
        cx = rhs.cx;
        cy = rhs.cy;
        if(rhs.extrinsics != nullptr){
            if(extrinsics == nullptr)
                extrinsics = new Float[16];
            memcpy(extrinsics, rhs.extrinsics, 16 *sizeof(Float));
        }
    }

    return *this;
}

CameraParams::~CameraParams(){
    if(extrinsics != nullptr)
        delete[] extrinsics;
}

Configuration::Configuration(int _width, int _height, const CameraParams* _cam0,
                             const CameraParams* _cam1 ,Float _scale,
                             Method _method, Mode _mode ,bool _calibrated,
                             Float _min_depth, Float _max_depth) :
        width(_width),height(_height), cam0(nullptr), cam1(nullptr),
        scale(_scale), mode(_mode),calibrated(_calibrated),
        method(_method), min_depth(_min_depth), max_depth(_max_depth){

   cam0 = new CameraParams(*_cam0);
   if(_cam1 != nullptr)
       cam1 = new CameraParams(*_cam1);
}

class ImplBase{
public:
    ImplBase(const Configuration& config);
    virtual ~ImplBase();
    virtual void Run(cv::Mat& result) = 0;

    inline bool calibrated() const
    {return calibrated_;}
    inline const cv::Size& size() const
    {return size_;}
    inline std::deque<Frame>& buff()
    {return buff_;}

protected:
    bool calibrated_;
    Float scale_;
    cv::Size size_;
    std::deque<Frame> buff_;
    CameraParams cam0_, cam1_;
};

ImplBase::ImplBase(const Configuration& config):
    calibrated_(config.calibrated),
    scale_(config.scale){
    size_ = cv::Size(scale_ * config.width,
                     scale_ * config.height);
    cam0_ = *config.cam0;
    if(config.cam1 != nullptr)
        cam1_ = *config.cam1;
}

ImplBase::~ImplBase(){

}

class ImplSGM : public ImplBase {
public:
    ImplSGM(const Configuration& config):
        ImplBase(config){}
    virtual ~ImplSGM(){}

    virtual void Run(cv::Mat& result) override;
};

class ImplSGMCL : public ImplBase {
public:    
    ImplSGMCL(const Configuration& config);
    virtual ~ImplSGMCL();

    virtual void Run(cv::Mat& result) override;

private:
    sgm_cl::CLContext* ctx_;
    sgm_cl::StereoSGMCL* impl_;
    Eigen::Matrix3f intrinsics_;
};

ImplSGMCL::ImplSGMCL(const Configuration& config)
                    :ImplBase(config){
    int disp_range = 32;
    intrinsics_ = Eigen::Matrix3f::Identity();
    intrinsics_(0,0) = scale_ * cam0_.fx;
    intrinsics_(1,1) = scale_ * cam0_.fy;
    intrinsics_(0,2) = scale_ * cam0_.cx;
    intrinsics_(1,2) = scale_ * cam0_.cy;

    sgm_cl::SGMProperties prop(scale_ * config.width,
                               scale_ * config.height,
                               disp_range);
    ctx_ = new sgm_cl::CLContext;
    impl_ = new sgm_cl::StereoSGMCL(prop, ctx_);
}

ImplSGMCL::~ImplSGMCL(){
    delete impl_;
    delete ctx_;
}

class ImplSGBM : public ImplBase {
public:
    ImplSGBM(const Configuration& config):
        ImplBase(config){}
    virtual ~ImplSGBM() {}

    virtual void Run(cv::Mat& result) override;
};

DepthSolver::DepthSolver(const Configuration* config)
    : impl_(nullptr){
    if(config != nullptr)
        Initialize(config);
}

DepthSolver::~DepthSolver(){
    if(impl_ != nullptr)
        delete impl_;
}

bool DepthSolver::Initialize(const Configuration *config){
    if(config == nullptr)
        throw Exception("initialize with invalid configuration");
    if(impl_ != nullptr)
        throw Exception("initialized twice");

    if(config->scale > 1.0f || config->scale <= 0.0f)
        throw Exception("invalid scale");
    if(config->min_depth <= 0.0f || config->max_depth <= 0.0f ||
       config->max_depth <= config->min_depth)
        throw Exception("invalid depth range");
    if(config->width <= 0 || config->height <= 0)
        throw Exception("invalid image dimension");

    if(config->mode == Mode::SPATIAL_STEREO)
        MAX_BUFF_SIZE = 1;

    switch(config->method) {
        case Method::SGM_SIMD:
            impl_ = new ImplSGM(*config);break;
        case Method::SGM_OCL:
            impl_ = new ImplSGMCL(*config);break;
        case Method::SGBM_OCV:
            impl_ = new ImplSGBM(*config);break;
    }

    return true;
}

void DepthSolver::Execute(Frame &cur_frame){
    size_t num_pixels = impl_->size().height *
                        impl_->size().width;
    cv::Mat res_temp(impl_->size(), CV_16U,
                     cur_frame.depth());
    impl_->buff().emplace_back(cur_frame);
    if(impl_->buff().size() > MAX_BUFF_SIZE)
        impl_->buff().pop_front();
    impl_->Run(res_temp);    
}

//-----------implementations for depth estimation---------------
void ImplSGM::Run(cv::Mat &result){
    ///TODO
}

void ImplSGMCL::Run(cv::Mat &result){
    if(Frame::mode() == MOTION_STEREO){
        if(buff_.size() < 2)
            return;

        float min_baseline = 0.06f;
        auto cur_f = buff_.rbegin();
        if(cur_f->empty())
            return;

        Eigen::Matrix4f pose_cur = Eigen::Matrix4f(cur_f->pose0()).inverse();
        Eigen::Vector3f tc = pose_cur.block<3,1>(0,3);

        for(auto ref_f = cur_f + 1; ref_f != buff_.rend(); ref_f++){
            if(ref_f->empty())
                continue;
            Eigen::Matrix4f pose_ref = Eigen::Matrix4f(ref_f->pose0()).inverse();
            const Eigen::Vector3f& tr = pose_ref.block<3,1>(0, 3);
            float base_length = (tr - tc).norm();
            if(base_length > min_baseline){
                impl_->Run(cur_f->image0(), pose_cur,
                           ref_f->image0(), pose_ref,
                           intrinsics_, result.data);
                return;
            }
        }

    }else{
        ///TODO
    }
}

void ImplSGBM::Run(cv::Mat &result){
//TODO
}

}

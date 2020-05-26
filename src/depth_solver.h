#ifndef DEPTHSOLVER_H
#define DEPTHSOLVER_H

#include <string>
#include <exception>

namespace ds{

typedef float Float;
typedef unsigned char Byte;

enum Method{
    //semi-global matching accelerated with SIMD
    SGM_SIMD,
    //semi-global matching accelerated with OpenCL
    SGM_OCL,
    //semi-global matching with opencv's SGBM
    SGBM_OCV
};

enum Mode{
    //for monocular case
    MOTION_STEREO,
    //for binocular case
    SPATIAL_STEREO
};

class Frame{
public:
    inline static void set_cv_depth(int depth)
    {cv_depth_ = depth;}
    inline static int cv_depth()
    {return cv_depth_;}
    inline static void set_mode(Mode mode)
    {mode_ = mode;}
    inline static Mode mode()
    {return mode_;}

private:
    //depth for one pixel(default is CV_8UC1)
    static int cv_depth_;
    //input mode, default is monocular case
    static Mode mode_;

public:    
    Frame(unsigned int _id, int _width, int _height,
          const Byte* _image0, const Float* _pose0,
          const Byte* _image1 = nullptr,
          const Float* _pose1 = nullptr);
    ~Frame() noexcept;

    Frame(const Frame& rhs);
    Frame& operator=(const Frame& rhs);

    inline Byte* image0() {return image0_;}
    inline Byte* image1() {return image1_;}
    inline Float* pose0() {return pose0_;}
    inline Float* pose1() {return pose1_;}
    inline uint16_t* depth() {return depth_;}

    inline bool empty() const {return empty_;}
    inline unsigned int get_id() const{return id_;}
    inline int width() const {return width_;}
    inline int height() const {return height_;}
    inline const Byte* image0() const {return image0_;}
    inline const Byte* image1() const {return image1_;}
    inline const Float* pose0() const{return pose0_;}
    inline const Float* pose1() const{return pose1_;}
    inline const uint16_t* depth() const{return depth_;}

private:
    //check this Frame is empty or not
    bool empty_;
    int width_, height_;
    unsigned int id_;
    //left image data
    Byte* image0_;
    //left image pose of 4*4 matrix in column-major(T_wc)
    Float* pose0_;
    //right image data
    Byte* image1_;
    //right image pose of 4*4 matrix in column-major(T_wc)
    Float* pose1_;
    //depth map
    uint16_t* depth_;

private:
    void Destruct() noexcept;
    int* refcount_;
};

struct CameraParams{
    //camera intrinsics
    Float fx, fy, cx, cy;
    //camera extrinscs matrix of 4*4 in column-major(T_bs)
    Float* extrinsics;

    CameraParams();
    CameraParams(Float _fx, Float _fy,
                 Float _cx, Float _cy,
                 const Float* _extrin);
    CameraParams(const CameraParams& rhs);
    ~CameraParams();

    CameraParams& operator=(const CameraParams& rhs);
};

struct Configuration{
    Method method;
    Mode mode;
    //scale factor for input images
    Float scale;
    //lower bound of depth range(in meter)
    Float min_depth;
    //upper bound of depth range(in meter)
    Float max_depth;
    //width and height of input images
    int width, height;
    //true if stereo rig has been calibrated
    bool calibrated;
    //parameters for left camera
    CameraParams* cam0;
    //parameters for right camera
    CameraParams* cam1;

    Configuration(int _width, int _height, const CameraParams* _cam0,
                  const CameraParams* _cam1 = nullptr,Float _scale = 1.0f,
                  Method _method = Method::SGM_OCL,
                  Mode _mode = Mode::MOTION_STEREO,bool _calibrated = true,                  
                  Float _min_depth = 0.4f, Float _max_depth = 2.5f);
};

class Exception : public std::exception{
public:
    virtual const char* what() const noexcept
        override{return msg_.c_str();}
    Exception() : msg_(){}
    Exception(const char* msg) : msg_(msg){}
    virtual ~Exception() = default;

private:
    std::string msg_;
};

class ImplBase;

class DepthSolver {
public:
    DepthSolver(const Configuration* config = nullptr);
    ~DepthSolver();
    DepthSolver(const DepthSolver&) = delete;
    DepthSolver& operator=(const DepthSolver&) = delete;

    //initialize with configuration paramters
    bool Initialize(const Configuration* config);
    //output depth map as result, each pixel is of 16-bits width(in millimeter)
    void Execute(Frame& cur_frame);

private:
    ImplBase* impl_;

};

}

#endif // DEPTHSOLVER_H

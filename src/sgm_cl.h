#ifndef SGM_CL_H
#define SGM_CL_H

#include <CL/cl.h>
#include <Eigen/Dense>

#include <string.h>
#include <iostream>
#include <cstdio>
#include <vector>
#include <map>
#include <string>
#include <fstream>
#include <sstream>
#include <chrono>

namespace sgm_cl{

typedef unsigned short ushort;
typedef ushort feature_type;
typedef ushort cost_type;

struct Dim3{
    Dim3(int _x = 1, int _y = 1, int _z = 1):
         x(_x), y(_y), z(_z) {}
    inline void Reset()
    {x = 1; y = 1; z = 1;}

    int x, y, z;
};

struct ArgumentPropereties {
    ArgumentPropereties(void* ptr = nullptr, size_t argsize = 0) :
                        arg_ptr(ptr), sizeof_arg(argsize) {}
    void* arg_ptr;
    size_t sizeof_arg;
};

enum MemFlag {
    MEM_FLAG_READ_WRITE = 1 << 0,
    MEM_FLAG_WRITE_ONLY = 1 << 1,
    MEM_FLAG_READ_ONLY = 1 << 2,
    MEM_FLAG_USE_HOST_PTR = 1 << 3,
    MEM_FLAG_ALLOC_HOST_PTR = 1 << 4,
    MEM_FLAG_COPY_HOST_PTR = 1 << 5
};

enum SyncMode {
    SYNC_MODE_ASYNC = 0,
    SYNC_MODE_BLOCKING = 1
};

enum CostAggreProp {
    COST_AGGRE_4_PATHS,
    COST_AGGRE_8_PATHS
};

class CLContext {
public:
    CLContext(int platform_id = 0, int device_id = 0, int num_streams = 1);
    ~CLContext();
    CLContext(const CLContext& context) = delete;
    CLContext& operator=(const CLContext& context) = delete;
    CLContext(CLContext&& context) noexcept;
    CLContext& operator=(CLContext&& context) noexcept;

    cl_context GetCLContext() const {return cl_context_;}
    cl_device_id GetDevId() const {return cl_device_id_;}
    cl_command_queue GetCommandQueue(int id) const;
    void Finish(int command_queue) const;
    inline const std::string& CLInfo() const {return cl_info_;}

public:
    std::string platform_vendor_, device_name_;
    std::string ocl_dev_version_, ocl_c_version_;
    cl_ulong local_mem_size_, global_mem_size_;
    cl_uint max_compute_units_;
    cl_uint max_workitem_dimensions_;
    size_t* max_workitem_sizes_;
    size_t  max_workgroup_size_;

private:
    std::string GetPlatformInfo(cl_platform_id platform_id, int info_name) const;
    std::string GetDevInfo(cl_device_id dev_id, int info_name) const;

    std::vector<cl_command_queue> cl_command_queues_;
    cl_context cl_context_;
    cl_device_id cl_device_id_;

    std::string cl_info_;
};

class CLBuffer {
public:
    CLBuffer(const CLContext* ctx, size_t size, MemFlag flag = MEM_FLAG_READ_WRITE,
             void* host_ptr = nullptr);
    ~CLBuffer();
    template<typename T> inline
    void Clear(const T* pattern, size_t offset, size_t size
               ,int command_queue = 0);
    void Write(const void* data, SyncMode block_queue = SYNC_MODE_BLOCKING,
               int command_queue = 0);
    void Read(void* data, SyncMode block_queue = SYNC_MODE_BLOCKING,
              int command_queue = 0) const;
    ArgumentPropereties GetArgumentPropereties() const;

private:
    void Write(const void* data, size_t offset, size_t size,
               SyncMode block_queue,int command_queue);
    void Read(void* data, size_t offset, size_t size, SyncMode block_queue,
              int command_queue) const;
    mutable cl_mem buffer_;
    MemFlag flag_;
    size_t size_;
    const CLContext* context_;
};

class CLKernel {
public:
    CLKernel(const CLContext* context, cl_program program, const std::string& kernel_name);
    ~CLKernel();

   void Launch(int queue_id, Dim3 gd, Dim3 bd);

   template <typename... Types> void SetArgs(Types&&... args);

private:
   void SetArgs(int num_args, void** argument, size_t* argument_sizes);
   inline void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof);
   template <typename T, typename... Types>
   void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                      T&& arg, Types&&... Fargs);
   template <typename... Types>
   void FillArgVector(int arg_idx, void** arg_address, size_t* arg_sizeof,
                      CLBuffer* arg, Types&&... Fargs);

    cl_kernel kernel_;
    const CLContext* context_;
};

class CLProgram{
public:
    CLProgram(const std::string& source_path="", const CLContext* context=nullptr
                          , const std::string& compilation_options="-I \"./\"");
    CLProgram(const char* source, size_t num_bytes, const CLContext* context=nullptr,
              const std::string& compilation_options="-I \"./\"");
    ~CLProgram();
    bool CreateProgram(const char* source, size_t size_src,
                       const std::string& compilation_options);
    bool CreateKernels();
    CLKernel* GetKernel(const std::string& kernel_name);
    inline void SetCLContext(const CLContext& context) {context_ = &context;}

private:
    const CLContext* context_;
    cl_program cl_program_;    
    std::map<std::string, CLKernel*> kernels_;
};

class CLProfiler{
public:
    typedef std::chrono::steady_clock::time_point time_point;
    typedef std::chrono::duration<double, std::milli> duration;

public:
    CLProfiler();
    ~CLProfiler();
    void Tick();
    void Tock();
    inline const std::string& Info() const
    {return prof_info_;}

private:
    size_t cnt_;
    std::string prof_info_;
    time_point st_;
    time_point ed_;
    double min_elapse_;
    double max_elapse_;
    double total_elapse_;
};

struct SGMProperties{
    int width;
    int height;
    int disp_range;
    float min_depth;
    float max_depth;
    float uniqueness;

    int P1;
    int P2;
    int P2_MIN;
    CostAggreProp agg_prop;

    SGMProperties(int _width, int _height, int _disp_range, float _min_depth = 0.4f,
                  float _max_depth = 2.5f, int _P1 = 7, int _P2 = 100, int _P2_MIN=17,
                  float uniqueness = 0.98, CostAggreProp _agg_prop = COST_AGGRE_8_PATHS):
                  width(_width),height(_height), disp_range(_disp_range), min_depth(_min_depth),
                  max_depth(_max_depth), P1(_P1), P2(_P2), P2_MIN(_P2_MIN),agg_prop(_agg_prop) {}
};

class StereoSGMCL{
public:
    StereoSGMCL(const SGMProperties& properties, const CLContext* ctx = nullptr);
    bool Init(const CLContext* ctx);
    void Run(const void* curr_img, const Eigen::Matrix4f& curr_pose,
             const void* ref_img,  const Eigen::Matrix4f& ref_pose,
             const Eigen::Matrix3f& K, void* output);
    ~StereoSGMCL();

private:
    std::string CreateProgramOptions();
    void ConfigKernels();

    void ClearCostBuff();
    void MakeLookupBuffer();
    void CensusTransform();
    void MakeCostVolume(const float* proj_params);
    void AggregateCost();
    void WinnerTakeAll();
    void PostProcess(void* dest);

private:
    SGMProperties* properties_;
    CLProgram* sgm_prog_;
    const CLContext* context_;

    CLKernel* census_transform_kernel_;
    CLKernel* clear_cost_volume_kernel_;
    CLKernel* make_cost_volume_kernel_;    

    CLKernel* aggre_cost_horizontal_kernel_;
    CLKernel* aggre_cost_vertical_kernel_;
    CLKernel* aggre_cost_oblique0_kernel_;
    CLKernel* aggre_cost_oblique1_kernel_;

    CLKernel* winner_take_all_kernel_;

    CLBuffer *d_curr_img_, *d_curr_census_,
             *d_ref_img_, *d_ref_census_,
             *d_cost_volume_, *d_aggre_cost_,
             *d_lookup_depth_,*d_proj_params_,
             *d_depth_, *d_disparity_;

    Dim3 bd_clr_, bd_census_, bd_cost_, bd_aggre_, bd_wta_;
    Dim3 gd_clr_, gd_census_, gd_cost_, gd_aggre_obl_,
         gd_aggre_v_, gd_aggre_h_, gd_wta_;
};

#include "sgm_cl.inl"
}

#endif

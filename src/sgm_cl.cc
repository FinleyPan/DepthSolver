#include "sgm_cl.h"
#include "sgm_source_path.h"

#include <tmmintrin.h>

#define OCL_BUILD_LOG_SIZE (1<<20)

namespace  sgm_cl{
//for definitions of macros in OpenCL program
static constexpr char const* IMAGE_WIDTH_DEF = "IMG_W";
static constexpr char const* IMAGE_HEIGHT_DEF = "IMG_H";
static constexpr char const* DISPARITY_RANGE_DEF = "DISP_RANGE";

static constexpr char const* CLR_BLOCK_WIDTH_DEF = "CLR_BLOCK_WIDTH";
static constexpr char const* CLR_BLOCK_HEIGHT_DEF = "CLR_BLOCK_HEIGHT";
static constexpr char const* CLR_RAW_COST_DEF = "CLR_RAW_COST";
static constexpr char const* CLR_AGG_COST_DEF = "CLR_AGG_COST";
static constexpr char const* MAX_COST_DEF = "MAX_COST";
static constexpr char const* INVALID_COST_DEF = "INVALID_COST";

static constexpr char const* CENSUS_WINDOW_WIDTH_DEF = "CENSUS_WINDOW_WIDTH";
static constexpr char const* CENSUS_WINDOW_HEIGHT_DEF = "CENSUS_WINDOW_HEIGHT";
static constexpr char const* CENSUS_BLOCK_SIZE_DEF = "CENSUS_BLOCK_SIZE";
static constexpr char const* LINES_PER_BLOCK_DEF = "LINES_PER_BLOCK";

static constexpr char const* NUM_DISP_SUBGROUPS_DEF = "NUM_DISP_SUBGROUPS";

static constexpr char const* AGG_STRIDE_DEF = "AGG_STRIDE";
static constexpr char const* PENALTY1_DEF = "P1";
static constexpr char const* PENALTY2_DEF = "P2";

static constexpr char const* SCALED_THRESH_DEF = "SCALED_THRESH";
static constexpr char const* INVALID_DISP_DEF = "INVALID_DISP";

//for census
static constexpr int CENSUS_WINDOW_WIDTH  = 5;
static constexpr int CENSUS_WINDOW_HEIGHT = 5;
static constexpr int CENSUS_BLOCK_SIZE = 128;
static constexpr int LINES_PER_BLOCK = 16;
//for clearing cost volume
static constexpr int CLR_BLOCK_WIDTH = 16;
static constexpr int CLR_BLOCK_HEIGHT = 16;
static constexpr int CLR_RAW_COST = 257;
static constexpr int CLR_AGG_COST = 0;
static constexpr int CLR_DISP_STEP = 2;
static constexpr int MAX_COST = USHRT_MAX;
static constexpr int INVALID_COST = 16;
//for building cost volume
static constexpr int NUM_DISP_SUBGROUPS = 16;
static constexpr int CV_BLOCK_WIDTH  = 8;
static constexpr int CV_BLOCK_HEIGHT = 8;
//for cost aggregation
static constexpr int AGG_STRIDE = 8;
//for winner-take-all
static constexpr int WTA_BLOCK_WIDTH = 16;
static constexpr int WTA_BLOCK_HEIGHT = 16;
static constexpr float INVALID_DISPARITY = -10.0f;

/**
 * definitions for helper functions
 */

static void HandleError(cl_int err, const std::string& msg){
    if(err != CL_SUCCESS){
        printf("[OpenCL Error] in %s !: %d \n", msg.c_str(), err);        
        exit(EXIT_FAILURE);
    }
}

template<typename T>
static void AddDefinition(std::string& options, const std::string& def, T val){
    std::ostringstream oss;
    oss << val;
    options += std::string("-D ") + def + "=" + oss.str() + " ";
}

static void AddDefinition(std::string& options, const std::string& def){
    options += std::string("-D ") + def + " ";
}

inline void SortSwap(__m128& a, __m128& b){
    __m128 temp = a;
    a = _mm_min_ps(a, b);
    b = _mm_max_ps(temp, b);
}

static void MedianFilter(float* disp_src, float* disp_dst, int width, int height){
    float* dispImgTemp = disp_dst;
    float* line1 = disp_src;
    float* line2 = disp_src + width;
    float* line3 = disp_src + 2 * width;

    float* end = disp_src + width*height;

    disp_dst += width;
    __m128 lastMedian = _mm_setzero_ps();

    do {
        const __m128 line1_reg = _mm_load_ps(line1);
        const __m128 line1_reg_next = _mm_load_ps(line1 + 4);
        __m128 store0 = line1_reg;
        __m128 store1 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line1_reg_next), _mm_castps_si128(line1_reg), 4));
        __m128 store2 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line1_reg_next), _mm_castps_si128(line1_reg), 8));

        const __m128 line2_reg = _mm_load_ps(line2);
        const __m128 line2_reg_next = _mm_load_ps(line2 + 4);
        __m128 store3 = line2_reg;
        __m128 store4 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line2_reg_next), _mm_castps_si128(line2_reg), 4));
        __m128 store5 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line2_reg_next), _mm_castps_si128(line2_reg), 8));

        const __m128 line3_reg = _mm_load_ps(line3);
        const __m128 line3_reg_next = _mm_load_ps(line3 + 4);
        __m128 store6 = line3_reg;
        __m128 store7 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line3_reg_next), _mm_castps_si128(line3_reg), 4));
        __m128 store8 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(line3_reg_next), _mm_castps_si128(line3_reg), 8));

        // find median by sorting network
        SortSwap(store1, store2);
        SortSwap(store4, store5);
        SortSwap(store7, store8);
        SortSwap(store0, store1);
        SortSwap(store3, store4);
        SortSwap(store6, store7);
        SortSwap(store1, store2);
        SortSwap(store4, store5);
        SortSwap(store7, store8);
        SortSwap(store0, store3);
        SortSwap(store5, store8);
        SortSwap(store4, store7);
        SortSwap(store3, store6);
        SortSwap(store1, store4);
        SortSwap(store2, store5);
        SortSwap(store4, store7);
        SortSwap(store4, store2);
        SortSwap(store6, store4);
        SortSwap(store4, store2);

        const __m128i c = _mm_alignr_epi8(_mm_castps_si128(store4), _mm_castps_si128(lastMedian), 12);
        _mm_store_si128((__m128i*)disp_dst, c);
        lastMedian = store4;

        disp_dst += 4; line1 += 4; line2 += 4; line3 += 4;

    } while (line3 + 4 + 4 <= end);

    memcpy(dispImgTemp, disp_src, sizeof(float)*(width + 1));
    memcpy(dispImgTemp + width*height - width - 1 - 3, disp_src + width*height - width - 1 - 3, sizeof(float)*(width + 1 + 3));
}

static int GetCLMemFlag(MemFlag mem_flag)
{
    int ret = 0;
    switch(mem_flag){
       case MEM_FLAG_READ_WRITE:
           ret = ret | CL_MEM_READ_WRITE; break;
       case MEM_FLAG_READ_ONLY:
           ret = ret | CL_MEM_READ_ONLY; break;
       case MEM_FLAG_WRITE_ONLY:
           ret = ret | CL_MEM_WRITE_ONLY; break;
       case MEM_FLAG_USE_HOST_PTR:
           ret = ret | CL_MEM_USE_HOST_PTR; break;
       case MEM_FLAG_ALLOC_HOST_PTR:
           ret = ret | CL_MEM_ALLOC_HOST_PTR; break;
       case MEM_FLAG_COPY_HOST_PTR:
           ret = ret | CL_MEM_COPY_HOST_PTR; break;
    }
    return ret;
}

static void MakeProjMatrix(Eigen::Matrix<float,3,4>& P, const Eigen::Matrix3f& K,
                           const Eigen::Matrix4f& left_pose,
                           const Eigen::Matrix4f& right_pose){
    const Eigen::Matrix3f RT = left_pose.block<3,3>(0,0).transpose();
    const Eigen::Vector3f t  = left_pose.block<3,1>(0,3);
    const Eigen::Matrix3f KI = K.inverse();

    const Eigen::Matrix3f& R2 = right_pose.block<3,3>(0,0);
    const Eigen::Vector3f& t2 = right_pose.block<3,1>(0,3);
    Eigen::Matrix3f RR = R2 * RT;
    P.block<3,3>(0,0) = K * RR * KI;
    P.block<3,1>(0,3) = K * (t2 - RR * t);
}


/**
 * definitions for members of CLContext
 */
CLContext::CLContext(int platform_id, int device_id, int num_streams) {
    cl_platform_id p_id;
    cl_int err = 0;
    cl_uint num_platforms, num_divices;
    std::vector<cl_platform_id> p_ids;
    std::vector<cl_device_id> d_ids;

    clGetPlatformIDs(0, nullptr, &num_platforms);
    if(num_platforms > 0){
        p_ids.resize(num_platforms);
        clGetPlatformIDs(num_platforms,p_ids.data(),nullptr);
        if(platform_id < 0 || platform_id >= int(num_platforms)) {
            printf("Incorrect platform id %d!\n",platform_id);
            exit(EXIT_FAILURE);
        }
        p_id = p_ids[platform_id];
    }else {
        printf("Not found any platforms\n");
        exit(EXIT_FAILURE);
    }

    clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, 0, nullptr, &num_divices);
    if(num_divices > 0){
        d_ids.resize(num_divices);
        clGetDeviceIDs(p_id, CL_DEVICE_TYPE_ALL, num_divices,d_ids.data(),nullptr);
        if(device_id < 0 || device_id >= int(num_divices)){
            printf("Incorrect device id %d!\n",device_id);
            exit(EXIT_FAILURE);
        }
        cl_device_id_ = d_ids[device_id];
    }
    else{
        printf("Not found any devices\n");
        exit(EXIT_FAILURE);
    }

    cl_context_properties prop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)p_id,0};
    cl_context_ = clCreateContext(prop, 1, &cl_device_id_, nullptr, nullptr, &err);
    HandleError(err, "creating context");
    printf("OpenCL context created! \n");

    cl_command_queues_.resize(num_streams);
    for(int i=0; i < num_streams; i++){
        cl_command_queues_[i] = clCreateCommandQueue(cl_context_,
                                                     cl_device_id_,
                                                     0, &err);        
        HandleError(err, "creating ClCommandQueue");
    }

    std::ostringstream oss;
    max_workitem_sizes_ = nullptr;
    size_t num_bytes = 0;
    platform_vendor_ = GetPlatformInfo(p_id,CL_PLATFORM_VENDOR) + " "
                      +GetPlatformInfo(p_id,CL_PLATFORM_VERSION);
    device_name_     = GetDevInfo(cl_device_id_, CL_DEVICE_NAME);
    ocl_dev_version_ = GetDevInfo(cl_device_id_, CL_DEVICE_VERSION);
    ocl_c_version_   = GetDevInfo(cl_device_id_, CL_DEVICE_OPENCL_C_VERSION);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_GLOBAL_MEM_SIZE, 0, nullptr, &num_bytes);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_GLOBAL_MEM_SIZE, num_bytes,
                    &global_mem_size_, nullptr);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_LOCAL_MEM_SIZE, 0, nullptr, &num_bytes);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_LOCAL_MEM_SIZE, num_bytes,
                    &local_mem_size_, nullptr);
    clGetDeviceInfo(cl_device_id_,CL_DEVICE_MAX_COMPUTE_UNITS, 0, nullptr, &num_bytes);
    clGetDeviceInfo(cl_device_id_,CL_DEVICE_MAX_COMPUTE_UNITS, num_bytes,
                    &max_compute_units_, nullptr);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                    0, nullptr, &num_bytes);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS, num_bytes,
                    &max_workitem_dimensions_, nullptr);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_GROUP_SIZE, 0,
                    nullptr, &num_bytes);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_GROUP_SIZE, num_bytes,
                    &max_workgroup_size_, nullptr);
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_ITEM_SIZES, 0,
                    nullptr, &num_bytes);
    max_workitem_sizes_ = new size_t[num_bytes /sizeof(size_t)];
    clGetDeviceInfo(cl_device_id_, CL_DEVICE_MAX_WORK_ITEM_SIZES, num_bytes,
                    max_workitem_sizes_, nullptr);
    oss<<"Selected platform vendor: "<<platform_vendor_<<"\n"
       <<"Selected device name: "    <<device_name_    <<"\n"
       <<"Selected device OpenCL device version: "
       <<ocl_dev_version_<<"\n"
       <<"Selected device OpenCL C device version: "
       <<ocl_c_version_  <<"\n"
       <<"Device maximum Work Item Dimensions: "
       <<max_workitem_dimensions_<<"\n"
       <<"Device maximum Work Item Sizes: ";
    for(int i=0; i<num_bytes / sizeof(size_t); i++)
        oss<<max_workitem_sizes_[i]<<" ";
    oss<<"\n"
       <<"Device maximum number of compute units: "<<max_compute_units_<<"\n"
       <<"Device maximum Work Group Size: "<<max_workgroup_size_<<"\n"
       <<"Device Local Memory Size: " <<local_mem_size_/(1<<10)<<" (KB)\n"
       <<"Device Global Memory Size: "<< global_mem_size_/(1<<20)<<" (MB)\n";
    cl_info_ = oss.str();
}

CLContext::~CLContext(){
    delete []max_workitem_sizes_;
    for(auto& cq : cl_command_queues_)
        clReleaseCommandQueue(cq);
    clReleaseContext(cl_context_);
}

CLContext::CLContext(CLContext&& right) noexcept :cl_command_queues_(right.cl_command_queues_),
    cl_info_(right.cl_info_), cl_context_(right.cl_context_), cl_device_id_(right.cl_device_id_){
    right.cl_command_queues_.clear();
    right.cl_info_ = "";
    right.cl_context_ = nullptr;
    right.cl_device_id_ = nullptr;
}

CLContext& CLContext::operator=(CLContext&& right) noexcept{
    if(this != &right){
        cl_command_queues_ = right.cl_command_queues_;
        cl_info_ = right.cl_info_;
        cl_context_ = right.cl_context_;
        cl_device_id_ = right.cl_device_id_;

        right.cl_command_queues_.clear();
        right.cl_info_ = "";
        right.cl_context_ = nullptr;
        right.cl_device_id_ = nullptr;
    }
    return *this;
}

cl_command_queue CLContext::GetCommandQueue(int id) const{
    return cl_command_queues_.at(id);
}

void CLContext::Finish(int command_queue_id) const{
    cl_int err = clFinish(cl_command_queues_[command_queue_id]);
    HandleError(err, "finishing command queue");
}

std::string CLContext::GetPlatformInfo(cl_platform_id platform_id, int info_name) const{
    size_t info_size = 0;
    clGetPlatformInfo(platform_id, info_name, 0, nullptr, &info_size);
    std::string str;
    str.resize(info_size);
    clGetPlatformInfo(platform_id, info_name, info_size, &str[0], nullptr);
    return std::move(str);
}

std::string CLContext::GetDevInfo(cl_device_id dev_id, int info_name) const{
    size_t info_size = 0;
    clGetDeviceInfo(dev_id, info_name, 0, nullptr, &info_size);
    std::string str;
    str.resize(info_size);
    clGetDeviceInfo(dev_id, info_name, info_size, &str[0], nullptr);
    return std::move(str);
}

/**
 * definitions for members of CLProgram
 */
CLProgram::CLProgram(const std::string& source_path, const CLContext* context,
                     const std::string& compilation_options):context_(context){
    std::ifstream file_in;
    file_in.open(source_path);
    char* data_src = nullptr;
    if(file_in.is_open()){
        std::string str{std::istreambuf_iterator<char>(file_in),
                        std::istreambuf_iterator<char>()};
        size_t num_bytes_src = str.size();
        data_src = new char[num_bytes_src];
        memcpy(data_src, str.data(),num_bytes_src);
        CreateProgram(data_src, num_bytes_src, compilation_options);
        CreateKernels();
        delete []data_src;
    }else {
        printf("Cannot open %s!\n", source_path.c_str());
        exit(EXIT_FAILURE);
    }
    file_in.close();
}

CLProgram::CLProgram(const char* src, size_t num_bytes,const CLContext* context,
                     const std::string& compilation_options):context_(context){
    CreateProgram(src, num_bytes, compilation_options);
    CreateKernels();
}

CLProgram::~CLProgram(){
    for(auto& item : kernels_){
        delete item.second;
        item.second = nullptr;
    }
    if(cl_program_ != nullptr){
        cl_int err = clReleaseProgram(cl_program_);
        HandleError(err, "releasing program");
    }
}

bool CLProgram::CreateProgram(const char* source, size_t size_src,
                              const std::string& compilation_options){
    cl_int err;
    cl_program_ = clCreateProgramWithSource(context_->GetCLContext(),1,
                                            &source, &size_src,&err);
    HandleError(err, "creating program with source data");
    cl_device_id dev_id = context_->GetDevId();
    err = clBuildProgram(cl_program_,1,&dev_id,compilation_options.c_str(),
                         nullptr,nullptr);
    if(err != CL_SUCCESS){
        char* log = new char[OCL_BUILD_LOG_SIZE];
        clGetProgramBuildInfo(cl_program_,dev_id,CL_PROGRAM_BUILD_LOG,
                              OCL_BUILD_LOG_SIZE,log, nullptr);
        printf("OCL program build log: %s\n", log);
        delete[] log;
    }
    HandleError(err, "building program.");
    return true;
}

bool CLProgram::CreateKernels(){
    cl_uint num_kernels = 0;
    cl_int err;
    err = clCreateKernelsInProgram(cl_program_, 0, nullptr, &num_kernels);
    if(num_kernels == 0)
        err = CL_INVALID_BINARY;
    if(err != CL_SUCCESS){
        std::string build_log;
        size_t log_size = 0;
        clGetProgramBuildInfo(cl_program_, context_->GetDevId(),
                              CL_PROGRAM_BUILD_LOG,0,nullptr,&log_size);
        build_log.resize(log_size);
        clGetProgramBuildInfo(cl_program_, context_->GetDevId(),
                              CL_PROGRAM_BUILD_LOG,log_size,&build_log[0],nullptr);
        printf("%s \n",build_log.c_str());
        HandleError(err,"creating kernels");

    }
    return true;
}

CLKernel* CLProgram::GetKernel(const std::string &kernel_name){
    CLKernel* kernel = nullptr;
    auto iter = kernels_.find(kernel_name);
    if(iter != kernels_.end()){
        kernel = iter->second;
    }else{
        kernel = new CLKernel(context_, cl_program_, kernel_name);
        kernels_.insert(std::make_pair(kernel_name,kernel));
    }
    if(kernel == nullptr){
        printf("kernel has been deleted or failed to create!\n");
        exit(EXIT_FAILURE);
    }
    return kernel;
}

/**
 * definitions for members of CLKernel
 */
CLKernel::CLKernel(const CLContext* context, cl_program program,
                   const std::string& kernel_name):context_(context){
    cl_int err = CL_SUCCESS;
    kernel_ = clCreateKernel(program, kernel_name.c_str(),&err);
    HandleError(err, "creating kernel: " + kernel_name);
}

void CLKernel::SetArgs(int num_args, void **argument, size_t *argument_sizes){
    cl_int err = CL_SUCCESS;
    for(int i=0; i<num_args; i++){
        err = clSetKernelArg(kernel_, cl_uint(i), argument_sizes[i], argument[i]);
        HandleError(err, "in setting kernel arguments");
    }
}

void CLKernel::Launch(int queue_id, Dim3 gd, Dim3 bd){
    size_t global_w_offset[3] = {0, 0, 0};
    size_t global_w_size[3] = {
                      size_t(gd.x * bd.x),
                      size_t(gd.y * bd.y),
                      size_t(gd.z * bd.z)};
    size_t local_w_size[3]= {size_t(bd.x),size_t(bd.y),size_t(bd.z)};
    cl_int err = clEnqueueNDRangeKernel(context_->GetCommandQueue(queue_id),
                                    kernel_,3,global_w_offset,global_w_size,
                                          local_w_size,0, nullptr, nullptr);
    HandleError(err, "enqueuing kernel");
}

CLKernel::~CLKernel(){
    cl_int err = clReleaseKernel(kernel_);
    HandleError(err, "releasing kernel objects");
}

/**
 * definitions for members of CLBuffer
 */

CLBuffer::CLBuffer(const CLContext * ctx, size_t size, MemFlag flag,
                   void * host_ptr):context_(ctx),size_(size),flag_(flag){
    cl_int err;
    buffer_ = clCreateBuffer(context_->GetCLContext(),
                             GetCLMemFlag(flag),
                             size_, nullptr, &err);
    HandleError(err, "creating buffer");
    if(host_ptr)
        Write(host_ptr, SYNC_MODE_BLOCKING, 0);

}

CLBuffer::~CLBuffer(){
    cl_int err = clReleaseMemObject(buffer_) ;
    HandleError(err, "in releasing buffer");
}

ArgumentPropereties CLBuffer::GetArgumentPropereties() const{
    return ArgumentPropereties(&buffer_, sizeof(buffer_));
}

void CLBuffer::Write(const void * data, SyncMode block_queue,int command_queue){
    Write(data, 0, size_, block_queue, command_queue);
}

void CLBuffer::Write(const void * data, size_t offset, size_t size,
                     SyncMode block_queue,int command_queue){
    cl_bool b_Block = block_queue == SYNC_MODE_BLOCKING ? CL_TRUE : CL_FALSE;
    cl_int err = clEnqueueWriteBuffer(context_->GetCommandQueue(command_queue),
                                       buffer_, b_Block, offset, size, data, 0,
                                                             nullptr, nullptr);
    HandleError(err, "enqueuing writing buffer");
}

void CLBuffer::Read(void *data, SyncMode block_queue, int command_queue) const{
    Read(data, 0, size_, block_queue, command_queue);
}

void CLBuffer::Read(void *data, size_t offset, size_t size, SyncMode block_queue,
                    int command_queue) const {
    cl_bool b_Block = block_queue == SYNC_MODE_BLOCKING ? CL_TRUE : CL_FALSE;
    cl_int err = clEnqueueReadBuffer(context_->GetCommandQueue(command_queue),
                                      buffer_, b_Block, offset, size, data, 0,
                                                            nullptr, nullptr);
    HandleError(err, "enqueuing reading buffer");
}

/**
 * definitions for members of CLprofiler
 */
CLProfiler::CLProfiler(): cnt_(0), total_elapse_(0.0f),
               min_elapse_(1000.0f), max_elapse_(0.0f)
{}

CLProfiler::~CLProfiler() {}

void CLProfiler::Tick(){
    st_ = std::chrono::steady_clock::now();
}

void CLProfiler::Tock(){
    ed_ = std::chrono::steady_clock::now();
    double elapse = duration(ed_ - st_).count();
    cnt_ += 1;
    total_elapse_ += elapse;
    if(elapse > max_elapse_)
        max_elapse_ = elapse;
    if(elapse < min_elapse_)
        min_elapse_ = elapse;

    if(cnt_ > 2){
        double aver_elapse = (total_elapse_ - max_elapse_ - min_elapse_)/(cnt_ - 2);
        prof_info_ = "loop number: "   + std::to_string(cnt_)
                  + ", min elapse: "   + std::to_string(min_elapse_)
                  + ", max elapse: "   + std::to_string(max_elapse_)
                + ", average elapse: " + std::to_string(aver_elapse);
    }
    else
        prof_info_ = "loop number: "   + std::to_string(cnt_);
}

/**
 * definitions for members of StereoSGMCL
 */
StereoSGMCL::StereoSGMCL(const SGMProperties& properties, const CLContext* ctx):
                         properties_(new SGMProperties(properties)), context_(nullptr){
    Init(ctx);
}

StereoSGMCL::~StereoSGMCL(){
    delete sgm_prog_;

    delete d_curr_img_;
    delete d_curr_census_;   
    delete d_ref_img_;
    delete d_ref_census_;
    delete d_cost_volume_;
    delete d_aggre_cost_;
    delete d_lookup_depth_;
    delete d_proj_params_;
    delete d_depth_;

    delete properties_;
}

bool StereoSGMCL::Init(const CLContext *ctx) {
    if(!ctx)
        return false;
    context_    = ctx;
    int width   = properties_->width;
    int height  = properties_->height;
    size_t area = width * height;

    //create buffers
    //construct lookup table for depth samples
    MakeLookupBuffer();
    d_curr_img_     = new CLBuffer(context_, area);
    d_curr_census_  = new CLBuffer(context_, sizeof(feature_type) * area);    
    d_ref_img_      = new CLBuffer(context_, area);
    d_ref_census_   = new CLBuffer(context_, sizeof(feature_type) * area);
    d_proj_params_  = new CLBuffer(context_, sizeof(float) * 24);
    d_cost_volume_  = new CLBuffer(context_, sizeof(cost_type) * area * properties_->disp_range);
    d_aggre_cost_   = new CLBuffer(context_, sizeof(cost_type) * area * properties_->disp_range);
    d_depth_        = new CLBuffer(context_, sizeof(ushort) * area);
    d_disparity_    = new CLBuffer(context_, sizeof(float) * area);

    //initialize kernels
    sgm_prog_ = new CLProgram(SGM_SRC_PATH, context_, CreateProgramOptions());
    census_transform_kernel_ =  sgm_prog_->GetKernel("census_transform_kernel");
    make_cost_volume_kernel_ =  sgm_prog_->GetKernel("make_cost_volume_kernel");
    clear_cost_volume_kernel_ = sgm_prog_->GetKernel("clear_cost_volume_kernel");   

    aggre_cost_horizontal_kernel_ = sgm_prog_->GetKernel("aggre_cost_horizontal_kernel");
    aggre_cost_vertical_kernel_   = sgm_prog_->GetKernel("aggre_cost_vertical_kernel");
    if(properties_-> agg_prop == COST_AGGRE_8_PATHS){
        aggre_cost_oblique0_kernel_ = sgm_prog_->GetKernel("aggre_cost_oblique0_kernel");
        aggre_cost_oblique1_kernel_ = sgm_prog_->GetKernel("aggre_cost_oblique1_kernel");
    }

    winner_take_all_kernel_ = sgm_prog_->GetKernel("winner_take_all_kernel1");

    ConfigKernels();
    return true;
}

void StereoSGMCL::ConfigKernels(){
    //for clearing cost volume
    clear_cost_volume_kernel_->SetArgs(d_cost_volume_, d_aggre_cost_);
    bd_clr_.x = CLR_BLOCK_WIDTH * CLR_DISP_STEP;
    bd_clr_.y = CLR_BLOCK_HEIGHT;
    gd_clr_.x = (properties_->width  + CLR_BLOCK_WIDTH  - 1) / CLR_BLOCK_WIDTH;
    gd_clr_.y = (properties_->height + CLR_BLOCK_HEIGHT - 1) / CLR_BLOCK_HEIGHT;

    //for census transform
    int width_per_block  = CENSUS_BLOCK_SIZE - CENSUS_WINDOW_WIDTH + 1;
    int height_per_block = LINES_PER_BLOCK;
    bd_census_ = Dim3(CENSUS_BLOCK_SIZE);
    gd_census_ = Dim3((properties_->width  + width_per_block  - 1) /  width_per_block,
                      (properties_->height + height_per_block - 1) / height_per_block);

    //for making cost volume
    width_per_block  = CV_BLOCK_WIDTH;
    height_per_block = CV_BLOCK_HEIGHT;
    bd_cost_ = Dim3(CV_BLOCK_WIDTH * NUM_DISP_SUBGROUPS, height_per_block);
    gd_cost_ = Dim3((properties_->width  + width_per_block  - 1) / width_per_block,
                    (properties_->height + height_per_block - 1) / height_per_block);
    make_cost_volume_kernel_->SetArgs(d_cost_volume_, d_curr_census_, d_ref_census_,
                                     d_lookup_depth_,d_proj_params_);

    //for cost aggregation    
    bd_aggre_ = Dim3(properties_->disp_range / 4, AGG_STRIDE);
    aggre_cost_horizontal_kernel_->SetArgs(d_aggre_cost_, d_cost_volume_);
    aggre_cost_vertical_kernel_  ->SetArgs(d_aggre_cost_, d_cost_volume_);
    gd_aggre_h_ = Dim3((properties_->height + AGG_STRIDE - 1)/ AGG_STRIDE);
    gd_aggre_v_ = Dim3((properties_->width  + AGG_STRIDE - 1)/ AGG_STRIDE);
    if(properties_->agg_prop == COST_AGGRE_8_PATHS){
        const int num_obl_paths = properties_->width + properties_->height - 1;
        aggre_cost_oblique0_kernel_->SetArgs(d_aggre_cost_, d_cost_volume_);
        aggre_cost_oblique1_kernel_->SetArgs(d_aggre_cost_, d_cost_volume_);
        gd_aggre_obl_ = Dim3((num_obl_paths + AGG_STRIDE - 1) / AGG_STRIDE);
    }

    //for WTA
    bd_wta_ = Dim3(WTA_BLOCK_WIDTH, WTA_BLOCK_HEIGHT);
    gd_wta_ = Dim3((properties_-> width + bd_wta_.x - 1) / bd_wta_.x,
                   (properties_->height + bd_wta_.y - 1) / bd_wta_.y);
    winner_take_all_kernel_->SetArgs(d_disparity_, d_aggre_cost_);
}

std::string StereoSGMCL::CreateProgramOptions(){
    std::string options;

    AddDefinition(options, IMAGE_WIDTH_DEF, properties_->width);
    AddDefinition(options, IMAGE_HEIGHT_DEF, properties_->height);
    AddDefinition(options, DISPARITY_RANGE_DEF, properties_->disp_range);

    AddDefinition(options, CLR_BLOCK_WIDTH_DEF, CLR_BLOCK_WIDTH);
    AddDefinition(options, CLR_BLOCK_HEIGHT_DEF, CLR_BLOCK_HEIGHT);
    AddDefinition(options, CLR_RAW_COST_DEF, CLR_RAW_COST);
    AddDefinition(options, CLR_AGG_COST_DEF, CLR_AGG_COST);
    AddDefinition(options, MAX_COST_DEF, MAX_COST);
    AddDefinition(options, INVALID_COST_DEF, INVALID_COST);

    AddDefinition(options, CENSUS_WINDOW_WIDTH_DEF, CENSUS_WINDOW_WIDTH);
    AddDefinition(options, CENSUS_WINDOW_HEIGHT_DEF, CENSUS_WINDOW_HEIGHT);
    AddDefinition(options, CENSUS_BLOCK_SIZE_DEF, CENSUS_BLOCK_SIZE);
    AddDefinition(options, LINES_PER_BLOCK_DEF, LINES_PER_BLOCK);

    AddDefinition(options, NUM_DISP_SUBGROUPS_DEF, NUM_DISP_SUBGROUPS);

    AddDefinition(options, AGG_STRIDE_DEF, AGG_STRIDE);
    AddDefinition(options, PENALTY1_DEF, properties_->P1);
    AddDefinition(options, PENALTY2_DEF, properties_->P2);    

    AddDefinition(options, SCALED_THRESH_DEF, properties_->uniqueness);
    AddDefinition(options, INVALID_DISP_DEF, INVALID_DISPARITY);
    return std::move(options);
}

void StereoSGMCL::MakeLookupBuffer(){
    int disp_range = properties_->disp_range;
    float* h_lookup = new float[disp_range];
    float max_disp = 1.0f / properties_->min_depth;
    float min_disp = 1.0f / properties_->max_depth;
    for(int level = 0; level < disp_range; level++){
        h_lookup[level] = 1.0f / ((max_disp - min_disp)*level
                               / (disp_range - 1) + min_disp);
    }
    d_lookup_depth_ = new CLBuffer(context_, sizeof(float) * disp_range,
                                   MEM_FLAG_READ_WRITE, h_lookup);
    delete []h_lookup;
}

void StereoSGMCL::PostProcess(void* dest){
    size_t area = properties_->width * properties_->height;
    float b=  1.0f / properties_->max_depth;
    float a= (1.0f / properties_->min_depth - b)/
             (properties_->disp_range - 1);
    float* disp_raw = (float*)_mm_malloc(sizeof(float) * area, 16);
    float* disp_refined = (float*)_mm_malloc(sizeof(float) * area, 16);
    memset(disp_raw, 0, sizeof(float) * area);
    memset(disp_refined, 0, sizeof(float) * area);
    ushort* depth = static_cast<ushort*>(dest);

    d_disparity_->Read(disp_raw);
    MedianFilter(disp_raw, disp_refined,properties_->width,properties_->height);
    for(int i=0; i<properties_->height; i++){
        size_t off_v = i * properties_->width;
        for(int j=0; j<properties_->width; j++){
            float d = disp_refined[off_v + j];
            if(d < 0.0f){
                depth[off_v + j] = (ushort)0;
                continue;
            }
            depth[off_v + j] = (ushort)(1000.0f/(a * d + b));
        }
    }

    _mm_free(disp_raw);
    _mm_free(disp_refined);
}

void StereoSGMCL::Run(const void* curr_img, const Eigen::Matrix4f& curr_pose,
                      const void* ref_img,  const Eigen::Matrix4f& ref_pose,
                      const Eigen::Matrix3f& K, void* output){    
    d_curr_img_->Write(curr_img);    
    d_ref_img_->Write(ref_img);

    /*clear cost buffer*/    
    ClearCostBuff();    

    /*calculate census transform*/    
    CensusTransform();    

    /*make cost volume*/    
    Eigen::Matrix<float,3,4> P = Eigen::Matrix<float,3,4>::Zero();
    MakeProjMatrix(P, K, curr_pose, ref_pose);
    MakeCostVolume(P.data());    

    /*aggregate costs*/    
    AggregateCost();    

    /*winner take all*/    
    WinnerTakeAll();    

    /*postprocessing and get depth*/    
    PostProcess(output);    
}

void StereoSGMCL::ClearCostBuff(){        
    clear_cost_volume_kernel_->Launch(0, gd_clr_, bd_clr_);
    context_->Finish(0);
}

void StereoSGMCL::CensusTransform(){
    census_transform_kernel_->SetArgs(d_curr_census_, d_curr_img_);
    census_transform_kernel_->Launch(0, gd_census_, bd_census_);
    context_->Finish(0);

    census_transform_kernel_->SetArgs(d_ref_census_, d_ref_img_);
    census_transform_kernel_->Launch(0, gd_census_, bd_census_);
    context_->Finish(0);
}

void StereoSGMCL::MakeCostVolume(const float* proj_params){    
    d_proj_params_->Write(proj_params);
    make_cost_volume_kernel_->Launch(0, gd_cost_, bd_cost_);
    context_->Finish(0);
}

void StereoSGMCL::AggregateCost(){  
  aggre_cost_horizontal_kernel_->Launch(0, gd_aggre_h_, bd_aggre_);
  context_->Finish(0);  
  aggre_cost_vertical_kernel_  ->Launch(0, gd_aggre_v_, bd_aggre_);
  context_->Finish(0);

   if(properties_->agg_prop == COST_AGGRE_8_PATHS){
        aggre_cost_oblique0_kernel_->Launch(0, gd_aggre_obl_,bd_aggre_);
        context_->Finish(0);
        aggre_cost_oblique1_kernel_->Launch(0, gd_aggre_obl_,bd_aggre_);
        context_->Finish(0);
   }
}

void StereoSGMCL::WinnerTakeAll(){
    winner_take_all_kernel_->Launch(0, gd_wta_, bd_wta_);
    context_->Finish(0);
}

}

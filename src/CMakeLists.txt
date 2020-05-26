configure_file(${CMAKE_CURRENT_SOURCE_DIR}/sgm_source_path.h.in
               ${CMAKE_CURRENT_SOURCE_DIR}/sgm_source_path.h)

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIR})

add_library(depth SHARED depth_solver.cpp sgm_cl.cc)
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)

target_link_libraries(depth opencv_core ${OpenCL_LIBRARIES})

include_directories("${PROJECT_SOURCE_DIR}/src")
link_directories("${PROJECT_BINARY_DIR}/lib")

add_executable(pc_maker pc_maker.cpp utils.cpp)
target_link_libraries(pc_maker depth opencv_highgui opencv_imgcodecs)
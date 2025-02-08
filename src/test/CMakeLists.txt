add_executable(Test_CNN main.cpp)
target_include_directories(Test_CNN PRIVATE convolution)
target_link_libraries(Test_CNN PRIVATE CNN_Lib)

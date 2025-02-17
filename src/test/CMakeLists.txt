add_executable(Test_CNN testlib.cpp)
target_include_directories(Test_CNN PRIVATE convolution)
target_link_libraries(Test_CNN PRIVATE CNN_Lib)

add_executable(Test_CNN2 testlib2.cpp)
target_include_directories(Test_CNN2 PRIVATE convolution)
target_link_libraries(Test_CNN2 PRIVATE CNN_Lib2)

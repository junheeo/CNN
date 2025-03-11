add_executable(Test_CNN testlib.cpp)
target_include_directories(Test_CNN PRIVATE convolution)
target_link_libraries(Test_CNN PRIVATE CNN_Lib)

add_executable(Test_CNN2 testlib2.cpp)
target_include_directories(Test_CNN2 PRIVATE convolution)
target_link_libraries(Test_CNN2 PRIVATE CNN_Lib2)

add_executable(Test_Cereal testcereal.cpp)
target_include_directories(Test_Cereal 
    PUBLIC external/cereal/include/cereal/archives
    PUBLIC external/cereal/include/cereal/types)
# cereal is a header-only library so no need for target_link_libraries()

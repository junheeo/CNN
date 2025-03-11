add_library(CNN_Lib INTERFACE)
target_include_directories(CNN_Lib INTERFACE .)

add_library(CNN_Lib2 INTERFACE)
target_include_directories(CNN_Lib2 
    INTERFACE .
    INTERFACE external/cereal/include/cereal/archives
    INTERFACE external/cereal/include/cereal/types)

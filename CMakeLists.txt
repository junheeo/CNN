cmake_minimum_required(VERSION 3.29.4)
project(CNN CXX)
set(CMAKE_CXX_STANDARD 20)
add_subdirectory(src)

set(CMAKE_EXPORT_COMPILE_COMMANDS 1)

#export CXX=/usr/local/opt/llvm/bin/clang++
#cmake -S . -B bin
#cmake --build bin

#./bin/src/test/Test_CNN

#git add .
#git commit -m "commit_name"
#git status
#git remote add origin https://github.com/junheeo/CNN.git
#git remote add origin git@github.com:junheeo/CNN.git
#git push -u origin main

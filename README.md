This is the project's top level directory for implementing, training, and testing a convolutional neural network written in pure C++.


This is the history of training this network:  
avgTrainLossPerEpoch = 2.42706,2.30815,2.30339,2.30265,2.30251,2.30249,2.30238,2.30254,2.30256,2.30252,2.30246,2.30247,2.30231,2.30228,2.30181,2.30112,2.29743,2.26442,2.16874,2.0719,  
avgTestLossPerEpoch = 2.31474,2.30464,2.30313,2.30279,2.30268,2.30287,2.30281,2.30275,2.30266,2.30278,2.3027,2.30268,2.30184,2.30158,2.30115,2.30153,2.29564,2.4053,2.10234,2.01921,  
avgTestAccuracyPerEpoch = 0.0985,0.1014,0.1005,0.1012,0.1011,0.1002,0.1025,0.102,0.1026,0.1024,0.103,0.1031,0.1084,0.1104,0.1115,0.1141,0.1301,0.1511,0.1939,0.2335,  

Currently 20 epochs have been trained. Until epoch 11, the learning rate has been 0.001. However, because the network suggested to be stuck in a saddle point or local minimum with high loss value, starting from epoch 12 the learning rate is 0.005.


Compiling this project in C++20 has not raised any errors.


The project has dependency of git submodule called 'cereal' stored in the cnn/external/ directory. 
Cereal is used so that the parameters of the convolutional neural network would be stored in a portable binary format. 
When downloaded on MacOS, Cereal had problems with the c++ 'include' statements in their files not being able to find other files. 
This is solved in the author's case by manually editing the directories reference of the 'include' with relative paths whenever compiling cnn/src/test/testcereal.cpp raised an error. 
Take any necessary measures to resolve this issue in your usecase.


use CMake to build the project:
```
export CXX=/your/c++/compiler/location
```
```
cmake -DSKIP_PERFORMANCE_COMPARISON=ON -DBUILD_SANDBOX=OFF -S . -B bin
```
```
cmake --build bin
```

Suppose that the issue with cereal submodule is resolved.
Then to train and test the model, change directory to cnn/bin/src/model/ and then run the Train_CNN executable file.
In a UNIX-based system, running the run_Train_CNN.sh shellscript should suffice.

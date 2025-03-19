Inside train.cpp, traintest() function performs minibatch training and inference on the entire train and test dataset.
Inside testall.cpp, testall() function performs inference on the entire test dataset.
Currently train.cpp is configured to only run the traintest() function.

data_batch_*.bin files are the train CIFAR-10 dataset.
test_batch.bin file is the test CIFAR-10 dataset.
Both dataset are shuffled according to the source.

model_*.bin files contain the parameter values of the convolutional neural network.
Training will NOT update the model_*.bin files here, but rather the ones in the cnn/bin/src/model/ directory once the project is built.

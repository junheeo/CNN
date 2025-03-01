#include "convolution2.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>

struct filebuffer{
    unsigned char * arr;
    size_t length;
};

filebuffer readFileData(const std::string& name) {
    std::filesystem::path inputFilePath{name};
    size_t length = std::filesystem::file_size(inputFilePath);
    if (length == 0) {
        std::cerr<<"File "<<name<<" is empty"<<std::endl;
        throw 1;
    }
    unsigned char * pbuffer = new unsigned char [length];
    std::ifstream inputFile(name, std::ios_base::binary);
    inputFile.read((char*)pbuffer, length);
    inputFile.close();
    filebuffer buffer = {pbuffer, length};
    return buffer;
}

struct imagedata_t {
    double * arr;
    size_t length;
    size_t inx;
};

void copyFromInx(imagedata_t & imageData, filebuffer & buffer){
    size_t imgCurrInx = imageData.inx;
    for(int i=0;i<buffer.length;++i){
        double imageVal = (double) ((unsigned int) buffer.arr[i]); 
        imageData.arr[imgCurrInx] = imageVal;
        ++imgCurrInx;
    }
    imageData.inx = imgCurrInx;
}

/*
data_batch_1.bin
data_batch_2.bin
data_batch_3.bin
data_batch_4.bin
data_batch_5.bin
test_batch.bin
*/
void setTrainData(imagedata_t & trainData){
    /* 50000 images
     * 1 value for class
     * 3x32x32 size of image 
     * 50000x3073 = 153650000 */
    trainData.arr = new double [153650000];
    trainData.length = 153650000;
    trainData.inx = 0;
    std::array<std::string, 5> fileNames = 
        {"data_batch_1.bin",
        "data_batch_2.bin",
        "data_batch_3.bin",
        "data_batch_4.bin",
        "data_batch_5.bin"};
    for(auto & fileName : fileNames){
        filebuffer buffer;
        buffer = readFileData(fileName);
        copyFromInx(trainData, buffer);
        delete [] buffer.arr;
        /*std::cout<<"    setTrainData: trainData.inx = "<<trainData.inx<<std::endl;*/
    }
    trainData.inx=0;
}
void setTestData(imagedata_t & testData){
    /* 10000 images
     * 1value for class
     * 3*32*32 size of image
     * 10000*3073 = 30730000 */
    testData.arr = new double [30730000];
    testData.length = 30730000;
    testData.inx=0;
    std::string fileName = "test_batch.bin";
    filebuffer buffer;
    buffer = readFileData(fileName);
    copyFromInx(testData, buffer);
    delete [] buffer.arr;
    /*std::cout<<"    setTestData: testData.inx = "<<testData.inx<<std::endl;*/
    testData.inx=0;
}

void setImageToTensorAndVector(imagedata_t & imageData, tensor3d * & ptensor3d, vector1d * pvector1d, int pInx){
    dim3_t tensorInx;
    size_t imageDataInx = imageData.inx;
    if(imageDataInx % 3073 != 0){
        std::cerr<<"setImageToTensorAndVector(): imageDataInx % 3073 = "<<imageDataInx % 3073<<" should equal zero"<<std::endl;
    }
    /* set pvector1d*/
    pvector1d[pInx].setZero(pvector1d[pInx].size);
    int classInx = imageData.arr[imageDataInx]; /* typecase double -> int */
    /*std::cout<<"    setImageToTensorAndVector: classInx = "<<classInx<<std::endl;*/
    pvector1d[pInx].setVal(classInx, 1);
    ++imageDataInx;
    /* set ptensor3d */
    for(int zinx=0;zinx<3;++zinx){
        tensorInx.d = zinx;
    for(int xinx=0;xinx<32;++xinx){
        tensorInx.w = xinx;
    for(int yinx=0;yinx<32;++yinx){
        tensorInx.h = yinx;
        double imageVal = imageData.arr[imageDataInx] / 255;
        ptensor3d[pInx].setVal(tensorInx, imageVal);
        
        ++imageDataInx;
    }
    }
    }
    imageData.inx = imageDataInx;
}

void testSetTrainTestData(imagedata_t & trainData, imagedata_t & testData);

int main(){
    imagedata_t trainData;
    imagedata_t testData;
    setTrainData(trainData);
    setTestData(testData);

    /*testSetTrainTestData(trainData, testData);*/

    const int numTrainImage = 50000;
    const int numTestImage = 10000;
    const int epochs = 500;
    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(3,32,32,batchsize);
    tensor3d * Yl_1 = newZeroTensor3dArr(3,34,34,batchsize);
    tensor3d * Yl_2 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_3 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_4 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_5 = newZeroTensor3dArr(32,34,34,batchsize);
    tensor3d * Yl_6 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_7 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_8 = newZeroTensor3dArr(32,32,32,batchsize);
    tensor3d * Yl_9 = newZeroTensor3dArr(32,16,16,batchsize);
    tensor3d * Yl_10 = newZeroTensor3dArr(32,18,18,batchsize);
    tensor3d * Yl_11 = newZeroTensor3dArr(32,16,16,batchsize);
    tensor3d * Yl_12 = newZeroTensor3dArr(32,16,16,batchsize);
    tensor3d * Yl_13 = newZeroTensor3dArr(32,16,16,batchsize);
    tensor3d * Yl_14 = newZeroTensor3dArr(32,8,8,batchsize);
    tensor3d * Yl_15 = newZeroTensor3dArr(4,8,8,batchsize);
    tensor3d * Yl_16 = newZeroTensor3dArr(4,8,8,batchsize);
    tensor3d * Yl_17 = newZeroTensor3dArr(4,8,8,batchsize);
    vector1d * Yffl_17 = newVector1dArrFromTensor3dArr(Yl_17,batchsize);
    vector1d * Yffl_18 = newZeroVector1dArr(10,batchsize);
    vector1d * Yffl_19 = newZeroVector1dArr(10,batchsize);

    vector1d * Y_truth = newZeroVector1dArr(10,batchsize);

    tensorZeroPad padl_1 {Yl_0,Yl_1,batchsize};
    conv2d convl_2 {3,32,3,3,Yl_1,Yl_2,1,false,batchsize};
    tensorBatchNorm batchnorml_3 {Yl_2,Yl_3,batchsize};
    tensorRelu relul_4 {32,32,32,Yl_3,Yl_4,batchsize};
    tensorZeroPad padl_5 {Yl_4,Yl_5,batchsize};
    conv2d convl_6 {32,32,3,3,Yl_5,Yl_6,1,false,batchsize};
    tensorBatchNorm batchnorml_7 {Yl_6,Yl_7,batchsize};
    tensorRelu relul_8 {32,32,32,Yl_7,Yl_8,batchsize};
    tensorMaxPool pooll_9 {Yl_8,Yl_9,batchsize};
    tensorZeroPad padl_10 {Yl_9,Yl_10,batchsize};
    conv2d convl_11 {32,32,3,3,Yl_10,Yl_11,1,false,batchsize};
    tensorBatchNorm batchnorml_12 {Yl_11,Yl_12,batchsize};
    tensorRelu relul_13 {32,16,16,Yl_12,Yl_13,batchsize};
    tensorMaxPool pooll_14 {Yl_13,Yl_14,batchsize};
    conv2d convl_15 {32,4,1,1,Yl_14,Yl_15,1,false,batchsize};
    tensorBatchNorm batchnorml_16 {Yl_15,Yl_16,batchsize};
    tensorRelu relul_17 {4,8,8,Yl_16,Yl_17,batchsize};
    v1dAffineTransform affineffl_18 {Yffl_17,Yffl_18,batchsize};
    v1dsoftmax softmaxffl_19 {Yffl_18,Yffl_19,batchsize};
    v1dCrossEntropyLoss lossl {Yffl_19,Y_truth,batchsize};

    int epoch = 0;
    /*for(int epoch=0;epoch<epochs;++epoch)*/{
        for(int batch=0;batch<2/*numTrainImage/batchsize*/;++batch){
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                setImageToTensorAndVector(trainData, Yl_0, Y_truth, batchInx);

                
            }
            
        }
    }
    trainData.inx=0;


    std::cout<<"end of main successfully reached."<<std::endl;
    return 0;
}










void testSetTrainTestData(imagedata_t & trainData, imagedata_t & testData){
    std::cout<<"testData's first three values are "<<testData.arr[0]<<" "<<testData.arr[1]<<" "<<testData.arr[2]<<std::endl; 
    std::cout<<"testData's first few label values(0~9) "<<testData.arr[0]<<" "<<testData.arr[3073]<<" "<<testData.arr[3073*2]<<" "<<testData.arr[3073*3]<<" "<<testData.arr[3073*4]<<std::endl<<std::endl;

    std::cout<<"trainData's 30730000th, 30730001th, ... data are "<<trainData.arr[30730000]<<" "<<trainData.arr[30730001]<<" "<<trainData.arr[30730002]<<" "<<trainData.arr[30730000+3]<<" "<<trainData.arr[30730000+4]<<" "<<trainData.arr[30730000+5]<<" "<<trainData.arr[30730000+6]<<" "<<trainData.arr[30730000+7]<<" "<<std::endl;
    std::cout<<"trainData's 10000th, ... label values(0~9) "<<trainData.arr[3073*10000]<<" "<<trainData.arr[3073*10001]<<" "<<trainData.arr[3073*10002]<<" "<<trainData.arr[3073*10003]<<" "<<trainData.arr[3073*10004]<<std::endl<<std::endl;

    std::cout<<"trainData.length should be 153650000 "<<trainData.length<<std::endl;
    std::cout<<"testData.length should be 30730000 "<<testData.length<<std::endl<<std::endl;

    const int numTrainImage = 50000;
    const int numTestImage = 10000;
    const int epochs = 500;
    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(3,32,32,batchsize);

    vector1d * y_truth = newZeroVector1dArr(10,batchsize);

    int epoch = 0;
    /*for(int epoch=0;epoch<epochs;++epoch)*/{
    for(int batch=0;batch<2/*numTrainImage/batchsize*/;++batch){
        for(int batchInx=0;batchInx<batchsize;++batchInx){
            setImageToTensorAndVector(trainData, Yl_0, y_truth, batchInx);
        }
        std::cout<<"batch "<<batch<<" **********************"<<std::endl;
        for(int batchInx=0;batchInx<batchsize;++batchInx){
            /*std::cout<<"    Yl_0["<<batchInx<<"] = "<<std::endl;
            Yl_0[batchInx].printMatrixForm();*/
            std::cout<<"    y_truth["<<batchInx<<"] = "<<std::endl;
            y_truth[batchInx].printVector();
        }
    }
    }
    trainData.inx=0;
}

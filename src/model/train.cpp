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
        std::cout<<"    setTrainData: trainData.inx = "<<trainData.inx<<std::endl;
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
    std::cout<<"    setTestData: testData.inx = "<<testData.inx<<std::endl;
    testData.inx=0;
}

void testSetTrainTestData(imagedata_t & trainData, imagedata_t & testData){
    std::cout<<"testData's first three values are "<<testData.arr[0]<<" "<<testData.arr[1]<<" "<<testData.arr[2]<<std::endl; 
    std::cout<<"testData's first few label values(0~9) "<<testData.arr[0]<<" "<<testData.arr[3073]<<" "<<testData.arr[3073*2]<<" "<<testData.arr[3073*3]<<" "<<testData.arr[3073*4]<<std::endl<<std::endl;

    std::cout<<"trainData's 30730000th, 30730001th, ... data are "<<trainData.arr[30730000]<<" "<<trainData.arr[30730001]<<" "<<trainData.arr[30730002]<<" "<<trainData.arr[30730000+3]<<" "<<trainData.arr[30730000+4]<<" "<<trainData.arr[30730000+5]<<" "<<trainData.arr[30730000+6]<<" "<<trainData.arr[30730000+7]<<" "<<std::endl;
    std::cout<<"trainData's 10000th, ... label values(0~9) "<<trainData.arr[3073*10000]<<" "<<trainData.arr[3073*10001]<<" "<<trainData.arr[3073*10002]<<" "<<trainData.arr[3073*10003]<<" "<<trainData.arr[3073*10004]<<std::endl<<std::endl;

    std::cout<<"trainData.length should be 153650000 "<<trainData.length<<std::endl;
    std::cout<<"testData.length should be 30730000 "<<testData.length<<std::endl;
}

int main(){
    imagedata_t trainData;
    imagedata_t testData;
    setTrainData(trainData);
    setTestData(testData);

    testSetTrainTestData(trainData, testData);

    return 0;
}

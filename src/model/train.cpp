#include "convolution2.hpp"
#include <iostream>
#include <fstream>
#include <filesystem>
#include <array>

#include <ctime>

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

int testtrain();

int main(){


    testtrain();


    return 0;
}

int testtrain(){
    std::vector<double> avgTrainLossPerEpoch;
    std::vector<double> avgTestLossPerEpoch;

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

        std::clock_t start, finish;
        double duration;
        start = clock();

        double currEpochAvgLoss=0;

        for(int batch=0;batch<1/*numTrainImage/batchsize*/;++batch){
            
            tensor3d * dLdYl_0 = newZeroTensor3dArr(3,32,32,batchsize);
            tensor3d * dLdYl_1 = newZeroTensor3dArr(3,34,34,batchsize);
            tensor3d * dLdYl_2 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor4d * dLdWl_2 = newZeroTensor4dArr(3,32,3,3,batchsize);
            tensor3d * dLdYl_3 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor3d * dLdGl_3 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdBl_3 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdYl_4 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor3d * dLdYl_5 = newZeroTensor3dArr(32,34,34,batchsize);
            tensor3d * dLdYl_6 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor4d * dLdWl_6 = newZeroTensor4dArr(32,32,3,3,batchsize);
            tensor3d * dLdYl_7 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor3d * dLdGl_7 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdBl_7 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdYl_8 = newZeroTensor3dArr(32,32,32,batchsize);
            tensor3d * dLdYl_9 = newZeroTensor3dArr(32,16,16,batchsize);
            tensor3d * dLdYl_10 = newZeroTensor3dArr(32,18,18,batchsize);
            tensor3d * dLdYl_11 = newZeroTensor3dArr(32,16,16,batchsize);
            tensor4d * dLdWl_11 = newZeroTensor4dArr(32,32,3,3,batchsize);
            tensor3d * dLdYl_12 = newZeroTensor3dArr(32,16,16,batchsize);
            tensor3d * dLdGl_12 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdBl_12 = newZeroTensor3dArr(32,1,1,batchsize);
            tensor3d * dLdYl_13 = newZeroTensor3dArr(32,16,16,batchsize);
            tensor3d * dLdYl_14 = newZeroTensor3dArr(32,8,8,batchsize);
            tensor3d * dLdYl_15 = newZeroTensor3dArr(4,8,8,batchsize);
            tensor4d * dLdWl_15 = newZeroTensor4dArr(32,4,1,1,batchsize);
            tensor3d * dLdYl_16 = newZeroTensor3dArr(4,8,8,batchsize);
            tensor3d * dLdGl_16 = newZeroTensor3dArr(4,1,1,batchsize);
            tensor3d * dLdBl_16 = newZeroTensor3dArr(4,1,1,batchsize);
            tensor3d * dLdYl_17 = newZeroTensor3dArr(4,8,8,batchsize);
            vector1d * dLdYffl_17 = newVector1dArrFromTensor3dArr(dLdYl_17,batchsize);
            vector1d * dLdYffl_18 = newZeroVector1dArr(10,batchsize);
            tensor3d * dLdWffl_18 = newZeroTensor3dArr(1,10,256,batchsize);
            vector1d * dLdBffl_18 = newZeroVector1dArr(10,batchsize);
            vector1d * dLdYffl_19 = newZeroVector1dArr(10,batchsize);

            padl_1.setGradientTensors(dLdYl_0,dLdYl_1);
            convl_2.setGradientTensors(dLdYl_1,dLdYl_2,dLdWl_2,nullptr);
            batchnorml_3.setGradientTensors(dLdYl_2,dLdYl_3,dLdGl_3,dLdBl_3);
            relul_4.setGradientTensors(dLdYl_3,dLdYl_4);
            padl_5.setGradientTensors(dLdYl_4,dLdYl_5);
            convl_6.setGradientTensors(dLdYl_5,dLdYl_6,dLdWl_6,nullptr);
            batchnorml_7.setGradientTensors(dLdYl_6,dLdYl_7,dLdGl_7,dLdBl_7);
            relul_8.setGradientTensors(dLdYl_7,dLdYl_8);
            pooll_9.setGradientTensors(dLdYl_8,dLdYl_9);
            padl_10.setGradientTensors(dLdYl_9,dLdYl_10);
            convl_11.setGradientTensors(dLdYl_10,dLdYl_11,dLdWl_11,nullptr);
            batchnorml_12.setGradientTensors(dLdYl_11,dLdYl_12,dLdGl_12,dLdBl_12);
            relul_13.setGradientTensors(dLdYl_12,dLdYl_13);
            pooll_14.setGradientTensors(dLdYl_13,dLdYl_14);
            convl_15.setGradientTensors(dLdYl_14,dLdYl_15,dLdWl_15,nullptr);
            batchnorml_16.setGradientTensors(dLdYl_15,dLdYl_16,dLdGl_16,dLdBl_16);
            relul_17.setGradientTensors(dLdYl_16,dLdYl_17);
            affineffl_18.setGradientTensors(dLdYffl_17,dLdYffl_18,dLdWffl_18,dLdBffl_18);
            softmaxffl_19.setGradientTensors(dLdYffl_18,dLdYffl_19);
            lossl.setGradientTensors(dLdYffl_19);
            

            /* train-forward */
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                setImageToTensorAndVector(trainData, Yl_0, Y_truth, batchInx);

                padl_1.zeropad(batchInx);
                convl_2.convolve(batchInx);
            }
            batchnorml_3.batchnorm();
            
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                relul_4.relu(batchInx);
                padl_5.zeropad(batchInx);
                convl_6.convolve(batchInx);
            }
            batchnorml_7.batchnorm();
            
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                relul_8.relu(batchInx);
                pooll_9.maxpool(batchInx);
                padl_10.zeropad(batchInx);
                convl_11.convolve(batchInx);
            }
            
            batchnorml_12.batchnorm();
            
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                relul_13.relu(batchInx);
                
                pooll_14.maxpool(batchInx);
                
                convl_15.convolve(batchInx);
                
            }
            
            batchnorml_16.batchnorm();
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                relul_17.relu(batchInx);
                affineffl_18.affine(batchInx);
                softmaxffl_19.softmax(batchInx);
            }
            double currBatchAvgLossVal = lossl.avgloss();
            std::cout<<"average current batch loss("<<batch<<") = "<<currBatchAvgLossVal<<std::endl;
            currEpochAvgLoss += currBatchAvgLossVal;
            

            /*{
                std::cout<<"Yl_0[0] = "<<std::endl;
                Yl_0[0].printMatrixForm();
                std::cout<<"Yl_1[0] = "<<std::endl;
                Yl_1[0].printMatrixForm();
                std::cout<<"Yl_2[0] = "<<std::endl;
                Yl_2[0].printMatrixForm();
                std::cout<<"Yl_3[0] = "<<std::endl;
                Yl_3[0].printMatrixForm();
                std::cout<<"Yl_4[0] = "<<std::endl;
                Yl_4[0].printMatrixForm();
                std::cout<<"Yl_5[0] = "<<std::endl;
                Yl_5[0].printMatrixForm();
                std::cout<<"Yl_6[0] = "<<std::endl;
                Yl_6[0].printMatrixForm();
                std::cout<<"Yl_7[0] = "<<std::endl;
                Yl_7[0].printMatrixForm();
                std::cout<<"Yl_8[0] = "<<std::endl;
                Yl_8[0].printMatrixForm();
                std::cout<<"Yl_9[0] = "<<std::endl;
                Yl_9[0].printMatrixForm();
                std::cout<<"Yl_10[0] = "<<std::endl;
                Yl_10[0].printMatrixForm();
                std::cout<<"Yl_11[0] = "<<std::endl;
                Yl_11[0].printMatrixForm();
                std::cout<<"Yl_12[0] = "<<std::endl;
                Yl_12[0].printMatrixForm();
                std::cout<<"Yl_13[0] = "<<std::endl;
                Yl_13[0].printMatrixForm();
                std::cout<<"Yl_14[0] = "<<std::endl;
                Yl_14[0].printMatrixForm();
                std::cout<<"Yl_15[0] = "<<std::endl;
                Yl_15[0].printMatrixForm();
                std::cout<<"Yl_16[0] = "<<std::endl;
                Yl_16[0].printMatrixForm();
                std::cout<<"Yl_17[0] = "<<std::endl;
                Yl_17[0].printMatrixForm();
                std::cout<<"Yffl_17[0] = "<<std::endl;
                Yffl_17[0].printVector();
                std::cout<<"Yffl_18[0] = "<<std::endl;
                Yffl_18[0].printVector();
                std::cout<<"Yffl_19[0] = "<<std::endl;
                Yffl_19[0].printVector();
                std::cout<<"Y_truth[0] = "<<std::endl;
                Y_truth[0].printVector();
            }*/

            /* train-backward */
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                lossl.computeGrad(batchInx);
                softmaxffl_19.computeGrad(batchInx);
                affineffl_18.computeGrad(batchInx);
            }
            affineffl_18.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                relul_17.computeGrad(batchInx);
            }
            batchnorml_16.computeGrad();
            batchnorml_16.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                convl_15.computeGrad(batchInx);
            }
            convl_15.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                pooll_14.computeGrad(batchInx);
                relul_13.computeGrad(batchInx);
            }
            batchnorml_12.computeGrad();
            batchnorml_12.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                convl_11.computeGrad(batchInx);
            }
            convl_11.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                padl_10.computeGrad(batchInx);
                pooll_9.computeGrad(batchInx);
                relul_8.computeGrad(batchInx);
            }
            batchnorml_7.computeGrad();
            batchnorml_7.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                convl_6.computeGrad(batchInx);
            }
            convl_6.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                padl_5.computeGrad(batchInx);
                relul_4.computeGrad(batchInx);
            }
            batchnorml_3.computeGrad();
            batchnorml_3.batchGD(0.002);
            for(int batchInx=0;batchInx<batchsize;++batchInx){
                convl_2.computeGrad(batchInx);
            }
            convl_2.batchGD(0.002);




            
            delete [] dLdYl_0;
            delete [] dLdYl_1;
            delete [] dLdYl_2;
            delete [] dLdWl_2;
            delete [] dLdYl_3;
            delete [] dLdGl_3;
            delete [] dLdBl_3;
            delete [] dLdYl_4;
            delete [] dLdYl_5;
            delete [] dLdYl_6;
            delete [] dLdWl_6;
            delete [] dLdYl_7;
            delete [] dLdGl_7;
            delete [] dLdBl_7;
            delete [] dLdYl_8;
            delete [] dLdYl_9;
            delete [] dLdYl_10;
            delete [] dLdYl_11;
            delete [] dLdWl_11;
            delete [] dLdYl_12;
            delete [] dLdGl_12;
            delete [] dLdBl_12;
            delete [] dLdYl_13;
            delete [] dLdYl_14;
            delete [] dLdYl_15;
            delete [] dLdWl_15;
            delete [] dLdYl_16;
            delete [] dLdGl_16;
            delete [] dLdBl_16;
            delete [] dLdYl_17;
            delete [] dLdYffl_17;
            delete [] dLdYffl_18;
            delete [] dLdWffl_18;
            delete [] dLdBffl_18;
            delete [] dLdYffl_19;

        }
        /* end of all batch in 1 epoch == current epoch done training */
        trainData.inx=0;
        currEpochAvgLoss /= (double) (numTrainImage/batchsize);
        avgTrainLossPerEpoch.push_back(currEpochAvgLoss);



        finish = clock();
        duration = (double)(finish - start) / CLOCKS_PER_SEC;
        std::cout << duration << "seconds" << std::endl;



        /* get ready for inference, especially batchnorm */
        batchnorml_16.endOfEpoch();
        batchnorml_12.endOfEpoch();
        batchnorml_7.endOfEpoch();
        batchnorml_3.endOfEpoch();



        /* do inference on test data == start current epoch testing */
        double currEpochAvgTestDataLoss=0;
        for(int testImgInx=0;testImgInx<3/*numTestImage*/;++testImgInx){

            setImageToTensorAndVector(testData, Yl_0, Y_truth, 0);

            padl_1.zeropad(0);
            convl_2.convolve(0);
            batchnorml_3.inference();
            relul_4.relu(0);
            padl_5.zeropad(0);
            convl_6.convolve(0);
            batchnorml_7.inference();
            relul_8.relu(0);
            pooll_9.maxpool(0);
            padl_10.zeropad(0);
            convl_11.convolve(0);
            batchnorml_12.inference();
            relul_13.relu(0);
            pooll_14.maxpool(0);
            convl_15.convolve(0);
            batchnorml_16.inference();
            relul_17.relu(0);
            affineffl_18.affine(0);
            softmaxffl_19.softmax(0);

            double currImgLoss = lossl.loss();
            std::cout<<"current image("<<testImgInx<<") loss = "<<currImgLoss<<std::endl;
            currEpochAvgTestDataLoss += currImgLoss;
        }
        /* end of test data */
        testData.inx=0;
        currEpochAvgTestDataLoss /= numTestImage;
        avgTestLossPerEpoch.push_back(currEpochAvgTestDataLoss);


    }


    std::cout<<"end of main successfully reached."<<std::endl;
    delete [] trainData.arr;
    delete [] testData.arr;
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

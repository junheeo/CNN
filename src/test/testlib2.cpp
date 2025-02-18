#include "convolution2.hpp"
#include <vector>

void testtensor(int batchsize){
    tensor3d B_random;
    B_random.setUniformRandom(3,2,2);
    std::cout<<"B_random(3,2,2) = "<<std::endl;
    B_random.printMatrixForm();
    tensor4d W_random;
    W_random.setUniformRandom(3,4,2,2);
    std::cout<<"W_random(3,4,2,2) = "<<std::endl;
    W_random.printMatrixForm();

    /* 2x3x4 */
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchinx1 {
    {
        {1.2, 1.3, 1.4, 1.5},
        {1.6, 1.7, 1.8, 1.9},
        {2.0, 2.1, 2.2, 2.3}
    },
    {
        {2.4, 2.5, 2.6, 2.7}, 
        {2.8, 2.9, 3.0, 3.1}, 
        {3.2, 3.3, 3.4, 3.5}
    }
    };

    tensor3d * Yl_0 = new tensor3d [batchsize];
    Yl_0[0].setVal(2,3,4,inputYl_0_batchinx1);
    std::cout<<"Yl_0[0] = "<<std::endl;
    Yl_0[0].printMatrixForm();

    /* 2x4x2x3 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1 {
        {
            {
                {5,6,7},
                {8,9,10}
            },
            {
                {17,18,19},
                {20,21,22}
            },
            {
                {29,30,31},
                {32,33,34}
            },
            {
                {41,42,43},
                {44,45,46}
            }
        },
        {
            {
                {11,12,13},
                {14,15,16}
            },
            {
                {23,24,25},
                {26,27,28}
            },
            {
                {35,36,37},
                {38,39,40}
            },
            {
                {47,48,49},
                {50,51,52}
            }
        }
    };

    tensor4d Wl_1;
    Wl_1.setVal(2,4,2,3, inputWl_1);
    std::cout<<"Wl_1 = "<<std::endl;
    Wl_1.printMatrixForm();
}

void testconv2d_convolve(){
    /* 2x3x4 */
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchinx1 {
    {
        {1.2, 1.3, 1.4, 1.5},
        {1.6, 1.7, 1.8, 1.9},
        {2.0, 2.1, 2.2, 2.3}
    },
    {
        {2.4, 2.5, 2.6, 2.7}, 
        {2.8, 2.9, 3.0, 3.1}, 
        {3.2, 3.3, 3.4, 3.5}
    }
    };

    /* 2x4x2x3 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1 {
        {
            {
                {5,6,7},
                {8,9,10}
            },
            {
                {17,18,19},
                {20,21,22}
            },
            {
                {29,30,31},
                {32,33,34}
            },
            {
                {41,42,43},
                {44,45,46}
            }
        },
        {
            {
                {11,12,13},
                {14,15,16}
            },
            {
                {23,24,25},
                {26,27,28}
            },
            {
                {35,36,37},
                {38,39,40}
            },
            {
                {47,48,49},
                {50,51,52}
            }
        }
    };
    /* 4x1x1 */
    std::vector<std::vector<std::vector<double>>> inputBl_1 {{{0.1}},{{0.2}},{{0.3}},{{0.4}}};

    int batchsize = 10;
    {
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,4,batchsize);
    Yl_0[0].setVal(2,3,4,inputYl_0_batchinx1);
    tensor3d * Yl_1 = newZeroTensor3dArr(4,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,4,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(4,2,2,batchsize);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,4,2,3,batchsize);

    conv2d convl_1_noB(2,4,2,3,Yl_0,Yl_1,
             dLdYl_0, dLdYl_1, dLdWl_1, 
             nullptr, 1,false,batchsize);
    convl_1_noB.setW(inputWl_1);
    convl_1_noB.convolve(0);
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    }

    {
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,4,batchsize);
    Yl_0[0].setVal(2,3,4,inputYl_0_batchinx1);
    tensor3d * Yl_1 = newZeroTensor3dArr(4,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,4,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(4,2,2,batchsize);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,4,2,3,batchsize);
    tensor3d * dLdBl_1 = newZeroTensor3dArr(2,1,1,batchsize);


    conv2d convl_1_yesB(2,4,2,3,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, dLdBl_1, 
              1,true,batchsize);
    convl_1_yesB.setW(inputWl_1);
    convl_1_yesB.setB(inputBl_1);
    convl_1_yesB.convolve(0);
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    }
}

void testconv2d_computeGrad(){
    /* 3x2x2 */
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex1 {
        {
            {0.1, 0.2},
            {0.3, 0.4},
        },
        {
            {0.5, 0.6},
            {0.7, 0.8}
        },
        {
            {0.9, 1.0},
            {1.1, 1.2}
        }
    };
    /* 2x3x3 */
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex1 {
        {
            {10, 11, 12},
            {13, 14, 15},
            {16, 17, 18}
        },
        {
            {19, 20, 21},
            {22, 23, 24},
            {25, 26, 27}
        }
    };
    /* 2x3x2x2 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1     {
        {
            {
                {100, 101},
                {102, 103}
            },
            {
                {200, 201},
                {202, 203}
            },
            {
                {300, 301},
                {302, 303}
            }
        },
        {
            {
                {104, 105},
                {106, 107}
            },
            {
                {204, 205},
                {206, 207}
            },
            {
                {304, 305},
                {306, 307}
            }
        }
    };
    /* 3x1x1 */
    std::vector<std::vector<std::vector<double>>> inputBl_1 {{{1}},{{2}},{{3}}};

    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    Yl_0[0].setVal(2,3,3,inputYl_0_batchindex1);
    tensor3d * Yl_1 = newZeroTensor3dArr(3,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    dLdYl_1[0].setVal(3,2,2,inputdLdYl_1_batchindex1);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,3,2,2,batchsize);
    tensor3d * dLdBl_1 = newZeroTensor3dArr(3,1,1,batchsize);

    /*conv2d convl_1(2,3,2,2,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, dLdBl_1, 1,true);
              */
    conv2d convl_1(2,3,2,2,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, nullptr, 
              1,false,batchsize);

    convl_1.setW(inputWl_1);
    /*convl_1.setB(inputBl_1);*/
    convl_1.convolve(0);
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();

    convl_1.computeGrad(0);

    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();

    std::cout<<"dLdWl_1[0] = "<<std::endl;
    dLdWl_1[0].printMatrixForm();

    /*std::cout<<"dLdBl_1[0] = "<<std::endl;
    dLdBl_1[0].printMatrixForm();*/
}
void testtensorrelu_relu(){
    /* 3x2x2 */
    std::vector<std::vector<std::vector<double>>> inputdLdYl_2_batchindex0 {
        {
            {0.1, 0.2},
            {0.3, 0.4},
        },
        {
            {0.5, 0.6},
            {0.7, 0.8}
        },
        {
            {0.9, 1.0},
            {1.1, 1.2}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {10, 11, 12},
            {13, 14, 15},
            {16, 17, 18}
        },
        {
            {19, 20, 21},
            {22, 23, 24},
            {25, 26, 27}
        }
    };
    /* 2x3x2x2 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1     {
        {
            {
                {100, 101},
                {102, 103}
            },
            {
                {200, 201},
                {202, 203}
            },
            {
                {300, 301},
                {302, 303}
            }
        },
        {
            {
                {104, 105},
                {106, 107}
            },
            {
                {204, 205},
                {206, 207}
            },
            {
                {304, 305},
                {306, 307}
            }
        }
    };
    /* 3x1x1 */
    std::vector<std::vector<std::vector<double>>> inputBl_1 {{{1}},{{2}},{{3}}};

    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    Yl_0[0].setVal(2,3,3,inputYl_0_batchindex0);
    tensor3d * Yl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor3d * Yl_2 = newZeroTensor3dArr(3,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,3,2,2,batchsize);
    tensor3d * dLdBl_1 = newZeroTensor3dArr(3,1,1,batchsize);
    tensor3d * dLdYl_2 = newZeroTensor3dArr(3,2,2,batchsize);
    dLdYl_2[0].setVal(3,2,2,inputdLdYl_2_batchindex0);

    conv2d convl_1(2,3,2,2,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, dLdBl_1, 
              1,true,batchsize);
    convl_1.setW(inputWl_1);
    convl_1.setB(inputBl_1);
    convl_1.convolve(0);

    tensorRelu relul_2(3,2,2,Yl_1,Yl_2,
                dLdYl_1,dLdYl_2,batchsize);
    relul_2.relu(0);

    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_2[0] = relu(Yl_1[0]) = "<<std::endl;
    Yl_2[0].printMatrixForm();

    relul_2.computeGrad(0);
    std::cout<<"dLdYl_2[0] = "<<std::endl;
    dLdYl_2[0].printMatrixForm();
    std::cout<<"dLdYl_1[0] = "<<std::endl;
    dLdYl_1[0].printMatrixForm();

    convl_1.computeGrad(0);
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();
}
void testzeropadding(){
    /* 3x4x4 */
    std::vector<std::vector<std::vector<double>>> inputdLdYl_3_batchindex0 {
        {
            {0.11, 0.12, 0.13, 0.14},
            {0.21, 0.1 , 0.2 , 0.24},
            {0.31, 0.3 , 0.4 , 0.34},
            {0.41, 0.42, 0.43, 0.44}
        },
        {
            {0.51, 0.52, 0.53, 0.54},
            {0.61, 0.5 , 0.6 , 0.64},
            {0.71, 0.7 , 0.8 , 0.74},
            {0.81, 0.82, 0.83, 0.84}
        },
        {
            {0.91, 0.92, 0.93, 0.94},
            {1.01, 0.9 , 1.0 , 1.04},
            {1.11, 1.1 , 1.2 , 1.14},
            {1.21, 1.22, 1.23, 2.14}
        }
    };


    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {10, 11, 12},
            {13, 14, 15},
            {16, 17, 18}
        },
        {
            {19, 20, 21},
            {22, 23, 24},
            {25, 26, 27}
        }
    };
    /* 2x3x2x2 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1     {
        {
            {
                {100, 101},
                {102, 103}
            },
            {
                {200, 201},
                {202, 203}
            },
            {
                {300, 301},
                {302, 303}
            }
        },
        {
            {
                {104, 105},
                {106, 107}
            },
            {
                {204, 205},
                {206, 207}
            },
            {
                {304, 305},
                {306, 307}
            }
        }
    };
    /* 3x1x1 */
    std::vector<std::vector<std::vector<double>>> inputBl_1 {{{1}},{{2}},{{3}}};

    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    Yl_0[0].setVal(2,3,3,inputYl_0_batchindex0);
    tensor3d * Yl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor3d * Yl_2 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor3d * Yl_3 = newZeroTensor3dArr(3,4,4,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,3,2,2,batchsize);
    tensor3d * dLdBl_1 = newZeroTensor3dArr(3,1,1,batchsize);
    tensor3d * dLdYl_2 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor3d * dLdYl_3 = newZeroTensor3dArr(3,4,4,batchsize);
    dLdYl_3[0].setVal(3,4,4,inputdLdYl_3_batchindex0);
    

    conv2d convl_1(2,3,2,2,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, dLdBl_1, 
              1,true,batchsize);
    convl_1.setW(inputWl_1);
    convl_1.setB(inputBl_1);
    convl_1.convolve(0);

    tensorRelu relul_2(3,2,2,Yl_1,Yl_2,
                dLdYl_1,dLdYl_2,batchsize);
    relul_2.relu(0);

    tensorZeroPad padl_3 (Yl_2, Yl_3, dLdYl_2, dLdYl_3,batchsize);
    padl_3.zeropad(0);


    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_2[0] = relu(Yl_1[0]) = "<<std::endl;
    Yl_2[0].printMatrixForm();
    std::cout<<"Yl_3[0] = "<<std::endl;
    Yl_3[0].printMatrixForm();

    padl_3.computeGrad(0);
    std::cout<<"dLdYl_3[0] = "<<std::endl;
    dLdYl_3[0].printMatrixForm();
    relul_2.computeGrad(0);
    std::cout<<"dLdYl_2[0] = "<<std::endl;
    dLdYl_2[0].printMatrixForm();
    std::cout<<"dLdYl_1[0] = "<<std::endl;
    dLdYl_1[0].printMatrixForm();

    convl_1.computeGrad(0);
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();
}
void testmaxpool(){
    /* 2x6x6 */
    std::vector<std::vector<std::vector<double>>> inputYl_3_batchindex0 {
        {
            {10, 11, 12, 13, 14, 15},
            {16, 17, 18, 19, 20, 21},
            {22, 23, 24, 25, 26, 27},
            {28, 29, 30, 31, 32, 33},
            {34, 35, 36, 37, 38, 39},
            {40, 41, 42, 43, 44, 45}
        },
        {
            {46, 47, 48, 49, 50, 51},
            {52, 53, 54, 55, 56, 57},
            {58, 59, 60, 61, 62, 63},
            {64, 65, 66, 67, 68, 69},
            {70, 71, 72, 73, 74, 75},
            {76, 77, 78, 79, 80, 81}
        }
    };
    /* 2x2x2 */
    std::vector<std::vector<std::vector<double>>> inputdLdYl_4_batchindex0 {
        {
            {0.1, 0.2},
            {0.3, 0.4},
        },
        {
            {0.5, 0.6},
            {0.7, 0.8}
        }
    };

    int batchsize = 10;
    tensor3d * Yl_3 = newZeroTensor3dArr(2,6,6,batchsize);
    Yl_3[0].setVal(2,6,6,inputYl_3_batchindex0);
    tensor3d * Yl_4 = newZeroTensor3dArr(2,2,2,batchsize);


    tensor3d * dLdYl_3 = newZeroTensor3dArr(2,6,6,batchsize);
    tensor3d * dLdYl_4 = newZeroTensor3dArr(2,2,2,batchsize);
    dLdYl_4[0].setVal(2,2,2,inputdLdYl_4_batchindex0);

    tensorMaxPool maxpooll_4(Yl_3, Yl_4,
                            dLdYl_3, dLdYl_4, batchsize);

    maxpooll_4.maxpool(0);

    std::cout<<"Yl_4[0] = "<<std::endl;
    Yl_4[0].printMatrixForm();

    maxpooll_4.computeGrad(0);
    std::cout<<"dLdYl_4[0] = "<<std::endl;
    dLdYl_4[0].printMatrixForm();
    std::cout<<"dLdYl_3[0] = "<<std::endl;
    dLdYl_3[0].printMatrixForm();
}

int main(){
    /*testtensor(10);*/
    /*testconv2d_convolve();*/
    /*testconv2d_computeGrad();*/
    /*testtensorrelu_relu();*/
    /*testzeropadding();*/
    testmaxpool();
    return 0;
}

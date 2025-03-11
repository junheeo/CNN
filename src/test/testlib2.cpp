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

    /*conv2d convl_1(2,3,2,2,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, nullptr, 
              1,false,batchsize);*/
    conv2d convl_1(2,3,2,2,Yl_0,Yl_1,1,false,batchsize);


    convl_1.setW(inputWl_1);
    /*convl_1.setB(inputBl_1);*/
    convl_1.convolve(0);
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();

    convl_1.setGradientTensors(dLdYl_0, dLdYl_1, dLdWl_1, nullptr);
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

    /*tensorRelu relul_2(3,2,2,Yl_1,Yl_2,
                dLdYl_1,dLdYl_2,batchsize);*/
    tensorRelu relul_2(3,2,2,Yl_1,Yl_2,batchsize);

    relul_2.relu(0);

    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_2[0] = relu(Yl_1[0]) = "<<std::endl;
    Yl_2[0].printMatrixForm();

    relul_2.setGradientTensors(dLdYl_1,dLdYl_2);

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

    /* 2x3x3 */
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

    /*tensorZeroPad padl_3 (Yl_2, Yl_3, dLdYl_2, dLdYl_3,batchsize);*/
    tensorZeroPad padl_3 (Yl_2, Yl_3, batchsize);

    padl_3.zeropad(0);

    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_2[0] = relu(Yl_1[0]) = "<<std::endl;
    Yl_2[0].printMatrixForm();
    std::cout<<"Yl_3[0] = "<<std::endl;
    Yl_3[0].printMatrixForm();

    padl_3.setGradientTensors(dLdYl_2, dLdYl_3);

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

    /*tensorMaxPool maxpooll_4(Yl_3, Yl_4,
                            dLdYl_3, dLdYl_4, batchsize);*/
    tensorMaxPool maxpooll_4(Yl_3, Yl_4, batchsize);

    maxpooll_4.maxpool(0);

    std::cout<<"Yl_3[0] = "<<std::endl;
    Yl_3[0].printMatrixForm();
    std::cout<<"Yl_4[0] = "<<std::endl;
    Yl_4[0].printMatrixForm();

    maxpooll_4.setGradientTensors(dLdYl_3, dLdYl_4);

    maxpooll_4.computeGrad(0);
    std::cout<<"dLdYl_4[0] = "<<std::endl;
    dLdYl_4[0].printMatrixForm();
    std::cout<<"dLdYl_3[0] = "<<std::endl;
    dLdYl_3[0].printMatrixForm();
}
void test1x1conv(){
    /* 2x3x3 */
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
    /* 3x1x1 */
    std::vector<std::vector<std::vector<std::vector<double>>>> inputWl_1     {
        {
            {
                {1}
            },
            {
                {2}
            },
            {
                {3}
            }
        },
        {
            {
                {1}
            },
            {
                {2}
            },
            {
                {3}
            }
        }
    };
    int batchsize = 10;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    Yl_0[0].setVal(2,3,3,inputYl_0_batchindex0);
    tensor4d * Wl_1 = newZeroTensor4dArr(2,3,1,1,batchsize);
    Wl_1[0].setVal(2,3,1,1,inputWl_1);
    tensor3d * Yl_1 = newZeroTensor3dArr(3,3,3,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,3,3,batchsize);
    tensor4d * dLdWl_1 = newZeroTensor4dArr(2,3,1,1,batchsize);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(3,3,3,batchsize);

    conv2d convl_1(2,3,1,1,Yl_0,Yl_1,
              dLdYl_0, dLdYl_1, dLdWl_1, nullptr, 
              1,false,batchsize);

    convl_1.setW(inputWl_1);
    /*convl_1.setB(inputBl_1);*/
    convl_1.convolve(0);
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
}
void testbatchnorm(){
    /*
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 4},
            {7, 10},
        },
        {
            {2, 5},
            {8, 11}
        },
        {
            {3, 6},
            {9, 12}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex0 {
        {
            {0.1, 0.2},
            {0.3, 0.5},
        },
        {
            {0.51, 0.52},
            {0.53, 0.55}
        },
        {
            {4, 5},
            {6, 8}
        }
    };
    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(3,2,2,batchsize);
    for(int i=0;i<batchsize;++i){
        Yl_0[i].setVal(3,2,2,inputYl_0_batchindex0);
    }
    tensor3d * Yl_1 = newZeroTensor3dArr(3,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(3,2,2,batchsize);
    tensor3d * dLdgamma = newZeroTensor3dArr(3,1,1, 1);
    tensor3d * dLdbeta = newZeroTensor3dArr(3,1,1, 1);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(3,2,2,batchsize);
    for(int i=0;i<batchsize;++i){
        dLdYl_1[i].setVal(3,2,2,inputdLdYl_1_batchindex0);
    }

    tensorBatchNorm batchnorml_1(Yl_0,Yl_1,
            dLdYl_0,dLdYl_1,dLdgamma, dLdbeta, batchsize);
    batchnorml_1.batchnorm();
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_1[1] = "<<std::endl;
    Yl_1[1].printMatrixForm();

    batchnorml_1.computeGrad();
    std::cout<<"dLdYl_1[0] = "<<std::endl;
    dLdYl_1[0].printMatrixForm();
    std::cout<<"dLdYl_1[1] = "<<std::endl;
    dLdYl_1[1].printMatrixForm();
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();
    std::cout<<"dLdYl_0[1] = "<<std::endl;
    dLdYl_0[1].printMatrixForm();
    */

    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 2},
            {3, 4},
        },
        {
            {9, 10},
            {11, 12}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex1 {
        {
            {5, 6},
            {7, 8},
        },
        {
            {13, 14},
            {15, 16}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex0 {
        {
            {0.1, 0.2},
            {0.3, 0.4},
        },
        {
            {5.1, 5.2},
            {5.3, 5.4}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex1 {
        {
            {0.5, 0.6},
            {0.7, 0.8},
        },
        {
            {5.1, 5.2},
            {5.3, 5.4}
        }
    };

    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    Yl_0[0].setVal(2,2,2,inputYl_0_batchindex0);
    Yl_0[1].setVal(2,2,2,inputYl_0_batchindex1);
    tensor3d * Yl_1 = newZeroTensor3dArr(2,2,2,batchsize);

    tensor3d * dLdYl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    tensor3d * dLdgamma = newZeroTensor3dArr(2,1,1, 1);
    tensor3d * dLdbeta = newZeroTensor3dArr(2,1,1, 1);
    tensor3d * dLdYl_1 = newZeroTensor3dArr(2,2,2,batchsize);
    dLdYl_1[0].setVal(2,2,2,inputdLdYl_1_batchindex0);
    dLdYl_1[1].setVal(2,2,2,inputdLdYl_1_batchindex1);

    /*tensorBatchNorm batchnorml_1(Yl_0,Yl_1,
            dLdYl_0,dLdYl_1,dLdgamma, dLdbeta, batchsize);*/
    tensorBatchNorm batchnorml_1(Yl_0,Yl_1,batchsize);

    batchnorml_1.batchnorm();
    std::cout<<"Yl_1[0] = "<<std::endl;
    Yl_1[0].printMatrixForm();
    std::cout<<"Yl_1[1] = "<<std::endl;
    Yl_1[1].printMatrixForm();

    batchnorml_1.setGradientTensors(dLdYl_0,dLdYl_1,dLdgamma, dLdbeta);

    batchnorml_1.computeGrad();
    std::cout<<"dLdYl_1[0] = "<<std::endl;
    dLdYl_1[0].printMatrixForm();
    std::cout<<"dLdYl_1[1] = "<<std::endl;
    dLdYl_1[1].printMatrixForm();
    std::cout<<"dLdYl_0[0] = "<<std::endl;
    dLdYl_0[0].printMatrixForm();
    std::cout<<"dLdYl_0[1] = "<<std::endl;
    dLdYl_0[1].printMatrixForm();

    std::cout<<"dLdgamma = "<<std::endl;
    dLdgamma[0].printMatrixForm();
    std::cout<<"dLdbeta = "<<std::endl;
    dLdbeta[0].printMatrixForm();
}

void testvector1d(){
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 2},
            {3, 4},
        },
        {
            {9, 10},
            {11, 12}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex1 {
        {
            {5, 6},
            {7, 8},
        },
        {
            {13, 14},
            {15, 16}
        }
    };

    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    Yl_0[0].setVal(2,2,2,inputYl_0_batchindex0);
    Yl_0[1].setVal(2,2,2,inputYl_0_batchindex1);

    vector1d * Yffl_0 = newVector1dArrFromTensor3dArr(Yl_0,batchsize);
    std::cout<<"Yffl_0[0] = "<<std::endl;
    Yffl_0[0].printVector();
    std::cout<<"Yffl_0[1] = "<<std::endl;
    Yffl_0[1].printVector();
 
    vector1d * Yffl_1 = newZeroVector1dArr(4,batchsize);
    std::cout<<"Yffl_1[0] = "<<std::endl;
    Yffl_1[0].printVector();
    std::cout<<"Yffl_1[1] = "<<std::endl;
    Yffl_1[1].printVector();

    {
    vector1d Yffl_2 {3};
    std::cout<<"Yffl_2 = "<<std::endl;
    Yffl_2.printVector();
    std::cout<<"Yffl_2.setUniformRandom(3) = "<<std::endl;
    Yffl_2.setUniformRandom(3);
    Yffl_2.printVector();
    }
}
void testv1daffinetransform(){
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    std::vector<std::vector<double>> inputWffl_1 {
        {11,12,13,14,15,16,17,18},
        {21,22,23,24,25,26,27,28}
    };
    std::vector<double> inputbffl_1 {1,2};
    std::vector<double> inputdLdYffl_1 {0.1,0.2};

    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    Yl_0[0].setVal(2,2,2,inputYl_0_batchindex0);

    vector1d * Yffl_0 = newVector1dArrFromTensor3dArr(Yl_0,batchsize);
    vector1d * Yffl_1 = newZeroVector1dArr(2,batchsize);
    vector1d * dLdYffl_0 = newZeroVector1dArr(8,batchsize);
    vector1d * dLdYffl_1 = newZeroVector1dArr(2,batchsize);
    dLdYffl_1[0].setVal(inputdLdYffl_1);
    tensor3d * dLdWffl_1 = newZeroTensor3dArr(1,2,8,batchsize);
    vector1d * dLdbffl_1 = newZeroVector1dArr(2,batchsize);

    /*v1dAffineTransform affineffl_1 {Yffl_0, Yffl_1, dLdYffl_0, dLdYffl_1, dLdWffl_1, dLdbffl_1, batchsize};*/
    v1dAffineTransform affineffl_1 {Yffl_0, Yffl_1, batchsize};

    affineffl_1.setb(inputbffl_1);
    std::cout<<"affineffl_1.printb() = "<<std::endl;
    affineffl_1.printb();
    affineffl_1.setW(inputWffl_1);
    std::cout<<"affineffl_1.printW() = "<<std::endl;
    affineffl_1.printW();

    affineffl_1.setGradientTensors(dLdYffl_0, dLdYffl_1, dLdWffl_1, dLdbffl_1);

    affineffl_1.affine(0);
    std::cout<<"Yffl_0[0] = "<<std::endl;
    Yffl_0[0].printVector();
    std::cout<<"Yffl_1[0] = "<<std::endl;
    Yffl_1[0].printVector();

    affineffl_1.computeGrad(0);
    std::cout<<"dLYffl_1[0] = "<<std::endl;
    dLdYffl_1[0].printVector();
    std::cout<<"dLYffl_0[0] = "<<std::endl;
    dLdYffl_0[0].printVector();
    std::cout<<"dLdWffl_1[0] = "<<std::endl;
    dLdWffl_1[0].printMatrixForm();
    std::cout<<"dLdbffl_1[0] = "<<std::endl;
    dLdbffl_1[0].printVector();
}

void testv1dsoftmax(){
    std::vector<double> inputyffl_1_batchindex0 {1, 2, 3};
    std::vector<double> inputdLdyffl_2_batchindex0 {0.1, 0.2, 0.3};

    int batchsize = 2;
    vector1d * yffl_1 = newZeroVector1dArr(3,batchsize);
    yffl_1[0].setVal(inputyffl_1_batchindex0);
    vector1d * yffl_2 = newZeroVector1dArr(3,batchsize);
    vector1d * dLdyffl_1 = newZeroVector1dArr(3,batchsize);
    vector1d * dLdyffl_2 = newZeroVector1dArr(3,batchsize);
    dLdyffl_2[0].setVal(inputdLdyffl_2_batchindex0);

    /*v1dsoftmax softmaxffl_2 {yffl_1,yffl_2,dLdyffl_1,dLdyffl_2,batchsize};*/
    v1dsoftmax softmaxffl_2 {yffl_1,yffl_2,batchsize};

    softmaxffl_2.softmax(0);
    std::cout<<"yffl_1[0] = "<<std::endl;
    yffl_1[0].printVector();
    std::cout<<"yffl_2[0] = "<<std::endl;
    yffl_2[0].printVector();
    std::cout<<"answer = 0.090031, 0.244728, 0.665241"<<std::endl;

    softmaxffl_2.setGradientTensors(dLdyffl_1,dLdyffl_2);

    softmaxffl_2.computeGrad(0);
    std::cout<<"dLdyffl_2[0] = "<<std::endl;
    dLdyffl_2[0].printVector();
    std::cout<<"dLdyffl_1[0] = "<<std::endl;
    dLdyffl_1[0].printVector();
}

void testv1dcrossentropyloss(){
    std::vector<double> inputyffl_1_batchindex0 {0.1,0.6,0.3};
    std::vector<double> inputy_truth_batchindex0 {0,1,0};
    int batchsize = 2;
    vector1d * yffl_1 = newZeroVector1dArr(3,batchsize);
    yffl_1[0].setVal(inputyffl_1_batchindex0);
    yffl_1[1].setVal(inputyffl_1_batchindex0);
    vector1d * y_truth = newZeroVector1dArr(3,batchsize);
    y_truth[0].setVal(inputy_truth_batchindex0);
    y_truth[1].setVal(inputy_truth_batchindex0);
    vector1d * dLdyffl_1 = newZeroVector1dArr(3,batchsize);

    v1dCrossEntropyLoss lossl{yffl_1,y_truth,batchsize};

    double avgloss=lossl.avgloss();
    std::cout<<"avgloss = "<<avgloss<<std::endl;

    int accuracy = lossl.accuratePrediction();
    std::cout<<"accuracy = "<<accuracy<<std::endl;

    lossl.setGradientTensors(dLdyffl_1);
    lossl.computeGrad(0);

    std::cout<<"dLdyffl_1[0] = "<<std::endl;
    dLdyffl_1[0].printVector();
}

void saveloadtensorvector(){
    /*testing part 1: tensor3d, tensor4d, vector1d, conv2d*/
    {
    /* 3 */
    std::vector<double> someVector {0.1, 0.2, 0.3};

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

    {
    std::cout<<std::endl<<"Testing tensor4d, tensor3d, vector1d savefile and loadfile"<<std::endl;

    tensor4d t4_1 (2,4,2,3,inputWl_1);
    std::cout<<"t4_1 = "<<std::endl;
    t4_1.printMatrixForm();
    t4_1.saveToFile("t4_1_vals.txt");
    tensor4d t4_2 (2,4,2,3);
    t4_2.loadFromFile("t4_1_vals.txt");
    std::cout<<"t4_2 = "<<std::endl;
    t4_2.printMatrixForm(); std::cout<<std::endl;

    tensor3d t3_1 (4,1,1,inputBl_1);
    std::cout<<"t3_1 = "<<std::endl;
    t3_1.printMatrixForm();
    t3_1.saveToFile("t3_1_vals.txt");
    tensor3d t3_2 (4,1,1);
    t3_2.loadFromFile("t3_1_vals.txt");
    std::cout<<"t3_2 = "<<std::endl;
    t3_2.printMatrixForm(); std::cout<<std::endl;

    vector1d v1_1 (3);
    v1_1.setVal(someVector);
    v1_1.saveToFile("v1_1_vals.txt");
    std::cout<<"v1_1 = "<<std::endl;
    v1_1.printVector();
    vector1d v1_2 (3);
    v1_2.loadFromFile("v1_1_vals.txt");
    std::cout<<"v1_2 = "<<std::endl;
    v1_2.printVector();

    std::cout<<"testing done"<<std::endl;
    }


    int batchsize = 10;

    {
    std::cout<<std::endl<<"Testing file save load of W,B for conv2d"<<std::endl;
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

    /* W,B loaded conv2d version */
    std::cout<<"W,B for convl_1_yesB saved to file and then loaded to convl_1_yesB_1"<<std::endl;
    tensor3d * Yl_1_1 = newZeroTensor3dArr(4,2,2,batchsize);

    convl_1_yesB.saveWToFile("convl_1_yesB_W.txt");
    convl_1_yesB.saveBToFile("convl_1_yesB_B.txt");

    conv2d convl_1_yesB_1(2,4,2,3,Yl_0,Yl_1_1,
              dLdYl_0, dLdYl_1, dLdWl_1, dLdBl_1, 
              1,true,batchsize);

    convl_1_yesB_1.loadWFromFile("convl_1_yesB_W.txt");
    convl_1_yesB_1.loadBFromFile("convl_1_yesB_B.txt");

    convl_1_yesB_1.convolve(0);
    std::cout<<"Yl_1_1[0] = "<<std::endl;
    Yl_1_1[0].printMatrixForm();
    std::cout<<"testing done"<<std::endl;
    }

    }
    /* DONE testing part 1: tensor3d, tensor4d, vector1d, conv2d*/
    std::cout<<std::endl<<std::endl;
    /* testing part2: tensorBatchNorm */
    {
    std::cout<<std::endl<<"Testing file save load of tensorBatchNorm"<<std::endl;
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 2},
            {3, 4},
        },
        {
            {9, 10},
            {11, 12}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex1 {
        {
            {5, 6},
            {7, 8},
        },
        {
            {13, 14},
            {15, 16}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex0 {
        {
            {0.1, 0.2},
            {0.3, 0.4},
        },
        {
            {5.1, 5.2},
            {5.3, 5.4}
        }
    };
    std::vector<std::vector<std::vector<double>>> inputdLdYl_1_batchindex1 {
        {
            {0.5, 0.6},
            {0.7, 0.8},
        },
        {
            {5.1, 5.2},
            {5.3, 5.4}
        }
    };

    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    Yl_0[0].setVal(2,2,2,inputYl_0_batchindex0);
    Yl_0[1].setVal(2,2,2,inputYl_0_batchindex1);
    tensor3d * Yl_1 = newZeroTensor3dArr(2,2,2,batchsize);

    tensorBatchNorm batchnorml_1(Yl_0,Yl_1,batchsize);


    /*batchnorml_1.saveGToFile("Gl_1_vals.txt");
    batchnorml_1.saveBToFile("Bl_1_vals.txt");*/
    
    batchnorml_1.loadGFromFile("Gl_1_vals.txt");
    batchnorml_1.loadBFromFile("Bl_1_vals.txt");
    

    std::cout<<"testing done"<<std::endl<<std::endl;
    }
    /* DONE testing part2: tensorBatchNorm */

    /* testing part3: affinetransform */
    {
    std::cout<<"Testing file save load of v1dAffineTransform"<<std::endl;
    std::vector<std::vector<std::vector<double>>> inputYl_0_batchindex0 {
        {
            {1, 2},
            {3, 4},
        },
        {
            {5, 6},
            {7, 8}
        }
    };
    std::vector<std::vector<double>> inputWffl_1 {
        {11,12,13,14,15,16,17,18},
        {21,22,23,24,25,26,27,28}
    };
    std::vector<double> inputbffl_1 {1,2};
    std::vector<double> inputdLdYffl_1 {0.1,0.2};

    int batchsize = 2;
    tensor3d * Yl_0 = newZeroTensor3dArr(2,2,2,batchsize);
    Yl_0[0].setVal(2,2,2,inputYl_0_batchindex0);

    vector1d * Yffl_0 = newVector1dArrFromTensor3dArr(Yl_0,batchsize);
    vector1d * Yffl_1 = newZeroVector1dArr(2,batchsize);
    vector1d * dLdYffl_0 = newZeroVector1dArr(8,batchsize);
    vector1d * dLdYffl_1 = newZeroVector1dArr(2,batchsize);
    dLdYffl_1[0].setVal(inputdLdYffl_1);
    tensor3d * dLdWffl_1 = newZeroTensor3dArr(1,2,8,batchsize);
    vector1d * dLdbffl_1 = newZeroVector1dArr(2,batchsize);

    v1dAffineTransform affineffl_1 {Yffl_0, Yffl_1, batchsize};

    affineffl_1.setb(inputbffl_1);
    std::cout<<"affineffl_1.printb() = "<<std::endl;
    affineffl_1.printb();
    affineffl_1.setW(inputWffl_1);
    std::cout<<"affineffl_1.printW() = "<<std::endl;
    affineffl_1.printW();

    /*
    affineffl_1.setGradientTensors(dLdYffl_0, dLdYffl_1, dLdWffl_1, dLdbffl_1);

    affineffl_1.affine(0);
    std::cout<<"Yffl_0[0] = "<<std::endl;
    Yffl_0[0].printVector();
    std::cout<<"Yffl_1[0] = "<<std::endl;
    Yffl_1[0].printVector();
    */
    affineffl_1.saveWToFile("affine_W.txt");
    affineffl_1.saveBToFile("affine_B.txt");
    
    v1dAffineTransform affineffl_1_1 {Yffl_0, Yffl_1, batchsize};

    std::cout<<"before load:"<<std::endl;
    std::cout<<"affineffl_1_1.printb() = "<<std::endl;
    affineffl_1_1.printb();
    std::cout<<"affineffl_1_1.printW() = "<<std::endl;
    affineffl_1_1.printW();

    affineffl_1_1.loadWFromFile("affine_W.txt");
    affineffl_1_1.loadBFromFile("affine_B.txt");

    std::cout<<"after load:"<<std::endl;
    std::cout<<"affineffl_1_1.printb() = "<<std::endl;
    affineffl_1_1.printb();
    std::cout<<"should equal (1,2)"<<std::endl;
    std::cout<<"affineffl_1_1.printW() = "<<std::endl;
    affineffl_1_1.printW();
    std::cout<<"should equal  {11,12,13,14,15,16,17,18},"<<std::endl<<"{21,22,23,24,25,26,27,28}"<<std::endl<<std::endl;

    std::cout<<"testing done"<<std::endl<<std::endl;
    }
}

void testcerealsaveload(){
    /*
    std::cout<<"test cereal save load tensor3d"<<std::endl;
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

    tensor3d t3_1 (2,3,4,inputYl_0_batchinx1);
    std::cout<<"t3_1 = "<<std::endl;
    t3_1.printMatrixForm();
    t3_1.saveToFile("arr_vals.bin");
    tensor3d t3_2 (2,3,4);
    t3_2.loadFromFile("arr_vals.bin");
    std::cout<<"t3_2 = "<<std::endl;
    t3_2.printMatrixForm(); std::cout<<std::endl;
    */

    /*
    std::cout<<"test cereal save load tensor3d"<<std::endl;
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
    tensor4d t4_1 (2,4,2,3,inputWl_1);
    std::cout<<"t4_1 = "<<std::endl;
    t4_1.printMatrixForm();
    t4_1.saveToFile("arr_vals.bin");
    tensor4d t4_2 (2,4,2,3);
    t4_2.loadFromFile("arr_vals.bin");
    std::cout<<"t4_2 = "<<std::endl;
    t4_2.printMatrixForm(); std::cout<<std::endl;
    */

    /*
    std::vector<double> someVector {0.1, 0.2, 0.3};
    vector1d v1_1 (3);
    v1_1.setVal(someVector);
    v1_1.saveToFile("arr_vals.bin");
    std::cout<<"v1_1 = "<<std::endl;
    v1_1.printVector();
    vector1d v1_2 (3);
    v1_2.loadFromFile("arr_vals.bin");
    std::cout<<"v1_2 = "<<std::endl;
    v1_2.printVector();
    */
}

int main(){
    /*testtensor(10);*/
    /*testconv2d_convolve();*/
    /*testconv2d_computeGrad();*/
    /*testtensorrelu_relu();*/
    /*testzeropadding();*/
    /*testmaxpool();*/
    /*test1x1conv();*/
    /*testbatchnorm();*/
    /*testvector1d();*/
    /*testv1daffinetransform();*/
    /*testv1dsoftmax();*/
    /*testv1dcrossentropyloss();*/
    /*saveloadtensorvector();*/
    testcerealsaveload();
    return 0;
}

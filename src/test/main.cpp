#include<iostream>
#include"convolution.hpp"

void print_image(){
    std::cout<<"double image[2][3][4] = {\n"
"    {\n"
"        {1.2, 1.3, 1.4, 1.5}, \n"
"        {1.6, 1.7, 1.8, 1.9}, \n"
"        {2.0, 2.1, 2.2, 2.3}\n"
"    }, \n"
"    {\n"
"        {2.4, 2.5, 2.6, 2.7}, \n"
"        {2.8, 2.9, 3.0, 3.1}, \n"
"        {3.2, 3.3, 3.4, 3.5}\n"
"    }\n"
"};"<<std::endl;
}

void testYandXmatrix(){
    std::cout<<"testYandXmatrix()*-*-*-*-"<<std::endl;
    print_image();
    double image[2][3][4] = {
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
    dim3_t image_dims = {2,3,4};
    Ymatrix <2,3,4> Yl_0 (&image);
    std::cout<<"Yl_0(1,2,3) must be 3.5 : "<<Yl_0(1,2,3)<<std::endl;
    Yl_0.setVal(1,2,3,4.0);
    std::cout<<"Yl_0(1,2,3) must be 4.0 : "<<Yl_0(1,2,3)<<std::endl;
    Yl_0.setVal(1,2,3,3.5);
    std::cout<<"Yl_0(1,2,3) must be 3.5 : "<<Yl_0(1,2,3)<<std::endl;


    dim3_t& range_image = Yl_0.dim();
    std::cout<<"Yl_0's dims are "<<range_image.d<<" "<<range_image.w<<" "<<range_image.h<<std::endl;

    Xmatrix <2,3,4> Xl_0 (Yl_0, 2, 2, 3);
    Xl_0.setStart(0,0,1);
    std::cout<<"Xl_0 (Yl_0, 2, 2, 3) with Xl_0.setStart(0,0,1) : "<<std::endl;
    for(int z=0;z<2;++z){
    for(int x=0;x<2;++x){
        std::cout<<"   ";
    for(int y=0;y<3;++y){
        std::cout<<Xl_0(z,x,y)<<" ";
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }

    Xl_0.setStart(0,1,1);
    std::cout<<"Xl_0 (Yl_0, 2, 2, 3) with Xl_0.setStart(0,1,1) : "<<std::endl;
    for(int z=0;z<2;++z){
    for(int x=0;x<2;++x){
        std::cout<<"   ";
    for(int y=0;y<3;++y){
        std::cout<<Xl_0(z,x,y)<<" ";
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }

    std::cout<<"Xl_0 (Yl_0, 2, 2, 3) with Xl_0.setStart(0,1,1) : "<<std::endl;
    for(size_t z=0;z<2;++z){
    for(size_t x=0;x<2;++x){
        std::cout<<"   ";
    for(size_t y=0;y<3;++y){
        dim3_t col_position = {z,x,y};
        std::cout<<Xl_0(col_position)<<" ";
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }


    std::cout<<"*-*-*-*-"<<std::endl<<std::endl;
}

void testWmatrix() {
    double image[2][3][4] = {
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
    dim3_t image_dims = {2,3,4};
    Ymatrix <2,3,4> Yl_0 (&image);
    Xmatrix <2,3,4> Xl_0 (Yl_0, 2, 2, 3);/*2,2,3 is window size*/
    Xl_0.setStart(0,1,1);


    double Weightarrl_1[2][4][2][3] = {
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

    Wmatrix <2,4,2,3> Wl_1 (&Weightarrl_1);
    /*
    std::cout<<"Wl_1(p,q,r) = "<<std::endl;
    for(size_t p=0;p<2;++p){
    for(size_t q=0;q<4;++q){
    for(size_t r=0;r<2;++r){
        std::cout<<"    ";
    for(size_t s=0;s<3;++s){
        std::cout<<Wl_1(p,q,r,s)<<" ";
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }
    */
    std::cout<<"The result should be in ascending order:"<<std::endl;
    std::cout<<"Wl_1(size_T row, dim3_t col) = "<<std::endl;
    Wl_1.printMatrixForm();
    /*
    for(size_t rowInx=0;rowInx<Wl_1.rowDim;++rowInx){
        std::cout<<"row "<<rowInx<<std::endl;
    for(size_t z=0;z<Wl_1.colsDim.d;++z){
    for(size_t x=0;x<Wl_1.colsDim.w;++x){
        std::cout<<"    ";
    for(size_t y=0;y<Wl_1.colsDim.h;++y){
        dim3_t colInx = {z,x,y};
        std::cout<<Wl_1(rowInx, colInx)<<" ";
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }
        std::cout<<std::endl;
    }
    */
    std::cout<<std::endl;
    std::cout<<"Wl_1.transpose = "<<std::endl;
    Wl_1.printMatrixTransposeForm();
}

void testWmatrixMultiply(){
    double image[2][3][4] = {
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
    dim3_t image_dims = {2,3,4};
    Ymatrix <2,3,4> Yl_0 (&image);
    /* window size of Xl_0 is 2,2,3 */
    Xmatrix <2,3,4> Xl_0 (Yl_0, 2, 2, 3);
    Xl_0.setStart(0,1,1);

    double Weightarrl_1[2][4][2][3] = {
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

    Wmatrix <2,4,2,3> Wl_1 (&Weightarrl_1);

    /* size Yl_0 is 2,3,4
     * window size of Xl_0 is 2,2,3
     * size Wl_1 is 2,4,2,3 
     * therefore
     * size Yl_1 is 4, (3-2+1)=2, (4-3+1)=2 */
    double Yl_1arr[4][2][2];
    Ymatrix <4,2,2> Yl_1 (&Yl_1arr);

    /* Xmatrix<D_prev=2, W_prev=3, H_prev=4> Xl_0
     * Wmatrix<D_prev=2, D_curr=4, W_window=2, H_window=3> Wl_1
     * Ymatrix<D_curr=4, W_curr=2, H_curr=2> Yl_1
     * matMult<D_prev, D_curr, W_window, H_window, 
     *         W_prev, H_prev, W_curr, H_curr> */
    matMult<2,4,2,3,3,4,2,2>(Wl_1, Xl_0, Yl_1);
    std::cout<<"Yl_1 = "<<std::endl;
    Yl_1.printMatrixForm();

    /* one bias value for each windows(D_curr windows)*/
    double Biasarrl_1[4] = {
        0.1, 0.2, 0.3, 0.4
    };
    Bmatrix <4> Bl_1 (&Biasarrl_1);
    affineconv<2,4,2,3,3,4,2,2>(Wl_1, Bl_1, Xl_0, Yl_1);
    std::cout<<"Yl_1 = "<<std::endl;
    Yl_1.printMatrixForm();

}

void testConvGradient(){
    double Yarrl_0[2][3][3] = 
    {
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
    double Warrl_1[2][3][2][2] = 
    {
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
    double Barrl_1[3] = {1,2,3};
    /* dim of Yl_1 is 
     * D_curr = 3
     * W_curr = W_prev - W_window + 1 = 3-2+1 = 2
     * H_curr = H_prev - H_window + 1 = 3-2+1 = 2
     * */
    double Yarrl_1[3][2][2];

    double dLdYl_1[3][2][2]={
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
    
    Ymatrix<2,3,3> Yl_0 (&Yarrl_0);
    Xmatrix<2,3,3> Xl_0 (Yl_0, 2,2,2);
    Wmatrix<2,3,2,2> Wl_1 (&Warrl_1);
    Bmatrix<3> Bl_1 (&Barrl_1);
    Ymatrix<3,2,2> Yl_1 (&Yarrl_1);

    ConvGradients<3,0,0,0,2,2,0,0> gradl_2;
    gradl_2.dLdYprev = new double [3*2*2];
    /*copying gradient values into gradl_2.dLdYprev*/
    for(size_t z=0;z<3;++z){
    for(size_t x=0;x<2;++x){
    for(size_t y=0;y<2;++y){
        gradl_2.dLdYprev[gradl_2.YprevgradInx(z,x,y)] = dLdYl_1[z][x][y];
    }
    }
    }
    /*copying gradient values: done.*/

    /*<D_prev,D_curr,W_window,H_window,W_prev,H_prev,W_curr,H_curr> */
    ConvGradients<2,3,2,2,3,3,2,2> gradl_1;
    computeAffineConvGradients<3,0,0,0,2,2,0,0,2,3,2,2,3,3,2,2>(
            gradl_2,
            gradl_1,
            Wl_1,
            Bl_1,
            Xl_0
            );


    std::cout<<"dLdYl_1 = "<<std::endl;
    gradl_2.printdLdYprev();

    std::cout<<"dLdWl_1 = "<<std::endl;
    gradl_1.printdLdW();
    std::cout<<"dLdBl_1 = "<<std::endl;
    gradl_1.printdLdB();
    std::cout<<"answer: 1.0, 2.6, 3.2"<<std::endl;
    std::cout<<"dLdYl_0 = "<<std::endl;
    gradl_1.printdLdYprev();

    std::cout<<"dLdYl_0[1,0,0] = ";
    std::cout<<gradl_1.dLdYprev[gradl_1.YprevgradInx(1,0,0)];
}


int main(){
    /*testYandXmatrix();*/
    /*testWmatrix();*/
    /*testWmatrixMultiply();*/
    testConvGradient();
    return 0;
}

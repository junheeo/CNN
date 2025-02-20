#include<iostream>
#include<vector>
#include<random>
#include<cmath>

struct dim3_t {
    int d;
    int w;
    int h;
};

class tensor3d {
    double * arr;      /* dynamically allocated array */
    dim3_t arrdim;
    int dim3ToarrInx (dim3_t colInx){
        if(colInx.d>=arrdim.d || colInx.w>=arrdim.w || colInx.h>=arrdim.h){
            std::cerr<<"tensor3d index out of range : "<<colInx.d<<" "<<colInx.w<<" "<<colInx.h<<std::endl;
            throw 0;
        }
        int z = colInx.d;
        int x = colInx.w;
        int y = colInx.h;
        return (z * arrdim.w * arrdim.h) + (x * arrdim.h) + y;
    }
    dim3_t arrTodim3Inx (int i){
        int z = i / (arrdim.w * arrdim.h);
        int x = (i % (arrdim.w * arrdim.h)) / arrdim.h;
        int y = (i % (arrdim.w * arrdim.h)) % arrdim.h;
        if(z>=arrdim.d || x>=arrdim.w || y>=arrdim.h){
            std::cerr<<"tensor3d double * arr index out of range: "<<i<<" : "<<x<<" "<<y<<" "<<z<<std::endl;
        }
        dim3_t dim3Inx = {z,x,y};
        return dim3Inx;
    }
    public:
    tensor3d(){arr = nullptr; arrdim = {0,0,0};};
    tensor3d(int d, int w, int h){setZero(d, w, h);} /* initialize with 0 */
    tensor3d(int d, int w, int h, std::vector<std::vector<std::vector<double>>> inputVector){
        setVal(d,w,h,inputVector);
    }
    ~tensor3d(){                                    /* destructor */
        if(arr != nullptr){delete arr; arrdim = {0,0,0};}
    }
    
    double operator() (dim3_t colInx){
        return arr[dim3ToarrInx(colInx)];
    }
    void setVal (dim3_t colInx, double val){
        if(arr == nullptr){
            std::cerr<<"tensor3d: (private)arr is not dyn allocated"<<std::endl;
        }
        arr[dim3ToarrInx(colInx)] = val;
    }
    void setVal (int d, int w, int h, std::vector<std::vector<std::vector<double>>> & inputVector){
        if(arr != nullptr){
            delete [] arr;
        }

        arr = new double [d*w*h];
        arrdim = {d,w,h};
        dim3_t inx3d;
        for(int z=0;z<d;++z){
        for(int x=0;x<w;++x){
        for(int y=0;y<h;++y){
            inx3d = {z,x,y};
            (*this).setVal(inx3d, inputVector[z][x][y]);
        }
        }
        }
    }
    void setZero (int d, int w, int h){
        if(arr != nullptr){
            delete [] arr;
        }
        arrdim = {d,w,h};
        arr = new double [d*w*h]();
    }
    void setZero(){
        if(arrdim.d>0 && arrdim.w>0 && arrdim.h>0 && arr!=nullptr){
            (*this).setZero(arrdim.d, arrdim.w, arrdim.h);
        }else{
            std::cerr<<"setZero() : arr or arrdim of tensor3d is not initialized"<<std::endl;
            throw 0;
        }
    }
    void setUniformRandom(int d, int w, int h){

        (*this).setZero(d,w,h);

        std::mt19937 gen(123);
        std::uniform_real_distribution<> dis(-1, 1);

        dim3_t arrinx;
        for(int z=0;z<d;++z){
        for(int x=0;x<w;++x){
        for(int y=0;y<h;++y){
            arrinx = {z,x,y};
            (*this).setVal(arrinx, dis(gen));
        }
        }
        }
    }
    dim3_t dim(){
        return arrdim;
    }

    void printMatrixForm(){
        dim3_t colInx;
        for(int z=0;z<arrdim.d;++z){
        for(int x=0;x<arrdim.w;++x){
            std::cout<<"    ";
        for(int y=0;y<arrdim.h;++y){
            colInx = {z,x,y};
            std::cout<<(*this)(colInx)<<" ";
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
    }
};

class tensor4d {
    double * arr;      /* dynamically allocated array */
    int rowdim;
    dim3_t coldim;
    int dim4ToarrInx (int rowInx, dim3_t colInx){
        if(rowInx>=rowdim || colInx.d>=coldim.d || colInx.w>coldim.w || colInx.h>coldim.h){
            std::cerr<<"tensor4d index out of range: "<<colInx.d<<" "<<rowInx<<" "<<colInx.w<<" "<<colInx.h<<std::endl;
        }
        int z_prev = colInx.d;
        int z_curr = rowInx;
        int x = colInx.w;
        int y = colInx.h;
        return z_prev * (rowdim * coldim.w * coldim.h) + z_curr * (coldim.w * coldim.h) + x * (coldim.h) + y;
    }
    public:
    tensor4d() {arr = nullptr; rowdim=0; coldim = {0,0,0};}
    tensor4d(int d_prev, int d_curr, int w, int h, std::vector<std::vector<std::vector<std::vector<double>>>> inputVector) {
        setVal(d_prev, d_curr, w, h, inputVector);
    }
    tensor4d(int d_prev, int d_curr, int w, int h){
        setZero(d_prev, d_curr, w, h);
    }
    ~tensor4d(){                                        /* destructor */
        if(arr != nullptr){delete arr; rowdim = 0; coldim = {0,0,0};}
    }

    double operator() (int rowInx, dim3_t colInx){
        return arr[dim4ToarrInx(rowInx, colInx)];
    }
    double transpose(dim3_t rowInx, int colInx){
        return (*this)(colInx, rowInx);
    }
    void setVal(int rowInx, dim3_t colInx, double val){
        if(arr == nullptr){
            std::cerr<<"tensor3d: (private)arr is not dyn allocated"<<std::endl;
        }
        arr[dim4ToarrInx(rowInx, colInx)] = val;
    }
    void setVal(int d_prev, int d_curr, int w, int h, std::vector<std::vector<std::vector<std::vector<double>>>> & inputVector){
        if(arr != nullptr){
            delete [] arr;
        }
        arr = new double [d_prev*d_curr*w*h];
        rowdim = d_curr;
        coldim = {d_prev, w, h};

        int rowInx;
        dim3_t colInx;
        for(int z_prev=0;z_prev<d_prev;++z_prev){
        for(int z_curr=0;z_curr<d_curr;++z_curr){
            rowInx = z_curr;
        for(int x=0;x<w;++x){
        for(int y=0;y<h;++y){
            colInx = {z_prev, x, y};
            arr[dim4ToarrInx(rowInx, colInx)] = inputVector[z_prev][z_curr][x][y];
        }
        }
        }
        }
    }
    void setZero(int d_prev, int d_curr, int w, int h){
        if(arr != nullptr){
            delete [] arr;
        }
        rowdim = d_curr;
        coldim = {d_prev, w, h};
        arr = new double [d_prev*d_curr*w*h](); /**/
   }
    void setUniformRandom(int d_prev, int d_curr, int w, int h){
        (*this).setZero(d_prev, d_curr, w, h);

        std::mt19937 gen(123);
        std::uniform_real_distribution<> dis(-1, 1);

        int rowInx;
        dim3_t colInx;
        for(int z_prev=0;z_prev<d_prev;++z_prev){
        for(int z_curr=0;z_curr<d_curr;++z_curr){
            rowInx = z_curr;
        for(int x=0;x<w;++x){
        for(int y=0;y<h;++y){
            colInx = {z_prev, x, y};
            (*this).setVal(rowInx, colInx, dis(gen));
        }
        }
        }
        }
    }

    void printMatrixForm(){
        int rowInx;
        dim3_t colInx;
        for(int z_curr=0;z_curr<rowdim;++z_curr){
            rowInx = z_curr;
            std::cout<<"(row "<<rowInx<<")"<<std::endl;
        for(int z_prev=0;z_prev<coldim.d;++z_prev){
        for(int x=0;x<coldim.w;++x){
            std::cout<<"    ";
        for(int y=0;y<coldim.h;++y){
            colInx = {z_prev, x, y};
            std::cout<<(*this)(rowInx, colInx)<<" ";
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
    }
};

tensor3d * newZeroTensor3dArr(int d, int w, int h, int arrSize){
    tensor3d * parr = new tensor3d [arrSize];
    for(int i=0;i<arrSize;++i){
        parr[i].setZero(d,w,h);
    }
    return parr;
}
tensor4d * newZeroTensor4dArr(int d_prev, int d_curr, int w, int h, int arrSize){
    tensor4d * parr = new tensor4d [arrSize];
    for(int i=0;i<arrSize;++i){
        parr[i].setZero(d_prev,d_curr,w,h);
    }
    return parr;
}

class conv2d{
    int WBrowInx;
    dim3_t WcolInx;
    int stride;
    tensor4d W;
    tensor3d B;
    bool includeBias;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    tensor4d * dLdW;
    tensor3d * dLdB;
    int batchSize;

    class X_prev_t{
        tensor3d * Y_prev;
        dim3_t * startInx;
        dim3_t windowdim;
        public:
        dim3_t Y_prevdim;

        X_prev_t(tensor3d * Yprev, int d, int w, int h, int batchSize) : Y_prev(Yprev) {        /* constructor */
                                                windowdim = {d,w,h}; 
                                                Y_prevdim = Yprev[0].dim(); 
                                                startInx = new dim3_t [batchSize];
                                                for(int batchInx=0;batchInx<batchSize;++batchInx){
                                                    startInx[batchInx] = {0,0,0};
                                                }
        } 
        void setStart(dim3_t start_Inx, int batchInx) {startInx[batchInx] = {start_Inx.d, start_Inx.w, start_Inx.h};}
        double operator() (dim3_t colInx, int batchInx){
            dim3_t Y_prev_inx;
            if(colInx.d>=windowdim.d || colInx.w>=windowdim.w || colInx.h>=windowdim.h){
                std::cerr<<"X_prev: operator() index out of range "<<colInx.d<<" "<<colInx.w<<" "<<colInx.h<<std::endl;
                throw 0;
            }
            int z = startInx[batchInx].d + colInx.d;
            int x = startInx[batchInx].w + colInx.w;
            int y = startInx[batchInx].h + colInx.h;
            Y_prev_inx = {z,x,y};
            
            return Y_prev[batchInx](Y_prev_inx);
        }
        void setVal(dim3_t colInx, double val, int batchInx){
            dim3_t Y_prev_inx;
            if(colInx.d>=windowdim.d || colInx.w>=windowdim.w || colInx.h>=windowdim.h){
                std::cerr<<"X_prev: operator() index out of range "<<colInx.d<<" "<<colInx.w<<" "<<colInx.h<<std::endl;
                throw 0;
            }
            int z = startInx[batchInx].d + colInx.d;
            int x = startInx[batchInx].w + colInx.w;
            int y = startInx[batchInx].h + colInx.h;
            Y_prev_inx = {z,x,y};
            Y_prev[batchInx].setVal(Y_prev_inx, val);
        }
        void printMatrixForm(int batchInx){
            for(int z=0;z<windowdim.d;++z){
            for(int x=0;x<windowdim.w;++x){
                std::cout<<"    ";
            for(int y=0;y<windowdim.h;++y){
                dim3_t inx = {z,x,y};
                std::cout<<(*this)(inx, batchInx)<<" ";
            }
                std::cout<<std::endl;
            }
                std::cout<<std::endl;
            }
        }
    };


    public:
    conv2d(int d_prev, int d_curr, int w_window, int h_window, 
            tensor3d * Yl_prevarr, tensor3d * Yl_currarr, 
            tensor3d * dLdY_prevarr, tensor3d * dLdY_currarr, 
            tensor4d * dLdW_arr, tensor3d * dLdB_arr=nullptr, 
            int stride_=1, bool include_bias=true, int batch_size=10){ /*constructor*/
        dim3_t Yprevdim = Yl_prevarr[0].dim();
        dim3_t Ycurrdim = Yl_currarr[0].dim();
        if(Ycurrdim.d!=d_curr || Ycurrdim.w!=Yprevdim.w-w_window+1 || Ycurrdim.h!=Yprevdim.h-h_window+1 || d_prev!=Yprevdim.d){
            std::cerr<<"conv2d: dimensions don\'t match: "<<Ycurrdim.d<<"="<<d_curr<<" , ";
            std::cerr<<Ycurrdim.w<<"="<<Yprevdim.w<<"-"<<w_window<<"+1 , ";
            std::cerr<<Ycurrdim.h<<"="<<Yprevdim.h<<"-"<<h_window<<"+1 , ";
            std::cerr<<d_prev<<" "<<Yprevdim.h<<std::endl;
            throw 0;
        }

        WBrowInx = d_curr;
        WcolInx = {d_prev, w_window, h_window};
        stride = stride_;
        includeBias = include_bias;
        batchSize = batch_size;
        Yl_prev = Yl_prevarr;
        Yl_curr = Yl_currarr;
        dLdYl_prev = dLdY_prevarr;
        dLdYl_curr = dLdY_currarr;
        dLdW = dLdW_arr;
        dLdB = dLdB_arr;
        

        W.setUniformRandom(d_prev, d_curr, w_window, h_window);
        if(includeBias){
            B.setUniformRandom(d_curr, 1, 1);
        }
    }
    void setW(std::vector<std::vector<std::vector<std::vector<double>>>> & Wvector){
        int d_prev = WcolInx.d;
        int d_curr = WBrowInx;
        int w = WcolInx.w;
        int h = WcolInx.h;
        W.setVal(d_prev, d_curr, w, h, Wvector);
    }
    void setB(std::vector<std::vector<std::vector<double>>> & Bvector){
        B.setVal(WBrowInx, 1, 1, Bvector);
    }
    void convolve(int batchInx=0){
        dim3_t Yprevdim = Yl_prev[batchInx].dim();
        dim3_t Ycurrdim = Yl_curr[batchInx].dim();

        Yl_curr[batchInx].setZero();

        X_prev_t Xprev(Yl_prev, WcolInx.d, WcolInx.w, WcolInx.h, batchSize);

        dim3_t startInx;
        dim3_t Yl_currInx;

        for(int xcurr=0;xcurr<Ycurrdim.w;++xcurr){
        for(int ycurr=0;ycurr<Ycurrdim.h;++ycurr){


            startInx = {0, stride * xcurr, stride * ycurr};
            Xprev.setStart(startInx, batchInx);


        /*    Y_curr     =               W                             X_prev               +     B
         * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x1   D_currx(1x1)
         */
            /*std::cout<<"W dim = "<<WBrowInx<<" , "<<WcolInx.d<<" "<<WcolInx.w<<" "<<WcolInx.h<<std::endl;*/

            for(int zcurr=0;zcurr<WBrowInx;++zcurr){
                double tmpVal = 0;
                dim3_t colInx;
                for(int zprev=0;zprev<WcolInx.d;++zprev){
                for(int xwin=0;xwin<WcolInx.w;++xwin){
                for(int ywin=0;ywin<WcolInx.h;++ywin){
                    colInx = {zprev, xwin, ywin};

                    tmpVal += W(zcurr, colInx) * Xprev(colInx, batchInx);
                }
                }
                }
                if(includeBias){
                    dim3_t biasInx = {zcurr,0,0};
                    tmpVal += B(biasInx);
                }

                Yl_currInx = {zcurr, xcurr, ycurr};
                Yl_curr[batchInx].setVal(Yl_currInx, tmpVal);
            }
        }
        }
    }
    void computeGrad(int batchInx=0){
        dim3_t Yl_currdim = Yl_curr[batchInx].dim();
        
       
        
        X_prev_t Xprev(Yl_prev, WcolInx.d, WcolInx.w, WcolInx.h, batchSize);
        X_prev_t dLdXprev(dLdYl_prev, WcolInx.d, WcolInx.w, WcolInx.h, batchSize);

        dim3_t startInx;
        for(int xcurr=0;xcurr<Yl_currdim.w;++xcurr){
        for(int ycurr=0;ycurr<Yl_currdim.h;++ycurr){
            /* (1x1) is fixed value (xcurr,ycurr)
             *    Y_curr     =               W                             X_prev               +     B
             * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x1   D_currx(1x1)
             *
             *    dL/d(W X_prev) = dL/dY_curr * dY_curr/d(W X_prev) = dL/dYcurr * 1 = dL/d(Y_curr)
             *
             *    dL/dX_prev = dL/d(W X_prev) d(W X_prev)/dXprev = dL/d(Y_curr) d(W X_prev)/dXprev
             *
             *    dL/dX_prev             =               W.transpose               dL/dYcurr
             * (D_prevxW_windowxH_window)x1   (D_prevxW_windowxH_window)xD_curr  D_currx(1x1)
             */
            startInx = {0,stride*xcurr,stride*ycurr};
            dLdXprev.setStart(startInx, batchInx);

            dim3_t dLdYcurrInx;
            dim3_t rowInx;
            for(int zprev=0;zprev<WcolInx.d;++zprev){
            for(int xwin=0;xwin<WcolInx.w;++xwin){
            for(int ywin=0;ywin<WcolInx.h;++ywin){


                rowInx = {zprev,xwin,ywin};
                double tmpVal=0;
                for(int zcurr=0;zcurr<WBrowInx;++zcurr){
                    dLdYcurrInx = {zcurr, xcurr, ycurr};
                   tmpVal += W.transpose(rowInx, zcurr) * dLdYl_curr[batchInx](dLdYcurrInx);

                }
                
                dLdXprev.setVal(rowInx, dLdXprev(rowInx, batchInx) + tmpVal, batchInx);
            }
            }
            }

            /* (1x1) is fixed value (xcurr,ycurr)
             *
             *     Y_curr     =               W                               Xprev               +     B
             * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x(1x1)   D_currx(1x1)
             *
             *    dL/dW             =                 dL/dYcurr           Xprev.transpose
             * D_currx(D_prevxW_windowxH_window)    D_currx(1x1)  (1x1)x(D_prevxW_windowxH_window)
             */
            int dLdWrowInx;
            dim3_t dLdWcolInx;
            Xprev.setStart(startInx, batchInx);
            for(int zcurr=0;zcurr<WBrowInx;++zcurr){
            for(int zprev=0;zprev<WcolInx.d;++zprev){
            for(int xwin=0;xwin<WcolInx.w;++xwin){
            for(int ywin=0;ywin<WcolInx.h;++ywin){
                dLdWrowInx = zcurr;
                dLdWcolInx = {zprev,xwin,ywin};
                dim3_t dLdYcurrInx = {dLdWrowInx,xcurr,ycurr};
                double tmpVal = dLdYl_curr[batchInx](dLdYcurrInx) * Xprev(dLdWcolInx, batchInx);
                tmpVal += dLdW[batchInx](dLdWrowInx, dLdWcolInx);
                dLdW[batchInx].setVal(dLdWrowInx,dLdWcolInx, tmpVal);
            }
            }
            }
            }


            if(includeBias){
                if(dLdB == nullptr){
                    std::cerr<<"set to includeBias=true but dLdB=nullptr"<<std::endl;
                    throw 0;
                }
                /* (1x1) is fixed value (xcurr,ycurr)
                 *
                 *     Y_curr     =               W                               Xprev               +     B
                 * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x(1x1)   D_currx(1x1)
                 *
                 *     dLdB     =    dLdY_curr
                 * D_currx(1x1)     D_currx(1x1)
                 */
                dim3_t dLdBInx;
                for(int zcurr=0;zcurr<WBrowInx;++zcurr){
                    dLdBInx = {zcurr,0,0};
                    dLdYcurrInx = {zcurr,xcurr,ycurr};
                    double tmp = dLdB[batchInx](dLdBInx) + dLdYl_curr[batchInx](dLdYcurrInx);
                    dLdB[batchInx].setVal(dLdBInx, tmp);
                }
            }
        }
        }

    }
};

class tensorRelu{
    dim3_t dim;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    public:
    tensorRelu (int d, int w, int h, tensor3d * Yprev, tensor3d * Ycurr, 
                tensor3d * dLdYprev, tensor3d * dLdYcurr, int batch_size){
        dim = {d,w,h};
        Yl_prev = Yprev;
        Yl_curr = Ycurr;
        dLdYl_prev = dLdYprev;
        dLdYl_curr = dLdYcurr;
    }
    void relu(int batchInx=0){
        dim3_t YInx;
        for(int z=0;z<dim.d;++z){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            YInx = {z,x,y};
            double prevVal = Yl_prev[batchInx](YInx);
            if(prevVal >= 0){
                Yl_curr[batchInx].setVal(YInx, prevVal);
            }else{
                Yl_curr[batchInx].setVal(YInx, 0);
            }
        }
        }
        }
    }
    void computeGrad(int batchInx=0){
        dim3_t YInx;
        for(int z=0;z<dim.d;++z){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            YInx = {z,x,y};
            if(Yl_prev[batchInx](YInx) >= 0){
                dLdYl_prev[batchInx].setVal(YInx, dLdYl_curr[batchInx](YInx));
            } else {
                dLdYl_prev[batchInx].setVal(YInx, 0);
            }
        }
        }
        }
    }
};

class tensorZeroPad{
    dim3_t dimprev;
    dim3_t dimcurr;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    int padWidth;
    int padHeight;
    public:
    tensorZeroPad (tensor3d * Yprev, tensor3d * Ycurr, 
                   tensor3d * dLdYprev, tensor3d * dLdYcurr, int batch_size) : Yl_prev(Yprev), Yl_curr(Ycurr), dLdYl_prev(dLdYprev), dLdYl_curr(dLdYcurr) {
        
        if(Yprev[0].dim().d != Ycurr[0].dim().d){
            std::cerr<<"tensorZeroPad: Yprev and Ycurr should have same depth: "<<Yprev[0].dim().d<<" "<<Ycurr[0].dim().d<<std::endl;
            throw 0;
        }
        dimprev = Yprev[0].dim();
        dimcurr = Ycurr[0].dim();
        padWidth = (dimcurr.w - dimprev.w) / 2;
        padHeight = (dimcurr.h - dimprev.h) / 2;
    }
    void zeropad(int batchInx=0){
        dim3_t prevInx;
        dim3_t currInx;
        for(int zprev=0;zprev<dimprev.d;++zprev){
        for(int xprev=0;xprev<dimprev.w;++xprev){
        for(int yprev=0;yprev<dimprev.h;++yprev){
            int zcurr = zprev;
            int xcurr = xprev + padWidth;
            int ycurr = yprev + padHeight;
            prevInx = {zprev, xprev, yprev};
            currInx = {zcurr, xcurr, ycurr};
            Yl_curr[batchInx].setVal(currInx, Yl_prev[batchInx](prevInx));
        }
        }
        }
    }
    void computeGrad(int batchInx=0){
        dim3_t prevInx;
        dim3_t currInx;
        for(int zprev=0;zprev<dimprev.d;++zprev){
        for(int xprev=0;xprev<dimprev.w;++xprev){
        for(int yprev=0;yprev<dimprev.h;++yprev){
            int zcurr = zprev;
            int xcurr = xprev + padWidth;
            int ycurr = yprev + padHeight;
            prevInx = {zprev, xprev, yprev};
            currInx = {zcurr, xcurr, ycurr};
            dLdYl_prev[batchInx].setVal(prevInx, dLdYl_curr[batchInx](currInx));
        }
        }
        }
    }
};

class tensorMaxPool{
    dim3_t dimprev;
    dim3_t dimcurr;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    tensor3d * prevMaxPos;
    int Wwindow;
    int Hwindow;
    int batchSize;
    public:
    tensorMaxPool (tensor3d * Yprev, tensor3d * Ycurr, 
                   tensor3d * dLdYprev, tensor3d * dLdYcurr, int batch_size) : Yl_prev(Yprev), Yl_curr(Ycurr), dLdYl_prev(dLdYprev), dLdYl_curr(dLdYcurr), batchSize(batch_size) {
        dimprev = Yprev[0].dim();
        dimcurr = Ycurr[0].dim();
        if(dimprev.w % dimcurr.w != 0 || dimprev.h % dimcurr.h != 0){
            std::cerr<<"tensorMaxPool: width height of maxpool input is not divisible by that of output: "<<dimprev.w<<" "<<dimprev.h<<" -> "<<dimcurr.w<<" "<<dimcurr.h<<std::endl;
            throw 0;
        }
        Wwindow = dimprev.w / dimcurr.w;
        Hwindow = dimprev.h / dimcurr.h;
        prevMaxPos = newZeroTensor3dArr(dimprev.d, dimprev.w, dimprev.h, batch_size);
    }

    void maxpool(int batchInx=0){
        int maxVal;
        int zprev;
        int xprev;
        int yprev;
        dim3_t prevInx;
        dim3_t currInx;
        for(int zcurr=0;zcurr<dimcurr.d;++zcurr){
        for(int xcurr=0;xcurr<dimcurr.w;++xcurr){
        for(int ycurr=0;ycurr<dimcurr.h;++ycurr){
            zprev = zcurr;
            xprev = xcurr * Wwindow;
            yprev = ycurr * Hwindow;
            prevInx = {zprev, xprev, yprev};
            maxVal = Yl_prev[batchInx](prevInx);
            int zmax,xmax,ymax;
            zmax = zprev;
            int tmpVal;
            for(int i=0;i<Wwindow;++i){
            for(int j=0;j<Hwindow;++j){
                prevInx = {zprev, xprev+i, yprev+j};
                tmpVal = Yl_prev[batchInx](prevInx);
                if(maxVal < tmpVal){
                    maxVal = tmpVal;
                    xmax = xprev+i;
                    ymax = yprev+j;
                }
            }
            }
            currInx = {zcurr, xcurr, ycurr};
            Yl_curr[batchInx].setVal(currInx, maxVal);
            dim3_t maxInx = {zmax,xmax,ymax};
            prevMaxPos[batchInx].setVal(maxInx,1);
        }
        }
        }
        /*std::cout<<"prevMaxPos[0] = "<<std::endl;
        prevMaxPos[0].printMatrixForm();*/
    }
    void computeGrad(int batchInx=0){
        int zprev;
        int xprev;
        int yprev;
        dim3_t prevInx, currInx;
        for(int zcurr=0;zcurr<dimcurr.d;++zcurr){
        for(int xcurr=0;xcurr<dimcurr.w;++xcurr){
        for(int ycurr=0;ycurr<dimcurr.h;++ycurr){
            zprev = zcurr;
            xprev = xcurr * Wwindow;
            yprev = ycurr * Hwindow;
            for(int i=0;i<Wwindow;++i){
            for(int j=0;j<Hwindow;++j){
                prevInx = {zprev,xprev+i,yprev+j};
                if(prevMaxPos[batchInx](prevInx) == 1){
                    currInx = {zcurr,xcurr,ycurr};
                    dLdYl_prev[batchInx].setVal(prevInx, dLdYl_curr[batchInx](currInx));
                }
            }
            }
        }
        }
        }
        prevMaxPos[batchInx].setZero(dimprev.d, dimprev.w, dimprev.h);
    }
};

class tensorBatchNorm{
    dim3_t dim;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    int batchSize;
    vector<vector<double>> mus_per_depth;
    vector<vector<double>> sigmas_per_depth;
    double epsilon;
    tensor3d gamma;
    tensor3d beta;
    tensor3d dLdgamma;
    tensor3d dLdbeta;

    double mu(int z){
        double tmpVal = 0;
        dim3_t Yl_prevInx;
        for(int batchInx=0;batchInx<batchSize;++batchInx){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            Yl_prevInx = {z,x,y};
            tmpVal += Yl_prev[batchInx](Yl_prev);
        }
        }
        }
        return tmpVal;
    }
    double sigma2(int z, double mu){
        double tmpVal = 0;
        dim3_t Yl_prevInx;
        for(int batchInx=0;batchInx<batchSize;++batchInx){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            Yl_prevInx = {z,x,y};
            tmpVal += std::pow(Yl_prev[batchInx](Yl_prev) - mu, 2);
        }
        }
        }
        return tmpVal / (batchSize * dim.w * dim.h);
    }
    public:
    tensorBatchNorm (tensor * Yprev, tensor * Ycurr, tensor * dLdYprev, tensor * dLdYcurr, int batch_size) : 
                    Yl_prev(Yprev), Yl_curr(Ycurr), dLdYl_prev(dLdYprev), dLdYl_curr(dLdYcurr), batchSize(batch_size) {
        if(Yprev[0].dim().d != Ycurr[0].dim().d || Yprev[0].dim().w != Ycurr[0].dim().w || Yprev[0].dim().h != Ycurr[0].dim().h){
            std::cerr<<"tensorBatchNorm : Yprev and Ycurr dimensions do not match: ";
            std::cerr<<Yprev[0].dim().d<<" "<<Yprev[0].dim().w<<" "<<Yprev[0].dim().h;
            std::cerr<<" should equal "<<Ycurr[0].dim().d<<" "<<Ycurr[0].dim().w<<" "<<Ycurr[0].dim().h<<std::endl;
            throw 0;
        } 
        dim = Yprev[0].dim();
        epsilon = 0.00001;
        gamma.setUniformRandom(dim.d, 1, 1);
        beta.setUniformRandom(dim.d, 1, 1);
        dLdgamma.setZero(dim.d, 1, 1);
        dLdbeta.setZero(dim.d, 1, 1);
    }
    
    void batchNorm(){
        for(int z=0;z<dim.d;++z){
            double mu_B = mu(z);
            double sigma2_B = sigma2(z,mu_B);
            dim3_t zInx = {z,0,0};
            double g = gamma(zInx);
            double b = beta(zInx);

            /* for m = batchSize
             * trueBatchSize = batchSize * width * height
             * m = (batchInx,x,y)
             * x_(batchInx,x,y) = {x_(batchInx,x,y)^(z=1), ... ,x_(batchInx,x,y)^(z=d)}
             * x_hat_(batchInx,x,y)^(z) = ( x_(batchInx,x,y)^(z) - mu^(z) ) / sqrt(sigma2^(z) + epsilon)
             * */
            dim3_t i;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int x=0;x<dim.w;++x){
            for(int y=0;y<dim.h;++y){
                i={z,x,y};
                double x_m = Yl_curr[batchInx](i);
                double x_hat_m = (x_m - mu_B) / std::sqrt(sigma2_B + epsilon);
            }
            }
            }
        }
    }
};

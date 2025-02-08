#include <iostream>


struct dim3_t {
    size_t d;
    size_t w;
    size_t h;
};

template<size_t D, size_t W, size_t H>
class Xmatrix;

template<size_t D_prev, size_t D_curr, size_t W, size_t H>
class Wmatrix;

template<size_t D, size_t W, size_t H>
class Ymatrix{
    friend class Xmatrix<D,W,H>;
    double (*ptrarr)[D][W][H];
    public:
    dim3_t range;
    Ymatrix(double (*ptr_array)[D][W][H]) : ptrarr(ptr_array) 
        {range = {D,W,H};}
    double operator() (size_t z, size_t x, size_t y){
        if(z>=0 && x>=0 && y>=0 && z<D && x<W && y<H){
            return (*ptrarr)[z][x][y];
        } else {
            std::cout<<"Ymatrix index "<<z<<","<<x<<","<<y<<" out of range"<<std::endl;
            throw 0;
        }
    }
    double operator() (dim3_t position){
        return (*this)(position.d, position.w, position.h);
    }
    void setVal(size_t z, size_t x, size_t y, double val){
        (*ptrarr)[z][x][y]=val;
    }
    void setVal(dim3_t inx, double val){
        (*ptrarr)[inx.d][inx.w][inx.h]=val;
    }

    dim3_t& dim() {return range;}
    void printMatrixForm(){
        for(int z=0;z<range.d;++z){
            std::cout<<"depth "<<z<<std::endl;
        for(int x=0;x<range.w;++x){
            std::cout<<"    ";
        for(int y=0;y<range.h;++y){
            std::cout<<(*this)(z,x,y)<<" ";
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
    }
};

template<size_t D_y, size_t W_y, size_t H_y>
class Xmatrix{
    Ymatrix<D_y,W_y,H_y> whole;
    public:
    dim3_t windowSize;
    dim3_t beginInx;
    Xmatrix(Ymatrix<D_y,W_y,H_y> Y, size_t D_x, size_t W_x, size_t H_x) : whole(Y) {
        windowSize = {D_x, W_x, H_x};
        beginInx = {0, 0, 0};
    }
    void setStart(size_t z, size_t x, size_t y) {
        if( z < whole.range.d-beginInx.d+1 && 
            x < whole.range.w-beginInx.w+1 &&
            y < whole.range.h-beginInx.h+1
          ) {
            beginInx = {z, x, y};
        } else {
            std::cout<<"Xmatrix::setStart "<<z<<" "<<x<<" "<<y<<" out of range"<<std::endl;
            throw 0;
        }
    }
    double operator()(size_t z, size_t x, size_t y){
        if(z<windowSize.d && x<windowSize.w && y<windowSize.h){
            size_t whole_z = beginInx.d + z;
            size_t whole_x = beginInx.w + x;
            size_t whole_y = beginInx.h + y;
            return whole(whole_z, whole_x, whole_y);
        } else {
            std::cout<<"Xmatrix "<<z<<","<<x<<","<<y<<" index out of range"<<std::endl;
            throw 0;
        }
    }
    double operator()(dim3_t position){
        return (*this)(position.d, position.w, position.h);
    }
    void setVal(size_t z, size_t x, size_t y, double val){
        if(z<windowSize.d && x<windowSize.w && y<windowSize.h){
            size_t whole_z = beginInx.d + z;
            size_t whole_x = beginInx.w + x;
            size_t whole_y = beginInx.h + y;
            whole.setVal(whole_z, whole_x, whole_y, val);
        } else {
            std::cout<<"Xmatrix "<<z<<","<<x<<","<<y<<" index out of range"<<std::endl;
            throw 0;
        }
    }
    void setVal(dim3_t inx, double val){
        setVal(inx.d, inx.w, inx.h, val);
    }

};



template<size_t D_prev, size_t D_curr, size_t W, size_t H>
class Wmatrix{
    double (*ptrarr)[D_prev][D_curr][W][H];
    public:
    dim3_t colsDim;
    size_t rowDim;
        Wmatrix(double (*ptr_array)[D_prev][D_curr][W][H])
                : ptrarr(ptr_array) {
                    colsDim = {D_prev, W, H};
                    rowDim = D_curr;
                }
        double operator() (size_t z_prev, size_t z_curr, size_t x, size_t y){
            if( z_prev<colsDim.d &&
                z_curr<rowDim &&
                x<colsDim.w &&
                y<colsDim.h){
                return (*ptrarr)[z_prev][z_curr][x][y];
            }else{
                std::cout<<"Wmatrix index "<<z_prev<<" "<<z_curr<<" "<<x<<" "<<y<<" out of range"<<std::endl;
                throw 0;
            }
        }
        double operator() (size_t rowInx, dim3_t colInx){
            return (*this)(colInx.d, rowInx, colInx.w, colInx.h);
        }
        double transpose (dim3_t rowInx, size_t colInx){
            return (*this)(rowInx.d, colInx, rowInx.w, rowInx.h);
        }

        void printMatrixForm(){
            for(size_t rowInx=0;rowInx<rowDim;++rowInx){
            std::cout<<"row "<<rowInx<<std::endl;
            for(size_t z=0;z<colsDim.d;++z){
            for(size_t x=0;x<colsDim.w;++x){
                std::cout<<"    ";
            for(size_t y=0;y<colsDim.h;++y){
                dim3_t colInx = {z,x,y};
                std::cout<<(*this)(rowInx, colInx)<<" ";
            }
                std::cout<<std::endl;
            }
                std::cout<<std::endl;
            }
                std::cout<<std::endl;
            }

        }
        void printMatrixTransposeForm(){
            for(size_t z=0;z<colsDim.d;++z){
            for(size_t x=0;x<colsDim.w;++x){
            for(size_t y=0;y<colsDim.h;++y){
                dim3_t rowInx = {z,x,y};
                for(size_t colInx=0;colInx<rowDim;++colInx)
                {
                    std::cout<<"(col"<<colInx<<")"<<(*this).transpose(rowInx, colInx)<<" ";
                }
                std::cout<<std::endl;
            }
                std::cout<<std::endl;
            }
                std::cout<<std::endl;
            }
        }
};

template<size_t D_curr>
class Bmatrix{
    double (*ptrarr)[D_curr];
    public:
    double rowDim;
    Bmatrix(double (*ptr_array)[D_curr]) : ptrarr(ptr_array) {
        rowDim = D_curr;
    }
    double operator()(size_t rowInx){
        if(rowInx<rowDim){
            return (*ptrarr)[rowInx];
        }else{
            std::cout<<"Bmatrix index "<<rowInx<<" out of range "<<D_curr<<std::endl;
            throw 0;
        }
    }
};


template<size_t D_prev, size_t D_curr, size_t W_window, size_t H_window, size_t W_prev, size_t H_prev, size_t W_curr, size_t H_curr>
void matMult(Wmatrix<D_prev,D_curr,W_window,H_window> W,
            Xmatrix<D_prev,W_prev,H_prev> X,
            Ymatrix<D_curr,W_curr,H_curr> Y){
    /*calculate Y for Y=WX*/
    if(X.windowSize.d != D_prev){
        std::cout<<"window Xl_0 should cover the entire depth"<<std::endl;
        throw 2;
    }

    if(X.windowSize.w != W_window ||
       X.windowSize.h != H_window){
        std::cout<<"window size provided to matMult template is not same ase window size in Wmatrix parameter"<<std::endl;
        throw 2;
    }
    
    dim3_t Y_inx;
    double Y_inx_Val;
    size_t W_row_inx;
    dim3_t W_col_inx;
    dim3_t X_col_inx;
    /* x_curr and y_curr is position on output Y_l
     * and also position of window X_l on Y_l-1
     */
    for(size_t z_curr=0;z_curr<D_curr;++z_curr){
    for(size_t x_curr=0;x_curr<W_curr;++x_curr){
    for(size_t y_curr=0;y_curr<H_curr;++y_curr){
        Y_inx = {z_curr,x_curr,y_curr};
        X.setStart(0,x_curr,y_curr);

        Y_inx_Val=0.0;
        for(size_t z_prev=0;z_prev<D_prev;++z_prev){
        for(size_t x_prev=0;x_prev<W_window;++x_prev){
        for(size_t y_prev=0;y_prev<H_window;++y_prev){
            W_row_inx = z_curr;
            W_col_inx = {z_prev,x_prev,y_prev};
            X_col_inx = {z_prev,x_prev,y_prev};

            Y_inx_Val += W(W_row_inx,W_col_inx)*X(X_col_inx);
        }
        }
        }
        Y.setVal(Y_inx, Y_inx_Val);
    }
    }
    }
    
}

template<size_t D_prev, size_t D_curr, size_t W_window, size_t H_window, size_t W_prev, size_t H_prev, size_t W_curr, size_t H_curr>
void convolve(Wmatrix<D_prev,D_curr,W_window,H_window> W,
            Bmatrix<D_curr> B,
            Xmatrix<D_prev,W_prev,H_prev> X,
            Ymatrix<D_curr,W_curr,H_curr> Y){
    matMult<D_prev,D_curr,W_window,H_window,W_prev,H_prev,W_curr,H_curr>(W, X, Y);

    double biasVal;
    dim3_t Y_inx;
    for(size_t z=0;z<D_curr;++z){
        biasVal=B(z);
        for(size_t x=0;x<W_curr;++x){
        for(size_t y=0;y<H_curr;++y){
            Y_inx={z,x,y};
            Y.setVal(Y_inx, Y(Y_inx)+biasVal);
        }
        }
    }
}

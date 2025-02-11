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
            std::cerr<<"Ymatrix index "<<z<<","<<x<<","<<y<<" out of range"<<std::endl;
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
            std::cerr<<"Xmatrix::setStart "<<z<<" "<<x<<" "<<y<<" out of range"<<std::endl;
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
            std::cerr<<"Xmatrix "<<z<<","<<x<<","<<y<<" index out of range"<<std::endl;
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
            std::cerr<<"Xmatrix "<<z<<","<<x<<","<<y<<" index out of range"<<std::endl;
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
                std::cerr<<"Wmatrix index "<<z_prev<<" "<<z_curr<<" "<<x<<" "<<y<<" out of range"<<std::endl;
                throw 0;
            }
        }
        double operator() (size_t rowInx, dim3_t colInx){
            return (*this)(colInx.d, rowInx, colInx.w, colInx.h);
        }
        double transpose (dim3_t rowInx, size_t colInx){
            return (*this)(rowInx.d, colInx, rowInx.w, rowInx.h);
        }
        void setVal(size_t z_prev, size_t z_curr , size_t x, size_t y, double val){
            if(z_prev<colsDim.d && z_curr<rowDim && x<colsDim.w, y<colsDim.h){
                (*ptrarr)[z_prev][z_curr][x][y]=val;
            }else{
                std::cerr<<"Wmatrix setval index ";
                std::cerr<<z_prev<<" "<<z_curr<<" "<<x<<" "<<y<<" ";
                std::cerr<<"out of range ";
                std::cerr<<colsDim.d<<" "<<rowDim<<" "<<colsDim.w<<" "<<colsDim.h<<std::endl;
                throw 0;
            }
        }
        void setVal(size_t rowInx, dim3_t colInx, double val){
            setVal(colInx.d, rowInx, colInx.w, colInx.h, val);
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
            std::cerr<<"Bmatrix index "<<rowInx<<" out of range "<<D_curr<<std::endl;
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
        std::cerr<<"window Xl_0 should cover the entire depth"<<std::endl;
        throw 2;
    }

    if(X.windowSize.w != W_window ||
       X.windowSize.h != H_window){
        std::cerr<<"window size provided to matMult template is not same ase window size in Wmatrix parameter"<<std::endl;
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
void affineconv(Wmatrix<D_prev,D_curr,W_window,H_window> W,
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

template<size_t D_prev, size_t D_curr, size_t W_window, size_t H_window, size_t W_prev, size_t H_prev, size_t W_curr, size_t H_curr>
struct ConvGradients{
    double * dLdYprev = NULL;
    double * dLdW = NULL;
    double * dLdB = NULL;
    size_t YprevgradInx (size_t z_prev, size_t x, size_t y){
        return (W_prev*H_prev*z_prev) + (H_prev*x) + y;
        /*return W_prev*(D_prev*z_prev+x)+y;*/
    }
    size_t WgradInx (size_t z_prev, size_t z_curr, size_t x, size_t y){
        return (D_curr*W_window*H_window*z_prev) + (W_window*H_window*z_curr) + (H_window*x) + y;
        /*return W_window*(D_curr*(D_prev*z_prev+z_curr)+x)+y;*/
    }
    size_t BgradInx (size_t z_curr){
        return z_curr;
    }
    /* destructor */
    ~ConvGradients(){
        if(dLdYprev!=NULL){
            delete[] dLdYprev;
        }
        if(dLdW!=NULL){
            delete[] dLdW;
        }
        if(dLdB!=NULL){
            delete[] dLdB;
        }
    }

    void printdLdYprev(){
        for(size_t z=0;z<D_prev;++z){
            std::cout<<"depth "<<z<<std::endl;
        for(size_t x=0;x<W_prev;++x){
            std::cout<<"    ";
        for(size_t y=0;y<H_prev;++y){
            std::cout<<dLdYprev[YprevgradInx(z,x,y)]<<" ";
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
    }
    void printdLdW(){
        for(size_t z_prev=0;z_prev<D_prev;++z_prev){
            std::cout<<"***"<<std::endl;
        for(size_t z_curr=0;z_curr<D_curr;++z_curr){
            std::cout<<"    ";
        for(size_t x=0;x<W_window;++x){
        for(size_t y=0;y<H_window;++y){
            std::cout<<"("<<z_prev<<" "<<z_curr<<" "<<x<<" "<<y<<")";
            std::cout<<dLdW[WgradInx(z_prev,z_curr,x,y)]<<" ";
        }
            std::cout<<std::endl<<"    ";
        }
            std::cout<<std::endl;
        }
            std::cout<<std::endl;
        }
    }
    void printdLdB(){
        std::cout<<"    ";
        for(size_t z_curr=0;z_curr<D_curr;++z_curr){
            std::cout<<dLdB[BgradInx(z_curr)]<<" ";
        }
        std::cout<<std::endl;
    }
    void printdLdWmatrixform(){
        for(size_t z_curr=0;z_curr<D_curr;++z_curr){
            std::cout<<"row "<<z_curr<<std::endl;
        for(size_t z_prev=0;z_prev<D_prev;++z_prev){
            std::cout<<"    ";
        for(size_t x=0;x<W_prev;++x){
        for(size_t y=0;y<H_prev;++y){
            std::cout<<dLdW[WgradInx(z_prev,z_curr,x,y)]<<" ";
        }
        }
        }
            std::cout<<std::endl;
        }
    }
};


template<size_t D_prev2, size_t D_curr2, size_t W_window2, size_t H_window2, size_t W_prev2, size_t H_prev2, size_t W_curr2, size_t H_curr2, 
        size_t D_prev, size_t D_curr, size_t W_window, size_t H_window, size_t W_prev, size_t H_prev, size_t W_curr, size_t H_curr>
void computeAffineConvGradients(
            ConvGradients<D_prev2,D_curr2,W_window2,H_window2,W_prev2,H_prev2,W_curr2,H_curr2> & inputGrads, 
            ConvGradients<D_prev,D_curr,W_window,H_window,W_prev,H_prev,W_curr,H_curr> & outputGrads, 
            Wmatrix<D_prev,D_curr,W_window,H_window> W,
            Bmatrix<D_curr> B,
            Xmatrix<D_prev,W_prev,H_prev> Xprev){

    /* dLdYcurr is the gradient of Y of current layer */
    outputGrads.dLdYprev = new double [D_prev * W_prev * H_prev];
    outputGrads.dLdW = new double [D_prev * D_curr * W_window * H_window];
    outputGrads.dLdB = new double [D_curr];
    

    /* compute dLdYprev from dLdYcurr */
    for(size_t xcurr=0;xcurr<W_curr;++xcurr){
    for(size_t ycurr=0;ycurr<H_curr;++ycurr){
        Xprev.setStart(0, xcurr, ycurr);
        /*    Y_curr     =               W                             X_prev               +     B
         * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x1   D_currx(1x1)
         *
         *    dL/d(W X_prev) = dL/dY_curr * dY_curr/d(W X_prev) = dL/dYcurr * 1 = dL/d(Y_curr)
         *
         *    dL/dX_prev = dL/d(W X_prev) d(W X_prev)/dXprev = dL/d(Y_curr) d(W X_prev)/dXprev
         *
         *    dL/dX_prev             =               W.transpose               dL/dYcurr
         * (D_prevxW_windowxH_window)x1   (D_prevxW_windowxH_window)xD_curr  D_currx(1x1)
         */
        size_t z_Yprev;
        size_t x_Yprev;
        size_t y_Yprev;
        double temp_grad;
        dim3_t rowInx;
        for(size_t zprev=0;zprev<D_prev;++zprev){
        for(size_t xwin=0;xwin<W_window;++xwin){
        for(size_t ywin=0;ywin<H_window;++ywin){
            temp_grad = 0;
            rowInx = {zprev, xwin, ywin};
            for(size_t zcurr=0;zcurr<D_curr;++zcurr){
                temp_grad += W.transpose(rowInx, zcurr) * inputGrads.dLdYprev[inputGrads.YprevgradInx(zcurr, xcurr, ycurr)];
            }
            z_Yprev = zprev;
            x_Yprev = xcurr + xwin;
            y_Yprev = ycurr + ywin;
            /*
            std::cout<<"Y_prev inx "<<z_Yprev<<" "<<x_Yprev<<" "<<y_Yprev<<" "<<outputGrads.YprevgradInx(z_Yprev, x_Yprev, y_Yprev)<<std::endl;
            */
            
            outputGrads.dLdYprev[outputGrads.YprevgradInx(z_Yprev, x_Yprev, y_Yprev)] += temp_grad;

            
            z_Yprev=0;
            x_Yprev=0;
            y_Yprev=0;
        }
        }
        }
    }
    }
    
    /* compute dL/dW from dLdYcurr */
    for(size_t xcurr=0;xcurr<W_curr;++xcurr){
    for(size_t ycurr=0;ycurr<H_curr;++ycurr){
        Xprev.setStart(0,xcurr,ycurr);
        /* (1x1) is fixed value (xcurr,ycurr)
         *
         *     Y_curr     =               W                               Xprev               +     B
         * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x(1x1)   D_currx(1x1)
         *
         *    dL/dW             =                 dL/dYcurr           Xprev.transpose
         * D_currx(D_prevxW_windowxH_window)    D_currx(1x1)  (1x1)x(D_prevxW_windowxH_window)
         */
        dim3_t XprevInx;
        for(size_t zcurr=0;zcurr<D_curr;++zcurr){
        for(size_t zprev=0;zprev<D_prev;++zprev){
        for(size_t xwin=0;xwin<W_window;++xwin){
        for(size_t ywin=0;ywin<H_window;++ywin){
            XprevInx = {zprev,xwin,ywin};
            outputGrads.dLdW[outputGrads.WgradInx(zprev,zcurr,xwin,ywin)] += inputGrads.dLdYprev[inputGrads.YprevgradInx(zcurr,xcurr,ycurr)] * Xprev(XprevInx);
        }
        }
        }
        }
        /*
        std::cout<<"for("<<xcurr<<","<<ycurr<<")"<<std::endl;
        outputGrads.printdLdW();
        */
    }
    }


    /* compute dLdB from dLdYcurr */
    for(size_t xcurr=0;xcurr<W_curr;++xcurr){
    for(size_t ycurr=0;ycurr<H_curr;++ycurr){
        /* (1x1) is fixed value (xcurr,ycurr)
         *
         *     Y_curr     =               W                               Xprev               +     B
         * D_currx(1x1)    D_curr*(D_prevxW_windowxH_window)  (D_prevxW_windowxH_window)x(1x1)   D_currx(1x1)
         *
         *     dLdB     =    dLdY_curr
         * D_currx(1x1)     D_currx(1x1)
         */
        for(size_t zcurr=0;zcurr<D_curr;++zcurr){
            outputGrads.dLdB[outputGrads.BgradInx(zcurr)] += inputGrads.dLdYprev[inputGrads.YprevgradInx(zcurr,xcurr,ycurr)];

        }
    }
    }

}

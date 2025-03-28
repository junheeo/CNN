#include<iostream>
#include<vector>
#include<random>
#include<cmath>
#include<numeric>
#include<fstream>
#include<sstream>
#include<string>

#include "../../external/cereal/include/cereal/archives/portable_binary.hpp"
#include "../../external/cereal/include/cereal/types/vector.hpp"

struct dim3_t {
    int d;
    int w;
    int h;


    void printdim(){
        std::cout<<d<<" "<<w<<" "<<h;
    }
};

bool operator!= (dim3_t & lhval, dim3_t & rhval){
    if(lhval.d != rhval.d || lhval.w != rhval.w || lhval.h != rhval.h){
        return true;
    }else{
        return false;
    }
}

class vector1d;

class tensor3d {
    friend class vector1d;
    double * arr;      /* dynamically allocated array */
    dim3_t arrdim;
    int dim3ToarrInx (dim3_t colInx){
        if(colInx.d>=arrdim.d || colInx.w>=arrdim.w || colInx.h>=arrdim.h){
            std::cerr<<"tensor3d dim3ToarrInx() index out of range : "<<colInx.d<<" "<<colInx.w<<" "<<colInx.h<<std::endl;
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
            std::cerr<<"tensor3d double * arr index out of range: "<<i<<" : "<<z<<" "<<x<<" "<<y<<std::endl;
        }
        dim3_t dim3Inx = {z,x,y};
        return dim3Inx;
    }
    public:
    tensor3d(){arr = nullptr; arrdim = {0,0,0};};
    tensor3d(int d, int w, int h){arr = nullptr; setZero(d, w, h);} /* initialize with 0 */
    tensor3d(int d, int w, int h, std::vector<std::vector<std::vector<double>>> inputVector){
        arr = nullptr;
        setVal(d,w,h,inputVector);
    }
    ~tensor3d(){                                    /* destructor */
        if(arr != nullptr){delete [] arr; arrdim = {0,0,0};}
    }
    
    double operator() (dim3_t colInx){
        try{
            double val = arr[dim3ToarrInx(colInx)];
            return val;
        }catch(int errorInt){
            std::cerr<<"tensor3d operator(): index out of range"<<std::endl;
            throw errorInt;
        }
    }
    void setVal (dim3_t colInx, double val){
        if(arr == nullptr){
            std::cerr<<"tensor3d: (private)arr is not dyn allocated"<<std::endl;
        }
        try{
            arr[dim3ToarrInx(colInx)] = val;
        }catch(int errorInt){
            std::cerr<<"tensor3d setVal(dim3_t, val) : index out of range"<<std::endl;
            throw errorInt;
        }
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

    void saveToFile(std::string fileName){
        int arrSize = arrdim.d * arrdim.w * arrdim.h;
        if(arrSize==0){
            std::cerr<<"tensor3d saveToFile(): tensor size is 0, nothing to store "<<arrdim.d<<" "<<arrdim.w<<" "<<arrdim.h<<std::endl;
            throw 0;
        }

        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(arrSize);

        for (int i = 0; i < arrSize; i++) {
            archive(arr[i]);
        }

        file.close();
        /*
        int arrSize = arrdim.d * arrdim.w * arrdim.h;
        if(arrSize==0){
            std::cerr<<"tensor3d storeTensor(): tensor size is 0, nothing to store "<<arrdim.d<<" "<<arrdim.w<<" "<<arrdim.h<<std::endl;
            throw 0;
        }

        std::stringstream streamStorage;
        streamStorage<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[0];
        for(int i=1;i<arrSize;++i){
            streamStorage<<" "<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[i];
        }
        streamStorage<<"\n";

        std::ofstream outputFile (fileName.c_str());
        outputFile<<streamStorage.str();
        outputFile.close();
        */
    }

    void loadFromFile(std::string fileName){
        int arrSize = arrdim.d * arrdim.w * arrdim.h;

        std::ifstream infile(fileName.c_str(), std::ios::binary);
        if(!infile.is_open()){
            std::cerr<<"tensor3d loadFromFile(std::string) cannot find(open) file "<<fileName<<std::endl;
            throw 1;
        }
        cereal::PortableBinaryInputArchive inarchive(infile);

        int loadSize;
        inarchive(loadSize);

        if(loadSize != arrSize){
            std::cerr<<"tensor3d loadFromFile(std::string) there are "<<loadSize<<" many values in "<<fileName<<" but curr tensor3d object is dimension "<<arrdim.d<<" * "<<arrdim.w<<" * "<<arrdim.h<<" = "<<arrSize<<std::endl;
            throw 1;
        }
        
        for(int i=0;i<arrSize;++i){
            inarchive(arr[i]);
        }

        infile.close();
        /*
        int arrSize = arrdim.d * arrdim.w * arrdim.h;
        std::ifstream inputFile (fileName.c_str());
        if(!inputFile.is_open()){
            std::cerr<<"tensor3d loadFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        {
        int i=0;
        double val;
        while(inputFile>>std::setprecision(std::numeric_limits<double>::max_digits10 +3)>>std::fixed>>val){
            if(i==arrSize){
                std::cerr<<"tensor3d loadFromFile(std::string): there are more values in file "<<fileName<<" than size of arr of tensor3d obj"<<std::endl;
                throw 0;
            }
            arr[i] = val;
            ++i;
        }
        }
        inputFile.close();
        */
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
        arr = nullptr;
        setVal(d_prev, d_curr, w, h, inputVector);
    }
    tensor4d(int d_prev, int d_curr, int w, int h){
        arr = nullptr;
        setZero(d_prev, d_curr, w, h);
    }
    ~tensor4d(){                                        /* destructor */
        if(arr != nullptr){delete [] arr; rowdim = 0; coldim = {0,0,0};}
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
    int rowDim(){return rowdim;}
    dim3_t colDim(){return coldim;}

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

    void saveToFile(std::string fileName){
        int arrSize = rowdim * coldim.d * coldim.w * coldim.h;
        if(arrSize==0){
            std::cerr<<"tensor4d saveToFile(): tensor size is 0, nothing to store "<<rowdim<<" "<<coldim.d<<" "<<coldim.w<<" "<<coldim.h<<std::endl;
            throw 0;
        }

        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(arrSize);

        for (int i = 0; i < arrSize; i++) {
            archive(arr[i]);
        }

        file.close();

        /*
        int arrSize = rowdim * coldim.d * coldim.w * coldim.h;
        if(arrSize==0){
            std::cerr<<"tensor4d storeTensor(): tensor size is 0, nothing to store "<<rowdim<<" , "<<coldim.d<<" "<<coldim.w<<" "<<coldim.h<<std::endl;
            throw 0;
        }

        std::stringstream streamStorage;
        streamStorage<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[0];
        for(int i=1;i<arrSize;++i){
            streamStorage<<" "<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[i];
        }
        streamStorage<<"\n";

        std::ofstream outputFile (fileName.c_str());
        outputFile<<streamStorage.str();
        outputFile.close();
        */
    }

    void loadFromFile(std::string fileName){
        int arrSize = rowdim * coldim.d * coldim.w * coldim.h;

        std::ifstream infile(fileName.c_str(), std::ios::binary);
        if(!infile.is_open()){
            std::cerr<<"tensor4d loadFromFile(std::string) cannot find(open) file "<<fileName<<std::endl;
            throw 1;
        }
        cereal::PortableBinaryInputArchive inarchive(infile);

        int loadSize;
        inarchive(loadSize);

        if(loadSize != arrSize){
            std::cerr<<"tensor4d loadFromFile(std::string) there are "<<loadSize<<" many values in "<<fileName<<" but curr tensor4d object is dimension "<<rowdim<<" * "<<coldim.d<<" * "<<coldim.w<<" * "<<coldim.h<<" = "<<arrSize<<std::endl;
            throw 1;
        }
        
        for(int i=0;i<arrSize;++i){
            inarchive(arr[i]);
        }

        infile.close();
        
        /*
        int arrSize = rowdim * coldim.d * coldim.w * coldim.h;
        std::ifstream inputFile (fileName.c_str());
        if(!inputFile.is_open()){
            std::cerr<<"tensor4d loadFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        {
        int i=0;
        double val;
        while(inputFile>>std::fixed>>std::setprecision(std::numeric_limits<double>::max_digits10 +3)>>val){
            if(i==arrSize){
                std::cerr<<"tensor4d loadFromFile(std::string): there are more values in file "<<fileName;
                std::cerr<<" than size of arr of tensor4d obj "<<rowdim<<" , "<<coldim.d<<" "<<coldim.w<<" "<<coldim.h<<std::endl;
                throw 0;
            }
            arr[i] = val;
            ++i;
        }
        }
        inputFile.close();
        */
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
        ~X_prev_t(){delete [] startInx;}
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

    conv2d(int d_prev, int d_curr, int w_window, int h_window, 
            tensor3d * Yl_prevarr, tensor3d * Yl_currarr,int stride_=1, bool include_bias=true, int batch_size=10){
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

        dLdYl_prev = nullptr;
        dLdYl_curr = nullptr;
        dLdW = nullptr;
        dLdB = nullptr;

        W.setUniformRandom(d_prev, d_curr, w_window, h_window);
        if(includeBias){
            B.setUniformRandom(d_curr, 1, 1);
        }

    }

    void setGradientTensors(tensor3d * dLdY_prevarr, tensor3d * dLdY_currarr, 
                            tensor4d * dLdW_arr, tensor3d * dLdB_arr=nullptr){
        int d_curr = WBrowInx;
        int d_prev = WcolInx.d;
        int w_window = WcolInx.w;
        int h_window = WcolInx.h;

        dLdYl_prev = dLdY_prevarr;
        dLdYl_curr = dLdY_currarr;
        dLdW = dLdW_arr;
        dLdB = dLdB_arr;
        

        {
        dim3_t gradDim = dLdYl_prev[0].dim();
        dim3_t tensorDim = Yl_prev[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"conv2d setGradientTensors : dimension of dLdY_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdYl_curr[0].dim();
        dim3_t tensorDim = Yl_curr[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"conv2d setGradientTensors : dimension of dLdY_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;           
        }
        }
        {
        int gradrowDim = dLdW[0].rowDim();
        dim3_t gradcolDim = dLdW[0].colDim();
        int tensorrowDim = W.rowDim();
        dim3_t tensorcolDim = W.colDim();
        if((gradrowDim != tensorrowDim) || (gradcolDim != tensorcolDim)){
            std::cerr<<"conv2d setGradientTensors : dimension of dLdW_arr does not match"<<std::endl;
            std::cerr<<"    "<<gradcolDim.d<<" "<<gradrowDim<<" "<<gradcolDim.w<<" "<<gradcolDim.h<<std::endl;
            std::cerr<<"    "<<tensorcolDim.d<<" "<<tensorrowDim<<" "<<tensorcolDim.w<<" "<<tensorcolDim.h<<std::endl;
            throw 0;           

        }
        }
        if(includeBias){
        dim3_t gradDim = dLdB[0].dim();
        dim3_t tensorDim = B.dim();
        if(gradDim != tensorDim){
            std::cerr<<"conv2d setGradientTensors : dimension of dLdB does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
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

    void saveWToFile(std::string fileName){
        try{
            W.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"conv2d saveWToFile("<<fileName<<"): error thrown with W.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void loadWFromFile(std::string fileName){
        try{
            W.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"conv2d loadWFromFile("<<fileName<<"): error thrown with W.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void saveBToFile(std::string fileName){
        if(!includeBias){
            std::cerr<<"conv2d saveBToFile(std::string): includeBias == false, no B to save"<<std::endl;
            throw 0;
        }
        try{
            B.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"conv2d saveBToFile("<<fileName<<"): error thrown with B.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void loadBFromFile(std::string fileName){
        if(!includeBias){
            std::cerr<<"conv2d loadBToFile(std::string): includeBias == false, no B to save"<<std::endl;
            throw 0;
        }
        try{
            B.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"conv2d loadBFromFile("<<fileName<<"): error thrown with B.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
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
    
    void batchGD(double learnrate){
        /* int WBrowInx;
         * dim3_t WcolInx; */
        /* W have dimension (WBrowInx x WcolInx) */
        /* B have dimension (WBrowInx x 1x1) */
        /* set dLdW[0] as the batchwise average gradient */
        for(int z_curr=0;z_curr<WBrowInx;++z_curr){
            int rowInx = z_curr;
        for(int z_prev=0;z_prev<WcolInx.d;++z_prev){
        for(int x=0;x<WcolInx.w;++x){
        for(int y=0;y<WcolInx.h;++y){
            dim3_t colInx = {z_prev,x,y};
            double tmp=0;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
                tmp += dLdW[batchInx](rowInx, colInx);
            }
            dLdW[0].setVal(rowInx, colInx, tmp/batchSize);

            /* update W = W - learnrate * dLdW[0] */
            W.setVal(rowInx, colInx, W(rowInx,colInx) - learnrate * dLdW[0](rowInx,colInx));
        }
        }
        }
        }
        if(includeBias){
            /* set dLdB[0] as the batchwise average gradient */
            for(int z_curr=0;z_curr<WBrowInx;++z_curr){
                dim3_t BInx = {z_curr,0,0};
                double tmp=0;
                for(int batchInx=0;batchInx<batchSize;++batchInx){
                    tmp += dLdB[batchInx](BInx);
                }
                dLdB[0].setVal(BInx, tmp/batchSize);

                /* update B = B - learnrate * dLdB[0] */
                B.setVal(BInx, B(BInx) - learnrate * dLdB[0](BInx));
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
    tensorRelu (int d, int w, int h, tensor3d * Yprev, tensor3d * Ycurr, int batch_size){
        dim = {d,w,h};
        Yl_prev = Yprev;
        Yl_curr = Ycurr;
        dLdYl_prev = nullptr;
        dLdYl_curr = nullptr;
    }
    void setGradientTensors(tensor3d * dLdYprev, tensor3d * dLdYcurr){
        dLdYl_prev = dLdYprev;
        dLdYl_curr = dLdYcurr;
        {
        dim3_t gradDim = dLdYl_prev[0].dim();
        dim3_t tensorDim = Yl_prev[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorRelu setGradientTensors : dimension of dLdY_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdYl_curr[0].dim();
        dim3_t tensorDim = Yl_curr[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorRelu setGradientTensors : dimension of dLdY_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
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
    tensorZeroPad(tensor3d * Yprev, tensor3d * Ycurr, int batch_size) : Yl_prev(Yprev), Yl_curr(Ycurr){
        if(Yprev[0].dim().d != Ycurr[0].dim().d){
            std::cerr<<"tensorZeroPad: Yprev and Ycurr should have same depth: "<<Yprev[0].dim().d<<" "<<Ycurr[0].dim().d<<std::endl;
            throw 0;
        }
        dimprev = Yprev[0].dim();
        dimcurr = Ycurr[0].dim();
        padWidth = (dimcurr.w - dimprev.w) / 2;
        padHeight = (dimcurr.h - dimprev.h) / 2;
    }
    void setGradientTensors(tensor3d * dLdYprev, tensor3d * dLdYcurr){
        dLdYl_prev = dLdYprev;
        dLdYl_curr = dLdYcurr;
        {
        dim3_t gradDim = dLdYl_prev[0].dim();
        dim3_t tensorDim = Yl_prev[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorZeroPad setGradientTensors : dimension of dLdYl_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdYl_curr[0].dim();
        dim3_t tensorDim = Yl_curr[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"conv2d setGradientTensors : dimension of dLdYl_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
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
    tensorMaxPool (tensor3d * Yprev, tensor3d * Ycurr, int batch_size) : Yl_prev(Yprev), Yl_curr(Ycurr), batchSize(batch_size) {
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
    void setGradientTensors(tensor3d * dLdYprev, tensor3d * dLdYcurr){
        dLdYl_prev = dLdYprev;
        dLdYl_curr = dLdYcurr;
        {
        dim3_t gradDim = dLdYl_prev[0].dim();
        dim3_t tensorDim = Yl_prev[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorMaxPool setGradientTensors : dimension of dLdYl_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdYl_curr[0].dim();
        dim3_t tensorDim = Yl_curr[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorMaxPool setGradientTensors : dimension of dLdYl_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
    }

    void maxpool(int batchInx=0){

        prevMaxPos[batchInx].setZero(dimprev.d, dimprev.w, dimprev.h);

        int maxVal;
        int zprev;
        int xprev;
        int yprev;
        dim3_t prevInx;
        dim3_t currInx;
        /*std::cout<<std::endl;
        std::cout<<"    maxpool(): Yl_prev<"<<dimprev.d<<","<<dimprev.w<<","<<dimprev.h<<">"<<std::endl;
        std::cout<<"               Yl_curr<"<<dimcurr.d<<","<<dimcurr.w<<","<<dimcurr.h<<">"<<std::endl;
        std::cout<<"               Wwindow,Hwindow = "<<Wwindow<<","<<Hwindow<<std::endl;*/
        for(int zcurr=0;zcurr<dimcurr.d;++zcurr){
        for(int xcurr=0;xcurr<dimcurr.w;++xcurr){
        for(int ycurr=0;ycurr<dimcurr.h;++ycurr){
            zprev = zcurr;
            xprev = xcurr * Wwindow;
            yprev = ycurr * Hwindow;
            prevInx = {zprev, xprev, yprev};
            try{
                maxVal = Yl_prev[batchInx](prevInx);
            } catch (int errorInt){
                std::cout<<"    maxpool(): * accessing Yl_prev["<<batchInx<<"]({"<<zprev<<","<<xprev<<","<<yprev<<"})"<<std::endl;
                throw errorInt;
            }
            int zmax,xmax,ymax;
            zmax = zprev;
            xmax = xprev;
            ymax = yprev;
            int tmpVal;
            for(int i=0;i<Wwindow;++i){
            for(int j=0;j<Hwindow;++j){
                prevInx = {zprev, xprev+i, yprev+j};
                try{
                tmpVal = Yl_prev[batchInx](prevInx);
                }catch(int errorInt){
                std::cout<<"    maxpool(): ** accessing Yl_prev["<<batchInx<<"]({"<<zprev<<","<<xprev+i<<","<<yprev+j<<"})"<<std::endl;
                throw errorInt;
                }
                if(maxVal < tmpVal){
                    maxVal = tmpVal;
                    xmax = xprev+i;
                    ymax = yprev+j;
                }
            }
            }
            currInx = {zcurr, xcurr, ycurr};
            try{
            Yl_curr[batchInx].setVal(currInx, maxVal);
            }catch(int errorInt){
            std::cout<<"    maxpool(): *** accessing Yl_curr["<<batchInx<<"]({"<<zcurr<<","<<xcurr<<","<<ycurr<<"})"<<std::endl;
            throw errorInt;
            }
            dim3_t maxInx = {zmax,xmax,ymax};
            try{
            prevMaxPos[batchInx].setVal(maxInx,1);
            }catch(int errorInt){
            std::cout<<"    maxpool(): ****accessing prevMaxPos["<<batchInx<<"]({"<<zmax<<","<<xmax<<","<<ymax<<"})"<<std::endl;
            }
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
    }
};

class tensorBatchNorm{
    dim3_t dim;
    tensor3d * Yl_prev;
    tensor3d * Yl_curr;
    tensor3d * dLdYl_prev;
    tensor3d * dLdYl_curr;
    int batchSize;
    std::vector<std::vector<double>> mus_per_depth;
    std::vector<std::vector<double>> sigma2s_per_depth;
    std::vector<double> sum_mus_per_depth;
    std::vector<double> sum_sigma2s_per_depth;
    double epsilon;
    tensor3d gamma; /*dim.dx1x1*/
    tensor3d beta; /*dim.dx1x1*/
    tensor3d * dLdgamma;
    tensor3d * dLdbeta;

    double mu(int z){
        double tmpVal = 0;
        dim3_t Yl_prevInx;
        for(int batchInx=0;batchInx<batchSize;++batchInx){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            Yl_prevInx = {z,x,y};
            tmpVal += Yl_prev[batchInx](Yl_prevInx);
        }
        }
        }
        return tmpVal / (batchSize * dim.w * dim.h);
    }
    double sigma2(int z, double mu){
        double tmpVal = 0;
        dim3_t Yl_prevInx;
        for(int batchInx=0;batchInx<batchSize;++batchInx){
        for(int x=0;x<dim.w;++x){
        for(int y=0;y<dim.h;++y){
            Yl_prevInx = {z,x,y};
            tmpVal += std::pow(Yl_prev[batchInx](Yl_prevInx) - mu, 2);
        }
        }
        }
        return tmpVal / (batchSize * dim.w * dim.h);
    }
    public:
    tensorBatchNorm (tensor3d * Yprev, tensor3d * Ycurr, 
                    tensor3d * dLdYprev, tensor3d * dLdYcurr, 
                    tensor3d * dLdgamma_, tensor3d * dLdbeta_, int batch_size) : 
                    Yl_prev(Yprev), Yl_curr(Ycurr), dLdYl_prev(dLdYprev), 
                    dLdYl_curr(dLdYcurr), batchSize(batch_size), dLdgamma(dLdgamma_), dLdbeta(dLdbeta_) {
        if(Yprev[0].dim().d != Ycurr[0].dim().d || Yprev[0].dim().w != Ycurr[0].dim().w || Yprev[0].dim().h != Ycurr[0].dim().h){
            std::cerr<<"tensorBatchNorm : Yprev and Ycurr dimensions do not match: ";
            std::cerr<<Yprev[0].dim().d<<" "<<Yprev[0].dim().w<<" "<<Yprev[0].dim().h;
            std::cerr<<" should equal "<<Ycurr[0].dim().d<<" "<<Ycurr[0].dim().w<<" "<<Ycurr[0].dim().h<<std::endl;
            throw 0;
        } 
        dim = Yprev[0].dim();
        epsilon = 0.0000001;
        /*gamma.setUniformRandom(dim.d, 1, 1);
        beta.setUniformRandom(dim.d, 1, 1);*/
        gamma.setZero(dim.d,1,1);
        for(int z=0;z<dim.d;++z){dim3_t i={z,0,0};gamma.setVal(i,1);}
        beta.setZero(dim.d,1,1);


        mus_per_depth.resize(dim.d);
        sigma2s_per_depth.resize(dim.d);
        sum_mus_per_depth.resize(dim.d);
        sum_sigma2s_per_depth.resize(dim.d);
    }

    tensorBatchNorm (tensor3d * Yprev, tensor3d * Ycurr, int batch_size) : Yl_prev(Yprev), Yl_curr(Ycurr), batchSize(batch_size) {
        dLdYl_prev = nullptr; 
        dLdYl_curr = nullptr;
        dLdgamma = nullptr;
        dLdbeta = nullptr;
        if(Yprev[0].dim().d != Ycurr[0].dim().d || Yprev[0].dim().w != Ycurr[0].dim().w || Yprev[0].dim().h != Ycurr[0].dim().h){
            std::cerr<<"tensorBatchNorm : Yprev and Ycurr dimensions do not match: ";
            std::cerr<<Yprev[0].dim().d<<" "<<Yprev[0].dim().w<<" "<<Yprev[0].dim().h;
            std::cerr<<" should equal "<<Ycurr[0].dim().d<<" "<<Ycurr[0].dim().w<<" "<<Ycurr[0].dim().h<<std::endl;
            throw 0;
        } 
        dim = Yprev[0].dim();
        epsilon = 0.0000001;
        /*gamma.setUniformRandom(dim.d, 1, 1);
        beta.setUniformRandom(dim.d, 1, 1);*/
        gamma.setZero(dim.d,1,1);
        for(int z=0;z<dim.d;++z){dim3_t i={z,0,0};gamma.setVal(i,1);}
        beta.setZero(dim.d,1,1);


        mus_per_depth.resize(dim.d);
        sigma2s_per_depth.resize(dim.d);
        sum_mus_per_depth.resize(dim.d);
        sum_sigma2s_per_depth.resize(dim.d);
    }

    void setGradientTensors(tensor3d * dLdYprev, tensor3d * dLdYcurr, tensor3d * dLdgamma_, tensor3d * dLdbeta_){
        dLdYl_prev = dLdYprev; 
        dLdYl_curr = dLdYcurr;
        dLdgamma = dLdgamma_;
        dLdbeta = dLdbeta_;
        {
        dim3_t gradDim = dLdYl_prev[0].dim();
        dim3_t tensorDim = Yl_prev[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorBatchNorm setGradientTensors : dimension of dLdYl_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdYl_curr[0].dim();
        dim3_t tensorDim = Yl_curr[0].dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorBatchNorm setGradientTensors : dimension of dLdYl_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdgamma[0].dim();
        dim3_t tensorDim = gamma.dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorBatchNorm setGradientTensors : dimension of dLdgamma does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdbeta[0].dim();
        dim3_t tensorDim = beta.dim();
        if(gradDim != tensorDim){
            std::cerr<<"tensorBatchNorm setGradientTensors : dimension of dLdbeta does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
    }

    void saveGToFile(std::string fileName){
        try{
            gamma.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"tensorBatchNorm saveGToFile("<<fileName<<"): error thrown with gamma.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
        /*std::cout<<"saveGToFile(): gamma = "<<std::endl;
        gamma.printMatrixForm();*/
    }
    void loadGFromFile(std::string fileName){
        try{
            gamma.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"tensorBatchNorm loadGFromFile("<<fileName<<"): error thrown with gamma.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }

        /*std::cout<<"loadGFromFile(): gamma = "<<std::endl;
        gamma.printMatrixForm();*/
    }
    void saveBToFile(std::string fileName){
        try{
            beta.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"tensorBatchNorm saveBToFile("<<fileName<<"): error thrown with beta.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
        /*std::cout<<"saveBToFile(): beta = "<<std::endl;
        beta.printMatrixForm();*/
    }
    void loadBFromFile(std::string fileName){
        try{
            beta.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"tensorBatchNorm loadBFromFile("<<fileName<<"): error thrown with beta.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }

        /*std::cout<<"loadBFromFile(): beta = "<<std::endl;
        beta.printMatrixForm();*/
    }
    void saveSumMusToFile(std::string fileName){
        if(dim.d<1){
            std::cerr<<"tensorBatchNorm saveSumMusToFile("<<fileName<<") : error: dim.d = "<<dim.d<<" nothing to save"<<std::endl;
            throw 0;
        }

        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(dim.d);

        for (int i = 0; i < dim.d; ++i) {
            archive(sum_mus_per_depth[i]);
        }

        file.close();

        /*
        if(dim.d<1){
            std::cerr<<"tensorBatchNorm saveSumMusToFile("<<fileName<<") : error: dim.d = "<<dim.d<<" nothing to save"<<std::endl;
            throw 0;
        }
        std::stringstream streamStorage;
        streamStorage<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_mus_per_depth[0];
        for(int i=1;i<dim.d;++i){
            streamStorage<<" "<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_mus_per_depth[i];
        }
        streamStorage<<"\n";
        std::ofstream outputFile (fileName.c_str());
        outputFile<<streamStorage.str();
        outputFile.close();
        */
    }
    void loadSumMusFromFile(std::string fileName){
        std::ifstream infile(fileName.c_str(), std::ios::binary);
        if(!infile.is_open()){
            std::cerr<<"tensorBatchNorm loadSumMusFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        cereal::PortableBinaryInputArchive inarchive(infile);

        int loadSize;
        inarchive(loadSize);
        if(loadSize != dim.d){
            std::cerr<<"tensorBatchNorm loadSumMusFromFile(): file "<<fileName<<" has "<<loadSize<<" values but vector sum_mus_per_depth is size "<<dim.d<<std::endl;
            throw 1;
        }
        
        for(int i=0; i<dim.d; ++i){
            inarchive(sum_mus_per_depth[i]);
        }

        /*std::cout<<"sum_mus_per_depth =";
        for(int i=0; i<dim.d; ++i){
            std::cout<<" "<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_mus_per_depth[i];
        }
        std::cout<<std::endl;*/

        infile.close(); 

        /*
        std::ifstream inputFile (fileName.c_str());
        if(!inputFile.is_open()){
            std::cerr<<"tensorBatchNorm loadSumMusFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        int i=0;
        double val;
        while(inputFile>>std::setprecision(std::numeric_limits<double>::max_digits10 +3)>>std::fixed>>val){
            if(dim.d == i){
                std::cerr<<"tensorBatchNorm loadSumMusFromFile(std::string): there are more values in file "<<fileName<<" than size vector sum_mus_per_depth = "<<dim.d<<std::endl;
                throw 0;
            }
            sum_mus_per_depth[i] = val;
            ++i;
        }
        inputFile.close();
        */
    }
    void saveSumSigma2sToFile(std::string fileName){
        if(dim.d<1){
            std::cerr<<"tensorBatchNorm saveSumSigma2sToFile("<<fileName<<") : error: dim.d = "<<dim.d<<" nothing to save"<<std::endl;
            throw 0;
        }

        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(dim.d);

        for (int i = 0; i < dim.d; ++i) {
            archive(sum_sigma2s_per_depth[i]);
        }

        file.close();
        /*
        if(dim.d<1){
            std::cerr<<"tensorBatchNorm saveSumSigma2sToFile("<<fileName<<") : error: dim.d = "<<dim.d<<" nothing to save"<<std::endl;
            throw 0;
        }
        std::stringstream streamStorage;
        streamStorage<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_sigma2s_per_depth[0];
        for(int i=1;i<dim.d;++i){
            streamStorage<<" "<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_sigma2s_per_depth[i];
        }
        streamStorage<<"\n";
        std::ofstream outputFile (fileName.c_str());
        outputFile<<streamStorage.str();
        outputFile.close();
        */
    }
    void loadSumSigma2sFromFile(std::string fileName){
        std::ifstream infile(fileName.c_str(), std::ios::binary);
        if(!infile.is_open()){
            std::cerr<<"tensorBatchNorm loadSumsigma2sFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        cereal::PortableBinaryInputArchive inarchive(infile);

        int loadSize;
        inarchive(loadSize);
        if(loadSize != dim.d){
            std::cerr<<"tensorBatchNorm loadSumSigma2sFromFile(): file "<<fileName<<" has "<<loadSize<<" values but vector sum_sigma2s_per_depth is size "<<dim.d<<std::endl;
            throw 1;
        }
        
        for(int i=0; i<dim.d; ++i){
            inarchive(sum_sigma2s_per_depth[i]);
        }

        /*std::cout<<"    loadSumSigma2sFromFile("<<fileName<<"): sum_sigma2s_per_depth =";
        for(int i=0; i<dim.d; ++i){
            std::cout<<" "<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<sum_sigma2s_per_depth[i];
        }
        std::cout<<std::endl;*/

        infile.close(); 
        /*
        std::ifstream inputFile (fileName.c_str());
        if(!inputFile.is_open()){
            std::cerr<<"tensorBatchNorm loadSumSigma2sFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        int i=0;
        double val;
        while(inputFile>>std::fixed>>std::setprecision(std::numeric_limits<double>::max_digits10 +3)>>val){
            if(dim.d == i){
                std::cerr<<"tensorBatchNorm loadSumSigma2sFromFile(std::string): there are more values in file "<<fileName<<" than size vector sum_mus_per_depth = "<<dim.d<<std::endl;
                throw 0;
            }
            sum_sigma2s_per_depth[i] = val;
            ++i;
        }

        std::cout<<"    loadSumSigma2sFromFile("<<fileName<<"): sum_sigma2s_per_depth =";
        for(double a : sum_sigma2s_per_depth){
            std::cout<<" "<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<a;
        }
        std::cout<<std::endl;
        inputFile.close();
        */
    }
    
    void batchnorm(){
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
             * y_(batchInx,x,y)^(z) = gamma^(z) * x_hat_(batchInx,x,y)^(z) + beta^(z)
             * */
            dim3_t i;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int x=0;x<dim.w;++x){
            for(int y=0;y<dim.h;++y){
                i={z,x,y};
                double x_m = Yl_prev[batchInx](i);
                double x_hat_m = (x_m - mu_B) / std::sqrt(sigma2_B + epsilon);
                double y_m = g * x_hat_m + b;
                Yl_curr[batchInx].setVal(i, y_m);

                /*std::cout<<"("<<z<<" "<<x<<" "<<y<<") x_m "<<x_m<<" mu "<<mu_B<<" sigma2 "<<sigma2_B<<std::endl;*/
            }
            }
            }

            mus_per_depth[z].push_back(mu_B);
            sigma2s_per_depth[z].push_back(sigma2_B);
        }

        /*std::cout<<"mus_per_depth = "<<std::endl;*/
        int depth=0;
        for(auto & v: mus_per_depth){
            /*std::cout<<"depth "<<depth<<" ";*/
            for(auto & m: v){
                /*std::cout<<m<<" ";*/
            }
            /*std::cout<<std::endl;*/
            ++depth;
        }
    }
    void computeGrad(){
        /*std::cout<<"entered computeGrad()"<<std::endl;*/
        /*std::cout<<"create dLdx_hat"<<std::endl;*/

        std::vector<std::vector<std::vector<double>>> dLdx_hat;
        dLdx_hat.resize(batchSize);
        for(auto & v1 : dLdx_hat){
        v1.resize(dim.w);
        for(auto & v2 : v1){
        v2.resize(dim.h);
        }
        }

        /*std::cout<<"...success"<<std::endl*/

        for(int z=0;z<dim.d;++z){

            dim3_t zInx = {z,0,0};
            double g = gamma(zInx);
            double b = beta(zInx);
            double mu_B = mu(z);
            double sigma2_B = sigma2(z, mu_B);

            double dLdx_hat_sum=0;

            /*std::cout<<"z="<<z<<" g="<<g<<" b="<<b<<" mu_B="<<mu_B<<" sigma2_B="<<sigma2_B<<std::endl;*/

            /* compute dLdx_hat */
            /*std::cout<<"start compute dLdx_hat z="<<z<<std::endl;*/
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                dim3_t i={z,xinx,yinx};
                double dLdx_hat_val = dLdYl_curr[batchInx](i) * g;
                dLdx_hat[batchInx][xinx][yinx] = dLdx_hat_val;
                dLdx_hat_sum += dLdx_hat_val;
                /*std::cout<<"    dLdx_hat["<<batchInx<<"]["<<xinx<<"]["<<yinx<<"] = "<<dLdx_hat[batchInx][xinx][yinx]<<" = "<<dLdx_hat_val<<std::endl;*/
            }
            }
            }
            /*std::cout<<"    dLdx_hat_sum = "<<dLdx_hat_sum<<std::endl;
            std::cout<<"...success"<<std::endl;*/

            /* compute dLdsigma2 */
            /*std::cout<<"start compute dLdsigma2 z="<<z<<std::endl;*/
            double dLdsigma2 = 0;
            /*std::cout<<"    startForLoop(batchInx,xinx,yinx)"<<std::endl;*/
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                /*std::cout<<"    dLdsigma2 batchInx,x,y "<<batchInx<<" "<<xinx<<" "<<yinx<<std::endl;*/
                dim3_t i={z,xinx,yinx};
                double x = Yl_prev[batchInx](i);
                /*std::cout<<"    x = "<<x<<std::endl;*/
                dLdsigma2 += dLdx_hat[batchInx][xinx][yinx] * (x - mu_B);
                /*std::cout<<"    dLdx_hat["<<batchInx<<"]["<<xinx<<"]["<<yinx<<"] = "<<dLdx_hat[batchInx][xinx][yinx]<<std::endl;
                std::cout<<"    dLdsigma2 += "<<dLdx_hat[batchInx][xinx][yinx] * (x - mu_B)<<std::endl;*/
                /*std::cout<<"    dLdsigma2 += "<<dLdx_hat[batchInx][xinx][yinx]<<" * ("<<x<<" - "<<mu_B<<")"<<std::endl;*/
            }
            }
            }
            /*std::cout<<"    dLdsigma2 outside forloop dLdsigma2sum = "<<dLdsigma2<<std::endl;
            std::cout<<"    sigma2_B + epsilon = "<<sigma2_B + epsilon<<std::endl;*/
            dLdsigma2 = dLdsigma2 * (-0.5) * std::pow(sigma2_B + epsilon, -1.5);
            /*std::cout<<"...success dLdsigma2 = "<<dLdsigma2<<std::endl;*/

            /* compute dLdmu */
            /*std::cout<<"start compute dLdmu z="<<z<<std::endl;*/
            double dLdmu = dLdx_hat_sum / std::sqrt(sigma2_B + epsilon) * (-1);
            /* commented below tmp = 0 */
            /*{
            double tmp=0;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                dim3_t i={z,xinx,yinx};
                double x = Yl_prev[batchInx](i);
                tmp += x - mu_B;
            }
            }
            }
            tmp *= dLdsigma2 * 2 * (-1);
            tmp /= batchSize * dim.w * dim.h;
            std::cout<<"    tmp = "<<tmp<<std::endl;
            dLdmu += tmp;
            }*/
            /*std::cout<<"...success dLdmu = "<<dLdmu<<std::endl;*/

            /* compute dLdx */
            /*std::cout<<"start compute dLdx z="<<z<<std::endl;*/
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                /*std::cout<<"    batchInx "<<batchInx<<" xinx "<<xinx<<" yinx "<<yinx<<std::endl;*/
                dim3_t i={z,xinx,yinx};
                double x = Yl_prev[batchInx](i);
                /*std::cout<<"    x = "<<x<<" dLdx_hat = "<<dLdx_hat[batchInx][xinx][yinx]<<std::endl;
                std::cout<<"    m = "<<batchSize * dim.w * dim.h<<std::endl;*/
                double dLdx = dLdx_hat[batchInx][xinx][yinx] / std::sqrt(sigma2_B + epsilon) 
                            + dLdsigma2 * (x - mu_B) * 2 / (batchSize * dim.w * dim.h)
                            + dLdmu / (batchSize * dim.w * dim.h);

                dLdYl_prev[batchInx].setVal(i, dLdx);
            }
            }
            }
            /*std::cout<<"...success"<<std::endl;*/

            /* compute dLdgamma */
            {
            double tmp=0;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                dim3_t i={z,xinx,yinx};
                double x = Yl_prev[batchInx](i);
                double x_hat = (x - mu_B) / std::sqrt(sigma2_B + epsilon);
                tmp +=  dLdYl_curr[batchInx](i) * x_hat; 
            }
            }
            }
            dLdgamma[0].setVal(zInx, tmp);
            }

            /* compute dLdbeta */
            {
            double tmp=0;
            for(int batchInx=0;batchInx<batchSize;++batchInx){
            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                dim3_t i={z,xinx,yinx};
                tmp +=  dLdYl_curr[batchInx](i); 
            }
            }
            }
            dLdbeta[0].setVal(zInx, tmp);
            }
        }
    }
    void endOfEpoch(){

        for(int z=0;z<dim.d;++z){
            double sum_mus = std::accumulate(mus_per_depth[z].begin(), mus_per_depth[z].end(), 0.0);
            sum_mus_per_depth[z] = sum_mus / (mus_per_depth[z].size());
            double sum_sigma2s = std::accumulate(sigma2s_per_depth[z].begin(), sigma2s_per_depth[z].end(), 0.0);
            sum_sigma2s_per_depth[z] = sum_sigma2s / (sigma2s_per_depth[z].size());

            mus_per_depth[z].clear();
            sigma2s_per_depth[z].clear();
        }
        
    }

    void inference(int batchInx=0){
        /* once the network has been trained, use the normalization of the population 
         * x_hat = ( x - E ) / std::sqrt(Var + epsilon)
         * where
         * E = Expected_Value(mu_(batchInx,x,y))
         * Var = m / (m-1) * Expected_Value(sigma2_(batchInx,x,y))
         * where
         * effective batch size m = batchSize * dim.w * dim.h
         *
         * then batch normalization during inference is
         * y = gamma * x_hat + beta
         *   = gamma/sqrt(Var+epsilon) * x + (beta - gamma*E/std::sqrt(Var+epsilon))
         * */
        for(int z=0;z<dim.d;++z){
            double E = sum_mus_per_depth[z];
            double Var = sum_sigma2s_per_depth[z] * ((batchSize * dim.w * dim.h) / (batchSize * dim.w * dim.h - 1));

            /*std::cout<<"E = "<<E<<" = "<<sum_mus_per_depth[z]<<std::endl;
            std::cout<<"Var = "<<Var<<" = "<<sum_sigma2s_per_depth[z]<<" * "<<(batchSize * dim.w * dim.h)<<" / "<<(batchSize * dim.w * dim.h - 1)<<std::endl;*/


            dim3_t zInx = {z,0,0};
            double a = gamma(zInx) / std::sqrt(Var + epsilon);
            double b = beta(zInx) - ( gamma(zInx) * E ) / std::sqrt(Var + epsilon);

            for(int xinx=0;xinx<dim.w;++xinx){
            for(int yinx=0;yinx<dim.h;++yinx){
                dim3_t i = {z,xinx,yinx};
                Yl_curr[batchInx].setVal(i, a * Yl_prev[batchInx](i) + b);
            }
            }
        }
    }

    void batchGD(double learnrate){
        /* beta, gamma have dimension (dim.d x 1 x 1) */
        /* set dLdbeta[0] as the batchwise average gradient */
        /* set dLdgamma[0] as the batchwise average gradient */
        double tmpBeta=0;
        double tmpGamma=0;
        dim3_t zInx;
        for(int z=0;z<dim.d;++z){
            zInx = {z,0,0};
            for(int batchInx=0;batchInx<batchSize;++batchInx){
                tmpBeta += dLdbeta[batchInx](zInx);
                tmpGamma += dLdgamma[batchInx](zInx);
            }
            dLdbeta[0].setVal(zInx, tmpBeta/batchSize);
            dLdgamma[0].setVal(zInx, tmpGamma/batchSize);
            tmpBeta=0;
            tmpGamma=0;

            /* update beta = beta - learnrate * dLdbeta[0] */
            /* update gamma = gamma - learnrate * dLdgamma[0] */
            beta.setVal(zInx, beta(zInx) - learnrate * dLdbeta[0](zInx));
            gamma.setVal(zInx, gamma(zInx) - learnrate * dLdgamma[0](zInx));
        }
    }
};

class vector1d {
    double * arr;   /* dynamically allocated */
    tensor3d * pt3d;
    public:
    int size;

    vector1d () {arr = nullptr; pt3d = nullptr; size = 0;}  /* constructor */
    vector1d (int size_val) {arr = new double[size_val] (); pt3d = nullptr; size = size_val;}   /* zero initialization of array */
    vector1d (tensor3d & t3d) {(*this).setVal(t3d);}
    ~vector1d () {
        if(arr != nullptr){
            delete [] arr;
        }
        pt3d = nullptr;
        size = 0;
    }

    double operator() (int inx) {
        if(arr == nullptr && (*pt3d).arr == nullptr){
            std::cerr<<"vector1d operator() : arr of vector1d is not initialized"<<std::endl;
            throw 0;
        }
        double * a;
        if(arr == nullptr){
            a = (*pt3d).arr;
        } else {
            a = arr;
        }
        
        if(inx >= size){
            std::cerr<<"vector1d  operator() : index "<<inx<<" out of range : size = "<<size<<std::endl;
            throw 0;
        }
        return a[inx];
    }

    void setVal(int inx, double val){
        if(arr == nullptr && (*pt3d).arr == nullptr){
            std::cerr<<"vector1d setVal(int,double) : arr of vector1d is not initialized"<<std::endl;
            throw 0;
        }
        double * a;
        if(arr == nullptr){
            a = (*pt3d).arr;
        } else {
            a = arr;
        }
         if(inx >= size){
            std::cerr<<"vector1d  setVal(int,double) : index "<<inx<<" out of range : size = "<<size<<std::endl;
            throw 0;
        }
        a[inx] = val;
    }

    void setVal(std::vector<double> & v){
        if(v.size() != size){
            std::cerr<<"vector1d setVal(std::vector<double>) input vector has different dimensions with vector1d: "<<v.size()<<" != "<<size<<std::endl;
            throw 0;
        }
        for(int i=0;i<size;++i){
            arr[i] = v[i];
        }
    }

    void setVal(tensor3d & t3d){    /* reference of t3d */
        arr = nullptr; 
        pt3d = & t3d;   /* pointer of t3d */
        if(t3d.arr == nullptr){
            if(t3d.arrdim.d == 0 && t3d.arrdim.w == 0 && t3d.arrdim.h == 0){
                std::cerr<<"vector1d constructor: input param tensor3d is not initialized, have arrdim={0,0,0}"<<std::endl;
                throw 0;
            }else{
                t3d.setZero();
            }
        }
        size = t3d.arrdim.d * t3d.arrdim.w * t3d.arrdim.h;
    }

    void setZero(int size_val){
        if(pt3d != nullptr){
            std::cerr<<"vector1d setZero(int): this vector1d is initialized already with a tensor3d object. "<<std::endl;
            std::cerr<<"    use setZero() to initialize this vector1d (also initialize associated tensor3d object"<<std::endl;
            throw 0;
        }
        size = size_val;
        if(arr != nullptr){
            delete [] arr;
        }
        arr = new double [size] ();
    }

    /*void setZero(){
        if(arr == nullptr && pt3d == nullptr){
            std::cerr<<"vector1d setZero(): object should be initialized to use setZero(). initialize to zero using setZero(int) instead"<<std::endl;
            throw 0;
        }
        if(arr != nullptr){
            (*this).setZero(size);
        } else if(pt3d != nullptr){
            int d = (*pt3d).arrdim.d;
            int w = (*pt3d).arrdim.w;
            int h = (*pt3d).arrdim.h;
            if(d>0 && w>0 && h>0){
                (*pt3d).setZero(d,w,h);
            }else{
                std::cerr<<"vector1d setZero(): tensor3d object assigned to vector1d is not initialized: ";
                std::cerr<<"dimensions = "<<d<<" "<<w<<" "<<h<<std::endl;
                throw 0;
            }
        } else {
            std::cerr<<"vector1d setZero(): arr==nullptr && pt3d==nullptr : something's wrong. already should have been covered."<<std::endl;
            throw 0;
        }
    }*/

    void setUniformRandom(int size_val){
        if(pt3d != nullptr){
            std::cerr<<"vector1d setUniformRandom(int): this vector1d is initialized with a tensor3d."<<std::endl;
            std::cerr<<"    if you want to randomly initialize with the tensor3d, use setUniformRandom() instead"<<std::endl;
            throw 0;
        }
        (*this).setZero(size_val);

        std::mt19937 gen(123);
        std::uniform_real_distribution<> dis(-1, 1);

        for(int i=0;i<size_val;++i){
            (*this).setVal(i, dis(gen));
        }
    }

    /*void setUniformRandom(){
        if(arr!=nullptr){
            std::cerr<<"vector1d setUniformRandom(): this vector1d is initialized with internal arr."<<std::endl;
            std::cerr<<"    if you want to randomly initialize with internal arr, use setUniformRandom(int) instead"<<std::endl;
            throw 0;
        }
        dim3_t t_dim = (*pt3d).dim();
        (*pt3d).setUniformRandom(t_dim.d, t_dim.w, t_dim.h);
    }*/

    double max(){
         double maxVal = (*this)(0);
         double tmp;
         for(int i=1;i<size;++i){
             tmp = (*this)(i);
             if(tmp > maxVal){
                 maxVal = tmp;
             }
         }
         return maxVal;
    }
    
    void printVector(){
        std::cout<<"( ";
        for(int i=0;i<size;++i){
            std::cout<<(*this)(i)<<" ";
        }
        std::cout<<")"<<std::endl;
    }

    void saveToFile(std::string fileName){
        if(arr==nullptr){
            if(pt3d==nullptr){
                std::cerr<<"vector1d saveToFile(): arr and pt3d both nullptr, nothing to save"<<std::endl;
                throw 0;
            }else{
                std::cerr<<"vector1d saveToFile(): this vector1d is a proxy to a tensor3d - save the tensor3d instead directly"<<std::endl;
                throw 0;
            }
        }

        if(size==0){
            std::cerr<<"tensor4d saveToFile(): vector size is 0, nothing to store"<<std::endl;
            throw 1;
        }

        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(size);

        for (int i = 0; i < size; i++) {
            archive(arr[i]);
        }

        file.close();
        /*
        if(arr==nullptr){
            if(pt3d==nullptr){
                std::cerr<<"vector1d saveToFile(): arr and pt3d both nullptr, nothing to save"<<std::endl;
                throw 0;
            }else{
                std::cerr<<"vector1d saveToFile(): this vector1d is a proxy to a tensor3d - save the tensor3d instead directly"<<std::endl;
                throw 0;
            }
        }

        std::stringstream streamStorage;
        streamStorage<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[0];
        for(int i=1;i<size;++i){
            streamStorage<<" "<<std::fixed<<std::setprecision(std::numeric_limits<double>::max_digits10 +3)<<arr[i];
        }
        streamStorage<<"\n";

        std::ofstream outputFile (fileName.c_str());
        outputFile<<streamStorage.str();
        outputFile.close();
        */
    }

    void loadFromFile(std::string fileName){
        if(arr==nullptr){
            if(pt3d==nullptr){
                std::cerr<<"vector1d loadFromFile(): arr and pt3d both nullptr, nowhere to load"<<std::endl;
                throw 0;
            }else{
                std::cerr<<"vector1d loadFromFile(): this vector1d is a proxy to a tensor3d - load the tensor3d instead directly"<<std::endl;
                throw 0;
            }
        }

        std::ifstream infile(fileName.c_str(), std::ios::binary);
        if(!infile.is_open()){
            std::cerr<<"vector1d loadFromFile(std::string) cannot find(open) file "<<fileName<<std::endl;
            throw 1;
        }
        cereal::PortableBinaryInputArchive inarchive(infile);

        int loadSize;
        inarchive(loadSize);

        if(loadSize != size){
            std::cerr<<"vector1d loadFromFile(std::string) there are "<<loadSize<<" many values in "<<fileName<<" but curr vector1d object is size "<<size<<std::endl;
            throw 1;
        }
        
        for(int i=0;i<size;++i){
            inarchive(arr[i]);
        }

        infile.close();
        /*
        if(arr==nullptr){
            if(pt3d==nullptr){
                std::cerr<<"vector1d loadFromFile(): arr and pt3d both nullptr, nowhere to load"<<std::endl;
                throw 0;
            }else{
                std::cerr<<"vector1d loadFromFile(): this vector1d is a proxy to a tensor3d - load the tensor3d instead directly"<<std::endl;
                throw 0;
            }
        }
        std::ifstream inputFile (fileName.c_str());
        if(!inputFile.is_open()){
            std::cerr<<"vector1d loadFromFile(std::string): file "<<fileName<<" not found"<<std::endl;
            throw 0;
        }
        {
        int i=0;
        double val;
        while(inputFile>>std::fixed>>std::setprecision(std::numeric_limits<double>::max_digits10 +3)>>val){
            if(i==size){
                std::cerr<<"vector1d loadFromFile(std::string): there are more values in file "<<fileName<<" than size of arr of tensor3d obj"<<std::endl;
                throw 0;
            }
            arr[i] = val;
            ++i;
        }
        }
        inputFile.close();
        */
    }

};


vector1d * newZeroVector1dArr(int size, int arrSize){
    vector1d * v1darr = new vector1d [arrSize];
    for(int i=0;i<arrSize;++i){
        v1darr[i].setZero(size);
    }
    return v1darr;
}

vector1d * newVector1dArrFromTensor3dArr(tensor3d * t3darr, int arrSize){
    vector1d * v1darr = new vector1d [arrSize];
    for(int i=0;i<arrSize;++i){
        v1darr[i].setVal(t3darr[i]);
    }
    return v1darr;
}

class v1dAffineTransform{
    vector1d * x;
    vector1d * y;
    vector1d * dLdx;
    vector1d * dLdy;
    tensor3d * dLdW;
    vector1d * dLdb;

    int rows;
    int cols;

    tensor3d W;
    vector1d b;
    int batchSize;
    public:
    v1dAffineTransform (vector1d * y_prev, vector1d * y_curr,
                        vector1d * dLdy_prev, vector1d * dLdy_curr, tensor3d * dLdW_curr, vector1d * dLdb_curr, int batch_size) : 
                        x(y_prev), y(y_curr), 
                        dLdx(dLdy_prev), dLdy(dLdy_curr), dLdW(dLdW_curr), dLdb(dLdb_curr), batchSize(batch_size) {


        rows = y_curr[0].size;
        cols = y_prev[0].size;

        W.setUniformRandom(1,rows,cols);
        b.setUniformRandom(rows);
    }
    v1dAffineTransform (vector1d * y_prev, vector1d * y_curr, int batch_size) : x(y_prev), y(y_curr), batchSize(batch_size) {
        rows = y_curr[0].size;
        cols = y_prev[0].size;

        W.setUniformRandom(1,rows,cols);
        b.setUniformRandom(rows);
    }


    void setGradientTensors(vector1d * dLdy_prev, vector1d * dLdy_curr, tensor3d * dLdW_curr, vector1d * dLdb_curr){
        dLdx = dLdy_prev;
        dLdy = dLdy_curr;
        dLdW = dLdW_curr;
        dLdb = dLdb_curr;
        {
        int gradDim = dLdy_prev[0].size;
        int tensorDim = x[0].size;
        if(gradDim != tensorDim){
            std::cerr<<"v1affineTransform setGradientTensors : dimension of dLdy_prev does not match"<<std::endl;
            std::cerr<<"    "<<gradDim<<std::endl;
            std::cerr<<"    "<<tensorDim<<std::endl;
            throw 0;
        }
        }
        {
        int gradDim = dLdy_curr[0].size;
        int tensorDim = y[0].size;
        if(gradDim != tensorDim){
            std::cerr<<"v1affineTransform setGradientTensors : dimension of dLdy_curr does not match"<<std::endl;
            std::cerr<<"    "<<gradDim<<std::endl;
            std::cerr<<"    "<<tensorDim<<std::endl;
            throw 0;
        }
        }
        {
        dim3_t gradDim = dLdW[0].dim();
        dim3_t tensorDim = W.dim();
        if(gradDim != tensorDim){
            std::cerr<<"v1affineTransform setGradientTensors : dimension of dLdW does not match"<<std::endl;
            std::cerr<<"    "<<gradDim.d<<" "<<gradDim.w<<" "<<gradDim.h<<std::endl;
            std::cerr<<"    "<<tensorDim.d<<" "<<tensorDim.w<<" "<<tensorDim.h<<std::endl;
            throw 0;
        }
        }
        {
        int gradDim = dLdb[0].size;
        int tensorDim = b.size;
        if(gradDim != tensorDim){
            std::cerr<<"v1affineTransform setGradientTensors : dimension of dLdb does not match"<<std::endl;
            std::cerr<<"    "<<gradDim<<std::endl;
            std::cerr<<"    "<<tensorDim<<std::endl;
            throw 0;
        }
        }
    }


    void setW(std::vector<std::vector<std::vector<double>>> & Wvector){
        W.setVal(1,rows,cols,Wvector);
    }
    void setW(std::vector<std::vector<double>> & Wvector){
        dim3_t Winx;
        for(int row=0;row<rows;++row){
        for(int col=0;col<cols;++col){
            Winx = {0,row,col};
            W.setVal(Winx, Wvector[row][col]);
        }
        }
    }
    void setb(std::vector<double> & bvector){
        b.setVal(bvector);
    }

    void saveWToFile(std::string fileName){
        try{
            W.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"v1dAffineTransform saveWToFile("<<fileName<<"): error thrown with W.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void loadWFromFile(std::string fileName){
        try{
            W.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"v1dAffineTransform loadWFromFile("<<fileName<<"): error thrown with W.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void saveBToFile(std::string fileName){
        try{
            b.saveToFile(fileName);
        }catch(int errorInt){
            std::cerr<<"v1dAffineTransform saveBToFile("<<fileName<<"): error thrown with b.saveToFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }
    void loadBFromFile(std::string fileName){
        try{
            b.loadFromFile(fileName);
        }catch(int errorInt){
            std::cerr<<"v1dAffineTransform loadBFromFile("<<fileName<<"): error thrown with b.loadFromFile("<<fileName<<")"<<std::endl;
            throw errorInt;
        }
    }

    void printW(){ W.printMatrixForm(); }
    void printb(){ b.printVector(); }

    void affine(int batchInx=0) {

        dim3_t Winx;
        for(int row=0;row<rows;++row){
            double tmpVal = b(row);
        for(int col=0;col<cols;++col){
            Winx = {0,row,col};
            tmpVal += W(Winx) * x[batchInx](col);
        }
            y[batchInx].setVal(row, tmpVal);
        }
    }

    void computeGrad(int batchInx=0){
        
        /* compute dLdx */
        for(int c=0;c<cols;++c){
            double tmpVal=0;
        for(int r=0;r<rows;++r){
            dim3_t WInx = {0,r,c};
            tmpVal += W(WInx) * dLdy[batchInx](r);
        }
            dLdx[batchInx].setVal(c, tmpVal);
        }

        /* compute dLdW */
        for(int r=0;r<rows;++r){
        for(int c=0;c<cols;++c){
            dim3_t WInx = {0,r,c};
            dLdW[batchInx].setVal(WInx, dLdy[batchInx](r) * x[batchInx](c));
        }
        }

        /* compute dLdb */
        for(int r=0;r<rows;++r){
            dLdb[batchInx].setVal(r, dLdy[batchInx](r));
        }
    }

    void batchGD(double learnrate){
        if(batchSize>1){
            double tmp=0;
            dim3_t WInx;
            /* dLdW[0] is the batchwise average gradient */
            for(int r=0;r<rows;++r){
            for(int c=0;c<cols;++c){
                WInx = {0,r,c};
            for(int batchInx=0;batchInx<batchSize;++batchInx){
                tmp += dLdW[batchInx](WInx);
            }
                dLdW[0].setVal(WInx, tmp/batchSize);
                tmp=0;
            }
            }
            /* dLdb[0] is the batchwise average gradient */
            for(int r=0;r<rows;++r){
            for(int batchInx=0;batchInx<batchSize;++batchInx){
                tmp += dLdb[batchInx](r);
            }
                dLdb[0].setVal(r, tmp/batchSize);
                tmp=0;
            }
        }
        /* update W := W - learnrate * dLdW  */
        /* update b := b - learnrate * dLdb */
        dim3_t WInx;
        for(int r=0;r<rows;++r){
        for(int c=0;c<cols;++c){
            WInx = {0,r,c};
            W.setVal(WInx, W(WInx) - learnrate * dLdW[0](WInx));
        }
            b.setVal(r, b(r) - learnrate * dLdb[0](r));
        }
    }
};

class v1dsoftmax {
    vector1d * z;
    vector1d * y;
    vector1d * dLdz;
    vector1d * dLdy;
    int size;
    int batchSize;
    public:
    v1dsoftmax (vector1d * y_prev, vector1d * y_curr, vector1d * dLdy_prev, vector1d * dLdy_curr, int batch_size) : 
             z(y_prev), y(y_curr), dLdz(dLdy_prev), dLdy(dLdy_curr), batchSize(batch_size) {
        if(y_prev[0].size != y_curr[0].size){
            std::cerr<<"v1dsoftmax constructor: y_prev and y_curr should have same size: "<<y_prev[0].size<<" "<<y_curr[0].size<<std::endl;
        }
        size = z[0].size;
        
    }
    v1dsoftmax (vector1d * y_prev, vector1d * y_curr, int batch_size) :
                 z(y_prev), y(y_curr), batchSize(batch_size) {
        if(y_prev[0].size != y_curr[0].size){
            std::cerr<<"v1dsoftmax constructor: y_prev and y_curr should have same size: "<<y_prev[0].size<<" "<<y_curr[0].size<<std::endl;
        }
        size = z[0].size;
    }

    void setGradientTensors(vector1d * dLdy_prev, vector1d * dLdy_curr){
        dLdz = dLdy_prev;
        dLdy = dLdy_curr;
        {
        int gradDim = dLdz[0].size;
        int tensorDim = z[0].size;
        if(gradDim != tensorDim){
            std::cerr<<"v1dsoftmax setGradientTensors : dimension of dLdz does not match"<<std::endl;
            std::cerr<<"    "<<gradDim<<std::endl;
            std::cerr<<"    "<<tensorDim<<std::endl;
            throw 0;
        }
        }
        {
        int gradDim = dLdy[0].size;
        int tensorDim = y[0].size;
        if(gradDim != tensorDim){
            std::cerr<<"v1dsoftmax setGradientTensors : dimension of dLdy does not match"<<std::endl;
            std::cerr<<"    "<<gradDim<<std::endl;
            std::cerr<<"    "<<tensorDim<<std::endl;
            throw 0;
        }
        }
    }

    void softmax(int batchInx=0){     /* safe softmax */
        double maxz_val= z[batchInx].max();
        double sum_e_val=0;
        for(int j=0;j<size;++j){
            sum_e_val += std::exp(z[batchInx](j) - maxz_val);
        }

        for(int i=0;i<size;++i){
            y[batchInx].setVal(i, std::exp(z[batchInx](i) - maxz_val) / sum_e_val);
        }
    }

    void computeGrad(int batchInx=0){     /* dsoftmax(z_l)/dz_i = softmaz(z_l) * (delta(l,i) - softmax(z_i)) */
        for(int i=0;i<size;++i){
            double yVal = y[batchInx](i);
            double tmp=0;
            for(int l=0;l<size;++l){
                if(l == i){
                    tmp += dLdy[batchInx](l) * yVal * (1 - yVal);
                }else{
                    tmp += dLdy[batchInx](l) * (-1) * y[batchInx](l) * yVal;
                }
            }
            dLdz[batchInx].setVal(i, tmp);
        }
    }
    
};

class v1dCrossEntropyLoss{
    vector1d * y;
    vector1d * truth;
    vector1d * dLdy;
    int batchSize;
    public:
    v1dCrossEntropyLoss(vector1d * y_output, vector1d * y_truth, vector1d * dLdy_output, int batch_size) : 
                        y(y_output), truth(y_truth), dLdy(dLdy_output), batchSize(batch_size) {
        if(y[0].size != truth[0].size){
            std::cerr<<"v1dCrossEntropyLoss constructor: y and truth need to have same dimensions: "<<std::endl;
            std::cerr<<"    "<<y[0].size<<" != "<<truth[0].size<<std::endl;
            throw 0;
        }
    }
    v1dCrossEntropyLoss(vector1d * y_output, vector1d * y_truth, int batch_size) : 
                        y(y_output), truth(y_truth), batchSize(batch_size) {
        if(y[0].size != truth[0].size){
            std::cerr<<"v1dCrossEntropyLoss constructor: y and truth need to have same dimensions: "<<std::endl;
            std::cerr<<"    "<<y[0].size<<" != "<<truth[0].size<<std::endl;
            throw 0;
        }
    }
    void setGradientTensors(vector1d * dLdy_output){
        dLdy = dLdy_output;
        if(y[0].size != dLdy[0].size){
            std::cerr<<"v1dCrossEntropyLoss setGradientTensors : dLdy and y have different dimensions: "<<std::endl;
            std::cerr<<"    "<<dLdy[0].size<<" != "<<y[0].size<<std::endl;
            throw 0;
        }
    }

    double avgloss(){       /* absolute value of loss average over categories and batch */
        int dimSize = y[0].size;
        double averageloss=0;
        for(int batchInx=0;batchInx<batchSize;++batchInx){
            double tmp=0;
            for(int i=0;i<dimSize;++i){
                /*tmp += truth[batchInx](i) * std::log(y[batchInx](i));*/
                if(truth[batchInx](i) == 1){
                    tmp += std::log(y[batchInx](i));
                }
            }
            averageloss += tmp;
        }
        averageloss /= batchSize;
        return (-1) * averageloss;
    }

    double loss(int batchInx=0){
        int dimSize = y[batchInx].size;
        double averageloss=0;

        double tmp=0;
        for(int i=0;i<dimSize;++i){
            /*tmp += truth[0](i) * std::log(y[0](i));*/
            if(truth[batchInx](i) == 1){
                tmp += std::log(y[batchInx](i));
            }
        }
        averageloss += tmp;

        return (-1) * averageloss;
    }

    int accuratePrediction(int batchInx=0){
        int dimSize = y[batchInx].size;
        int truthInx=0;
        while(truthInx<dimSize && truth[batchInx](truthInx)!=1){
            ++truthInx;
        }

        int predictInx=0;
        if(dimSize>1){
            int tmpInx=0;
            while(tmpInx<dimSize){
                if(y[batchInx](tmpInx) > y[batchInx](predictInx)){
                    predictInx = tmpInx;
                }
                ++tmpInx;
            }
        }

        if(truthInx==predictInx){
            return 1;
        }else{
            return 0;
        }
    }

    void computeGrad(int batchInx=0){
        int sizeVal = dLdy[0].size;
        for(int i=0;i<sizeVal;++i){
            double tVal = truth[batchInx](i);
            double yVal = y[batchInx](i);
            if(tVal == 0.0){
                dLdy[batchInx].setVal(i, 0);
            }else{
                dLdy[batchInx].setVal(i, (-1) * tVal / yVal);
            }
        }
    }
};

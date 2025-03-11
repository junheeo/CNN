#include <iostream>
#include <fstream>
#include <string>
#include "../../external/cereal/include/cereal/archives/portable_binary.hpp"
#include "../../external/cereal/include/cereal/types/vector.hpp"

int testcereal1();
int testcereal2();

int main(){
    /*testcereal1();*/
    testcereal2();
    return 0;
}

int testcereal1() {
    // Initialize the array size
    int arrSize = 5;
    
    // Dynamically allocate an array of doubles
    double* arr = new double[arrSize];
    
    // Initialize the array with values 0.1, 0.2, 0.3, 0.4, 0.5
    for (int i = 0; i < arrSize; i++) {
        arr[i] = 0.1 * (i + 1);
    }
    
    // Print the array contents before serialization
    std::cout << "Array contents before serialization:" << std::endl;
    for (int i = 0; i < arrSize; i++) {
        std::cout << "arr[" << i << "] = " << arr[i] << std::endl;
    }
    
    // Open an output file stream for binary writing
    std::ofstream file("arr_vals.bin", std::ios::binary);
    
    // Create a portable binary output archive
    cereal::PortableBinaryOutputArchive archive(file);
    
    // First, save the array size
    archive(arrSize);
    
    // Then, save the array contents
    // We need to serialize each element individually since cereal doesn't
    // directly support raw pointers
    for (int i = 0; i < arrSize; i++) {
        archive(arr[i]);
    }
    
    // Close the file
    file.close();
    
    std::cout << "\nArray has been serialized to 'arr_vals.bin'" << std::endl;
    
    // Free the dynamically allocated memory
    delete[] arr;
    
    // Now let's deserialize to verify it worked
    std::ifstream infile("arr_vals.bin", std::ios::binary);
    cereal::PortableBinaryInputArchive inarchive(infile);
    
    // Read the array size
    int loadedSize;
    inarchive(loadedSize);
    
    // Allocate a new array
    double* loadedArr = new double[loadedSize];
    
    // Load each element
    for (int i = 0; i < loadedSize; i++) {
        inarchive(loadedArr[i]);
    }
    
    // Print the loaded array
    std::cout << "\nArray contents after deserialization:" << std::endl;
    for (int i = 0; i < loadedSize; i++) {
        std::cout << "loadedArr[" << i << "] = " << loadedArr[i] << std::endl;
    }
    
    // Free the second dynamically allocated array
    delete[] loadedArr;
    
    return 0;
}

class myClass{
    double * arr;
    int arrSize;
    public:
    myClass(){
        arrSize = 0;
        arr = nullptr;
    }
    myClass(int size){      /* constructor */
        arrSize = size;
        arr = new double [arrSize];
        for (int i = 0; i < arrSize; i++) {
            arr[i] = 0.1 * (i + 11);
        }
    }
    ~myClass(){
        delete [] arr;
        arrSize = 0;
    }
    

    void storeArr(std::string fileName){
        std::ofstream file(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryOutputArchive archive(file);

        archive(arrSize);   /* save arrSize */

        for (int i = 0; i < arrSize; i++) {
            archive(arr[i]);    /* save content of arr */
        }
        
        // Close the file
        file.close();
        
        std::cout << "\nArray has been serialized to " << fileName << std::endl;
    }

    void loadArr(std::string fileName){
        std::ifstream infile(fileName.c_str(), std::ios::binary);
        cereal::PortableBinaryInputArchive inarchive(infile);

        inarchive(arrSize);

        if(arr != nullptr){
            delete [] arr;
        }
        arr = new double [arrSize];


        for(int i=0;i<arrSize;++i){
            inarchive(arr[i]);
        }

        infile.close(); 
    }

    void printArr(){
        std::cout<<"arr =";
        for(int i=0;i<arrSize;++i){
            std::cout<<" "<<arr[i];
        }
        std::cout<<std::endl;
    }
};

int testcereal2(){

    myClass obj1 {7};
    obj1.storeArr("arr_vals.bin");

    myClass obj2 {};
    obj2.loadArr("arr_vals.bin");
    std::cout<<"obj2.";
    obj2.printArr();

    return 0;
}

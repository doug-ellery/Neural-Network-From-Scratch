#include "DataProcessing.h"
#include <vector>
#include <fstream>
#include <iostream>

//for converting the 4 4-byte values at the beginning from big-endian to little-endian
static uint32_t reverseBytes(uint32_t val){
    return ((val & 0xFF) << 24) | ((val & 0xFF00) << 8) |
        ((val & 0xFF0000) >> 8) | ((val >> 24) & 0xFF);
}


std::vector<float> get_MNIST_Images(const std::string& path){
    //use std::ios::binary for the file so we don't interpret the file as being characters
    //which could lead to us accidentally reading an EOF
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        std::cerr<<"Failed to open: "<<path<<"\n";
        exit(1);
    }
    //skip magic num at beginning
    file.seekg(4);
    //read num_images, rows, then cols; use reinterpret_cast because the 
    //read API requires a char * as the first argument
    uint32_t num_images = 0;
    file.read(reinterpret_cast<char*>(&num_images), 4);
    num_images = reverseBytes(num_images);
    uint32_t rows = 0;
    file.read(reinterpret_cast<char*>(&rows), 4);
    rows = reverseBytes(rows);
    uint32_t cols = 0;
    file.read(reinterpret_cast<char*>(&cols), 4);
    cols = reverseBytes(cols);
    //now read all bytes into out buffer, where 1 byte = 1 pixel
    //For MNIST the size of this buff should come out to be 60000 * 28 * 28
    std::vector<uint8_t> buff(num_images * rows * cols);
    file.read(reinterpret_cast<char*>(buff.data()), buff.size());
    std::vector<float> images(num_images * rows * cols);
    //push the normalized value of pixel into the output [0, 1.0]
    for(size_t i = 0; i < buff.size(); i++){
        images[i] = buff[i] / 255.0f;
    }
    return images;
}

std::vector<float> get_MNIST_Labels(const std:: string& path){
    //use std::ios::binary for the file so we don't interpret the file as being characters
    //which could lead to us accidentally reading an EOF
    std::ifstream file(path, std::ios::binary);
    if(!file.is_open()){
        std::cerr<<"Failed to open: "<<path<<"\n";
        exit(1);
    }
    //skip magic num at beginning
    file.seekg(4);
    //read num_labels first
    uint32_t num_labels = 0;
    file.read(reinterpret_cast<char*>(&num_labels), 4);
    num_labels = reverseBytes(num_labels);
    //read each label into a buff, each label = 1 byte
    std::vector<uint8_t> buff(num_labels);
    file.read(reinterpret_cast<char*>(buff.data()), buff.size());
    //convert labels from digits to 0's and 1's (the 1 represent the correct digit, everything else is a zero for a group of 10 floats)
    //a little bit of hard coding here with the 10 cuz I know that MNIST is just 0-9
    std::vector<float> proper_labels(10*num_labels);
    for(uint32_t buff_label = 0; buff_label < num_labels; buff_label++){
        for(int j = 0; j < 10; j++){
            //This line is a little bit weird, but basically for whatever group of 10 floats we are in, we want the digit at the index 
            //of buff[buff_label] to be a one to tell us which digit it is
            proper_labels[10 * buff_label + j] = j == buff[buff_label] ? 1.0f : 0.0f;
        }
    }
    return proper_labels;
}



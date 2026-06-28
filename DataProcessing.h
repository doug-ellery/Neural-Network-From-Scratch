#include <vector>
#include <string>

#ifndef DATAPROCESSING_H
#define DATAPROCESSING_H

std::vector<float> get_MNIST_Images(const std::string& path);
std::vector<float> get_MNIST_Labels(const std:: string& path);

#endif
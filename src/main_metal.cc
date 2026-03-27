#include "layers.hpp"
#include <iostream>
#include <fstream>
#include <memory>
#include <vector>

#define H 10
#define W 10
#define D 3

/* Print a tensor */
void print_tensor(const float* t, int depth, int height, int width) {
    for (int d = 0; d < depth; d++) {
        std::cout << "[\n";
        for (int h = 0; h < height; h++) {
            std::cout << "  [ ";
            for (int w = 0; w < width; w++) {
                std::cout << t[d * height * width + h * width + w];
                if (w < width - 1) std::cout << ", ";
            }
            std::cout << " ]\n";
        }
        std::cout << "]\n";
    }
}

/* Read a tensor from a file */
void read_from_file(std::string fname, float* data_buffer, size_t total_bytes) {
    std::ifstream file(fname, std::ios::binary);
    if (!file) {
        std::cerr << "Failed to open " << fname << std::endl;
        return;
    }
    file.read(reinterpret_cast<char*>(data_buffer), total_bytes);
    if (!file) {
        std::cerr << "Failed to read data from " << fname << std::endl;
        return;
    }
    std::cout << "Read " << total_bytes << " bytes from " << fname << std::endl;
    file.close();
}

/* Backtest network */
void test_network() {
    Tensor x({1, D, W, H});
    Tensor r_0({1, D, W, H});
    Tensor r_1({1, D, W, H});
    Conv2DLayer conv2d(3, 1);
    ReLULayer relu;
    MaxPoolLayer max_pool(3);

    std::vector<std::unique_ptr<Layer>> layers;
    // First block
    layers.push_back(std::make_unique<Conv2DLayer>(3, 1));
    layers.push_back(std::make_unique<ReLULayer>());
    layers.push_back(std::make_unique<MaxPoolLayer>(2, 2));

    read_from_file("/Users/matteo/Programming/C++/cnn/res/input.bin", x.data, D * W * H * sizeof(float));
    read_from_file("/Users/matteo/Programming/C++/cnn/res/filter.bin", conv2d.get_weights().data, 3 * 3 * sizeof(float));

    conv2d.forward(x, r_0);
    relu.forward(r_0, r_1);
    max_pool.forward(r_1, r_0);

    // std::cout << "Tensor x: \n";
    // print_tensor(x.data, D, W, H);
    // std::cout << "Weights: \n";
    // print_tensor(conv2d.get_weights().data, 1, 3, 3);
    // std::cout << "Conv2D result: \n";
    // print_tensor(r_0.data, D, W, H);
    // std::cout << "ReLU result: \n";
    // print_tensor(r_1.data, D, W, H);
    // std::cout << "MaxPool result: \n";
    // print_tensor(r_0.data, D, W, H);
}

int main() {
    test_network();
}

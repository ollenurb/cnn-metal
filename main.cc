#include "tensor.hpp"
#include <fstream>
#include <array>
#include <iostream>
#include <string>
#include <cstdint>

using BytesBuffer = std::array<char, 4>;

// Takes a 4 bytes buffer to convert it to its corresponding uint32 value
uint32_t bytes_to_int(const BytesBuffer& buffer) {
    return (static_cast<uint32_t>(buffer[0] & 0xFF) << 24) |
           (static_cast<uint32_t>(buffer[1] & 0xFF) << 16) |
           (static_cast<uint32_t>(buffer[2] & 0xFF) << 8) |
           (static_cast<uint32_t>(buffer[3] & 0xFF));
}

Tensor<3> load_data(const std::string& path) {
    std::ifstream file(path);
    BytesBuffer file_buffer;

    uint32_t dataset_size = 0;
    uint32_t image_width = 0;
    uint32_t image_height = 0;

    // Read header
    file.seekg(4, std::ios::cur);
    file.read(file_buffer.data(), file_buffer.size());
    dataset_size = bytes_to_int(file_buffer);
    file.read(file_buffer.data(), file_buffer.size());
    image_width = bytes_to_int(file_buffer);
    file.read(file_buffer.data(), file_buffer.size());
    image_height = bytes_to_int(file_buffer);

    // Read images
    // std::vector<char> images(dataset_size * image_width * image_height);
    size_t images_bytes = dataset_size * image_width * image_height;
    char* images = new char[images_bytes];
    file.read(images, images_bytes);
    // We need to convert bytes to actual floating point values
    Real* buffer = new Real[images_bytes];
    for (size_t i = 0; i < images_bytes; i++) {
        buffer[i] = static_cast<Real>(images[i]);
    }
    delete[] images;
    // TODO: return Tensor.frombuffer(data)
    return Tensor<3>::from_buffer(buffer, {dataset_size, image_width, image_height});
}

// Test program driver
int main(int argc, char** argv) {
    if (argc < 2) return -1;
    std::cout << "Reading " << argv[1] << std::endl;
    auto dataset = load_data(argv[1]);
    std::cout << dataset << std::endl;

    auto print_image = [](auto &image) {
        size_t height = image.shape()[0];
        size_t width = image.shape()[1];
        for (size_t h = 0; h < height; h++) {
            for (size_t w = 0; w < width; w++) {
                std::cout << (image(w, h) > 0 ? "#" : " ");
            }
            std::cout << "\n";
        }
    };

    for (size_t i = 0; i < 10; i++) {
        auto image = dataset(i);
        print_image(image);
    }
}

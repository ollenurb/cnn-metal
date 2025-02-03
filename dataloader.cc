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

void print_image(
    const std::vector<char>& images,
    uint32_t width,
    uint32_t height,
    uint32_t image_number
) {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t pixel = static_cast<uint8_t>(images[(image_number * width * height) + ((y * width) + x)]);
            std::cout << ((pixel > 0) ? "#" : ".");
        }
        std::cout << std::endl;
    }
}

void load_data(const std::string& path) {
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
    auto image_size = image_width * image_height;
    // TODO: return Tensor.frombuffer(data)
}

// Test program driver
int main(int argc, char** argv) {
    if (argc < 2) return -1;
    std::cout << "Reading " << argv[1] << std::endl;
    load_data(argv[1]);
}

#include <algorithm>
#include <array>
#include <functional>
#include <iomanip>
#include <numeric>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <cstddef>
#include <iostream>
#include <utility>

// Define Precision
using Real = float;

// Matrix multiplication kernel
inline void matmul_kernel(
    Real* res,
    Real* a,
    Real* b,
    size_t rows_a,
    size_t cols_a,
    size_t rows_b,
    size_t cols_b // One of these arguments is useless but we keep it for clarity
) {
    for (size_t r = 0; r < rows_a; ++r) {
        for (size_t c = 0; c < cols_b; ++c) {
            for (size_t k = 0; k < cols_a; ++k) {
                res[(r * cols_b) + c] += a[(r * cols_a) + k] * b[(k * cols_b) + c];
            }
        }
    }
}

// Tensor class definition. It's just a multi-dimensional view over a linear memory chunk.
// Data is accessed in a row-major fashion.
template <size_t Rank>
class Tensor {
private:
    size_t offset_;
    std::array<size_t, Rank> shape_;
    std::array<size_t, Rank> strides_;
    std::shared_ptr<Real[]> data_;

    // Check for shape consistency
    void validate_shape() {
        if (shape_.empty()) throw std::invalid_argument("Shape cannot be empty");
        for (auto dim : shape_) {
            if (dim == 0) throw std::invalid_argument("Dimension cannot be zero-sized");
        }
    }

    // Compute strides for this Tensor
    const void compute_strides() {
        size_t stride = 1;
        for (size_t i = Rank; i > 0; i--) {
            strides_[i-1] = stride;
            stride *= shape_[i-1];
        }
    }

public:
    // Private constructor for creating new tensors from shapes
    Tensor(const std::array<size_t, Rank>& shape) : offset_(0), shape_(shape), strides_{} {
        validate_shape();
        compute_strides();
        size_t read_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        // Create the chunk and assign it to the shared_ptr member
        data_.reset(new Real[read_size]);
    }

    // Private constructor for creating views
    Tensor(
        std::shared_ptr<Real[]>& data,
        size_t offset,
        const std::array<size_t, Rank>& shape,
        const std::array<size_t, Rank>& strides
    ) : data_(data), offset_(offset), shape_(shape), strides_(strides) {}

    template <typename... Args>
    static Tensor<Rank> create(Args&&... shape) {
        size_t offset = 0;
        return Tensor({ static_cast<size_t>(std::forward<Args>(shape))... });
    }

    // Full indexing - returns element reference
    template <typename... Args>
    requires (sizeof...(Args) == Rank)
    Real& operator()(Args... indices) {
        const std::array<size_t, Rank> idx = {static_cast<size_t>(indices)...};
        size_t offset = offset_;
        for (size_t i = 0; i < Rank; ++i) {
            if (idx[i] >= shape_[i]) {
                throw std::out_of_range("Index out of bounds");
            }
            offset += idx[i] * strides_[i];
        }
        return data_.get()[offset];
    }

    // Partial indexing - returns new tensor view
    template <typename... Args>
    requires (sizeof...(Args) < Rank)
    auto operator()(Args... indices) {
        const size_t NewRank = Rank - sizeof...(Args);
        const std::array<size_t, sizeof...(Args)> idx = {static_cast<size_t>(indices)...};
        size_t new_offset = offset_;
        // Compute new offset
        new_offset = std::inner_product(idx.begin(), idx.end(), strides_.begin(), offset_);
        // Compute new strides and shape
        std::array<size_t, NewRank> new_shape;
        std::array<size_t, NewRank> new_strides;

        std::copy(shape_.begin() + sizeof...(Args), shape_.end(), new_shape.begin());
        std::copy(strides_.begin() + sizeof...(Args), strides_.end(), new_strides.begin());
        // Return new Tensor view
        return Tensor<NewRank>(data_, new_offset, new_shape, new_strides);
    }

    // String representation
    friend std::ostream& operator<<(std::ostream& os, const Tensor& t) {
        os << "Tensor<" << Rank << ">(shape=[";
        for (size_t i = 0; i < Rank; ++i) {
            os << t.shape_[i];
            if (i < Rank - 1) os << ",";
        }
        os << "], strides=[";
        for (size_t i = 0; i < Rank; ++i) {
            os << t.strides_[i];
            if (i < Rank - 1) os << ",";
        }
        os << "])";
        return os;
    }

    // Matrix multiplication
    Tensor<Rank> matmul(const Tensor<Rank>& other) {
        // Check shape consistency
        if(shape_[Rank - 1] != other.shape()[Rank - 2]) {
            std::stringstream ss;
            ss << "Dimensions don't match. Got (M x "
                << shape_[Rank - 1] << ") @ ("
                << other.shape()[Rank - 2] << " x N)";
            throw std::invalid_argument(ss.str());
        }

        std::array<size_t, Rank> new_shape = shape_;
        new_shape[Rank - 1] = other.shape()[Rank - 1];
        auto result = Tensor<Rank>(new_shape);
        // For ranks higher than 2, we need to loop through the batch dimensions
        for (size_t r = 3; r <= Rank; r++) {
            if (shape_[Rank - r] != other.shape()[Rank - r]) {
                std::stringstream ss;
                ss << "Batch dimensions don't match at dim " << Rank - r
                    << ". Got " << shape_[Rank - r]
                    << " vs " << other.shape()[Rank - r];
                throw std::invalid_argument(ss.str());
            }
        }
        // Compute the total batch size by multiplying all dimensions except the last 2
        size_t batch_size = 1;
        for (size_t i = 0; i < Rank - 2; ++i) {
            batch_size *= shape_[i];
        }
        // For each batch...
        for (size_t batch = 0; batch < batch_size; ++batch) {
            // We need to compute the offsets for this batch
            size_t this_offset = offset_;
            size_t other_offset = other.offset_;
            size_t result_offset = result.offset_;
            // We basically compute the starting position of each matrix by
            // generating an index with Rank - 2. We use the % operation to
            // retrieve the index value of the current dimension
            for (size_t dim = 0; dim < Rank - 2; ++dim) {
                size_t idx = batch % shape_[dim];
                this_offset += idx * strides_[dim];
                other_offset += idx * other.strides_[dim];
                result_offset += idx * result.strides_[dim];
            }
            // Multiply the matrices, storing the result into the result Tensor's data buffer
            matmul_kernel(
                (result.data_.get() + result_offset),
                (data_.get() + this_offset),
                (other.data_.get() + other_offset),
                shape_[Rank - 2],
                shape_[Rank - 1],
                other.shape_[Rank - 2],
                other.shape_[Rank - 1]
            );
        }
        return result;
    }

    // Check the broadcasting conditions on the shape
    static bool is_broadcastable(const Tensor<Rank>& a, const Tensor<Rank>& b) {
        bool req_empty = false;
        // Check that dimensions are the same until we see the first dim = 1
        for (size_t i = Rank; i > 0; i--) {
            auto& cur_a = a.shape_[i - 1];
            auto& cur_b = b.shape_[i - 1];
            // Check empty
            if (req_empty) {
                if (cur_b != 1) return false;
            }
            // Check dims equals
            else {
                if (cur_a != cur_b) {
                    if (cur_b == 1) req_empty = true; else return false;
                }
            }
        }
        return true;
    }

    // Linearwise operators. These operators support broadcasting in the sense
    // that the other Tensor rank have only outer shape dimensions of size 1.
    // eg. Tensor(5, 4) + Tensor(4) will broadcast in Tensor(5, 4) + Tensor(1,
    // 4) where the Tensor is repeated along dim 0.

    template <size_t NewShape>
    Tensor<NewShape> broadcast(const Tensor<NewShape>& other) {

    }

    // Sum two tensors. Broadcasts dimensions along the axis that are 1.
    Tensor<Rank> operator+(const Tensor<Rank>& other) {
        // Check shape compatibility with broadcasting
        if (!is_broadcastable(*this, other)) {
            std::stringstream ss;
            ss << "Tensors " << *this << " and " << other << " are not broadcastable";
            throw std::invalid_argument(ss.str());
        }

        // Create result tensor with max of each dimension
        auto result = Tensor<Rank>(shape_);
        auto total_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        auto other_size = std::accumulate(other.shape_.begin(), other.shape_.end(), 1, std::multiplies<>());

        // Get raw buffer pointers
        float *const this_data = data_.get();
        float *const other_data = other.data_.get();
        float *const result_data = result.data_.get();

        for (size_t idx = 0; idx < total_size; ++idx) {
            auto other_idx = idx % other_size;
            // std::cout << "this_data[" << other_idx << "] - other_data[" << idx << "]\n";
            result_data[idx] = this_data[idx] + other_data[other_idx];
        }

        return result;
    }


    // Tensor properties
    const std::array<size_t, Rank>& shape() const { return shape_; }
    const std::array<size_t, Rank>& strides() const { return strides_; }
    size_t ndim() const { return shape_.size(); }
    size_t rank() const { return Rank; }
    const std::shared_ptr<Real[]>& raw_ptr() { return data_; };

    void print() const {
        auto data = data_.get();
        // Helper function for recursive dimension printing
        std::function<void(size_t, const std::array<size_t, Rank>&)> print_dim;
        print_dim = [&](size_t dim, const std::array<size_t, Rank>& current_idx) {
            // Base case, print the current element
            if (dim == Rank) {
                size_t linear_idx = std::inner_product(current_idx.begin(), current_idx.end(), strides_.begin(), offset_);
                std::cout << std::fixed << std::setprecision(4) << data[linear_idx] << " ";
                return;
            }
            std::cout << "[";

            // Recursive case, go to each element of the current dimension
            for (size_t i = 0; i < shape_[dim]; ++i) {
                if (i > 0 && dim == Rank-1) std::cout << " ";

                std::array<size_t, Rank> next_idx = current_idx;
                next_idx[dim] = i;
                print_dim(dim + 1, next_idx);

                if (i < shape_[dim] - 1) {
                    if (dim == Rank-1) std::cout << " ";
                    else {
                        std::cout << ",\n";
                        // Add spaces to "tab" the current dimension
                        for (size_t j = 0; j <= dim; j++) std::cout << " ";
                    }
                }
            }

            std::cout << "]";
            if (dim == 0) std::cout << "\n";
        };

        // Start recursive printing
        std::array<size_t, Rank> start_idx{};
        print_dim(0, start_idx);
    }
};

// Wrapper that automatically infer the rank of the tensor.
// For example, it avoids having to call:
// Tensor<3>::create(1, 2, 3), Tensor<2>::create(1, 2) etc..
template<typename... Args>
auto create(Args&&... args) {
    return Tensor<sizeof...(Args)>::create(std::forward<Args>(args)...);
}

int main(int argc, char** argv) {
    auto a = create(2, 2, 3, 4);
    auto b = create(1, 1, 1, 4);

    for (int i = 0; i < 4; i++) {
        b(0, 0, 0, i) = i + 1;
    }
    std::cout << std::setfill('=') << std::setw(50) << " B " << std::setfill('=') << std::setw(50) << "\n";
    b.print();

    for (int l = 0; l < 2; l++) {
        for (int k = 0; k < 2; k++) {
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 4; j++) {
                    a(l, k, i, j) = (k + 1) * (i + j);
                }
            }
        }
    }

    std::cout << std::setfill('=') << std::setw(50) << " A " << std::setfill('=') << std::setw(50) << "\n";
    a.print();

    std::cout << std::setfill('=') << std::setw(50) << " C " << std::setfill('=') << std::setw(50) << "\n";
    auto c = a + b;
    c.print();
}

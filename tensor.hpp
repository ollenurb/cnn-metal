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

// Apply elementwise the op to each element of a and b, saving the result to
// res. Broadcasted version of indexing assumes that b has been broadcasted to
// a.
inline void broadcasted_ewise_kernel(
    Real* res,
    Real* a,
    Real* b,
    std::function<Real(Real&, Real&)> op,
    size_t size_a,
    size_t size_b
) {
    for (size_t idx = 0; idx < size_a; idx++) {
        size_t b_idx = idx % size_b;
        res[idx] = op(a[idx], b[b_idx]);
    }
}

inline void scalar_ewise_kernel(
    Real *res,
    Real* a,
    Real b,
    std::function<Real(Real&, Real&)> op,
    size_t size_a
) {
    for (size_t idx = 0; idx < size_a; idx++) {
        res[idx] = op(a[idx], b);
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
    // Constructor that creates new tensors from shapes
    Tensor(const std::array<size_t, Rank>& shape) : offset_(0), shape_(shape), strides_{} {
        validate_shape();
        compute_strides();
        size_t read_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        // Create the chunk and assign it to the shared_ptr member
        data_.reset(new Real[read_size]);
    }

    Tensor(Real* data, const std::array<size_t, Rank>& shape) : offset_(0), shape_(shape), strides_{} {
        validate_shape();
        compute_strides();
        size_t read_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        // Create the chunk and assign it to the shared_ptr member
        data_.reset(data);
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

    // Array creation routine from raw buffer
    // ! This can potentially cause memory problems because the caller may
    // retain the raw pointer. From now on we assume that the resulting tensor
    // is the only borrower of this memory chunk, so it's totally fine to free
    // the memory after the last view on this raw buffer is freed.
    static Tensor<Rank> from_buffer(Real* buffer, const std::array<size_t, Rank> shape) {
        return Tensor(buffer, shape);
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


    // Linearwise operator that support broadcasting between tensors of
    // different shapes. Reading from right to left: dimensions must either
    // match exactly or one must be 1. Once a dimension of 1 is encountered in
    // the second tensor (other), all remaining dimensions to the left must also
    // be 1. Example: [2,2,3,4] can broadcast with [1,1,1,4] but not with
    // [1,2,1,4]. It is slightly different (and less flexible) than NumPy and.
    // Pytorch broadcasting rules but it allows computing broadcasting indices
    // more easily.
    Tensor<Rank> ewise_op(const Tensor<Rank>& other, std::function<Real(Real, Real)> op) {
        // Check shape compatibility with broadcasting rules
        if (!is_broadcastable(*this, other)) {
            std::stringstream ss;
            ss << "Tensors " << *this << " and " << other << " are not broadcastable";
            throw std::invalid_argument(ss.str());
        }

        // Create result tensor
        auto result = Tensor<Rank>(shape_);
        auto result_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        auto other_size = std::accumulate(other.shape_.begin(), other.shape_.end(), 1, std::multiplies<>());

        // Call kernel on raw buffer pointers
        broadcasted_ewise_kernel(
            result.data_.get(),
            data_.get(),
            other.data_.get(),
            op,
            result_size,
            other_size
        );

        return result;
    }

    Tensor<Rank> ewise_op(Real other, std::function<Real(Real, Real)> op) {
        auto result = Tensor<Rank>(shape_);
        auto result_size = std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<>());
        scalar_ewise_kernel(result.data_.get(), data_.get(), other, op, result_size);
        return result;
    }

    Tensor<Rank> operator+(const Tensor<Rank>& other) {
        return ewise_op(other, std::plus<Real>());
    }

    Tensor<Rank> operator*(const Tensor<Rank>& other) {
        return ewise_op(other, std::multiplies<Real>());
    }

    Tensor<Rank> operator/(const Tensor<Rank>& other) {
        return ewise_op(other, std::divides<Real>());
    }

    Tensor<Rank> operator+(Real other) {
        return ewise_op(other, std::plus<Real>());
    }

    Tensor<Rank> operator*(Real other) {
        return ewise_op(other, std::multiplies<Real>());
    }

    Tensor<Rank> operator/(Real other) {
        return ewise_op(other, std::divides<Real>());
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

namespace tensor {
    // Wrapper that automatically infer the rank of the tensor.
    // For example, it avoids having to call:
    // Tensor<3>::create(1, 2, 3), Tensor<2>::create(1, 2) etc..
    template<typename... Args>
    auto create(Args&&... args) {
        return Tensor<sizeof...(Args)>::create(std::forward<Args>(args)...);
    }
}

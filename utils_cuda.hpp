
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include "Matrix.hpp"

enum class Operation { ADD, SUBT };

template<typename T>
std::vector<T> matrix_vec_to_vec(const std::vector<Matrix<T>>& _v) {
    std::vector<T> a;
    a.reserve(_v.size() * _v[0].rows() * _v[0].cols());

    for (const auto& m : _v) {
        auto f = m.flat();
        a.insert(a.end(), f.cbegin(), f.cend());
    }

    return a;
}

template<typename T>
std::vector<Matrix<T>> vec_to_matrices_vec(const T* _data, const size_t& _rows, const size_t& _cols, const size_t& _batch) {
    std::vector<Matrix<T>> v;
    v.reserve(_batch);

    for (size_t b = 0, i = 0, r = 0, c = 0; b < _batch; ++b) {
        auto m = Matrix<T>(_rows, _cols);
        for (r = 0; r < _rows; ++r) {
            for (c = 0; c < _cols; ++c, ++i) {
                m[r][c] = _data[i];
            }
        }
        v.emplace_back(m);
    }

    return v;
}

void check_cuda_error(cudaError_t _err, const std::string _msg) {
    if (_err != cudaSuccess) {
        std::cerr << _msg << " - " << cudaGetErrorString(_err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

namespace CU {

    template<typename T>
    __global__ void dot_product_kernel_function(T* _c, const T* _a, const T* _b, const size_t _rows_a, const size_t _K, const size_t _cols_b) {
        int row = blockIdx.y * blockDim.y + threadIdx.y;
        int col = blockIdx.x * blockDim.x + threadIdx.x;

        if (row < _rows_a && col < _cols_b) {
            T sum = 0;

            for (int i = 0; i < _K; ++i)
                sum += _a[row * _K + i] * _b[i * _cols_b + col];

            _c[row * _cols_b + col] = sum;
        }
    }

    template<typename T>
    __global__ void add_kernel_function(T* _c, const T* _a, const T* _b, const size_t _rows, const size_t _cols) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < _rows && col < _cols) {
            int index = row * _cols + col;
            _c[index] = _a[index] + _b[index];
        }
    }

    template<typename T>
    __global__ void scalar_kernel_function(T* _b, const T* _a, const T _s, const size_t _rows, const size_t _cols) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < _rows && col < _cols) {
            int index = row * _cols + col;
            _b[index] = _a[index] * _s;
        }
    }

    template<typename T, typename Functor>
    __global__ void function_kernel_function(T* _b, const T* _a, const Functor _fu, const size_t _rows, const size_t _cols) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < _rows && col < _cols) {
            int index = row * _cols + col;
            _b[index] = _fu(_a[index]);
        }
    }

    template<typename T>
    __global__ void subt_kernel_function(T* _c, const T* _a, const T* _b, const size_t _rows, const size_t _cols) {
        int col = blockIdx.x * blockDim.x + threadIdx.x;
        int row = blockIdx.y * blockDim.y + threadIdx.y;

        if (row < _rows && col < _cols) {
            int index = row * _cols + col;
            _c[index] = _a[index] - _b[index];
        }
    }

    template<typename T>
    static Matrix<T> dot_product(const Matrix<T>& _a, const Matrix<T>& _b, const unsigned& _threads_per_block = 32) {

        if (_a.cols() != _b.rows())
            throw std::invalid_argument("Dimensiones incompatibles para multiplicacion");

        const auto ROWS{ _a.rows() };
        const auto K{ _b.rows() };
        const auto COLS{ _b.cols() };

        const auto SIZE_A{ ROWS * K * sizeof(T) };
        const auto SIZE_B{ K * COLS * sizeof(T) };
        const auto SIZE_C{ ROWS * COLS * sizeof(T) };

        T* dev_a{ nullptr };
        T* dev_b{ nullptr };
        T* dev_c{ nullptr };

        auto aux_a{ _a.flat() };
        auto aux_b{ _b.flat() };

        auto ho_a = aux_a.data();
        auto ho_b = aux_b.data();
        auto ho_c = new T[ROWS * COLS];

        check_cuda_error(cudaMalloc((void**)&dev_a, SIZE_A), "c_dot_product, cudaMalloc, dev_a");
        check_cuda_error(cudaMalloc((void**)&dev_b, SIZE_B), "c_dot_product, cudaMalloc, dev_b");
        check_cuda_error(cudaMalloc((void**)&dev_c, SIZE_C), "c_dot_product, cudaMalloc, dev_c");

        check_cuda_error(cudaMemcpy(dev_a, ho_a, SIZE_A, cudaMemcpyHostToDevice), "c_dot_product, cudaMemcpy, dev_a");
        check_cuda_error(cudaMemcpy(dev_b, ho_b, SIZE_B, cudaMemcpyHostToDevice), "c_dot_product, cudaMemcpy, dev_b");

        dim3 block_dim(_threads_per_block, _threads_per_block);
        dim3 grid_dim((static_cast<unsigned>(COLS) + block_dim.x - 1) / block_dim.x, (static_cast<unsigned>(ROWS) + block_dim.y - 1) / block_dim.y);

        dot_product_kernel_function << <grid_dim, block_dim >> > (dev_c, dev_a, dev_b, ROWS, K, COLS);

        check_cuda_error(cudaDeviceSynchronize(), "c_dot_product, cudaDeviceSynchronize");
        check_cuda_error(cudaMemcpy(ho_c, dev_c, SIZE_C, cudaMemcpyDeviceToHost), "c_dot_product, cudaMemcpy");

        auto c{ Matrix<T>(ho_c, ROWS * COLS, ROWS, COLS) };

        check_cuda_error(cudaFree(dev_a), "c_dot_product, cudaFree, dev_a");
        check_cuda_error(cudaFree(dev_b), "c_dot_product, cudaFree, dev_b");
        check_cuda_error(cudaFree(dev_c), "c_dot_product, cudaFree, dev_c");
        delete[] ho_c;

        return c;
    }

    template<typename T>
    static Matrix<T> outer_product(const Matrix<T>& _a, const Matrix<T>& _b, const unsigned& _threads_per_block = 32) {
        if (_a.cols() != 1 || _b.cols() != 1)
            throw std::invalid_argument("Para outer_product, ambas matrices deben ser vectores columna (cols=1)");
        return dot_product<T>(_a, _b.transpose(), _threads_per_block);
    }

    template<typename T>
    static Matrix<T> scalar_product(const Matrix<T>& _a, const T& _s, const unsigned& _threads_per_block = 32) {

        const auto ROWS{ _a.rows() };
        const auto COLS{ _a.cols() };

        const auto SIZE{ ROWS * COLS * sizeof(T) };

        T* dev_a{ nullptr };
        T* dev_b{ nullptr };

        auto aux_a{ _a.flat() };

        auto ho_a = aux_a.data();
        auto ho_b = new T[ROWS * COLS];

        check_cuda_error(cudaMalloc((void**)&dev_a, SIZE), "c_scalar_product, cudaMalloc, size_a");
        check_cuda_error(cudaMalloc((void**)&dev_b, SIZE), "c_scalar_product, cudaMalloc, size_b");

        check_cuda_error(cudaMemcpy(dev_a, ho_a, SIZE, cudaMemcpyHostToDevice), "c_scalar_product, cudaMemcpy, dev_a");

        dim3 block_dim(_threads_per_block, _threads_per_block);
        dim3 grid_dim((static_cast<unsigned>(COLS) + block_dim.x - 1) / block_dim.x, (static_cast<unsigned>(ROWS) + block_dim.y - 1) / block_dim.y);

        scalar_kernel_function << <grid_dim, block_dim >> > (dev_b, dev_a, _s, ROWS, COLS);

        check_cuda_error(cudaDeviceSynchronize(), "c_scalar_product, cudaDeviceSynchronize");
        check_cuda_error(cudaMemcpy(ho_b, dev_b, SIZE, cudaMemcpyDeviceToHost), "c_scalar_product, cudaMemcpy");

        auto b{ Matrix<T>(ho_b, ROWS * COLS, ROWS, COLS) };

        check_cuda_error(cudaFree(dev_a), "c_dot_product, cudaFree, dev_a");
        check_cuda_error(cudaFree(dev_b), "c_dot_product, cudaFree, dev_b");

        delete[] ho_b;

        return b;
    }

    template<typename T, typename Functor>
    static Matrix<T> apply_function(const Matrix<T>& _a, const Functor& _fu, const unsigned& _threads_per_block = 32) {

        const auto ROWS{ _a.rows() };
        const auto COLS{ _a.cols() };

        const auto SIZE{ ROWS * COLS * sizeof(T) };

        T* dev_a{ nullptr };
        T* dev_b{ nullptr };

        auto aux_a{ _a.flat() };

        auto ho_a = aux_a.data();
        auto ho_b = new T[ROWS * COLS];

        check_cuda_error(cudaMalloc((void**)&dev_a, SIZE), "c_apply_function, cudaMalloc, size_a");
        check_cuda_error(cudaMalloc((void**)&dev_b, SIZE), "c_apply_function, cudaMalloc, size_b");

        check_cuda_error(cudaMemcpy(dev_a, ho_a, SIZE, cudaMemcpyHostToDevice), "c_apply_function, cudaMemcpy, dev_a");

        dim3 block_dim(_threads_per_block, _threads_per_block);
        dim3 grid_dim((static_cast<unsigned>(COLS) + block_dim.x - 1) / block_dim.x, (static_cast<unsigned>(ROWS) + block_dim.y - 1) / block_dim.y);

        function_kernel_function<T, Functor> << <grid_dim, block_dim >> > (dev_b, dev_a, _fu, ROWS, COLS);

        check_cuda_error(cudaDeviceSynchronize(), "c_apply_function, cudaDeviceSynchronize");
        check_cuda_error(cudaMemcpy(ho_b, dev_b, SIZE, cudaMemcpyDeviceToHost), "c_apply_function, cudaMemcpy");

        auto b{ Matrix<T>(ho_b, ROWS * COLS, ROWS, COLS) };

        check_cuda_error(cudaFree(dev_a), "c_apply_function, cudaFree, dev_a");
        check_cuda_error(cudaFree(dev_b), "c_apply_function, cudaFree, dev_b");
        delete[] ho_b;

        return b;
    }

    template<typename T>
    static Matrix<T> basic_oper(const Matrix<T>& _a, const Matrix<T>& _b, const Operation& _o, const unsigned& _threads_per_block = 32) {

        if (_a.rows() != _b.rows() || _a.cols() != _b.cols())
            throw std::invalid_argument("Dimensiones incompatibles para suma");

        const auto ROWS{ _a.rows() };
        const auto COLS{ _a.cols() };

        const auto SIZE{ ROWS * COLS * sizeof(T) };

        T* dev_a{ nullptr };
        T* dev_b{ nullptr };
        T* dev_c{ nullptr };

        auto aux_a{ _a.flat() };
        auto aux_b{ _b.flat() };

        auto ho_a{ aux_a.data() };
        auto ho_b{ aux_b.data() };
        auto ho_c = new T[ROWS * COLS];

        check_cuda_error(cudaMalloc((void**)&dev_a, SIZE), "c_basic_oper, cudaMalloc, dev_a");
        check_cuda_error(cudaMalloc((void**)&dev_b, SIZE), "c_basic_oper, cudaMalloc, dev_b");
        check_cuda_error(cudaMalloc((void**)&dev_c, SIZE), "c_basic_oper, cudaMalloc, dev_c");

        check_cuda_error(cudaMemcpy(dev_a, ho_a, SIZE, cudaMemcpyHostToDevice), "c_basic_oper, cudaMemcpy, dev_a");
        check_cuda_error(cudaMemcpy(dev_b, ho_b, SIZE, cudaMemcpyHostToDevice), "c_basic_oper, cudaMemcpy, dev_b");

        dim3 block_dim(_threads_per_block, _threads_per_block);
        dim3 grid_dim((static_cast<unsigned>(COLS) + block_dim.x - 1) / block_dim.x, (static_cast<unsigned>(ROWS) + block_dim.y - 1) / block_dim.y);

        switch (_o) {
        case Operation::ADD:
            add_kernel_function << <grid_dim, block_dim >> > (dev_c, dev_a, dev_b, ROWS, COLS);
            break;
        case Operation::SUBT:
            subt_kernel_function << <grid_dim, block_dim >> > (dev_c, dev_a, dev_b, ROWS, COLS);
            break;
        default:
            throw std::invalid_argument("E");
        }

        check_cuda_error(cudaDeviceSynchronize(), "c_basic_oper, cudaDeviceSynchronize");
        check_cuda_error(cudaMemcpy(ho_c, dev_c, SIZE, cudaMemcpyDeviceToHost), "c_basic_oper, cudaMemcpy");

        auto c{ Matrix<T>(ho_c, ROWS * COLS, ROWS, COLS) };

        check_cuda_error(cudaFree(dev_a), "c_basic_oper, cudaFree, dev_a");
        check_cuda_error(cudaFree(dev_b), "c_basic_oper, cudaFree, dev_b");
        check_cuda_error(cudaFree(dev_c), "c_basic_oper, cudaFree, dev_c");
        delete[] ho_c;

        return c;
    }
}

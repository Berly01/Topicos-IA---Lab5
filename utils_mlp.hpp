
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cmath>
#include "Matrix.hpp"

enum class Initializer { U_XAVIER, N_XAVIER, HE, RANDOM };

enum class Optimizer { ADAM, RMS_PROP, NONE };

struct SigmoidFunctor {
    __device__ double operator()(const double& x) const {
        return 1.0 / (1.0 + exp(-x));
    }
    SigmoidFunctor() = default;
};
struct DSigmoidFunctor {
    __device__ double operator()(const double& x) const {
        auto f = SigmoidFunctor()(x);
        return f * (1.0 - f);
    }
    DSigmoidFunctor() = default;
};
struct ReluFunctor {
    __device__ double operator()(const double& x) const {
        return x > 0.0 ? x : 0.0;
    }
    ReluFunctor() = default;
};
struct DReluFunctor {
    __device__ double operator()(const double& x) const {
        return x > 0.0 ? 1.0 : 0.0;
    }
    DReluFunctor() = default;
};
struct TanhFunctor {
    __device__ double operator()(const double& x) const {
        return (1.0 - std::exp(-2 * x)) / (1.0 + std::exp(-2 * x));
    }
    TanhFunctor() = default;
};
struct DTanhFunctor {
    __device__ double operator()(const double& x) const {
        auto f = TanhFunctor()(x);
        return 1.0 - f * f;
    }
    DTanhFunctor() = default;
};
struct SoftmaxFunctor {
    Matrix<double> operator()(const Matrix<double>& m) const {
        const auto ROWS = m.rows();

        Matrix<double> result(ROWS, 1);

        double max_val = m[0][0];
        for (size_t i = 1; i < ROWS; ++i)
            if (m[i][0] > max_val) max_val = m[i][0];

        double sum_exp = 0.0;
        for (size_t i = 0; i < ROWS; ++i) {
            result[i][0] = std::exp(m[i][0] - max_val);
            sum_exp += result[i][0];
        }

        for (size_t i = 0; i < ROWS; ++i)
            result[i][0] /= sum_exp;

        return result;
    }
    SoftmaxFunctor() = default;
};

struct Hyperparameters {
    std::vector<size_t> layers = {2, 2, 3};
    Initializer initializer = Initializer::RANDOM;
    Optimizer optimizer = Optimizer::NONE;
    double learning_rate = 0.01;
    size_t batch = 16;
    size_t epochs = 100;
    double decay_rate = 0.9;
    double epsilon = 1e-8;
    double beta1 = 0.9;
    double beta2 = 0.999;
    double bias_init = 0.0;
    size_t timestep = 1;
    bool shuffle = false;
    bool debug = false;
};

struct SquareFunctor {
    __device__ double operator()(const double x) const {
        return x * x;
    }
    SquareFunctor() = default;
};


Matrix<double> xavier_uniform_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(6.0 / (_input_size + _output_size));
    std::uniform_real_distribution<> dist(-limit, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> xavier_normal_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / (_input_size + _output_size));
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> he_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    const double limit = std::sqrt(2.0 / _input_size);
    std::normal_distribution<> dist(0.0, limit);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

Matrix<double> random_init(const size_t& _input_size,
    const size_t& _output_size,
    std::mt19937& _gen) {

    std::uniform_real_distribution<> dist(0.0, 1.0);

    Matrix<double> m(_output_size, _input_size, 0.0);

    for (size_t r = 0; r < _output_size; ++r)
        for (size_t c = 0; c < _input_size; ++c)
            m[r][c] = dist(_gen);

    return m;
}

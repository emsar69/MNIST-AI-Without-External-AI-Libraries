#pragma once

#define NUMCPP_NO_USE_BOOST
#include <NumCpp.hpp>
#include <vector>
#include <fftw/fftw3.h>

std::vector<double> fft_convolve(const std::vector<double>& a, const std::vector<double>& b) {
    int n = a.size() + b.size() - 1;
    int N = 1;
    while (N < n) N *= 2;

    std::vector<double> A(N, 0), B(N, 0);
    for (size_t i = 0; i < a.size(); ++i) A[i] = a[i];
    for (size_t i = 0; i < b.size(); ++i) B[i] = b[i];

    fftw_complex* FA = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    fftw_complex* FB = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    fftw_complex* FC = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * (N / 2 + 1));
    std::vector<double> result(N);

    fftw_plan planA = fftw_plan_dft_r2c_1d(N, A.data(), FA, FFTW_ESTIMATE);
    fftw_plan planB = fftw_plan_dft_r2c_1d(N, B.data(), FB, FFTW_ESTIMATE);
    fftw_plan iplan = fftw_plan_dft_c2r_1d(N, FC, result.data(), FFTW_ESTIMATE);

    fftw_execute(planA);
    fftw_execute(planB);

    for (int i = 0; i < N / 2 + 1; ++i) {
        FC[i][0] = FA[i][0] * FB[i][0] - FA[i][1] * FB[i][1];
        FC[i][1] = FA[i][0] * FB[i][1] + FA[i][1] * FB[i][0];
    }

    fftw_execute(iplan);

    for (double& val : result) val /= N;

    fftw_destroy_plan(planA);
    fftw_destroy_plan(planB);
    fftw_destroy_plan(iplan);
    fftw_free(FA);
    fftw_free(FB);
    fftw_free(FC);

    result.resize(n);
    return result;
}

inline constexpr double ReLU(double x){
    return x > 0.0 ? x : 0;
}

inline constexpr double d_ReLU(double x){ // a is also ok
    return x > 0;
}

inline double Sigmoid(double x){
    return 1.0 / (1.0 + nc::exp(-x));
}

inline double d_Sigmoid(double a){
    return a * (1-a); // Where a = sigmoid(z) or a(z)
}

inline nc::NdArray<double> Softmax(const nc::NdArray<double>& x){
    auto s = x - x.max();
    s = nc::exp(s);
    return s/s.sum();
}

inline double MSE(const nc::NdArray<double>& predicted, const nc::NdArray<double>& original){
    return nc::mean(nc::power(predicted - original, 2.0)).item();
}

inline double cross_entropy(const nc::NdArray<double>& predicted, const nc::NdArray<double>& original){
    constexpr double epsilon = 1e-12;
    auto clipped = predicted.clip(epsilon, 1-epsilon);

    return -((nc::log(clipped) * original).sum().item());
}

enum ActivationFunction{
    mReLU,
    mSigmoid,
    mSoftmax
};

struct Activation{
    std::function<double(double)> scalar;
    std::function<nc::NdArray<double>(nc::NdArray<double>)> vector;

    Activation(){}
    Activation(std::function<double(double)> s, std::function<nc::NdArray<double>(const nc::NdArray<double>&)> v = nullptr) : scalar(s), vector(v) {}
};
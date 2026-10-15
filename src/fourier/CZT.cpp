#include <fourier/CZT.hpp>
#include <fourier/FFT.hpp>
#include <algorithm>
#include <bit>
#include <cmath>
#include <stdexcept>

namespace {

std::size_t convolution_size(std::size_t input_size, std::size_t output_size) {
    return input_size == 0 || output_size == 0 ? 1 : std::bit_ceil(input_size + output_size - 1);
}

}

namespace fourier {

struct CZT::Plan {
    std::size_t input_size;
    std::size_t output_size;
    FFT fft;
    ComplexVector chirp;
    ComplexVector kernel_spectrum;
    Plan(std::size_t n, std::size_t m, Real theta) : input_size(n), output_size(m), fft({convolution_size(n, m)}) {
        if (!std::isfinite(theta)) {
            throw std::invalid_argument("CZT requires a finite theta");
        }
        if (input_size == 0 || output_size == 0) {
            return;
        }
        chirp.resize(std::max(input_size, output_size));
        for (std::size_t k = 0; k < chirp.size(); ++k) {
            const long double index = static_cast<long double>(k);
            const long double phase = -0.5L * static_cast<long double>(theta) * index * index;
            const ComplexLong value = std::exp(ComplexLong{0.0L, phase});
            chirp[k] = {static_cast<Real>(value.real()), static_cast<Real>(value.imag())};
        }
        ComplexVector kernel(fft.shape()[0]);
        for (std::size_t k = 0; k < output_size; ++k) {
            kernel[k] = std::conj(chirp[k]);
        }
        for (std::size_t k = 1; k < input_size; ++k) {
            kernel[kernel.size() - k] = std::conj(chirp[k]);
        }
        kernel_spectrum = fft.forward(kernel);
    }
    ComplexVector compute(const ComplexVector& input) const {
        if (input.size() != input_size) {
            throw std::invalid_argument("CZT input size does not match its plan");
        }
        if (input_size == 0 || output_size == 0) {
            return ComplexVector(output_size);
        }
        ComplexVector values(fft.shape()[0]);
        for (std::size_t j = 0; j < input_size; ++j) {
            values[j] = input[j] * chirp[j];
        }
        values = fft.forward(values);
        for (std::size_t k = 0; k < values.size(); ++k) {
            values[k] *= kernel_spectrum[k];
        }
        values = fft.inverse(values);
        ComplexVector output(output_size);
        for (std::size_t m = 0; m < output_size; ++m) {
            output[m] = values[m] * chirp[m];
        }
        return output;
    }
};

CZT::CZT(std::size_t input_size, std::size_t output_size, Real theta) : plan(std::make_shared<const Plan>(input_size, output_size, theta)) {}
ComplexVector CZT::compute(const ComplexVector& input) const {
    return plan->compute(input);
}

}

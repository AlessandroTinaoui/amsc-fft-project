#include <fourier/DirectTransform.hpp>
#include <cmath>
#include <stdexcept>


namespace {
std::size_t shape_size(const std::vector<std::size_t>& shape) {
    std::size_t size = 1;
    for (const std::size_t extent : shape) {
        size *= extent;
    }
    return size;
}
}

namespace fourier {
ComplexVector DirectTransform::compute(const ComplexVector& input, const std::vector<std::size_t>& input_shape, const std::vector<std::size_t>& output_shape, const std::vector<Real>& theta) const {
    if (input_shape.empty() || input_shape.size() != output_shape.size() || input_shape.size() != theta.size()) {
        throw std::invalid_argument("DirectTransform requires matching nonzero ranks");
    }
    if (shape_size(input_shape) != input.size()) {
        throw std::invalid_argument("DirectTransform input size does not match its shape");
    }
    const std::size_t output_size = shape_size(output_shape);
    for (const Real angle : theta) {
        if (!std::isfinite(angle)) {
            throw std::invalid_argument("DirectTransform requires finite theta values");
        }
    }
    for (const Complex& value : input) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::invalid_argument("DirectTransform requires finite input values");
        }
    }
    ComplexVector output(output_size);
    for (std::size_t m = 0; m < output_size; ++m) {
        ComplexLong sum;
        for (std::size_t j = 0; j < input.size(); ++j) {
            std::size_t input_index = j;
            std::size_t output_index = m;
            long double phase = 0.0L;
            for (std::size_t axis = input_shape.size(); axis-- > 0;) {
                phase -= static_cast<long double>(theta[axis]) * static_cast<long double>(input_index % input_shape[axis]) * static_cast<long double>(output_index % output_shape[axis]);
                input_index /= input_shape[axis];
                output_index /= output_shape[axis];
            }
            const ComplexLong value{input[j].real(), input[j].imag()};
            sum += value * std::exp(ComplexLong{0.0L, phase});
        }
        output[m] = {static_cast<Real>(sum.real()), static_cast<Real>(sum.imag())};
    }
    return output;
}

}

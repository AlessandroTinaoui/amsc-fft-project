#include <fourier/DirectTransform.hpp>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace fourier {

ComplexVector DirectTransform::compute(
    const ComplexVector& input, std::size_t output_size, Real theta) const {
    if (!std::isfinite(theta)) {
        throw std::invalid_argument("DirectTransform requires a finite theta");
    }
    for (const auto& value : input) {
        if (!std::isfinite(value.real()) || !std::isfinite(value.imag())) {
            throw std::invalid_argument("DirectTransform requires finite input values");
        }
    }
    ComplexVector output(output_size);
    const auto max_value = static_cast<long double>(std::numeric_limits<Real>::max());
    for (std::size_t m = 0; m < output_size; ++m) {
        ComplexLong sum;
        for (std::size_t j = 0; j < input.size(); ++j) {
            const auto phase = -static_cast<long double>(theta) * static_cast<long double>(j)* static_cast<long double>(m);
            const ComplexLong value{input[j].real(), input[j].imag()};
            sum += value * std::exp(ComplexLong{0.0L, phase});
        }
        if (!std::isfinite(sum.real()) || !std::isfinite(sum.imag()) || std::abs(sum.real()) > max_value || std::abs(sum.imag()) > max_value) {
            throw std::overflow_error("DirectTransform result is not representable as a finite double");
        }
        output[m] = {static_cast<Real>(sum.real()), static_cast<Real>(sum.imag())};
    }
    return output;
}

}

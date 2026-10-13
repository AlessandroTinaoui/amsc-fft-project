#include <fourier/FFT.hpp>
#include <algorithm>
#include <cmath>
#include <numbers>
#include <stdexcept>
#include <utility>

namespace {

enum class Direction { Forward, Inverse };
class RadixTwoPlan {
public:
    explicit RadixTwoPlan(std::size_t size) : bit_reversal(size), twiddles(size / 2) {
        for (std::size_t j = 1; j < size; ++j) {
            bit_reversal[j] = (bit_reversal[j / 2] >> 1) | ((j & 1) * (size / 2));
        }
        const long double theta = -2.0L * std::numbers::pi_v<long double> / static_cast<long double>(size);
        for (std::size_t j = 0; j < twiddles.size(); ++j) {
            twiddles[j] = std::exp(fourier::ComplexLong{0.0L, theta * static_cast<long double>(j)});
        }
    }
    void transform_line(std::vector<fourier::ComplexLong>& values, Direction direction) const {
        const std::size_t size = values.size();
        for (std::size_t j = 0; j < size; ++j) {
            if (j < bit_reversal[j]) {
                std::swap(values[j], values[bit_reversal[j]]);
            }
        }
        for (std::size_t half = 1; half < size; half *= 2) {
            const std::size_t block_size = 2 * half;
            const std::size_t twiddle_stride = size / block_size;
            for (std::size_t base = 0; base < size; base += block_size) {
                for (std::size_t j = 0; j < half; ++j) {
                    const fourier::ComplexLong even = values[base + j];
                    const fourier::ComplexLong twiddle = direction == Direction::Inverse ? std::conj(twiddles[j * twiddle_stride]) : twiddles[j * twiddle_stride];
                    const fourier::ComplexLong odd = twiddle * values[base + j + half];
                    values[base + j] = even + odd;
                    values[base + j + half] = even - odd;
                }
            }
        }
    }
private:
    std::vector<std::size_t> bit_reversal;
    std::vector<fourier::ComplexLong> twiddles;
};

}

namespace fourier {

struct FFT::Plan {
    std::vector<std::size_t> shape;
    std::size_t size = 1;
    std::size_t scratch_size = 0;
    std::vector<std::size_t> strides;
    std::vector<RadixTwoPlan> axes;
    explicit Plan(std::vector<std::size_t> dimensions) : shape(std::move(dimensions)) {
        if (shape.empty()) {
            throw std::invalid_argument("FFT requires a nonzero rank");
        }
        for (const std::size_t extent : shape) {
            if (extent != 0 && (extent & (extent - 1)) != 0) {
                throw std::invalid_argument("FFT requires power-of-two axis lengths");
            }
            size *= extent;
            scratch_size = std::max(scratch_size, extent);
        }
        if (size == 0) {
            return;
        }
        strides.resize(shape.size());
        std::size_t stride = 1;
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            strides[axis] = stride;
            stride *= shape[axis];
        }
        axes.reserve(shape.size());
        for (const std::size_t extent : shape) {
            axes.emplace_back(extent);
        }
    }
    ComplexVector transform(const ComplexVector& input, Direction direction) const {
        if (size != input.size()) {
            throw std::invalid_argument("FFT input size does not match its shape");
        }
        if (size == 0) {
            return {};
        }
        std::vector<ComplexLong> values(input.begin(), input.end());
        std::vector<ComplexLong> line(scratch_size);
        for (std::size_t axis = shape.size(); axis-- > 0;) {
            const std::size_t extent = shape[axis];
            const std::size_t stride = strides[axis];
            const std::size_t block_size = extent * stride;
            line.resize(extent);
            for (std::size_t base = 0; base < size; base += block_size) {
                for (std::size_t offset = 0; offset < stride; ++offset) {
                    for (std::size_t j = 0; j < extent; ++j) {
                        line[j] = values[base + offset + j * stride];
                    }
                    axes[axis].transform_line(line, direction);
                    for (std::size_t m = 0; m < extent; ++m) {
                        values[base + offset + m * stride] = line[m];
                    }
                }
            }
        }
        ComplexVector output(size);
        const long double normalization = direction == Direction::Inverse ? static_cast<long double>(size) : 1.0L;
        for (std::size_t m = 0; m < size; ++m) {
            const ComplexLong value = values[m] / normalization;
            output[m] = {static_cast<Real>(value.real()), static_cast<Real>(value.imag())};
        }
        return output;
    }
};

FFT::FFT(std::vector<std::size_t> shape) : plan(std::make_shared<const Plan>(std::move(shape))) {}
ComplexVector FFT::forward(const ComplexVector& input) const {
    return plan->transform(input, Direction::Forward);
}

ComplexVector FFT::inverse(const ComplexVector& input) const {
    return plan->transform(input, Direction::Inverse);
}
const std::vector<std::size_t>& FFT::shape() const noexcept {
    return plan->shape;
}

}

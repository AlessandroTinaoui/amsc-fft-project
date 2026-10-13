#include <fourier/FFT.hpp>
#include <fourier/DirectTransform.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>

namespace {

using Values = fourier::ComplexVector;
constexpr double tolerance = 256.0 * std::numeric_limits<double>::epsilon();

void expect_values(const Values& actual, const Values& expected, double scale = 1.0) {
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t m = 0; m < expected.size(); ++m) {
        const double bound = tolerance * (scale + std::abs(expected[m]));
        EXPECT_LE(std::abs(actual[m] - expected[m]), bound) << "index " << m;
    }
}

Values make_input(std::size_t size) {
    Values input(size);
    for (std::size_t j = 0; j < size; ++j) {
        const double index = static_cast<double>(j);
        input[j] = {std::sin(0.7 * index) + 0.2 * std::cos(0.3 * index), std::cos(0.4 * index) - 0.1 * std::sin(0.9 * index)};
    }
    return input;
}

void expect_direct(const std::vector<std::size_t>& shape) {
    std::size_t size = 1;
    std::vector<double> theta;
    for (const std::size_t extent : shape) {
        size *= extent;
        theta.push_back(2.0 * std::numbers::pi / static_cast<double>(extent));
    }
    const Values input = make_input(size);
    const Values original = input;
    const fourier::FFT fft(shape);
    const fourier::DirectTransform direct;
    expect_values(fft.forward(input), direct.compute(input, shape, shape, theta), static_cast<double>(size));
    for (double& angle : theta) {
        angle = -angle;
    }
    Values inverse = direct.compute(input, shape, shape, theta);
    for (fourier::Complex& value : inverse) {
        value /= static_cast<double>(size);
    }
    expect_values(fft.inverse(input), inverse);
    EXPECT_EQ(input, original);
}

TEST(FFT, RadixTwoMatchesDirect) {
    for (const std::size_t size : {2, 4, 8, 16, 64, 256}) {
        SCOPED_TRACE(size);
        expect_direct({size});
    }
}

TEST(FFT, NonPowerOfTwoAxes) {
    for (const std::size_t size : {3, 6}) {
        SCOPED_TRACE(size);
        EXPECT_THROW(fourier::FFT({size}), std::invalid_argument);
    }
    EXPECT_THROW(fourier::FFT({4, 3}), std::invalid_argument);
    EXPECT_THROW(fourier::FFT({3, 4}), std::invalid_argument);
}

TEST(FFT, MultidimensionalMatchesDirect) {
    expect_direct({2, 4});
    expect_direct({8, 2});
    expect_direct({2, 4, 8});
    expect_direct({1, 4, 1, 2});
}

TEST(FFT, FrequencyOrderAndNormalization) {
    const fourier::FFT fft({2, 4});
    Values impulse(8);
    impulse[5] = {2.0, -1.0};
    expect_values(fft.forward(impulse), {{2.0, -1.0}, {-1.0, -2.0}, {-2.0, 1.0}, {1.0, 2.0}, {-2.0, 1.0}, {1.0, 2.0}, {2.0, -1.0}, {-1.0, -2.0}});
    Values expected(8);
    expected[0] = {2.0, -1.0};
    expect_values(fft.inverse(Values(8, {2.0, -1.0})), expected);
}

TEST(FFT, RoundTripAndRepeatedCalls) {
    const fourier::FFT fft({1024});
    const Values input = make_input(1024);
    expect_values(fft.inverse(fft.forward(input)), input);
    const fourier::FFT tensor_fft({4, 8, 16});
    const Values tensor = make_input(512);
    expect_values(tensor_fft.inverse(tensor_fft.forward(tensor)), tensor);
    expect_values(tensor_fft.forward(tensor_fft.inverse(tensor)), tensor);
}

TEST(FFT, BoundariesAndScales) {
    EXPECT_TRUE(fourier::FFT({2, 0, 4}).forward({}).empty());
    EXPECT_TRUE(fourier::FFT({0}).inverse({}).empty());
    const fourier::FFT singleton_fft({1, 1});
    expect_values(singleton_fft.forward({{2.0, -3.0}}), {{2.0, -3.0}});
    expect_values(singleton_fft.inverse({{2.0, -3.0}}), {{2.0, -3.0}});
    expect_values(fourier::FFT({2, 4}).forward(Values(8)), Values(8));
    expect_values(fourier::FFT({4}).forward({{1e12, 1e12}, {1.0, -2.0}, {-1e12, -1e12}, {0.0, 0.0}}), {{1.0, -2.0}, {2e12 - 2.0, 2e12 - 1.0}, {-1.0, 2.0}, {2e12 + 2.0, 2e12 + 1.0}});
    const fourier::FFT fft({2});
    const double large = std::numeric_limits<double>::max() / 8.0;
    EXPECT_EQ(fft.forward({{large, large}, {large, -large}})[0], fourier::Complex(2.0 * large, 0.0));
    const double small = std::numeric_limits<double>::min() * 4.0;
    EXPECT_EQ(fft.forward({{small, 2.0 * small}, {2.0 * small, -small}})[0], fourier::Complex(3.0 * small, small));
}

TEST(FFT, InvalidShape) {
    EXPECT_THROW(fourier::FFT({}), std::invalid_argument);
    const fourier::FFT fft({2, 2});
    EXPECT_THROW(fft.forward({{1.0, 0.0}}), std::invalid_argument);
    EXPECT_THROW(fft.inverse({{1.0, 0.0}}), std::invalid_argument);
    EXPECT_THROW(fourier::FFT({0, 2}).inverse({{1.0, 0.0}}), std::invalid_argument);
}
TEST(FFT, PlanOwnershipAndReuse) {
    std::vector<std::size_t> shape{2, 4};
    const fourier::FFT fft(shape);
    shape[0] = 4;
    EXPECT_EQ(fft.shape(), (std::vector<std::size_t>{2, 4}));
    fourier::FFT copy = fft;
    const Values input = make_input(8);
    const Values spectrum = fft.forward(input);
    expect_values(copy.forward(input), spectrum);
    expect_values(copy.inverse(spectrum), input);
    expect_values(fft.forward(Values(8)), Values(8));
    expect_values(fft.forward(input), spectrum);
    {
        const fourier::FFT temporary({4});
        copy = temporary;
    }
    EXPECT_EQ(copy.shape(), (std::vector<std::size_t>{4}));
    const Values small_input = make_input(4);
    expect_values(copy.inverse(copy.forward(small_input)), small_input);
    expect_values(fft.forward(input), spectrum);
}

}

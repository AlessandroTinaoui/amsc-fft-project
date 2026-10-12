#include <fourier/DirectTransform.hpp>
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <numbers>
#include <stdexcept>
#include <string>

namespace {

using Values = fourier::ComplexVector;
constexpr double tolerance = 64.0 * std::numeric_limits<double>::epsilon();

void expect_values(const Values& actual, const Values& expected, const std::string& name, double absolute_tolerance = tolerance) {
    SCOPED_TRACE(name);
    ASSERT_EQ(actual.size(), expected.size());
    for (std::size_t m = 0; m < expected.size(); ++m) {
        const auto bound = absolute_tolerance + tolerance * std::abs(expected[m]);
        EXPECT_LE(std::abs(actual[m] - expected[m]), bound) << "index " << m;
    }
}

TEST(DirectTransform, StandardDft) {
    const fourier::DirectTransform transform;
    const Values input{{1.0, 1.0}, {2.0, -1.0}, {3.0, 2.0}, {4.0, -2.0}};
    const double theta = 2.0 * std::numbers::pi / input.size();
    expect_values(transform.compute(input, 4, theta), {{10.0, 0.0}, {-1.0, 1.0}, {-2.0, 6.0}, {-3.0, -3.0}}, "standard DFT");
    expect_values(transform.compute(input, 2, theta), {{10.0, 0.0}, {-1.0, 1.0}}, "M < N");
}

TEST(DirectTransform, GeneralizedTransform) {
    const fourier::DirectTransform transform;
    const Values input{{1.0, 2.0}, {3.0, -1.0}};
    const double r = std::sqrt(3.0) / 2.0;
    expect_values(transform.compute(input, 5, std::numbers::pi / 3.0),{{4.0, 1.0}, {2.5 - r, 1.5 - 3.0 * r}, {-0.5 - r, 2.5 - 3.0 * r},{-2.0, 3.0}, {-0.5 + r, 2.5 + 3.0 * r}}, "generalized transform with M > N");
    expect_values(transform.compute(input, 3, -std::numbers::pi / 3.0),{{4.0, 1.0}, {2.5 + r, 1.5 + 3.0 * r}, {-0.5 + r, 2.5 + 3.0 * r}}, "negative theta");
}

TEST(DirectTransform, Impulse) {
    const fourier::DirectTransform transform;
    const Values input{{0.0, 0.0}, {0.0, 0.0}, {0.0, 0.0}, {2.0, -1.0}, {0.0, 0.0}};
    expect_values(transform.compute(input, 4, std::numbers::pi / 2.0),
        {{2.0, -1.0}, {1.0, 2.0}, {-2.0, 1.0}, {-1.0, -2.0}}, "shifted impulse");
}

TEST(DirectTransform, Boundaries) {
    const fourier::DirectTransform transform;
    expect_values(transform.compute({}, 3, 0.7), Values(3), "empty input");
    expect_values(transform.compute({{2.0, -3.0}}, 0, 0.7), {}, "zero outputs");
    expect_values(transform.compute({{2.0, -3.0}}, 4, 0.7), Values(4, {2.0, -3.0}), "singleton");
    expect_values(transform.compute({{1.0, 2.0}, {-3.0, 1.0}, {2.0, -4.0}}, 5, 0.0),Values(5, {0.0, -1.0}), "zero theta");
}

TEST(DirectTransform, ScalesAndCancellation) {
    const fourier::DirectTransform transform;
    expect_values(transform.compute({{1e12, 1e12}, {1.0, -2.0}, {-1e12, -1e12}}, 2, 0.0),Values(2, {1.0, -2.0}), "cancellation");
    const double large = std::numeric_limits<double>::max() / 8.0;
    expect_values(transform.compute({{large, large}, {large, -large}}, 1, 0.0),{{2.0 * large, 0.0}}, "large values", 0.0);
    const double small = std::numeric_limits<double>::min() * 4.0;
    expect_values(transform.compute({{small, 2.0 * small}, {2.0 * small, -small}}, 1, 0.0),{{3.0 * small, small}}, "small values", 0.0);
}

TEST(DirectTransform, Failures) {
    const double infinity = std::numeric_limits<double>::infinity();
    const double nan = std::numeric_limits<double>::quiet_NaN();
    for (double value : {infinity, -infinity, nan}) {
        EXPECT_THROW(fourier::DirectTransform{}.compute({{1.0, 0.0}}, 2, value), std::invalid_argument);
        EXPECT_THROW(fourier::DirectTransform{}.compute({{value, 0.0}}, 2, 0.7), std::invalid_argument);
        EXPECT_THROW(fourier::DirectTransform{}.compute({{0.0, value}}, 2, 0.7), std::invalid_argument);
    }
    const double max_value = std::numeric_limits<double>::max();
    EXPECT_THROW(fourier::DirectTransform{}.compute({{max_value, 0.0}, {max_value, 0.0}}, 1, 0.0), std::overflow_error);
}

}

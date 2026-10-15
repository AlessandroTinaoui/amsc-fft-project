#include <fourier/CZT.hpp>
#include <fourier/DirectTransform.hpp>
#include <fourier/FFT.hpp>
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

void expect_direct(const Values& input, std::size_t output_size, double theta) {
    const fourier::CZT czt(input.size(), output_size, theta);
    const Values original = input;
    double scale = 0.0;
    for (const fourier::Complex& value : input) {
        scale += std::abs(value);
    }
    expect_values(czt.compute(input), fourier::DirectTransform{}.compute(input, {input.size()}, {output_size}, {theta}), scale);
    EXPECT_EQ(input, original);
}

TEST(CZT, StandardDftAndFft) {
    const Values input{{1.0, 1.0}, {2.0, -1.0}, {3.0, 2.0}, {4.0, -2.0}};
    expect_values(fourier::CZT(4, 4, std::numbers::pi / 2.0).compute(input), {{10.0, 0.0}, {-1.0, 1.0}, {-2.0, 6.0}, {-3.0, -3.0}});
    expect_direct(make_input(7), 7, 2.0 * std::numbers::pi / 7.0);
    expect_direct(make_input(17), 17, 2.0 * std::numbers::pi / 17.0);
    const Values data = make_input(64);
    expect_values(fourier::CZT(64, 64, 2.0 * std::numbers::pi / 64.0).compute(data), fourier::FFT({64}).forward(data), 64.0);
}

TEST(CZT, GeneralizedTransform) {
    expect_direct(make_input(7), 3, 0.37);
    expect_direct(make_input(5), 11, -0.71);
    expect_direct(make_input(9), 8, 7.1);
    expect_direct(make_input(9), 9, 0.013);
    expect_direct(make_input(17), 31, std::numbers::pi / 3.0);
    expect_direct(make_input(257), 193, -0.43);
}

TEST(CZT, ShiftedImpulse) {
    Values input(5);
    input[3] = {2.0, -1.0};
    expect_values(fourier::CZT(5, 9, std::numbers::pi / 2.0).compute(input), {{2.0, -1.0}, {1.0, 2.0}, {-2.0, 1.0}, {-1.0, -2.0}, {2.0, -1.0}, {1.0, 2.0}, {-2.0, 1.0}, {-1.0, -2.0}, {2.0, -1.0}});
}

TEST(CZT, Boundaries) {
    expect_values(fourier::CZT(0, 5, 0.7).compute({}), Values(5));
    EXPECT_TRUE(fourier::CZT(3, 0, 0.7).compute(make_input(3)).empty());
    EXPECT_TRUE(fourier::CZT(0, 0, 0.7).compute({}).empty());
    expect_values(fourier::CZT(1, 1, 0.7).compute({{2.0, -3.0}}), {{2.0, -3.0}});
    expect_values(fourier::CZT(1, 5, -0.7).compute({{2.0, -3.0}}), Values(5, {2.0, -3.0}));
    expect_values(fourier::CZT(3, 5, 0.0).compute({{1.0, 2.0}, {-3.0, 1.0}, {2.0, -4.0}}), Values(5, {0.0, -1.0}));
    expect_direct(make_input(9), 1, 0.73);
    expect_direct(make_input(9), 13, 1e-12);
    expect_values(fourier::CZT(7, 11, 0.3).compute(Values(7)), Values(11));
}

TEST(CZT, ScalesAndCancellation) {
    for (const double scale : {1e100, 1e-100}) {
        SCOPED_TRACE(scale);
        Values input = make_input(11);
        for (fourier::Complex& value : input) {
            value *= scale;
        }
        expect_direct(input, 17, 0.37);
    }
    expect_direct({{1e8, 1e8}, {1.0, -2.0}, {-1e8, -1e8}}, 5, 0.0);
}

TEST(CZT, PlanOwnershipAndReuse) {
    const fourier::CZT czt(9, 13, -0.37);
    const Values input = make_input(9);
    const Values expected = czt.compute(input);
    fourier::CZT copy = czt;
    expect_values(copy.compute(input), expected);
    expect_values(czt.compute(Values(9)), Values(13));
    const Values changed(9, {1.0, -2.0});
    expect_values(czt.compute(changed), fourier::DirectTransform{}.compute(changed, {9}, {13}, {-0.37}), 27.0);
    expect_values(czt.compute(input), expected);
    {
        const fourier::CZT temporary(3, 5, 0.2);
        copy = temporary;
    }
    const Values small_input = make_input(3);
    expect_values(copy.compute(small_input), fourier::DirectTransform{}.compute(small_input, {3}, {5}, {0.2}), 3.0);
    expect_values(czt.compute(input), expected);
}

TEST(CZT, InvalidArguments) {
    const fourier::CZT czt(3, 5, 0.7);
    EXPECT_THROW(czt.compute({{1.0, 0.0}}), std::invalid_argument);
    EXPECT_THROW(fourier::CZT(0, 2, 0.7).compute({{1.0, 0.0}}), std::invalid_argument);
    for (const double theta : {std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
        EXPECT_THROW(fourier::CZT(3, 5, theta), std::invalid_argument);
    }
}

}

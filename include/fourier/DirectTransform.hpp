#pragma once

#include <fourier/Types.hpp>
#include <cstddef>

namespace fourier {

class DirectTransform {
public:
    ComplexVector compute(const ComplexVector& input, const std::vector<std::size_t>& input_shape, const std::vector<std::size_t>& output_shape, const std::vector<Real>& theta) const;
};

}

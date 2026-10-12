#pragma once

#include <fourier/Types.hpp>
#include <cstddef>

namespace fourier {

class DirectTransform {
public:
    ComplexVector compute(const ComplexVector& input, std::size_t output_size, Real theta) const;
};

}

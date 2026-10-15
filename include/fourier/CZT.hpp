#pragma once

#include <fourier/Types.hpp>
#include <cstddef>
#include <memory>

namespace fourier {

class CZT {
public:
    CZT(std::size_t input_size, std::size_t output_size, Real theta);
    ComplexVector compute(const ComplexVector& input) const;
private:
    struct Plan;
    std::shared_ptr<const Plan> plan;
};

}

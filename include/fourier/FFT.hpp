#pragma once

#include <fourier/Types.hpp>
#include <cstddef>
#include <memory>

namespace fourier {

class FFT {
public:
    explicit FFT(std::vector<std::size_t> shape);
    ComplexVector forward(const ComplexVector& input) const;
    ComplexVector inverse(const ComplexVector& input) const;
    const std::vector<std::size_t>& shape() const noexcept;
private:
    struct Plan;
    std::shared_ptr<const Plan> plan;
};

}

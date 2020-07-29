#include <torch/extension.h>

#include <iostream>
#include <vector>

// Forward declarations

std::vector<torch::Tensor> mcubes_cpu(
    torch::Tensor func,
    float threshold
);


// Pybind11 exports
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("mcubes_cpu", &mcubes_cpu, "Marching cubes (CPU)");
}

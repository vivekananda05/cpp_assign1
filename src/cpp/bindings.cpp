

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "tensor.h"
#include "layers.h"
#include "optimizers.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(custom_dl_framework, m) {
    m.doc() = "Custom Deep Learning Framework";
    
    // Tensor class
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def(py::init<const std::vector<int>&>(), 
             py::arg("shape"))
        .def(py::init<const std::vector<int>&, bool>(), 
             py::arg("shape"), py::arg("requires_grad") = false)
        .def(py::init<const std::vector<float>&, const std::vector<int>&>(), 
             py::arg("data"), py::arg("shape"))
        .def(py::init<const std::vector<float>&, const std::vector<int>&, bool>(),
             py::arg("data"), py::arg("shape"), py::arg("requires_grad") = false)
        .def("getShape", &Tensor::getShape)
        .def("getData", &Tensor::getData)
        .def("getTotalSize", &Tensor::getTotalSize)
        .def("zeros", &Tensor::zeros)
        .def("ones", &Tensor::ones)
        .def("randomNormal", &Tensor::randomNormal, 
             py::arg("mean") = 0.0f, py::arg("stddev") = 1.0f)
        .def("randomUniform", &Tensor::randomUniform,
             py::arg("low") = 0.0f, py::arg("high") = 1.0f)
        .def("xavierUniform", &Tensor::xavierUniform)
        .def("matmul", &Tensor::matmul)
        .def("conv2d", &Tensor::conv2d)
        .def("maxPool2d", &Tensor::maxPool2d)
        .def("relu", &Tensor::relu)
        .def("sigmoid", &Tensor::sigmoid)
        .def("softmax", &Tensor::softmax, py::arg("axis") = -1)
        .def("reshape", &Tensor::reshape)
        .def("flatten", &Tensor::flatten)
        .def("backward", &Tensor::backward)
        .def("zeroGrad", &Tensor::zeroGrad)
        .def("__repr__", &Tensor::toString)
        .def("__add__", [](const Tensor& a, const Tensor& b) {
            return a.elementwiseAdd(b);
        })
        .def("__mul__", [](const Tensor& a, const Tensor& b) {
            return a.elementwiseMultiply(b);
        });
    
    // Layer base class
    py::class_<Layer>(m, "Layer")
        .def("getName", &Layer::getName)
        .def("isTrainable", &Layer::isTrainable)
        .def("forward", &Layer::forward)
        .def("backward", &Layer::backward)
        .def("getParameters", &Layer::getParameters)
        .def("getGradients", &Layer::getGradients)
        .def("zeroGrad", &Layer::zeroGrad);
    
    // Conv2d layer
    py::class_<Conv2d, Layer>(m, "Conv2d")
        .def(py::init<int, int, int, int, int, std::string>(),
             py::arg("in_channels"), py::arg("out_channels"), 
             py::arg("kernel_size"), py::arg("stride") = 1,
             py::arg("padding") = 0, py::arg("name") = "Conv2d")
        .def("getNumParameters", &Conv2d::getNumParameters)
        .def("getMACs", &Conv2d::getMACs)
        .def("getFLOPs", &Conv2d::getFLOPs);
    
    // MaxPool2d layer
    py::class_<MaxPool2d, Layer>(m, "MaxPool2d")
        .def(py::init<int, int, int, std::string>(),
             py::arg("kernel_size") = 2, py::arg("stride") = 2,
             py::arg("padding") = 0, py::arg("name") = "MaxPool2d")
        .def("getFLOPs", &MaxPool2d::getFLOPs);
    
    // Linear layer
    py::class_<Linear, Layer>(m, "Linear")
        .def(py::init<int, int, std::string>(),
             py::arg("in_features"), py::arg("out_features"),
             py::arg("name") = "Linear")
        .def("getNumParameters", &Linear::getNumParameters)
        .def("getMACs", &Linear::getMACs)
        .def("getFLOPs", &Linear::getFLOPs);
    
    // ReLU layer
    py::class_<ReLU, Layer>(m, "ReLU")
        .def(py::init<std::string>(), py::arg("name") = "ReLU");
    
    // Utility functions
    m.def("crossEntropyLoss", &utils::crossEntropyLoss);
    m.def("mseLoss", &utils::mseLoss);
    m.def("accuracy", &utils::accuracy);
    m.def("normalizeImage", &utils::normalizeImage);
    m.def("isCUDAAvailable", &utils::isCUDAAvailable);
    m.def("setDevice", &utils::setDevice);
    m.def("cudaSynchronize", &utils::cudaSynchronize);
    
    // Timer class
    py::class_<utils::Timer>(m, "Timer")
        .def(py::init<>())
        .def("start", &utils::Timer::start)
        .def("elapsed", &utils::Timer::elapsed);
    
    // ModelStats struct
    py::class_<utils::ModelStats>(m, "ModelStats")
        .def(py::init<>())
        .def_readwrite("total_parameters", &utils::ModelStats::total_parameters)
        .def_readwrite("total_macs", &utils::ModelStats::total_macs)
        .def_readwrite("total_flops", &utils::ModelStats::total_flops)
        .def_readwrite("memory_mb", &utils::ModelStats::memory_mb);
}
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "parser/cxxopts.hpp"

#include "networks/yolov4.h"
#include "networks/yolov4tiny.h"

// Don't remove unused include, necessary to correctly load and register tensorrt yolo plugin
#include "layers/yololayer.h"

#include "utils/logging.h"
static Logger gLogger;

#include <iostream>

#define DEVICE 0
#define BATCH_SIZE 1

using namespace nvinfer1;

enum NETWORKS {
    YOLOV4,
    YOLOV4TINY
};

int main(int argc, char** argv) {
    cxxopts::Options options(argv[0], "--- YOLOV4 TRITON TENSORRT ---");

    options.add_options()
        ("n,network", "Network to optimize, either \"yolov4\" or \"yolov4tiny\"", cxxopts::value<std::string>()->default_value("yolov4"))
        ("h,help", "Print help screen");

    NETWORKS network;

    // Parse and check options
    try {
        auto result = options.parse(argc, argv);

        if (result.count("help")) {
            std::cout << options.help() << std::endl;
            exit(0);
        }

        auto network_string = result["network"].as<std::string>();
        if (network_string.compare("yolov4") == 0) {
            network = NETWORKS::YOLOV4;
        }
        else if (network_string.compare("yolov4tiny") == 0) {
            network = NETWORKS::YOLOV4TINY;
        }
        else {
            std::cout << "[Error] Network to optimize must be either \"yolov4\" or \"yolov4tiny\"" << std::endl;
            std::cout << options.help({""}) << std::endl;
            exit(0);
        }
    }
    catch(cxxopts::OptionException exception) {
        std::cout << "[Error] " << exception.what() << std::endl;
        std::cout << options.help() << std::endl;
        exit(0);
    }

    cudaSetDevice(DEVICE);

    std::cout << "[Info] Creating builder" << std::endl;
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine;
    if(network == NETWORKS::YOLOV4) {
        std::cout << "[Info] Creating model yolov4" << std::endl;
        engine = yolov4::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT, "yolov4.wts");
    }
    else if(network == NETWORKS::YOLOV4TINY) {
        std::cout << "[Info] Creating model yolov4tiny" << std::endl;
        engine = yolov4tiny::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT, "yolov4tiny.wts");
    }
    assert(engine != nullptr);

    std::cout << "[Info] Serializing model to engine file" << std::endl;
    // Serialize the engine
    IHostMemory* modelStream{nullptr};
    modelStream = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();

    assert(modelStream != nullptr);
    std::ofstream p(network == NETWORKS::YOLOV4TINY ? "yolov4tiny.engine" : "yolov4.engine", std::ios::binary);
    if (!p) {
        std::cerr << "[Error] Could not open engine output file " << (network == NETWORKS::YOLOV4TINY ? "yolov4tiny.engine" : "yolov4.engine") << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    std::cout << "[Info] Done" << std::endl;

    return 0;
}
#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "networks/yolov4.h"

#include "utils/logging.h"
static Logger gLogger;

#include <iostream>

#define DEVICE 0
#define BATCH_SIZE 1

using namespace nvinfer1;

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    std::cout << "Creating builder" << std::endl;
    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    std::cout << "Creating model" << std::endl;
    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = yolov4::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT, "yolov4.wts");
    assert(engine != nullptr);

    std::cout << "Serializing model to engine file" << std::endl;
    // Serialize the engine
    IHostMemory* modelStream{nullptr};
    modelStream = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();

    assert(modelStream != nullptr);
    std::ofstream p("yolov4.engine", std::ios::binary);
    if (!p) {
        std::cerr << "Could not open engine output file yolov4.engine" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    std::cout << "Done" << std::endl;

    return 0;
}
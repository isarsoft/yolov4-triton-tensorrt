#include "NvInfer.h"
#include "cuda_runtime_api.h"

#include "networks/yolov4.h"

#include "utils/logging.h"
static Logger gLogger;

#define DEVICE 0
#define BATCH_SIZE 1

using namespace nvinfer1;

int main(int argc, char** argv) {
    cudaSetDevice(DEVICE);

    // Create builder
    IBuilder* builder = createInferBuilder(gLogger);
    IBuilderConfig* config = builder->createBuilderConfig();

    // Create model to populate the network, then set the outputs and create an engine
    ICudaEngine* engine = yolov4::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT, "../yolov4.wts");
    assert(engine != nullptr);

    // Serialize the engine
    IHostMemory* modelStream{nullptr};
    modelStream = engine->serialize();

    // Close everything down
    engine->destroy();
    builder->destroy();

    assert(modelStream != nullptr);
    std::ofstream p("yolov4.engine", std::ios::binary);
    if (!p) {
        std::cerr << "could not open plan output file" << std::endl;
        return -1;
    }
    p.write(reinterpret_cast<const char*>(modelStream->data()), modelStream->size());
    modelStream->destroy();

    return 0;
}
#include "NvInfer.h"
#include "cuda_runtime_api.h"
#include <string>
#include "parser/cxxopts.hpp"


#include "networks/yolov4p5.h"

#ifndef THEIRS

#include "networks/yolov4tiny.h"
#include "networks/yolov4tiny3l.h"
#include "layers/yololayer.h"
#endif
// Don't remove unused include, necessary to correctly load and register tensorrt yolo plugin
#include "utils/logging.h"


static Logger gLogger;

#include <iostream>

#define DEVICE 0
// #define BATCH_SIZE 1

 using namespace nvinfer1;

enum NETWORKS { YOLOV4, YOLOV4P5, YOLOV4TINY, YOLOV4TINY3L };

int main(int argc, char **argv) {
  cxxopts::Options options(argv[0], "--- YOLOV4 TRITON TENSORRT ---");

  options.add_options()
    ("n,network",
     "Network to optimize, either \"yolov4\", \"yolov5-p5 \", "
     "\"yolov4tiny\" or \"yolov4tiny3l\"",
     cxxopts::value<std::string>()->default_value("yolov4"))
    ("w,weights",
     "Path to network weights",
     cxxopts::value<std::string>()->default_value("idontexist"))
    ("b,batch_size",
     "Batch size to be used",
     cxxopts::value<int>()->default_value("1"))
    ("input_h",
     "Input height",

     cxxopts::value<int>()->default_value("608"))
    ("input_w",
     "Input weight",
     cxxopts::value<int>()->default_value("608"))
    ("h,help", "Print help screen");

  NETWORKS network;
  int BATCH_SIZE = 1;
  int INPUT_H = 0;
  int INPUT_W = 0;
  std::string WEIGHTS_PATH = "idontexist";
  // Parse and check options
  try {
    auto result = options.parse(argc, argv);

    if (result.count("help")) {

      std::cout << options.help() << std::endl;
      exit(0);
    }
    WEIGHTS_PATH= result["weights"].as<std::string>();
    BATCH_SIZE = result["batch_size"].as<int>();
    INPUT_H = result["input_h"].as<int>();
    INPUT_W = result["input_w"].as<int>();
    auto network_string = result["network"].as<std::string>();
    if (network_string.compare("yolov4") == 0) {
      network = NETWORKS::YOLOV4;
    } 
     else if (network_string.compare("yolov4p5") == 0) {
      network = NETWORKS::YOLOV4P5;
    }
    else if (network_string.compare("yolov4tiny") == 0) {
      network = NETWORKS::YOLOV4TINY;
    } else if (network_string.compare("yolov4tiny3l") == 0) {
      network = NETWORKS::YOLOV4TINY3L;
    } else {
      std::cout << "[Error] Network to optimize must be either \"yolov4\" , "
                   "\"yolov4tiny\" , \"yolov4tiny3l \" or \"yolov4p5\" "
                << std::endl;
      std::cout << options.help({""}) << std::endl;
      exit(0);
    }
  } catch (cxxopts::OptionException exception) {
    std::cout << "[Error] " << exception.what() << std::endl;
    std::cout << options.help() << std::endl;
    exit(0);
  }

  cudaSetDevice(DEVICE);

  std::cout << "[Info] Creating builder" << std::endl;
  // Create builder
  IBuilder *builder = createInferBuilder(gLogger);
  IBuilderConfig *config = builder->createBuilderConfig();

  // Create model to populate the network, then set the outputs and create an
  // engine
  float gd = 1.0f;
  float gw = 1.0f;
  ICudaEngine *engine;
  if (network == NETWORKS::YOLOV4) {
    std::cout << "[Info] Creating model yolov4" << std::endl;
    // engine = yolov4::createEngine(BATCH_SIZE, builder, config, DataType::kFLOAT,
 // "yolov4.wts");
  } else if (network == NETWORKS::YOLOV4P5) {
    std::cout << "[Info] Creating model yolov4p5" << std::endl;
    auto params = yolov4p5::yolov4p5Parameters();
    params.WEIGHTS_PATH= WEIGHTS_PATH;
    params.BATCH_SIZE = BATCH_SIZE;
    params.INPUT_H = INPUT_H;
    params.INPUT_W = INPUT_W;
    engine = yolov4p5::createEngine(params, builder, config,
                                    DataType::kFLOAT, gd,gw);
  } else if (network == NETWORKS::YOLOV4TINY) {
    std::cout << "[Info] Creating model yolov4tiny" << std::endl;
    // engine = yolov4tiny::createEngine(BATCH_SIZE, builder, config,
                                      // DataType::kFLOAT, "yolov4tiny.wts");
  } else if (network == NETWORKS::YOLOV4TINY3L) {
    std::cout << "[Info] Creating model yolov4tiny3l" << std::endl;
    // engine = yolov4tiny3l::createEngine(BATCH_SIZE, builder, config,
                                        // DataType::kFLOAT, "yolov4tiny3l.wts");
  }
  assert(engine != nullptr);

  std::cout << "[Info] Serializing model to engine file" << std::endl;
  // Serialize the engine
  IHostMemory *modelStream{nullptr};
  modelStream = engine->serialize();

  // Close everything down
  engine->destroy();
  builder->destroy();

  assert(modelStream != nullptr);
  std::string engine_name = "";
  if (network == NETWORKS::YOLOV4TINY3L) {
    engine_name = "yolov4tiny3l.engine";
  } else if (network == NETWORKS::YOLOV4TINY) {
    engine_name = "yolov4tiny.engine";
  } else if (network == NETWORKS::YOLOV4) {
    engine_name = "yolov4.engine";
  } else if (network == NETWORKS::YOLOV4P5) {
    engine_name = "yolov4p5.engine";
  }
  std::ofstream p(engine_name.c_str(), std::ios::binary);
  if (!p) {
    std::cerr << "[Error] Could not open engine output file " << engine_name
              << std::endl;
    return -1;
  }
  p.write(reinterpret_cast<const char *>(modelStream->data()),
          modelStream->size());
  modelStream->destroy();

  std::cout << "[Info] Done" << std::endl;

  return 0;
}

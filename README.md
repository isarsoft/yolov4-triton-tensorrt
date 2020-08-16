# YOLOv4 on Triton Inference Server with TensorRT

![GitHub release (latest by date including pre-releases)](https://img.shields.io/github/v/release/Isarsoft/yolov4-triton-tensorrt?include_prereleases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository shows how to deploy YOLOv4 as an optimized [TensorRT](https://github.com/NVIDIA/tensorrt) engine to [Triton Inference Server](https://github.com/NVIDIA/triton-inference-server).

Triton Inference Server takes care of model deployment with many out-of-the-box benefits, like a GRPC and HTTP interface, automatic scheduling on multiple GPUs, shared memory (even on GPU), health metrics and memory resource management.

TensorRT will automatically optimize throughput and latency of our model by fusing layers and chosing the fastest layer implementations for our specific hardware. We will use the TensorRT API to generate the network from scratch and add all non-supported layers as a plugin.

## Build TensorRT engine

There are no dependencies needed to run this code, except a working docker environment with GPU support. We will run all compilation inside the TensorRT NGC container to avoid having to install TensorRT natively.

Run the following to get a running TensorRT container with our repo code:

```bash
cd yourworkingdirectoryhere
git clone git@github.com:isarsoft/yolov4-triton-tensorrt.git
docker run --gpus all -it --rm -v $(pwd)/yolov4-triton-tensorrt:/yolov4-triton-tensorrt nvcr.io/nvidia/tensorrt:20.06-py3
```

Docker will download the TensorRT container. You need to select the version (in this case 20.06) according to the version of Triton that you want to use later to ensure the TensorRT versions match. Matching NGC version tags use the same TensorRT version.

Inside the container run the following to compile our code:

```bash
cd /yolov4-triton-tensorrt
mkdir build
cd build
cmake ..
make
```

This will generate two files (`liblayerplugin.so` and `main`). The library contains all unsupported TensorRT layers and the executable will build us an optimized engine in a second.

Download the weights for this network from [Google Drive](https://drive.google.com/drive/folders/1YUDVgEefnk2HENpGMwq599Yj45i_7-iL?usp=sharing). Instructions on how to generate this weight file from the original darknet config and weights can be found [here](https://github.com/wang-xinyu/tensorrtx/tree/master/yolov4). Place the weight file in the same folder as the executable `main`. Then run the following to generate a serialized TensorRT engine optimized for your GPU:

```bash
./main
```

This will generate a file called `yolov4.engine`, which is our serialized TensorRT engine. Together with `liblayerplugin.so` we can now deploy to Triton Inference Server.

Before we do this we can test the engine with standalone TensorRT by running:

```bash
cd /workspace/tensorrt/bin
./trtexec --loadEngine=/yolov4-triton-tensorrt/build/yolov4.engine --plugins=/yolov4-triton-tensorrt/build/liblayerplugin.so
```

```
(...)
[I] Starting inference threads
[I] Warmup completed 1 queries over 200 ms
[I] Timing trace has 204 queries over 3.00185 s
[I] Trace averages of 10 runs:
[I] Average on 10 runs - GPU latency: 14.5469 ms - Host latency: 16.1718 ms (end to end 16.1964 ms, enqueue 2.69769 ms)
[I] Average on 10 runs - GPU latency: 13.1222 ms - Host latency: 14.7452 ms (end to end 14.7681 ms, enqueue 2.89363 ms)
(...)
[I] GPU Compute
[I] min: 12.241 ms
[I] max: 15.0692 ms
[I] mean: 13.1447 ms
```

## Deploy to Triton Inference Server

We need to create our model repository file structure first:

```bash
# Create model repository
cd yourworkingdirectoryhere
mkdir -p triton-deploy/model_repository/yolov4/1/
mkdir triton-deploy/plugins

# Copy engine and plugins
cp yolov4-triton-tensorrt/build/yolov4.engine triton-deploy/models/yolov4/1/model.plan
cp yolov4-triton-tensorrt/build/liblayerplugin.so triton-deploy/plugins/
```

Now we can start Triton with this model repository:

```bash
docker run --gpus all --rm --shm-size=1g --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 -p8000:8000 -p8001:8001 -p8002:8002 -v$(pwd)/triton-deploy/models:/models -v$(pwd)/triton-deploy/plugins:/plugins --env LD_PRELOAD=/plugins/liblayerplugin.so nvcr.io/nvidia/tritonserver:20.06-py3 tritonserver --model-repository=/models --strict-model-config=false --grpc-infer-allocation-pool-size=16 --log-verbose 1
```

This should give us a running Triton instance with our yolov4 model loaded. You can check out what to do next in the [Triton Documentation](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html).

## How to run model in your code

Triton has a very easy C++, Go and Python SDK with examples on how to run inference when the model is deployed on the server. It supports shared memory for basically zero copy latency when you run the code on the same device. This repo will be extended with a full implementation of such a client in the future, but it's really not hard to do by looking at the examples: https://github.com/NVIDIA/triton-inference-server/tree/master/src/clients

## Benchmark

To benchmark the performance of the model, we can run [Tritons Performance Client](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/optimization.html#perf-client).

To run the perf_client, get the Triton Client SDK docker container.

```bash
docker run -it --ipc=host --net=host nvcr.io/nvidia/tritonserver:20.06-py3-clientsdk /bin/bash
cd install/bin
./perf_client (...argumentshere)
# Example
./perf_client -m yolov4 -u 127.0.0.1:8001 -i grpc --concurrency-range 4
```

The following benchmarks were taken on a system with `2 x Nvidia 2080 TI` GPUs and an `AMD Ryzen 9 3950X` 16 Core CPU and with batchsize 1.

| concurrency / precision | FP32                                | FP16                                |
|-------------------------|-------------------------------------|-------------------------------------|
| 1                       | 44 infer/sec, latency 22633 usec    | 62.4 infer/sec, latency 15986 usec  |
| 2                       | 84.2 infer/sec, latency 23677 usec  | 136.2 infer/sec, latency 14675 usec |
| 4                       | 100.2 infer/sec, latency 39946 usec | 154.2 infer/sec, latency 19443 usec |
| 8                       | 99.2 infer/sec, latency 80552 usec  | 171 infer/sec, latency 46780 usec   |


## Tasks in this repo

- [x] Layer plugin working with trtexec and Triton
- [x] FP16 optimization
- [ ] INT8 optimization
  - [ ] Implement calibrator
  - [ ] Add support for INT8 in custom layers
  - [ ] Optional: use ReLU instead of Mish for layer fusion speedup
- [ ] General optimizations (using [this darknet->onnx->tensorrt export](https://github.com/Tianxiaomo/pytorch-YOLOv4#5-onnx2tensorrt-evolving) with --best flag gives 572 FPS (batchsize 8) and 392 FPS (batchsize 1) without full INT8 calibration)
- [ ] YOLOv4 tiny (example is [here](https://github.com/tjuskyzhang/yolov4-tiny-tensorrt))
- [ ] Add Triton client code in python
- [ ] Add image pre and postprocessing code

INT8 will give another big boost (maybe 2x - 3x ?) in performance, as the Tensor Cores on Nvidia GPUs will be activated. A first naive implementation did not result in performance improvements, because the custom layers do not support INT8 and have FP32 outputs, which breaks the optimization at multiple stages in the network. Optionally we can deactivate Mish and use standard ReLU instead. The weights and config for this are in the darknet repo.

## Acknowledgments

The initial codebase is from [Wang Xinyu](https://github.com/wang-xinyu) in his [TensorRTx](https://github.com/wang-xinyu/tensorrtx) repo. He had the idea to implement YOLO using only the TensorRT API and its very nice he shares this code. This repo has the purpose to deploy this engine and plugin to Triton and to add additional perfomance improvements as well as benchmarks.
#pragma once
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/utility.hpp>
#include <fstream>

#include "grpc_client.h"
#include "http_client.h"

namespace ni = nvidia::inferenceserver;
namespace nic = nvidia::inferenceserver::client;



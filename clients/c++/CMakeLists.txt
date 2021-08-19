cmake_minimum_required(VERSION 3.0)

project(yolov4-triton-cpp-client)
message(STATUS ${PROJECT_NAME}) 
message(STATUS $ENV{TritonClientThirdParty_DIR})
message(STATUS $ENV{TritonClientBuild_DIR})
set(CMAKE_CXX_STANDARD 17)



set(CURL_DIR "$ENV{TritonClientThirdParty_DIR}/curl/lib/cmake/CURL")
set(Protobuf_DIR "$ENV{TritonClientThirdParty_DIR}/protobuf/lib/cmake/protobuf")
set(gRPC_DIR "$ENV{TritonClientThirdParty_DIR}/grpc/lib/cmake/grpc")
set(c-ares_DIR "$ENV{TritonClientThirdParty_DIR}/c-ares/lib/cmake/c-ares")
set(TritonClient_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonClient")
set(TritonCommon_DIR "$ENV{TritonClientBuild_DIR}/lib/cmake/TritonCommon")

message(STATUS  ${CURL_DIR})
message(STATUS  ${Protobuf_DIR})
message(STATUS  ${gRPC_DIR})
message(STATUS  ${TritonClient_DIR})

find_package(OpenCV REQUIRED)
find_package(TritonCommon REQUIRED)
find_package(TritonClient REQUIRED)


add_executable(${PROJECT_NAME} ${PROJECT_SOURCE_DIR}/yolov4-client.cpp)
target_include_directories(
    ${PROJECT_NAME} 
    PRIVATE ${OpenCV_INCLUDE_DIRS} $ENV{TritonClientBuild_DIR}/include
  )
target_link_directories(${PROJECT_NAME} PRIVATE $ENV{TritonClientBuild_DIR}/lib)
target_link_libraries(${PROJECT_NAME} 
PRIVATE
grpcclient
httpclient
${OpenCV_LIBS}
)


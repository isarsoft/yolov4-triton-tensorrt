#pragma once
#include "common.hpp"
#include "Yolo.hpp"

namespace Triton{

    enum ScaleType { NONE = 0, YOLOV4 = 1};

    enum ProtocolType { HTTP = 0, GRPC = 1 };


    struct TritonModelInfo {
        std::string output_name_;
        std::vector<std::string> output_names_;
        std::string input_name_;
        std::string input_datatype_;
        // The shape of the input
        int input_c_;
        int input_h_;
        int input_w_;
        // The format of the input
        std::string input_format_;
        int type1_;
        int type3_;
        int max_batch_size_;

        std::vector<int64_t> shape_;

    };


    void setModel(TritonModelInfo& yoloModelInfo, const int batch_size){
        yoloModelInfo.output_names_ = std::vector<std::string>{"prob"};
        yoloModelInfo.input_name_ = "data";
        yoloModelInfo.input_datatype_ = std::string("FP32");
        // The shape of the input
        yoloModelInfo.input_c_ = Yolo::INPUT_C;
        yoloModelInfo.input_w_ = Yolo::INPUT_W;
        yoloModelInfo.input_h_ = Yolo::INPUT_H;
        // The format of the input
        yoloModelInfo.input_format_ = "FORMAT_NCHW";
        yoloModelInfo.type1_ = CV_32FC1;
        yoloModelInfo.type3_ = CV_32FC3;
        yoloModelInfo.max_batch_size_ = 32;
        yoloModelInfo.shape_.push_back(batch_size);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_c_);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_h_);
        yoloModelInfo.shape_.push_back(yoloModelInfo.input_w_);

    }

    union TritonClient
    {
        TritonClient()
        {
            new (&httpClient) std::unique_ptr<nic::InferenceServerHttpClient>{};
        }
        ~TritonClient() {}

        std::unique_ptr<nic::InferenceServerHttpClient> httpClient;
        std::unique_ptr<nic::InferenceServerGrpcClient> grpcClient;
    };



    std::vector<uint8_t> Preprocess(
        const cv::Mat& img, const std::string& format, int img_type1, int img_type3,
        size_t img_channels, const cv::Size& img_size, const ScaleType scale)
    {
        // Image channels are in BGR order. Currently model configuration
        // data doesn't provide any information as to the expected channel
        // orderings (like RGB, BGR). We are going to assume that RGB is the
        // most likely ordering and so change the channels to that ordering.
        std::vector<uint8_t> input_data;
        cv::Mat sample;
        cv::cvtColor(img, sample,  cv::COLOR_BGR2RGB);
        cv::Mat sample_resized;
        if (sample.size() != img_size)
        {
            cv::resize(sample, sample_resized, img_size);
        }
        else
        {
            sample_resized = sample;
        }

        cv::Mat sample_type;
        sample_resized.convertTo(
            sample_type, (img_channels == 3) ? img_type3 : img_type1);
    
        cv::Mat sample_final;
        sample.convertTo(
            sample_type, (img_channels == 3) ? img_type3 : img_type1);
        const int INPUT_W = Yolo::INPUT_W;
        const int INPUT_H = Yolo::INPUT_H;
        int w, h, x, y;
        float r_w = INPUT_W / (sample_type.cols * 1.0);
        float r_h = INPUT_H / (sample_type.rows * 1.0);
        if (r_h > r_w)
        {
            w = INPUT_W;
            h = r_w * sample_type.rows;
            x = 0;
            y = (INPUT_H - h) / 2;
        }
        else
        {
            w = r_h * sample_type.cols;
            h = INPUT_H;
            x = (INPUT_W - w) / 2;
            y = 0;
        }
        cv::Mat re(h, w, CV_8UC3);
        cv::resize(sample_type, re, re.size(), 0, 0, cv::INTER_CUBIC);
        cv::Mat out(INPUT_H, INPUT_W, CV_8UC3, cv::Scalar(128, 128, 128));
        re.copyTo(out(cv::Rect(x, y, re.cols, re.rows)));
        out.convertTo(sample_final, CV_32FC3, 1.f / 255.f);


        // Allocate a buffer to hold all image elements.
        size_t img_byte_size = sample_final.total() * sample_final.elemSize();
        size_t pos = 0;
        input_data.resize(img_byte_size);

        // (format.compare("FORMAT_NCHW") == 0)
        //
        // For CHW formats must split out each channel from the matrix and
        // order them as BBBB...GGGG...RRRR. To do this split the channels
        // of the image directly into 'input_data'. The BGR channels are
        // backed by the 'input_data' vector so that ends up with CHW
        // order of the data.
        std::vector<cv::Mat> input_bgr_channels;
        for (size_t i = 0; i < img_channels; ++i)
        {
            input_bgr_channels.emplace_back(
                img_size.height, img_size.width, img_type1, &(input_data[pos]));
            pos += input_bgr_channels.back().total() *
                input_bgr_channels.back().elemSize();
        }

        cv::split(sample_final, input_bgr_channels);

        if (pos != img_byte_size)
        {
            std::cerr << "unexpected total size of channels " << pos << ", expecting "
                << img_byte_size << std::endl;
            exit(1);
        }

        return input_data;
    }


    auto
    PostprocessYoloV4(
        nic::InferResult* result,
        const size_t batch_size,
        const std::vector<std::string>& output_names, const bool batching)
    {
        if (!result->RequestStatus().IsOk())
        {
            std::cerr << "inference  failed with error: " << result->RequestStatus()
                << std::endl;
            exit(1);
        }

        std::vector<float> detections;
        std::vector<int64_t> shape;


        float* outputData;
        size_t outputByteSize;
        for (auto outputName : output_names)
        {
            if (outputName == "prob")
            { 
                result->RawData(
                    outputName, (const uint8_t**)&outputData, &outputByteSize);

                nic::Error err = result->Shape(outputName, &shape);
                detections = std::vector<float>(outputByteSize / sizeof(float));
                std::memcpy(detections.data(), outputData, outputByteSize);
                if (!err.IsOk())
                {
                    std::cerr << "unable to get data for " << outputName << std::endl;
                    exit(1);
                }
            }

        }

        return make_tuple(detections, shape);
    }


    ScaleType
    ParseScale(const std::string& str)
    {
        if (str == "NONE")
        {
            return ScaleType::NONE;
        }
        else if (str == "YOLOV4")
        {
            return ScaleType::YOLOV4;
        }
        std::cerr << "unexpected scale type \"" << str
            << "\", expecting NONE, INCEPTION or VGG" << std::endl;
        exit(1);

        return ScaleType::NONE;
    }

    ProtocolType
    ParseProtocol(const std::string& str)
    {
        std::string protocol(str);
        std::transform(protocol.begin(), protocol.end(), protocol.begin(), ::tolower);
        if (protocol == "http")
        {
            return ProtocolType::HTTP;
        }
        else if (protocol == "grpc")
        {
            return ProtocolType::GRPC;
        }

        std::cerr << "unexpected protocol type \"" << str
            << "\", expecting HTTP or gRPC" << std::endl;
        exit(1);

        return ProtocolType::HTTP;
    }

    bool
    ParseType(const std::string& dtype, int* type1, int* type3)
    {
        if (dtype.compare("UINT8") == 0)
        {
            *type1 = CV_8UC1;
            *type3 = CV_8UC3;
        }
        else if (dtype.compare("INT8") == 0)
        {
            *type1 = CV_8SC1;
            *type3 = CV_8SC3;
        }
        else if (dtype.compare("UINT16") == 0)
        {
            *type1 = CV_16UC1;
            *type3 = CV_16UC3;
        }
        else if (dtype.compare("INT16") == 0)
        {
            *type1 = CV_16SC1;
            *type3 = CV_16SC3;
        }
        else if (dtype.compare("INT32") == 0)
        {
            *type1 = CV_32SC1;
            *type3 = CV_32SC3;
        }
        else if (dtype.compare("FP32") == 0)
        {
            *type1 = CV_32FC1;
            *type3 = CV_32FC3;
        }
        else if (dtype.compare("FP64") == 0)
        {
            *type1 = CV_64FC1;
            *type3 = CV_64FC3;
        }
        else
        {
            return false;
        }

        return true;
    }


}




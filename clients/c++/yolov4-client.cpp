#include "Yolo.hpp"
#include "Triton.hpp"



static const std::string keys = 
    "{ help h   | | Print help message. }"
    "{ video v | video.mp4 | video name}"
    "{ serverAddress  s  | localhost:8001 | Path to server address}"
    "{ verbose vb | false | Verbose mode, true or false}"
    "{ protocol p | grpc | Protocol type, grpc or http}"
    "{ labelsFile l | ../coco.names | path to  coco labels names}"
    "{ batch b | 1 | Batch size}";


int main(int argc, const char* argv[])
{
    cv::CommandLineParser parser(argc, argv, keys);
    if (parser.has("help")){
        parser.printMessage();
        return 0;  
    }

    std::string serverAddress = parser.get<std::string>("serverAddress");
    bool verbose = parser.get<bool>("verbose");
    std::string videoName;
    videoName = parser.get<std::string>("video");
    Triton::ProtocolType protocol; 
    if(parser.get<std::string>("protocol") == "grpc")
        protocol = Triton::ProtocolType::GRPC;
    else protocol = Triton::ProtocolType::HTTP;      
    const size_t batch_size = parser.get<size_t>("batch");

    Triton::ScaleType scale = Triton::ScaleType::YOLOV4;
    std::string preprocess_output_filename;
    std::string modelName = "yolov4";
    std::string modelVersion = "";
    std::string url(serverAddress);
    
    nic::Headers httpHeaders;

    const std::string fileName = parser.get<std::string>("labelsFile"); 

    std::cout << "Server address: " << serverAddress << std::endl;
    std::cout << "Video name: " << videoName << std::endl;
    std::cout << "Protocol:  " << parser.get<std::string>("protocol") << std::endl;
    std::cout << "Path to labels name:  " << fileName << std::endl;

    Triton::TritonClient tritonClient;
    nic::Error err;
    if (protocol == Triton::ProtocolType::HTTP)
    {
        err = nic::InferenceServerHttpClient::Create(
            &tritonClient.httpClient, url, verbose);
    }
    else
    {
        err = nic::InferenceServerGrpcClient::Create(
            &tritonClient.grpcClient, url, verbose);
    }
    if (!err.IsOk())
    {
        std::cerr << "error: unable to create client for inference: " << err
                  << std::endl;
        exit(1);
    }

    Triton::TritonModelInfo yoloModelInfo;
    Triton::setModel(yoloModelInfo, batch_size);

    nic::InferInput *input;
    err = nic::InferInput::Create(
        &input, yoloModelInfo.input_name_, yoloModelInfo.shape_, yoloModelInfo.input_datatype_);
    if (!err.IsOk())
    {
        std::cerr << "unable to get input: " << err << std::endl;
        exit(1);
    }

    std::shared_ptr<nic::InferInput> input_ptr(input);

    std::vector<nic::InferInput *> inputs = {input_ptr.get()};
    std::vector<const nic::InferRequestedOutput *> outputs;

    for (auto output_name : yoloModelInfo.output_names_)
    {
        nic::InferRequestedOutput *output;
        err =
            nic::InferRequestedOutput::Create(&output, output_name);
        if (!err.IsOk())
        {
            std::cerr << "unable to get output: " << err << std::endl;
            exit(1);
        }
        else
            std::cout << "Created output " << output_name << std::endl;
        outputs.push_back(std::move(output));
    }

    nic::InferOptions options(modelName);
    options.model_version_ = modelVersion;

    cv::Mat frame;
    std::vector<uint8_t> input_data;
    std::vector<cv::Mat> frameBatch;
    std::vector<std::vector<uint8_t>> input_data_raw;

    cv::VideoCapture cap(videoName);

    Yolo::coco_names = Yolo::readLabelNames(fileName);

    if(Yolo::coco_names.size() != Yolo::CLASS_NUM){
        std::cerr << "Wrong labels filename or wrong path to file: " << fileName << std::endl;
        exit(1);
    }

    while (cap.read(frame))
    {
        frameBatch.push_back(frame.clone());
        if (frameBatch.size() < batch_size)
        {
            continue;
        }

        // Reset the input for new request.
        err = input_ptr->Reset();
        if (!err.IsOk())
        {
            std::cerr << "failed resetting input: " << err << std::endl;
            exit(1);
        }

        for (size_t batchId = 0; batchId < batch_size; batchId++)
        {
            input_data_raw.push_back(Triton::Preprocess(
                frameBatch[batchId], yoloModelInfo.input_format_, yoloModelInfo.type1_, yoloModelInfo.type3_,
                yoloModelInfo.input_c_ , cv::Size(yoloModelInfo.input_w_, yoloModelInfo.input_h_), scale));
            err = input_ptr->AppendRaw(input_data_raw[batchId]);
            if (!err.IsOk())
            {
                std::cerr << "failed setting input: " << err << std::endl;
                exit(1);
            }
        }

        nic::InferResult *result;
        std::unique_ptr<nic::InferResult> result_ptr;
        if (protocol == Triton::ProtocolType::HTTP)
        {
            err = tritonClient.httpClient->Infer(
                &result, options, inputs, outputs);
        }
        else
        {
            err = tritonClient.grpcClient->Infer(
                &result, options, inputs, outputs);
        }
        if (!err.IsOk())
        {
            std::cerr << "failed sending synchronous infer request: " << err
                      << std::endl;
            exit(1);
        }
        
        const int DETECTION_SIZE = sizeof(Yolo::Detection) / sizeof(float);
        const int OUTPUT_SIZE = Yolo::MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;
        auto [detections, shape] = Triton::PostprocessYoloV4(result, batch_size, yoloModelInfo.output_names_, yoloModelInfo.max_batch_size_ != 0);
        std::vector<std::vector<Yolo::Detection>> batch_res(batch_size);    
        const float *prob = detections.data();        
        for (size_t batchId = 0; batchId < batch_size; batchId++) 
        {
            auto& res = batch_res[batchId];
            Yolo::nms(res, &prob[batchId * OUTPUT_SIZE]);
        }
        for (size_t batchId = 0; batchId < batch_size; batchId++) 
        {
            auto& res = batch_res[batchId];
            cv::Mat img = frameBatch.at(batchId);
            for (size_t j = 0; j < res.size(); j++) {
                cv::Rect r = Yolo::get_rect(img, res[j].bbox);
                cv::rectangle(img, r, cv::Scalar(0x27, 0xC1, 0x36), 2);
                cv::putText(img, Yolo::coco_names[(int)res[j].class_id], cv::Point(r.x, r.y - 1), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar(0xFF, 0xFF, 0xFF), 2);
            }
            cv::imshow("video feed " + std::to_string(batchId), img);
            cv::waitKey(1);
        }
        frameBatch.clear();
        input_data_raw.clear();
    }

    return 0;
}
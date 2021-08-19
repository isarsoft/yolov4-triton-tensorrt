#pragma once
#include "common.hpp"

namespace Yolo
{
    static constexpr int CHECK_COUNT = 3;
    static constexpr float IGNORE_THRESH = 0.1f;
    static constexpr int MAX_OUTPUT_BBOX_COUNT = 1000;
    static constexpr int CLASS_NUM = 80;
    static constexpr int INPUT_C = 3;
    static constexpr int INPUT_H = 608;
    static constexpr int INPUT_W = 608;

    static constexpr int LOCATIONS = 4;

    struct Detection{
        //x y w h
        float bbox[LOCATIONS];
        float det_confidence;
        float class_id;
        float class_confidence;
    };

    std::vector<std::string> coco_names;

    std::vector<std::string> readLabelNames(const std::string& fileName)
    {
        std::vector<std::string> classes;
        std::ifstream ifs(fileName.c_str());
        std::string line;
        while (getline(ifs, line))
        classes.push_back(line);
        return classes;   
    }


    cv::Rect get_rect(cv::Mat& img, float bbox[4]) 
    {
        int l, r, t, b;
        float r_w = INPUT_W / (img.cols * 1.0);
        float r_h = INPUT_H / (img.rows * 1.0);
        if (r_h > r_w) {
            l = bbox[0] - bbox[2]/2.f;
            r = bbox[0] + bbox[2]/2.f;
            t = bbox[1] - bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
            b = bbox[1] + bbox[3]/2.f - (INPUT_H - r_w * img.rows) / 2;
            l = l / r_w;
            r = r / r_w;
            t = t / r_w;
            b = b / r_w;
        } else {
            l = bbox[0] - bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
            r = bbox[0] + bbox[2]/2.f - (INPUT_W - r_h * img.cols) / 2;
            t = bbox[1] - bbox[3]/2.f;
            b = bbox[1] + bbox[3]/2.f;
            l = l / r_h;
            r = r / r_h;
            t = t / r_h;
            b = b / r_h;
        }
        return cv::Rect(l, t, r-l, b-t);
    }


    float iou(float lbox[4], float rbox[4]) 
    {
        float interBox[] = {
            std::max(lbox[0] - lbox[2]/2.f , rbox[0] - rbox[2]/2.f), //left
            std::min(lbox[0] + lbox[2]/2.f , rbox[0] + rbox[2]/2.f), //right
            std::max(lbox[1] - lbox[3]/2.f , rbox[1] - rbox[3]/2.f), //top
            std::min(lbox[1] + lbox[3]/2.f , rbox[1] + rbox[3]/2.f), //bottom
        };

        if(interBox[2] > interBox[3] || interBox[0] > interBox[1])
            return 0.0f;

        float interBoxS =(interBox[1]-interBox[0])*(interBox[3]-interBox[2]);
        return interBoxS/(lbox[2]*lbox[3] + rbox[2]*rbox[3] -interBoxS);
    }


    bool cmp(const Detection& a, const Detection& b) 
    {
        return a.det_confidence > b.det_confidence;
    }



    void nms(std::vector<Yolo::Detection>& res, const float* output, float nms_thresh = 0.4) {
        const float BBOX_CONF_THRESH = 0.5;
        const int BATCH_SIZE = 1;
        const int DETECTION_SIZE = sizeof(Detection) / sizeof(float);
        const int OUTPUT_SIZE = MAX_OUTPUT_BBOX_COUNT * DETECTION_SIZE + 1;  // we assume the yololayer outputs no more than MAX_OUTPUT_BBOX_COUNT boxes that conf >= 0.1
        std::map<float, std::vector<Detection>> m;
        for (int i = 0; i < output[0] && i < Yolo::MAX_OUTPUT_BBOX_COUNT; i++) {
            if (output[1 + DETECTION_SIZE * i + 4] <= BBOX_CONF_THRESH) continue;
            Detection det;
            memcpy(&det, &output[1 + DETECTION_SIZE * i], DETECTION_SIZE * sizeof(float));
            if (m.count(det.class_id) == 0) m.emplace(det.class_id, std::vector<Detection>());
            m[det.class_id].push_back(det);
        }
        for (auto it = m.begin(); it != m.end(); it++) {
            //std::cout << it->second[0].class_id << " --- " << std::endl;
            auto& dets = it->second;
            std::sort(dets.begin(), dets.end(), cmp);
            for (size_t m = 0; m < dets.size(); ++m) {
                auto& item = dets[m];
                res.push_back(item);
                for (size_t n = m + 1; n < dets.size(); ++n) {
                    if (iou(item.bbox, dets[n].bbox) > nms_thresh) {
                        dets.erase(dets.begin()+n);
                        --n;
                    }
                }
            }
        }
    }


}





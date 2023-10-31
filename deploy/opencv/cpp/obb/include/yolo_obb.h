/**
@author: CVHub
@date: 2023-10-28
**/

#pragma once

#include <iostream>
#include <cassert>
#include <opencv2/opencv.hpp>
#include <Eigen/Dense>

struct OBB {
    float cx;
    float cy;
    float longside;
    float shortside;
    float theta_pred;
    float max_class_score;
    int class_idx;
};

struct RotatedBox {
    cv::Point2f center;
    float w;
    float h;
    float theta;
};


class Yolov5_OBB
{
public:
    Yolov5_OBB()
    {
    }
    ~Yolov5_OBB() {}
    bool readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda);
    Eigen::MatrixXd detect(cv::Mat &image, cv::dnn::Net &net);
    void draw_polys(cv::Mat srcImg, Eigen::MatrixXd results);
    void draw_polys(cv::Mat srcImg, Eigen::MatrixXd results, std::string out_path);

private:
    cv::Mat preprocess(cv::Mat &srcImg);
    std::vector<cv::Mat> inference(cv::Mat &blob, cv::dnn::Net &net);
    Eigen::MatrixXd postprocess(std::vector<cv::Mat> &outputs);
    void letterBox(const cv::Mat &image, cv::Mat &outImage,
                   cv::Vec4d &params,
                   const cv::Size &newShape = cv::Size(640, 640),
                   bool autoShape = false,
                   bool scaleFill = false,
                   bool scaleUp = true,
                   int stride = 32,
                   const cv::Scalar &color = cv::Scalar(114, 114, 114));
    Eigen::MatrixXd rbox2poly(const std::vector<OBB>& obboxes);
    Eigen::MatrixXd scale_polys(Eigen::MatrixXd polys);

    const int _netWidth = 1024;
    const int _netHeight = 1024;

    cv::Vec4d _params; // [ratio_x, ratio_y, dw, dh]
    bool _hideLabel = true;
    bool _doLetterBox = true;
    float _iouThreshold = 0.20;
    float _confThreshold = 0.25;
    int _imgWidth;
    int _imgHeight;
    
    std::vector<std::string> _className = {
        "plane", "baseball-diamond", "bridge", "ground-track-field", 
        "small-vehicle", "large-vehicle", "ship", "tennis-court", 
        "basketball-court", "storage-tank", "soccer-ball-field", "roundabout",
        "harbor", "swimming-pool", "helicopter", "container-crane"
    };
    std::vector<cv::Scalar> _box_colors;
    cv::Scalar _text_color = cv::Scalar(255, 255, 255);
};

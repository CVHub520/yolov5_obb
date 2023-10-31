/**
@author: CVHub
@date: 2023-10-28
**/

#include "yolo_obb.h"

bool Yolov5_OBB::readModel(cv::dnn::Net &net, std::string &netPath, bool isCuda = false)
{
    try
    {
        net = cv::dnn::readNet(netPath);
    }
    catch (const std::exception &)
    {
        return false;
    }
    // cuda
    if (isCuda)
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    }
    // cpu
    else
    {
        net.setPreferableBackend(cv::dnn::DNN_BACKEND_DEFAULT);
        net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    }

    for (size_t i = 0; i < this->_className.size(); i++)
    {
        int b = rand() % 256;
        int g = rand() % 256;
        int r = rand() % 256;
        this->_box_colors.push_back(cv::Scalar(b, g, r));
    }
    return true;
}

void Yolov5_OBB::letterBox(const cv::Mat &image, cv::Mat &outImage, cv::Vec4d &params, const cv::Size &newShape,
                           bool autoShape, bool scaleFill, bool scaleUp, int stride, const cv::Scalar &color)
{
    if (false)
    {
        int maxLen = MAX(image.rows, image.cols);
        outImage = cv::Mat::zeros(cv::Size(maxLen, maxLen), CV_8UC3);
        image.copyTo(outImage(cv::Rect(0, 0, image.cols, image.rows)));
        params[0] = 1;
        params[1] = 1;
        params[3] = 0;
        params[2] = 0;
    }

    cv::Size shape = image.size();
    float r = std::min((float)newShape.height / (float)shape.height,
                       (float)newShape.width / (float)shape.width);
    if (!scaleUp)
        r = std::min(r, 1.0f);

    float ratio[2]{r, r};
    int new_un_pad[2] = {(int)std::round((float)shape.width * r), (int)std::round((float)shape.height * r)};

    auto dw = (float)(newShape.width - new_un_pad[0]);
    auto dh = (float)(newShape.height - new_un_pad[1]);
    if (autoShape)
    {
        dw = (float)((int)dw % stride);
        dh = (float)((int)dh % stride);
    }
    else if (scaleFill)
    {
        dw = 0.0f;
        dh = 0.0f;
        new_un_pad[0] = newShape.width;
        new_un_pad[1] = newShape.height;
        ratio[0] = (float)newShape.width / (float)shape.width;
        ratio[1] = (float)newShape.height / (float)shape.height;
    }

    dw /= 2.0f;
    dh /= 2.0f;

    if (shape.width != new_un_pad[0] && shape.height != new_un_pad[1])
    {
        cv::resize(image, outImage, cv::Size(new_un_pad[0], new_un_pad[1]));
    }
    else
    {
        outImage = image.clone();
    }

    int top = int(std::round(dh - 0.1f));
    int bottom = int(std::round(dh + 0.1f));
    int left = int(std::round(dw - 0.1f));
    int right = int(std::round(dw + 0.1f));
    params[0] = ratio[0];
    params[1] = ratio[1];
    params[2] = left;
    params[3] = top;
    cv::copyMakeBorder(outImage, outImage, top, bottom, left, right, cv::BORDER_CONSTANT, color);
}

cv::Mat Yolov5_OBB::preprocess(cv::Mat &srcImg)
{
    cv::Mat blob, netInputImg;
    this->_imgWidth = srcImg.cols;
    this->_imgHeight = srcImg.rows;

    letterBox(srcImg, netInputImg, this->_params, cv::Size(this->_netWidth, this->_netHeight));
    /*
        1. Scale input pixel values to 0 to 1
        2. BGR -> RGB
        3. HWC -> NCHW
        4. uint8 -> float32
    */
    cv::dnn::blobFromImage(netInputImg, blob, 1 / 255.0, cv::Size(this->_netWidth, this->_netHeight), cv::Scalar(0, 0, 0), true, false);

    return blob;
}

std::vector<cv::Mat> Yolov5_OBB::inference(cv::Mat &blob, cv::dnn::Net &net)
{
    net.setInput(blob);
    std::vector<cv::Mat> outputs;
    net.forward(outputs, net.getUnconnectedOutLayersNames());
    return outputs;
}

Eigen::MatrixXd Yolov5_OBB::postprocess(std::vector<cv::Mat> &outputs)
{
    int nc = outputs[0].size[2] - 5 - 180; // number of classes

    cv::Mat out(outputs[0].size[1], outputs[0].size[2], CV_32F, outputs[0].ptr<float>());

    std::vector<cv::RotatedRect> bboxes;
    std::vector<float> scores;
    std::vector<OBB> generate_boxes;

    for (int i = 0; i < out.rows; ++i) {
        float cx = out.at<float>(i, 0);
        float cy = out.at<float>(i, 1);
        float longside = out.at<float>(i, 2);
        float shortside = out.at<float>(i, 3);
        float obj_score = out.at<float>(i, 4);

        if (obj_score < this->_confThreshold)
            continue;

        cv::Mat class_scores = out.row(i).colRange(5, 5 + nc);
        class_scores *= obj_score;
        double minV, maxV;
        cv::Point minI, maxI;
        cv::minMaxLoc(class_scores, &minV, &maxV, &minI, &maxI);

        int class_idx = maxI.x;
        float max_class_score = maxV;
        if (max_class_score < this->_confThreshold)
            continue;
        scores.push_back(max_class_score);

        cv::Mat theta_scores = out.row(i).colRange(5 + nc, out.row(i).cols);
        cv::minMaxLoc(theta_scores, &minV, &maxV, &minI, &maxI);
        float theta_idx = maxI.x;
        float theta_pred = (theta_idx - 90) / 180 * M_PI;

        bboxes.push_back(cv::RotatedRect(cv::Point2f(cx, cy), cv::Size2f(longside, shortside), theta_pred));

        OBB obb;
        obb.cx = cx;
        obb.cy = cy;
        obb.longside = longside;
        obb.shortside = shortside;
        obb.theta_pred = theta_pred;
        obb.max_class_score = max_class_score;
        obb.class_idx = class_idx;
        generate_boxes.push_back(obb);
    }

    std::vector<int> indices;
    cv::dnn::NMSBoxes(bboxes, scores, this->_confThreshold, this->_iouThreshold, indices);

    std::vector<OBB> det;
    for (int idx : indices) {

        
        det.push_back(generate_boxes[idx]);
    }

    Eigen::MatrixXd pred_poly = rbox2poly(det);
    Eigen::MatrixXd scaled_poly = scale_polys(pred_poly);

    return scaled_poly;

}

Eigen::MatrixXd Yolov5_OBB::rbox2poly(const std::vector<OBB>& obboxes) {
    Eigen::MatrixXd polys(obboxes.size(), 10);

    for (size_t i = 0; i < obboxes.size(); ++i) {
        float cx = obboxes[i].cx;
        float cy = obboxes[i].cy;
        float w = obboxes[i].longside;
        float h = obboxes[i].shortside;
        float theta = obboxes[i].theta_pred;
        float conf = obboxes[i].max_class_score;
        int cls_id = obboxes[i].class_idx;

        double Cos = std::cos(theta);
        double Sin = std::sin(theta);

        Eigen::Vector2d vector1(w / 2 * Cos, -w / 2 * Sin);
        Eigen::Vector2d vector2(-h / 2 * Sin, -h / 2 * Cos);

        Eigen::Vector2d point1 = Eigen::Vector2d(cx, cy) + vector1 + vector2;
        Eigen::Vector2d point2 = Eigen::Vector2d(cx, cy) + vector1 - vector2;
        Eigen::Vector2d point3 = Eigen::Vector2d(cx, cy) - vector1 - vector2;
        Eigen::Vector2d point4 = Eigen::Vector2d(cx, cy) - vector1 + vector2;

        polys(i, 0) = point1[0];
        polys(i, 1) = point1[1];
        polys(i, 2) = point2[0];
        polys(i, 3) = point2[1];
        polys(i, 4) = point3[0];
        polys(i, 5) = point3[1];
        polys(i, 6) = point4[0];
        polys(i, 7) = point4[1];
        polys(i, 8) = conf;
        polys(i, 9) = cls_id;
    }

    return polys;
}

Eigen::MatrixXd Yolov5_OBB::scale_polys(Eigen::MatrixXd polys)
{
    double gain = std::min((double)this->_netHeight / this->_imgHeight, (double)this->_netWidth / this->_imgWidth);  // gain = resized / raw
    Eigen::Vector2d pad((this->_netWidth - this->_imgWidth * gain) / 2, (this->_netHeight - this->_imgHeight * gain) / 2);  // wh padding

    Eigen::MatrixXd padMatrix = Eigen::MatrixXd::Constant(polys.rows(), 2, pad[0]);
    padMatrix.col(1).array() = pad[1];

    for (int i = 0; i < polys.rows(); ++i) {
        polys(i, 0) = (polys(i, 0) - pad[0]) / gain;
        polys(i, 2) = (polys(i, 2) - pad[0]) / gain;
        polys(i, 4) = (polys(i, 4) - pad[0]) / gain;
        polys(i, 6) = (polys(i, 6) - pad[0]) / gain;
        polys(i, 1) = (polys(i, 1) - pad[1]) / gain;
        polys(i, 3) = (polys(i, 3) - pad[1]) / gain;
        polys(i, 5) = (polys(i, 5) - pad[1]) / gain;
        polys(i, 7) = (polys(i, 7) - pad[1]) / gain;
    }

    return polys;
}

Eigen::MatrixXd Yolov5_OBB::detect(cv::Mat &srcImg, cv::dnn::Net &net)
{
    cv::Mat blob = preprocess(srcImg);
    std::vector<cv::Mat> outputs = inference(blob, net);
    Eigen::MatrixXd results = postprocess(outputs);
    return results;
}

void Yolov5_OBB::draw_polys(cv::Mat srcImg, Eigen::MatrixXd results)
{
    int lw = std::max(static_cast<int>(std::round((srcImg.rows + srcImg.cols) / 2 * 0.003)), 2);

    for (int i = 0; i < results.rows(); ++i) {
        
        float conf = results(i, 8);
        int cls_id = results(i, 9);
        cv::Scalar color = this->_box_colors[cls_id];
        std::string label = this->_className[cls_id] + " " + std::to_string(conf);

        std::vector<cv::Point> polygon_list;
        for (int j = 0; j < 4; ++j) {
            cv::Point point(static_cast<int>(results(i, j * 2)), static_cast<int>(results(i, j * 2 + 1)));
            polygon_list.push_back(point);
        }

        std::vector<std::vector<cv::Point>> contours;
        contours.push_back(polygon_list);
        
        cv::drawContours(srcImg, contours, -1, color, lw, cv::LINE_AA);

        cv::Point labelPosition;
        labelPosition.x = std::min(std::min(polygon_list[0].x, polygon_list[2].x), std::min(polygon_list[1].x, polygon_list[3].x));
        labelPosition.y = std::min(std::min(polygon_list[0].y, polygon_list[2].y), std::min(polygon_list[1].y, polygon_list[3].y));

        if (!label.empty() && !this->_hideLabel) {
            int tf = std::max(lw - 1, 1);
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, lw / 3, tf, nullptr);
            int labelHeight = textSize.height;
            cv::rectangle(srcImg, labelPosition, cv::Point(labelPosition.x + textSize.width + 1, labelPosition.y + int(1.5 * labelHeight)), color, -1, cv::LINE_AA);
            cv::putText(srcImg, label, cv::Point(labelPosition.x, labelPosition.y + labelHeight), cv::FONT_HERSHEY_SIMPLEX, lw / 3, this->_text_color, tf, cv::LINE_AA);
        }
    }

    cv::imshow("Results", srcImg);
    cv::waitKey(0);
}

void Yolov5_OBB::draw_polys(cv::Mat srcImg, Eigen::MatrixXd results, std::string out_path)
{
    int lw = std::max(static_cast<int>(std::round((srcImg.rows + srcImg.cols) / 2 * 0.003)), 2);

    for (int i = 0; i < results.rows(); ++i) {
        
        float conf = results(i, 8);
        int cls_id = results(i, 9);
        cv::Scalar color = this->_box_colors[cls_id];
        std::string label = this->_className[cls_id] + " " + std::to_string(conf);

        std::vector<cv::Point> polygon_list;
        for (int j = 0; j < 4; ++j) {
            cv::Point point(static_cast<int>(results(i, j * 2)), static_cast<int>(results(i, j * 2 + 1)));
            polygon_list.push_back(point);
        }

        std::vector<std::vector<cv::Point>> contours;
        contours.push_back(polygon_list);
        
        cv::drawContours(srcImg, contours, -1, color, lw, cv::LINE_AA);

        cv::Point labelPosition;
        labelPosition.x = std::min(std::min(polygon_list[0].x, polygon_list[2].x), std::min(polygon_list[1].x, polygon_list[3].x));
        labelPosition.y = std::min(std::min(polygon_list[0].y, polygon_list[2].y), std::min(polygon_list[1].y, polygon_list[3].y));

        if (!label.empty() && !this->_hideLabel) {
            int tf = std::max(lw - 1, 1);
            cv::Size textSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, lw / 3, tf, nullptr);
            int labelHeight = textSize.height;
            cv::rectangle(srcImg, labelPosition, cv::Point(labelPosition.x + textSize.width + 1, labelPosition.y + int(1.5 * labelHeight)), color, -1, cv::LINE_AA);
            cv::putText(srcImg, label, cv::Point(labelPosition.x, labelPosition.y + labelHeight), cv::FONT_HERSHEY_SIMPLEX, lw / 3, this->_text_color, tf, cv::LINE_AA);
        }
    }

    cv::imshow("Results", srcImg);

}
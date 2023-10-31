#include <fstream>
#include "yolo_obb.h"

int main()
{
    bool use_cuda = true;
    std::string root_path = "/home/cvhub/workspace/projects/cplusplus/yolov5_obb/";
    std::string model_path = root_path + "model/yolov5m_obb_csl_dotav15.onnx";
    std::string image_path = root_path + "image/demo.jpg";
    std::string out_path = root_path + "image/demo_out.jpg";

    Yolov5_OBB yolov5_obb;
    cv::dnn::Net net;

    if (yolov5_obb.readModel(net, model_path, use_cuda))
    {
        std::cout << "Initialize yolov5-obb model ok!" << std::endl;
    }
    else
    {
        std::cout << "Initialize yolov5-obb model error!" << std::endl;
        return -1;
    }

    cv::Mat srcImg = cv::imread(image_path);
    Eigen::MatrixXd results = yolov5_obb.detect(srcImg, net);

    if (!out_path.empty()) {
        yolov5_obb.draw_polys(srcImg, results, out_path);
    }
    else
    {
        yolov5_obb.draw_polys(srcImg, results);
    }

    return 0;
}
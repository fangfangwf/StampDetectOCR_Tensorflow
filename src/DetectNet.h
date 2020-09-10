#pragma once
#include<iostream>
#include"dataStructures.h"
#include <opencv2/dnn.hpp>

class DetectNet 
{
public:
	_declspec(dllexport) static DetectNet* getInstance(std::string yoloClassesFile, std::string yoloModelConfiguration, std::string yoloModelWeights);
	_declspec(dllexport) const std::vector<std::string>& getDetectClasses();
	_declspec(dllexport) const cv::dnn::Net& getDNNNet();
	_declspec(dllexport) void  getDetectResultsFromImage(cv::Mat& img, std::vector<BoundingBox>& bBoxes,
									float detect_conf_threshold = 0.20, 
									float nms_threshold = 0.4);

protected:
	DetectNet(std::string yoloClassesFile, std::string yoloModelConfiguration, std::string yoloModelWeights);

	std::vector<std::string> m_detectClasses;
	cv::dnn::Net m_dnnNet;

	static DetectNet* m_pInstance;

	void pageLayoutAnalysis(cv::Mat& img, std::vector<BoundingBox>& bBoxes, std::vector<std::string>& m_detectClasses);
};
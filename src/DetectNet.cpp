#include"DetectNet.h"
#include"dataStructures.h"
#include<fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/opencv.hpp>

DetectNet * DetectNet::m_pInstance = NULL;

DetectNet::DetectNet(std::string yoloClassesFile, std::string yoloModelConfiguration, std::string yoloModelWeights)
{
	m_detectClasses.clear();
	std::ifstream ifs(yoloClassesFile.c_str());
	std::string line;
	while (getline(ifs, line)) m_detectClasses.push_back(line);

	m_dnnNet = cv::dnn::readNetFromDarknet(yoloModelConfiguration, yoloModelWeights);
	m_dnnNet.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
	m_dnnNet.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
}

const cv::dnn::Net& DetectNet::getDNNNet()
{
	return m_dnnNet;
}

const std::vector<std::string>& DetectNet::getDetectClasses()
{
	return m_detectClasses;
}

DetectNet* DetectNet::getInstance(std::string yoloClassesFile, std::string yoloModelConfiguration, std::string yoloModelWeights)
{
	if (NULL == m_pInstance)
	{
		m_pInstance = new DetectNet(yoloClassesFile, yoloModelConfiguration, yoloModelWeights);
	}
	return m_pInstance;
}

void DetectNet::pageLayoutAnalysis(cv::Mat& img, 
								   std::vector<BoundingBox>& bBoxes, 
								   std::vector<std::string>& detectClasses)
{
	int img_h = img.rows;
	int img_w = img.cols;

	bool isTypeOrStampTitle = false;

	cv::Rect aim_roi;
	for (int i = 0; i < bBoxes.size(); i++)
	{
		BoundingBox bBox = bBoxes[i];
		int classId = bBox.classID;
		if ((detectClasses[classId] == "type") || (detectClasses[classId] == "stamp_title"))
		{
			isTypeOrStampTitle = true;
			aim_roi = bBox.roi;
			break;
		}
	}

	int rotate = 0; // Ðý×ª½Ç¶È
	if (isTypeOrStampTitle)
	{
		int aim_h = aim_roi.height;
		int aim_w = aim_roi.width;

		if (aim_h < aim_w)
		{
			if (aim_roi.y > int(1.0 / 2 * img_h)) rotate = 180;
		}
		else
		{
			if (aim_roi.x > int(1.0 / 2 * img_w)) rotate = 90; else rotate = 270;
		}
		//std::cout << "Rotate:" << rotate << std::endl;
		cv::Mat temp, dst;

		for (int i = 0; i < (rotate / 90); i++)
		{
			transpose(img, temp);
			flip(temp, dst, 0);
			img = dst;
		}

		for (int i = 0; i < bBoxes.size(); i++)
		{
			BoundingBox bBox = bBoxes[i];
			cv::Rect temp_roi = bBox.roi;
			cv::Rect new_roi = temp_roi;
			if (rotate == 90)
			{
				new_roi.x = temp_roi.y;
				new_roi.y = img_w - temp_roi.x - temp_roi.width;
				new_roi.height = temp_roi.width;
				new_roi.width = temp_roi.height;
			}

			if (rotate == 180)
			{
				new_roi.x = img_w - temp_roi.x - temp_roi.width;
				new_roi.y = img_h - temp_roi.y - temp_roi.height;
			}
			if (rotate == 270)
			{
				new_roi.x = img_w - temp_roi.y - temp_roi.height;
				new_roi.y = temp_roi.x;
				new_roi.height = temp_roi.width;
				new_roi.width = temp_roi.height;
			}
			bBoxes[i].roi = new_roi;
			bBoxes[i].img = cv::Mat(img, new_roi);
		}
	}
}

void DetectNet::getDetectResultsFromImage(cv::Mat& img, std::vector<BoundingBox>& bBoxes,
	float detect_conf_threshold,float nms_threshold )
{
	cv::Mat detect_blob;
	double scalefactor = 1 / 255.0;
	cv::Size size = cv::Size(416, 416);
	cv::Scalar mean = cv::Scalar(0, 0, 0);
	bool swapRB = false;
	bool crop = false;

	cv::dnn::blobFromImage(img, detect_blob, scalefactor, size, mean, swapRB, crop);
	std::vector<cv::String> detect_names;

	std::vector<int> detect_out_layers = m_dnnNet.getUnconnectedOutLayers();
	std::vector<cv::String> detect_layers_names = m_dnnNet.getLayerNames();

	detect_names.resize(detect_out_layers.size());
	for (size_t i = 0; i < detect_out_layers.size(); i++)
	{
		detect_names[i] = detect_layers_names[detect_out_layers[i] - 1];
	}

	std::vector<cv::Mat> detect_net_output;
	m_dnnNet.setInput(detect_blob);
	m_dnnNet.forward(detect_net_output, detect_names);
	
	std::vector<int> detect_class_ids;
	std::vector<float> detect_confidences;
	std::vector<cv::Rect> detect_boxes;

	for (size_t i = 0; i < detect_net_output.size(); i++)
	{
		float* data = (float*)detect_net_output[i].data;
		for (int j = 0; j < detect_net_output[i].rows; j++, data += detect_net_output[i].cols)
		{
			cv::Mat scores = detect_net_output[i].row(j).colRange(5, detect_net_output[i].cols);
			cv::Point classId;
			double confidence;
			cv::minMaxLoc(scores, 0, &confidence, 0, &classId);
			if (confidence > detect_conf_threshold)
			{
				cv::Rect box; int cx, cy;
				cx = (int)(data[0] * img.cols);
				cy = (int)(data[1] * img.rows);
				box.width = (int)(data[2] * img.cols);
				box.height = (int)(data[3] * img.rows);
				box.x = cx - box.width / 2;  //left
				box.y = cy - box.height / 2;  //top
				detect_boxes.push_back(box);
				detect_class_ids.push_back(classId.x);
				detect_confidences.push_back((float)confidence);
			}
		}
	}
	// perform non-maxima suppression
	std::vector<int> indices;
	cv::dnn::NMSBoxes(detect_boxes, detect_confidences, detect_conf_threshold, nms_threshold, indices);
	bBoxes.clear();
	for (auto it = indices.begin(); it != indices.end(); ++it)
	{
		BoundingBox bBox;
		bBox.roi = detect_boxes[*it];
		bBox.classID = detect_class_ids[*it];
		bBox.confidence = detect_confidences[*it];
		bBox.boxID = (int)bBoxes.size();
		bBox.img = cv::Mat(img, bBox.roi);
		bBox.text = "";
		bBoxes.push_back(bBox);
	}
	pageLayoutAnalysis(img, bBoxes, m_detectClasses);
}
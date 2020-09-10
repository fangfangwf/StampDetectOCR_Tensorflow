#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS

#include <io.h>
#include <iostream>
#include <numeric>
#include <fstream>
#include <locale>
#include <Windows.h>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/opencv.hpp>
#include "PutText.h"
#include "dataStructures.h"
#include "OCRNet.h"
#include "DetectNet.h"
#include <ctime>


using namespace tensorflow;
using namespace std;

void  get_all_files(string srcPath, vector<string>& files)
{
	intptr_t handle;
	struct _finddata_t fileInfo;
	handle = _findfirst(srcPath.c_str(), &fileInfo);
	if (handle == -1) { return; }
	do
	{
		files.push_back(string(fileInfo.name));
	} while (!_findnext(handle, &fileInfo));
	_findclose(handle);
}


void detect_ocr()
{
	string src_path_file = "../data/detect/*.jpg";
	string src_path = "../data/detect/";
	string dst_path = "../predict/images/";

	vector<string> files;
	get_all_files(src_path_file, files);
	clock_t time_start;
	time_start = clock();
	for (int k = 0; k < files.size(); k++)
	{
		string file = files[k];
		string file_name = src_path + file;

		size_t index = file.find_last_of(".");
		string shortname = file.substr(0, index);
		string save_file_name = dst_path + shortname + "_predict.jpg";
		cout << save_file_name << endl;
		cv::Mat img = cv::imread(file_name);
		std::vector<BoundingBox> bBoxes;

		std::string yoloBasePath = "../model/yolo/";
		std::string yoloClassesFile = yoloBasePath + "jz.names";
		std::string yoloModelConfiguration = yoloBasePath + "yolov3_jz.cfg";
		std::string yoloModelWeights = yoloBasePath + "yolov3_jz.weights";
		DetectNet* m_detectNet = DetectNet::getInstance(yoloClassesFile, yoloModelConfiguration, yoloModelWeights);
		m_detectNet->getDetectResultsFromImage(img, bBoxes);


		std::string char_path = "../model/char_dict/jz_char.txt";
		std::string model_path = "../model/crnn/shadownet_2020-04-25-15-06-30-99000.pb";
		OCRNet* m_ocrNet = OCRNet::getInstance(model_path, char_path);
		m_ocrNet->getOCRResultsFromBBoxes(bBoxes, m_detectNet->getDetectClasses());


		cv::Mat visImg = img.clone();
		for (auto it = bBoxes.begin(); it != bBoxes.end(); it++)
		{  
			int top, left, width, height;
			top = (*it).roi.y;
			left = (*it).roi.x;
			width = (*it).roi.width;
			height = (*it).roi.height;
			cv::rectangle(visImg, cv::Point(left, top), cv::Point(left + width, top + height), cv::Scalar(255, 0, 0), 2);
			string label = (*it).text;
			if (label.length() > 0)
			{
				int baseLine;
				cv::Size label_size = getTextSize(label, cv::FONT_ITALIC, 0.5, 1, &baseLine);
				top = max(top, label_size.height);
				putTextZH(visImg, label.c_str(), cv::Point(left, top - round(1.5*label_size.height)), Scalar(0, 0, 255), 25, "Î¢ÈíÑÅºÚ", false, false);
			}
		}
		cv::imwrite(save_file_name, visImg);
	}
	clock_t time_end;
	time_end = clock();
	double total_time = (double)(time_end - time_start) / CLOCKS_PER_SEC;
	std::cout << "Total time is " << total_time << std::endl;
	std::cout << "FPS:" << files.size() / total_time << std::endl;

}

int main()
{
	detect_ocr();
	return 0;
}



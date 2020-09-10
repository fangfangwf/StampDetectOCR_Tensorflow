#include"OCRNet.h"
#include"dataStructures.h"
#include<fstream>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc/types_c.h>  
#include <opencv2/opencv.hpp>
#include"tensorflow/core/public/session.h"
#include"tensorflow/core/platform/env.h"

OCRNet* OCRNet::m_pInstance = NULL;

OCRNet::OCRNet(std::string model_path, std::string char_path)
{
	m_ocr_char_list.clear();
	std::string line;
	std::ifstream ifs(char_path.c_str());
	while (getline(ifs, line)) m_ocr_char_list.push_back(line);

	tensorflow::GraphDef graph_def;
	TF_CHECK_OK(tensorflow::ReadBinaryProto(tensorflow::Env::Default(), model_path, &graph_def));
	m_ocr_session = std::unique_ptr<tensorflow::Session>(tensorflow::NewSession(tensorflow::SessionOptions()));
	TF_CHECK_OK(m_ocr_session->Create(graph_def));
}

OCRNet* OCRNet::getInstance(std::string model_path, std::string char_path)
{
	if (NULL == m_pInstance)
	{
		m_pInstance = new OCRNet(model_path, char_path);
	}
	return m_pInstance;
}

const std::vector<std::string>& OCRNet::getOCRCharList()
{
	return m_ocr_char_list;
}

std::unique_ptr<tensorflow::Session>& OCRNet::getOCRSession()
{
	return m_ocr_session;
}


void OCRNet::getOCRResultsFromBBoxes(std::vector<BoundingBox>& bBoxes, const std::vector<std::string>& detect_classes)
{
	std::string input_name = "input:0";
	std::string output_name = "CTCBeamSearchDecoder:1";
	int channels = 3;
	int input_height = 32;
	int input_width = 280;
	float input_std = 255.0;

	tensorflow::Tensor input_tensor(tensorflow::DT_FLOAT, tensorflow::TensorShape({ 1, input_height, input_width, channels }));

	for (int i = 0; i < bBoxes.size(); i++)
	{
		int index = bBoxes[i].classID;
		if (detect_classes[index] == "stamp")
		{
			std::cout << "Is stamp." << std::endl;
			continue;
		}
		cv::Mat img = bBoxes[i].img;
		cv::Mat img_resize;
		//int channels = img.channels();

		img.convertTo(img, CV_32F);
		cv::resize(img, img_resize, cv::Size(input_width, input_height), cv::INTER_LINEAR);

		img_resize = img_resize / 255.0;

		float* source_data = (float*)img_resize.data;

		auto input_tensor_mapped = input_tensor.tensor<float, 4>();

		for (int i = 0; i < input_height; i++)
		{
			float *source_row = source_data + (i*input_width*channels);
			for (int j = 0; j < input_width; j++)
			{
				float* source_pixel = source_row + (j*channels);
				for (int c = 0; c < channels; c++)
				{
					float* source_value = source_pixel + c;
					input_tensor_mapped(0, i, j, c) = *source_value;
				}
			}
		}

		std::vector<std::pair<std::string, tensorflow::Tensor>> inputs = { { input_name, input_tensor } };

		std::vector<tensorflow::Tensor> outputs;

		//TF_CHECK_OK(session->Run(inputs, { output_name }, {}, &outputs));
		TF_CHECK_OK(m_ocr_session->Run(inputs, { output_name }, {}, &outputs));
		tensorflow::Tensor output = std::move(outputs.at(0));
		auto out_shape = output.shape();
		auto out_val = output.tensor<tensorflow::int64, 1>();
		std::string out_str = "";

		for (int i = 0; i < output.dim_size(0); i++)
		{
			int value = (int)out_val(i);
			out_str += m_ocr_char_list[value];
		}
		bBoxes[i].text = out_str;
		//std::cout << out_str << std::endl;
	}
}

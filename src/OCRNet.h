#define COMPILER_MSVC
#define NOMINMAX
#define PLATFORM_WINDOWS
#pragma once
#include<iostream>
#include"dataStructures.h"
#include"tensorflow/core/public/session.h"
#include"tensorflow/core/platform/env.h"

class OCRNet
{
public:
	_declspec(dllexport) static OCRNet* getInstance(std::string model_path, std::string char_path);
	_declspec(dllexport) const std::vector<std::string>& getOCRCharList();
	_declspec(dllexport) std::unique_ptr<tensorflow::Session>& getOCRSession();

	_declspec(dllexport) void getOCRResultsFromBBoxes(std::vector<BoundingBox>& bBoxes,const std::vector<std::string>& detect_classes);


private:
	OCRNet(std::string model_path, std::string char_path);
	static OCRNet *m_pInstance;

	std::vector<std::string> m_ocr_char_list;
    std::unique_ptr<tensorflow::Session> m_ocr_session;
};

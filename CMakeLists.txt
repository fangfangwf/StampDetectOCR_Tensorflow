cmake_minimum_required(VERSION 2.8)
project(JinZhongBankSolution_CPU)
set(CMAKE_CXX_STANDARD 11)

set(ThirdParty_dir "${PROJECT_SOURCE_DIR}/3rdparty")
set(Opencv_DIR "${ThirdParty_dir}/opencv")
find_package(Opencv 4.1.0 REQUIRED)


include_directories(${ThirdParty_dir}/tensorflow/include
					${OpenCV_INCLUDE_DIRS}
                    )

message(STATUS "${ThirdParty_dir}")
message(STATUS "${PROJECT_SOURCE_DIR}")
message(STATUS "${PROJECT_BINARY_DIR}")
link_directories(${ThirdParty_dir}/tensorflow/lib
				 ${OpenCV_LIBRARY_DIRS}
				)
link_libraries(tensorflow.lib)	
add_definitions(${OpenCV_DEFINITIONS})
message(STATUS "${OpenCV_DEFINITIONS}")

# set(LIBRARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/DetectOcr")
# message(STATUS "${LIBRARY_OUTPUT_PATH}")
# add_library(DetectOcr SHARED src/PutText.h src/PutText.cpp src/dataStructures.h src/OCRNet.h src/OCRNet.cpp src/DetectNet.h src/DetectNet.cpp)
# target_link_libraries(DetectOcr ${OpenCV_LIBRARIES})


#set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/Release)
add_executable(JinZhongBankSolution_CPU src/main.cpp src/PutText.h src/PutText.cpp src/dataStructures.h src/OCRNet.h src/OCRNet.cpp src/DetectNet.h src/DetectNet.cpp)
target_link_libraries(JinZhongBankSolution_CPU ${OpenCV_LIBRARIES})

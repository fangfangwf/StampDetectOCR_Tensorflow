#ifndef dataStructures_h
#define dataStructures_h

#include <vector>
#include <string>
#include <opencv2/core.hpp>


struct LidarPoint { // single lidar point in space
    double x,y,z,r; // x,y,z in [m], r is point reflectivity
};

struct BoundingBox { // bounding box around a classified object (contains both 2D and 3D data)
    
    int boxID; // unique identifier for this bounding box
    int trackID; // unique identifier for the track to which this bounding box belongs
    
    cv::Rect roi; // 2D region-of-interest in image coordinates
    int classID; // ID based on class file provided to YOLO framework
    double confidence; // classification trust
	std::string text;  // ocr results
	cv::Mat img;       // detect aim image
};

#endif /* dataStructures_h */

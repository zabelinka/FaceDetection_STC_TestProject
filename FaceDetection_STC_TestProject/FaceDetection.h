#ifndef FACE_DETECTION_H
#define FACE_DETECTION_H

#include <iostream> 

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;

class FaceDetector
{
public:
	FaceDetector();
	cv::Mat getFace(const cv::Mat &frame);

private:
	cv::Mat detectFace(const cv::Mat &frame_gray);

private:
	CascadeClassifier classifier;
};




FaceDetector::FaceDetector(){
	// Load the cascade
	if (!classifier.load("haarcascade_frontalface_default.xml")){
		cout << "Error loading cascade.\n" << endl;
	}
}


Mat FaceDetector::getFace(const Mat &frame){
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);

	Mat face;

	// Apply the classifier to the frame
	if (frame_gray.data)
	{
		face = detectFace(frame_gray);
	}

	return face;
}

Mat FaceDetector::detectFace(const Mat &frame_gray){
	std::vector<Rect> faces;
	Mat face;
	int neighbors = 3;

	while (true){
		classifier.detectMultiScale(frame_gray, faces, 1.1, neighbors, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		cout << "FaceDetector:\tNeighbors: " << neighbors << " Face count:" << faces.size() << endl;

		if (faces.size() == 0){
			cout << "Faces not found. minNeighbors paramether = " << neighbors << endl;
			break;
		}

		if (faces.size() == 1){
			face = frame_gray(faces[0]);
			break;
		}

		neighbors++;
	}

	return face;
}



#endif

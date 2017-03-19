#ifndef EDGE_DETECTION_H
#define EDGE_DETECTION_H

#include <iostream> 

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

double SobelDeviation(const Mat&img)
{
	Mat dx, dy, gradient;
	Sobel(img, dx, CV_32F, 1, 0, 3);
	Sobel(img, dy, CV_32F, 0, 1, 3);
	magnitude(dx, dy, gradient);

	// convert dx, dy back to CV_8U
	Mat abs_grad_x, abs_grad_y;
	convertScaleAbs(dx, abs_grad_x);
	convertScaleAbs(dy, abs_grad_y);
	convertScaleAbs(gradient, gradient);

	// imshow("Sobel dx", abs_grad_x);
	// imshow("Sobel dy", abs_grad_y);
	imshow("Sobel Gradient", gradient);

	Scalar mean;
	Scalar deviation;
	meanStdDev(gradient, mean, deviation);

	cout << "Sobel" << endl;
	cout << "mean " << mean << endl;
	cout << "deviation " << deviation << endl;
	return deviation[0];
}


double LaplacianDeviation(Mat image){
	Mat laplacianOutput;
	cv::Laplacian(image, laplacianOutput, CV_8U, 3, 1, 0, cv::BORDER_DEFAULT);

	imshow("Laplacian", laplacianOutput);

	Scalar mean;
	Scalar deviation;
	meanStdDev(laplacianOutput, mean, deviation);

	cout << "Laplacian" << endl;
	cout << "mean " << mean << endl;
	cout << "deviation: " << deviation << endl;

	return deviation[0];
}

#endif
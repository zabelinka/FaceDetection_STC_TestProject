#include <iostream>
#include <algorithm>   
#include <vector>      

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

#include "FaceDetection.h"
#include "FourierTransformation.h"

using namespace cv;
using namespace std;


double variance_of_sobel(const Mat&img);

double variance_of_laplacian(Mat image){
	Mat laplacianOutput;
	cv::Laplacian(image, laplacianOutput, CV_8U, 3, 1, 0, cv::BORDER_DEFAULT);
	imshow("laplacian", laplacianOutput);
	Scalar mean;
	Scalar deviation;
	meanStdDev(laplacianOutput, mean,deviation);
	cout << "Laplacian" << endl;
	cout << "mean " << mean << endl;
	cout << "deviation: " << deviation << endl;

	double variance = deviation[0];
	return variance;
}





int main()
{
	// load image
	Mat frame = imread("adele-0.jpg");
	if (!frame.data){
		cout << "File not loaded." << endl;
		return -1;
	}

	// face detection
	FaceDetector faceDetector = FaceDetector();
	Mat face = faceDetector.getFace(frame);

	// measure sharpness

	// Method 1: Using Fourier Transformation and contrast
	Mat spectrum = fourierSpectrum(face);
	imshow("Spectrum magnitude", spectrum);
	Mat histogram = calcHistogram(spectrum);
	imshow("Spectrum histogram", histogram);

	measureContrast(face);
	

	// Method 2: Using Sobel mask and Laplacian

	/*cout << variance_of_sobel(face) << endl;
	variance_of_laplacian(face);*/

	waitKey();
	return 0;
}




double variance_of_sobel(const Mat&img)
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

	// imshow("sobel_dx", abs_grad_x);
	// imshow("sobel_dy", abs_grad_y);
	imshow("sobel_gradient", gradient);

	Scalar mean;
	Scalar deviation;
	meanStdDev(gradient, mean, deviation);

	cout << "Sobel" << endl;
	cout << "mean " << mean << endl;
	cout << "deviation " << deviation << endl;
	return deviation[0];
}


#include <iostream>
#include <algorithm>   
#include <vector>      

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

#include "FaceDetection.h"
#include "FourierTransformation.h"
#include "EdgeDetection.h"

using namespace cv;
using namespace std;

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
	/*Mat spectrum = fourierSpectrum(face);
	imshow("Spectrum magnitude", spectrum);
	Mat histogram = calcHistogram(spectrum);
	imshow("Spectrum histogram", histogram);

	measureContrast(face);*/
	

	// Method 2: Using Sobel mask and Laplacian

	cout << "Sobel Deviation: " << SobelDeviation(face) << endl;
	cout << "Laplacian Deviation: " << LaplacianDeviation(face) << endl;

	waitKey();
	return 0;
}






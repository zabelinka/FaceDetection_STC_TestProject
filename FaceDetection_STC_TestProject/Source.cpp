#include <iostream>
#include <algorithm>   
#include <vector>      

#include <opencv2/opencv.hpp>
#include "opencv2/core/core.hpp"

using namespace cv;
using namespace std;


Mat detectFace(Mat, CascadeClassifier );
Mat fourierSpectrum(Mat frame);
void measureContrast(Mat);
void histogram(Mat);
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
	CascadeClassifier faceCascade;

	// Load the cascade
	if (!faceCascade.load("haarcascade_frontalface_default.xml")){
		printf("--(!)Error loading\n");
		return (-1);
	}

	// Read the image file
	Mat frame = imread("adele-3.jpg");
	if (!frame.data){
		cout << "File not loaded." << endl;
		return -1;
	}
	cvtColor(frame, frame, COLOR_BGR2GRAY);

	Mat face = Mat();

	// Apply the classifier to the frame
	if (!frame.empty()){
		face = detectFace(frame, faceCascade);
	}
	else{
		cout << "Frame is not loaded." << endl;
	}

	cout << variance_of_sobel(face) << endl;
	imshow("Face", face);    // Show the result
	variance_of_laplacian(face);

	/*Mat spectrum = fourierSpectrum(face);
	imshow("Spectrum magnitude", spectrum);

	measureContrast(face);
*/
	waitKey();
	return 0;
}



Mat detectFace(Mat frame_gray, CascadeClassifier faceCascade)
{
	std::vector<Rect> faces;
	Mat faceROI = Mat();
	int neighbors = 3;

	while (true){
		faceCascade.detectMultiScale(frame_gray, faces, 1.1, neighbors, 0 | CV_HAAR_SCALE_IMAGE, Size(30, 30));
		cout << "Neighbors: " << neighbors << " Face count:" << faces.size() << endl;

		if (faces.size() == 0){
			cout << "Faces not found. minNeighbors paramether = " << neighbors << endl;
			break;
		}

		for (size_t i = 0; i < faces.size(); i++)
		{
			// cv::rectangle(frame_gray, cv::Rect(faces[i].x, faces[i].y, faces[i].width, faces[i].height), Scalar(255, 0, 0));
			faceROI = frame_gray(faces[i]);
		}

		if (faces.size() == 1){
			break;
		}

		neighbors++;
	}
	
	return faceROI;		
}

Mat fourierSpectrum(Mat frame)
{
	//cvtColor(frame, frame, COLOR_BGR2GRAY);

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(frame.rows);
	int n = getOptimalDFTSize(frame.cols); // on the border add zero values
	copyMakeBorder(frame, padded, 0, m - frame.rows, 0, n - frame.cols, BORDER_CONSTANT, Scalar::all(0));

	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexI;
	merge(planes, 2, complexI);         // Add to the expanded another plane with zeros

	dft(complexI, complexI);            // this way the result may fit in the source matrix

	// compute the magnitude and switch to logarithmic scale
	// => log(1 + sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
	split(complexI, planes);                   // planes[0] = Re(DFT(I), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], planes[0]);// planes[0] = magnitude
	Mat magI = planes[0];

	magI += Scalar::all(1);                    // switch to logarithmic scale
	log(magI, magI);

	// crop the spectrum, if it has an odd number of rows or columns
	magI = magI(Rect(0, 0, magI.cols & -2, magI.rows & -2));

	// rearrange the quadrants of Fourier image  so that the origin is at the image center
	int cx = magI.cols / 2;
	int cy = magI.rows / 2;

	Mat q0(magI, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
	Mat q1(magI, Rect(cx, 0, cx, cy));  // Top-Right
	Mat q2(magI, Rect(0, cy, cx, cy));  // Bottom-Left
	Mat q3(magI, Rect(cx, cy, cx, cy)); // Bottom-Right

	Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
	q0.copyTo(tmp);
	q3.copyTo(q0);
	tmp.copyTo(q3);

	q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
	q2.copyTo(q1);
	tmp.copyTo(q2);

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a
	// viewable image form (float between values 0 and 1).

	histogram(magI);

	return magI;
}

void measureContrast(Mat frame)
{
	const int cols = frame.cols;
	const int rows = frame.rows;

	int maxColor = 0;
	int minColor = 255;
	for (int y = 0; y < rows; y++){
		for (int x = 0; x < cols; x++){

			int color = frame.at<uchar>(y, x);
			if (color < minColor){
				minColor = color;
			}
			if (color > maxColor){
				maxColor = color;
			}
		}
	}
	double contrast = (double)(maxColor - minColor) / (maxColor + minColor);
	
	cout << maxColor << endl;
	cout << minColor << endl;
	cout << contrast << endl;
}

void histogram(Mat src){
	/// Establish the number of bins
	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat hist;

	/// Compute the histograms:
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	/// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());
	
	/// Draw for each channel
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	imshow("calcHist Demo", histImage);
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
	// imshow("sobel_gradient", gradient);

	Scalar mean;
	Scalar deviation;
	meanStdDev(gradient, mean, deviation);

	cout << "Sobel" << endl;
	cout << "mean " << mean << endl;
	cout << "deviation " << deviation << endl;
	return deviation[0];
}


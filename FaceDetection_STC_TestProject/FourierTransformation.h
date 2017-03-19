#ifndef FOURIER_TRANSFORMATION_H
#define FOURIER_TRANSFORMATION_H

#include <iostream> 

#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;

Mat fourierSpectrum(Mat &frame);
void measureContrast(const Mat &frame);
Mat calcHistogram(const Mat &src);


Mat fourierSpectrum(Mat &frame)
{
	if (frame.channels() != 1){
		cvtColor(frame, frame, COLOR_BGR2GRAY);
	}

	Mat padded;                            //expand input image to optimal size
	int m = getOptimalDFTSize(frame.rows);
	int n = getOptimalDFTSize(frame.cols); 
	// on the border add zero values
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

	normalize(magI, magI, 0, 1, CV_MINMAX); // Transform the matrix with float values into a viewable image form (float between values 0 and 1).
	return magI;
}

Mat calcHistogram(const Mat &src){

	normalize(src, src, 0, 255, NORM_MINMAX, -1, Mat());

	// Establish the number of bins
	int histSize = 256;

	float range[] = { 0, 256 };
	const float* histRange = { range };

	Mat hist;

	// Compute the histograms:
	calcHist(&src, 1, 0, Mat(), hist, 1, &histSize, &histRange);

	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound((double)hist_w / histSize);

	Mat histImage(hist_h, hist_w, CV_8UC3, Scalar(0, 0, 0));

	// Normalize the result to [ 0, histImage.rows ]
	normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat());

	// Draw
	for (int i = 1; i < histSize; i++)
	{
		line(histImage, Point(bin_w*(i - 1), hist_h - cvRound(hist.at<float>(i - 1))),
			Point(bin_w*(i), hist_h - cvRound(hist.at<float>(i))),
			Scalar(255, 0, 0), 2, 8, 0);
	}

	return histImage;
}

void measureContrast(const Mat &frame)
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

	cout << "Max Intencity: " << maxColor << endl;
	cout << "Min Intencity: " << minColor << endl;
	cout << "Contrast: " << contrast << endl;
}
#endif
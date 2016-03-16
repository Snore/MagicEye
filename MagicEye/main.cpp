#include <opencv2/core/core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>

#include "CardFinder.h"
#include "CardDatabase.h"


//FOR TESTING ONLY
void drawHistogram(cv::MatND & histogram, const int sbins, const int hbins)
{
	// DEBUG ONLY
	double frameMaxVal = 0;
	cv::minMaxLoc(histogram, 0, &frameMaxVal, 0, 0);

	int scale = 10;
	cv::Mat histImg = cv::Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			float binVal = histogram.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / frameMaxVal);
			rectangle(histImg, cv::Point(h*scale, s*scale),
				cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				cv::Scalar::all(intensity),
				CV_FILLED);
		}
	}
	cv::imshow("H-S Histogram", histImg);
}


//FOR TESTING ONLY
void jumpXFrames(cv::VideoCapture & cap, const int framesToJump)
{
	for (int index = 0; index < framesToJump; ++index)
	{
		//throw away
		cap.grab();
	}

}


int main(int argc, char** argv)
{
	CardDatabase cdb;
	cdb.loadSet(CardDetails::RTR);  //RTR //ISD

	while (true)
	{
		MagicCard test = cdb.getCard();
		//test.analyzeCardImage();
		cv::imshow("Card Image", test.loadCardImage());
		drawHistogram(test.getFrameHistogram(), CardMeasurements::SaturationBins, CardMeasurements::HueBins);
		std::cout << test.toString() << "\n";

		if (cv::waitKey(0) == 'q')
		{
			break;
		}
	}

	
	/// Video test
	/*
	cv::VideoCapture cap("Assets\\videos\\Pro Tour Return to Ravnica- Finals.mp4");
	cv:VideoCapture * cap = new VideoCapture("Assets\\videos\\Pro Tour Return to Ravnica- Finals.mp4")
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video file.\n";
		return -1;
	}

	/// Display
	cv::namedWindow("Magic eye", CV_WINDOW_AUTOSIZE);
	//cv::imshow("Magic eye", image);

	jumpXFrames(cap, 1251);
	unsigned int frameNumber = 1251;
	CardFinder cardFinder;
	while (true)
	{
		cv::Mat frame;
		bool success = cap.read(frame);
		if (!success)
		{
			std::cout << "Cannot read frame.\n";
			break;
		}
		else
		{
			cardFinder.findAllCards(frame);
			std::cout << "Frame: ";
			std::cout << frameNumber++;
			std::cout << std::endl;
		}
		cv::imshow("Magic eye", frame);
		if (cv::waitKey(0) == 'q')
		{
			break;
		}
	}
	*/

	return 0;
}
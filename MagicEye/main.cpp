#include <opencv2/core/core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>

#include "CardFinder.h"
#include "CardDatabase.h"

/// FOR DEMO ONLY
#include <chrono>
CardDatabase cdb;
std::vector<TableCard> foundCards;
cv::Mat scene;

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

// FOR DEMO ONLY
void mouseEventCallback(int mouseEvent, int x, int y, int flags, void* userData)
{
	if (mouseEvent == CV_EVENT_LBUTTONDOWN)
	{
		std::cout << "Mouse pointer X: " << x << " Y: " << y << "\n";

		cv::Point mousePoint(x, y);
		for (auto tableCardItr = foundCards.cbegin(); tableCardItr != foundCards.cend(); ++tableCardItr)
		{
			// find which rectangle contains the mouse pointer if any
			if (tableCardItr->getBoundingRect().contains(mousePoint))
			{
				std::chrono::time_point<std::chrono::system_clock> start, end;
				start = std::chrono::system_clock::now();

				std::vector<MagicCard*> matches = cdb.returnMostAlike(tableCardItr->getMagicCard(), 5);

				end = std::chrono::system_clock::now();
				std::chrono::duration<double> elapsed_seconds = end - start;
				cv::imshow("1st match", matches[0]->loadCardImage());
				cv::imshow("2nd match", matches[1]->loadCardImage());
				cv::imshow("3rd match", matches[2]->loadCardImage());
				cv::imshow("4th match", matches[3]->loadCardImage());
				cv::imshow("5th match", matches[4]->loadCardImage());
				std::cout << "Done finding best match.\nElapsed time: " << elapsed_seconds.count() << " seconds\n";
				break;
			}
		}
	}
}


int main(int argc, char** argv)
{
	/// Database / real loop
	///CardDatabase cdb; TODO FOR DEMO ONLY
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	//cdb.loadSet(CardDetails::RTR); // Return to Ravnica block
	//cdb.loadSet(CardDetails::GTC);
	//cdb.loadSet(CardDetails::DGM);
	//cdb.loadSet(CardDetails::ISD); // Innistrad block
	//cdb.loadSet(CardDetails::DKA);
	cdb.loadSet(CardDetails::AVR);
	//cdb.loadSet(CardDetails::ROE); // Zendikar block
	//cdb.loadSet(CardDetails::_10E); // Tenth edition
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Done loading card database.\nElapsed time: " << elapsed_seconds.count() << " seconds\n";
	std::cout << "loaded : " << cdb.toString() << "\n";
	
	/// Video test
	cv::VideoCapture cap("Assets\\videos\\Pro Tour Return to Ravnica- Finals.mp4"); //1251
	//cv::VideoCapture cap("Assets\\videos\\Pro Tour Avacyn Restored Top 8 Finals.mp4"); //13940
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video file.\n";
		return -1;
	}

	/// Display
	cv::namedWindow("Magic eye", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback("Magic eye", mouseEventCallback, NULL);
	//cv::imshow("Magic eye", image);

	jumpXFrames(cap, 4208);
	unsigned int frameNumber = 4208; //4208 for multi cards and stacked cards // 1251 for beginning
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
			frame = cv::imread("Assets\\videos\\reasonable2.jpg", CV_LOAD_IMAGE_COLOR);
			foundCards = cardFinder.findAllCards(frame);
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
	return 0;
}
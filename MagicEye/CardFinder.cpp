#include "CardFinder.h"
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG
#include <climits>



CardFinder::CardFinder()
{
}


CardFinder::~CardFinder()
{
}


int CardFinder::findAllCards(cv::Mat & scene)
{
	cv::Mat bwScene;
	cv::cvtColor(scene, bwScene, CV_BGR2GRAY);
	cv::threshold(bwScene, bwScene, 40, UCHAR_MAX, cv::THRESH_BINARY_INV);

	cv::imshow("scene", bwScene);

	return 0;
}


cv::Rect CardFinder::findPlayField(cv::Mat & scene)
{
	return cv::Rect();
}

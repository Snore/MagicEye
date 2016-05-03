#include "TableCard.h"
#include <opencv2\imgproc.hpp>

// Debug
#include <sstream>
#include <opencv2\highgui.hpp>



TableCard::TableCard()
	:
	_assumedCard_ptr(NULL),
	_boundingBoxInScene()
{
}


TableCard::TableCard(const cv::RotatedRect boundingBox, const cv::Mat & cardImage)
	:
	_boundingBoxInScene(boundingBox)
{
	// add a black border
	cv::Mat cardImageWithBorder;
	cv::copyMakeBorder(cardImage, cardImageWithBorder, 10, 10, 10, 10, cv::BORDER_CONSTANT, CV_RGB(0, 0, 0));

	// check to see if we need to flip, will do nothing if nothing needs to be done
	makeRightsideUp(cardImageWithBorder);

	/// Make a new magic card and pass in ROI of rotated image
	_assumedCard_ptr = std::make_shared<MagicCard>(cardImageWithBorder);
	_assumedCard_ptr->deepAnalyze();
}


TableCard::TableCard(const TableCard & obj)
	:
	_assumedCard_ptr(obj._assumedCard_ptr),
	_boundingBoxInScene(obj._boundingBoxInScene)
{
	// Everything else is covered
}


TableCard::~TableCard()
{
	// Let the shared_ptr object dereference itself
	_assumedCard_ptr = NULL;
}


cv::Rect TableCard::getBoundingRect() const
{
	return _boundingBoxInScene.boundingRect();
}


cv::RotatedRect TableCard::getMinimumBoundingRect() const
{
	return _boundingBoxInScene;
}


MagicCard* TableCard::getMagicCard() const
{
	// Maybe should return a weak pointer
	return _assumedCard_ptr.get();
}


bool TableCard::isPointInside(cv::Point point) const
{
	return true;
}


void TableCard::makeRightsideUp(cv::Mat & cardImage) const
{
	const double inletRatio = 0.05; // 23 pixel in from 460 px width

	/// Could probably optimize with floodfill?
	// cut out the black border
	cv::Mat card = cardImage.clone();
	cv::cvtColor(card, card, CV_BGR2GRAY);
	cv::threshold(card, card, 40, UCHAR_MAX, CV_THRESH_BINARY);

	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(card, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Merge all contours points into one
	std::vector<cv::Point> allPoints;
	for (int outer = 0; outer < contours.size(); ++outer)
	{
		for (std::vector<cv::Point>::iterator pointItr = contours[outer].begin(); pointItr != contours[outer].end(); ++pointItr)
		{
			allPoints.push_back(*pointItr);
		}
	}

	cv::Rect cardROI = cv::boundingRect(allPoints);

	if (cardROI.area() == 0)
		return;

	// Sample the two vertical trace lines
	const int leftColumn = static_cast<int>(cardROI.width * inletRatio);
	const int rightColumn = cardROI.width - leftColumn;

	cv::Mat cardAlone = cardImage(cardROI);
	cv::Rect leftColumnROI = cv::Rect(leftColumn, 0, 1, cardAlone.rows);
	cv::Rect rightColumnROI = cv::Rect(rightColumn, 0, 1, cardAlone.rows);
	cv::Mat leftColumnBW, rightColumnBW;

	cv::cvtColor(cardAlone(leftColumnROI), leftColumnBW, CV_BGR2GRAY);
	cv::cvtColor(cardAlone(rightColumnROI), rightColumnBW, CV_BGR2GRAY);

	// make ranges for upper and lower halves
	const double rangePercent = 0.4;
	const int topUpperRangeIndex = 0;
	const int bottomUpperRangeIndex = static_cast<int>(leftColumnBW.rows * rangePercent);
	const int topLowerRangeIndex = leftColumnBW.rows - bottomUpperRangeIndex;
	const int bottomLowerRangeIndex = leftColumnBW.rows;

	// Find means ans standard deviations
	cv::Scalar topMeanLeft, topStdvLeft, bottomMeanLeft, bottomStdvLeft;
	cv::meanStdDev(leftColumnBW.rowRange(topUpperRangeIndex, topLowerRangeIndex), topMeanLeft, topStdvLeft);
	cv::meanStdDev(leftColumnBW.rowRange(topLowerRangeIndex, bottomLowerRangeIndex), bottomMeanLeft, bottomStdvLeft);

	//cv::Scalar topMeanRight, topStdvRight, bottomMeanRight, bottomStdvRight;
	//cv::meanStdDev(leftColumnBW.rowRange(topUpperRangeIndex, topLowerRangeIndex), topMeanRight, topStdvRight);
	//cv::meanStdDev(leftColumnBW.rowRange(topLowerRangeIndex, bottomLowerRangeIndex), bottomMeanRight, bottomStdvRight);

	// DECIDE if these fetures tell us that the card is rightside up or not
	const double topDelta = topMeanLeft[0] - topStdvLeft[0];
	const double bottomDelta = bottomMeanLeft[0] - bottomStdvLeft[0];

	//
	bool isUpsideDown = topDelta > bottomDelta;
	if (isUpsideDown)
	{
		cv::flip(cardImage, cardImage, -1);
	}
}
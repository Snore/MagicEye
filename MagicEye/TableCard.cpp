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

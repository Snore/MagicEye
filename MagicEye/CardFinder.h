#pragma once
#include <opencv2/core/core.hpp>
#include <vector>
#include "TableCard.h"

class CardFinder
{
public:
	CardFinder();
	~CardFinder();

	std::vector<TableCard> findAllCards(cv::Mat & scene);
	std::vector<cv::Rect> findAllCards(cv::Mat & scene, int); // Remember

private:
	std::vector<TableCard> _foundCards;

	// functions
	cv::Mat findPlayField(const cv::Mat & scene) const;
	void identifyCardsInRegion(const cv::Mat & ROI, const cv::Point ROIOffset, std::vector<TableCard>& runningList) const;
	TableCard extractCardImage(const cv::Mat & fromScene, const cv::RotatedRect boundingRect, const cv::Point worldPosition) const;
	bool isContourConsumedByAnother(const std::vector<cv::Point> contour, const std::vector<cv::Point2f> consumedBy) const;
	void outlineRotatedRectangle(cv::Mat & scene, const cv::RotatedRect RR) const;
	void blackoutRotatedRectangle(cv::Mat & scene, const cv::RotatedRect RR) const;
};


#pragma once
#include <opencv2/core/core.hpp>
#include <memory>
#include "MagicCard.h"

class TableCard
{
public:
	TableCard();
	TableCard(const cv::RotatedRect boundingBox, const cv::Mat & cardImage);
	TableCard(const TableCard & obj);
	~TableCard();

	cv::Rect getBoundingRect() const;
	cv::RotatedRect getMinimumBoundingRect() const;
	MagicCard* getMagicCard() const;
	bool isPointInside(cv::Point point) const;

private:
	std::shared_ptr<MagicCard> _assumedCard_ptr;
	cv::RotatedRect _boundingBoxInScene;
};


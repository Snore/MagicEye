#pragma once
#include <opencv2/core/core.hpp>
#include <vector>

class CardFinder
{
public:
	CardFinder();
	~CardFinder();

	int findAllCards(cv::Mat & scene);

private:
	cv::Rect findPlayField(cv::Mat & scene);
};


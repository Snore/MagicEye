#pragma once
#include <opencv2/core/core.hpp>
#include <string>

#include "json\json.h"
#include "CardDetails.h"

namespace CardMeasurements
{
	const int HueBins = 16;
	const int SaturationBins = 4;
}

class MagicCard
{
public:
	MagicCard(const Json::Value json_card, const CardDetails::CardSet set);
	MagicCard(const std::string imagePath);
	MagicCard(const std::string name, const std::string imagePath, const CardDetails::CardSet set, const CardDetails::Type type);
	~MagicCard();

	cv::Mat loadCardImage() const;
	cv::Mat getFrameHistogram() const;
	void setCardFrameColor(const CardDetails::FrameColor fcolor);

	// DEBUG
	std::string toString() const;

private:
	// card properties
	std::string _name;
	std::string _imageFilePath;
	CardDetails::CardSet _set;
	CardDetails::Type _type;

	// Card image properties
	CardDetails::FrameColor _fcolor;
	cv::Rect _artROI;
	cv::Rect _textROI;
	cv::Rect _borderlessROI;

	//functions
	void locateCardRegions(); // Needs to be called before other functions
	cv::Rect findBorderlessROI(cv::Mat & wholeCardImage) const;
	cv::Mat getFrameOnlyMask() const;
	cv::Mat getBorderlessCardImage() const;
};


#include "MagicCard.h"
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>
#include <iostream>
#include <climits>
#include <sstream>


MagicCard::MagicCard(const Json::Value json_card, const CardDetails::CardSet set)
{
	// get card name, image path, and set
	_name = json_card["name"].asString();
	_imageFilePath = "Assets\\AllSets\\" + CardDetails::CardSet_name.at(set) + "\\" + json_card["imageName"].asString() + ".jpg";
	_set = set;

	// discern if card is creature or not
	Json::Value card_types = json_card["types"];
	Json::ValueIterator type_itr = std::find(card_types.begin(), card_types.end(), "Creature");
	if (type_itr == card_types.end())
	{
		_type = CardDetails::Instant;
	}
	else
	{
		_type = CardDetails::Creature;
	}

	// set card border color
	_fcolor = CardDetails::Unsure;

	// Analyze card
	locateCardRegions();
}


MagicCard::MagicCard(const std::string imagePath)
	:
	_name(""),
	_imageFilePath(imagePath),
	_set(CardDetails::ALA),
	_type(CardDetails::Unidentified),
	_fcolor(CardDetails::Unsure)
{
	// Analyze card
	locateCardRegions();
}


MagicCard::MagicCard(const std::string name, const std::string imagePath, const CardDetails::CardSet set, const CardDetails::Type type)
	:
	_name(name),
	_imageFilePath(imagePath),
	_set(set),
	_type(type),
	_fcolor(CardDetails::Unsure)
{
	// Analyze card
	locateCardRegions();
}


MagicCard::~MagicCard()
{
}


cv::Mat MagicCard::loadCardImage() const
{
	cv::Mat cardImage = cv::imread(_imageFilePath, CV_LOAD_IMAGE_COLOR);
	if (!cardImage.data)
	{
		std::cerr << "Failed to load image for " << _name << " located at: " << _imageFilePath << "\n";
		assert(false);
	}

	return cardImage;
}

void MagicCard::locateCardRegions()
{
	// find the outer boarder of the card
	cv::Mat card = loadCardImage();
	//cv::imshow("Card", card);
	_borderlessROI = findBorderlessROI(card);
	card = card(_borderlessROI);
	//cv::imshow("Now borderless", card);

	// discern the card art from the card's inner border
	const int cardHeight = _borderlessROI.height;
	const int cardWidth = _borderlessROI.width;
	const double topRatio = 60.0 / 660.0;
	const double leftRatio = 20.0 / 440.0;
	const double bottomRatio = 290.0 / 660.0;
	const double cardArtWidthRatio = 420.0 / 460.0;
	const double cardArtHeightRatio = 310.0 / 660.0;
	cv::Point ulArtPoint(static_cast<int>(cardWidth * leftRatio), static_cast<int>(cardHeight * topRatio));
	cv::Point brArtPoint(ulArtPoint.x + static_cast<int>(cardWidth*cardArtWidthRatio), ulArtPoint.y + static_cast<int>(cardHeight * cardArtHeightRatio));
	_artROI = cv::Rect(ulArtPoint, brArtPoint);

	// MORE TESTING
	//cv::imshow("Just Art", histoCard(_artROI));

	// discern the card text box from the card's inner border
	const double textTopRatio = 0.6378787878787879;
	const double textHeightRatio = 0.2969;
	_textROI = _artROI;
	_textROI.y = static_cast<int>(_borderlessROI.height * textTopRatio);
	_textROI.height = static_cast<int>(_borderlessROI.height * textHeightRatio);

	// MORE TESTING
	//cv::imshow("Just Text", histoCard(_textROI));

	// MORE TESTING
	//cv::Mat test = card(_borderlessROI).clone();
	//test.setTo(0, getFrameOnlyMask());
	//cv::imshow("Just frame", test);
}


cv::Mat MagicCard::getFrameHistogram() const
{
	cv::Mat cardHLS;
	cv::cvtColor(getBorderlessCardImage(), cardHLS, CV_BGR2HLS);

	int histSize[] = { CardMeasurements::HueBins, CardMeasurements::SaturationBins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	const float * ranges[] = { hranges, sranges };
	cv::Mat frameHist;
	int channels[] = { 0, 1 };

	cv::calcHist(&cardHLS, 1, channels, getFrameOnlyMask(), frameHist, 2, histSize, ranges, true, false);

	return frameHist;
}


void MagicCard::setCardFrameColor(const CardDetails::FrameColor fcolor)
{
	_fcolor = fcolor;
}


std::string MagicCard::toString() const
{
	std::stringstream description;

	//print card name
	description << "Name: " << _name << "\n";

	// print frame color
	description << "Frame color: ";
	switch (_fcolor)
	{
	case CardDetails::Red:
		description << "Red\n";
		break;
	case CardDetails::Blue:
		description << "Blue\n";
		break;
	case CardDetails::White:
		description << "White\n";
		break;
	case CardDetails::Black:
		description << "Black\n";
		break;
	case CardDetails::Green:
		description << "Green\n";
		break;
	case CardDetails::Multi:
		description << "Multi\n";
		break;
	case CardDetails::Colorless:
		description << "Colorless\n";
		break;
	default:
		description << "Who Knows?\n";
		break;
	}

	return description.str();
}


cv::Rect MagicCard::findBorderlessROI(cv::Mat & wholeCardImage) const
{
	cv::Mat card = wholeCardImage.clone();
	cv::cvtColor(card, card, CV_BGR2GRAY);
	cv::threshold(card, card, 40, UCHAR_MAX, CV_THRESH_BINARY);
	//cv::imshow("thresh", card);  // DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(card, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// Draw contours // DEBUG
	/*
	cv::Mat contourCanvas = cv::Mat::zeros(card.size(), card.type());
	for (int i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(contourCanvas, contours, i, CV_RGB(255, 255, 255));
	}
	cv::imshow("contours", contourCanvas);
	*/

	// Merge all contours points into one
	std::vector<cv::Point> allPoints;
	for (int outer = 0; outer < contours.size(); ++outer)
	{
		for (std::vector<cv::Point>::iterator pointItr = contours[outer].begin(); pointItr != contours[outer].end(); ++pointItr)
		{
			allPoints.push_back(*pointItr);
		}
	}

	return cv::boundingRect(allPoints);
}


cv::Mat MagicCard::getFrameOnlyMask() const
{
	// Inverse mask for testing
	//cv::Mat frameOnlyMask = cv::Mat::zeros(_borderlessROI.size(), CV_8UC1);
	//cv::rectangle(frameOnlyMask, _artROI, CV_RGB(1, 1, 1), CV_FILLED);
	//cv::rectangle(frameOnlyMask, _textROI, CV_RGB(1, 1, 1), CV_FILLED);

	cv::Mat frameOnlyMask = cv::Mat::ones(_borderlessROI.size(), CV_8UC1);
	cv::rectangle(frameOnlyMask, _artROI, CV_RGB(0, 0, 0), CV_FILLED);
	cv::rectangle(frameOnlyMask, _textROI, CV_RGB(0, 0, 0), CV_FILLED);
	return frameOnlyMask;
}


cv::Mat MagicCard::getBorderlessCardImage() const
{
	cv::Mat cardImage = loadCardImage();
	return cardImage(_borderlessROI);
}
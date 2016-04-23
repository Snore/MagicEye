#pragma once
#include <vector>
#include "MagicCard.h"

class CardDatabase
{
public:
	CardDatabase();
	~CardDatabase();

	bool loadSet(const CardDetails::CardSet set);
	std::vector<MagicCard*> returnMostAlike(MagicCard const * const cardToMatch_ptr, const int groupSize) const; //progenator for card matching
	CardDetails::FrameColor getCardColor(const MagicCard* card) const;
	std::string toString() const;

	// Debug
	MagicCard getCard();
	MagicCard getCard(const int index) const;

private:
	// Master card list
	std::vector<MagicCard*> _masterList;
	std::vector<MagicCard*>::iterator _masterItr;

	// Card frame histograms
	cv::Mat _greenFrameHistogram;
	cv::Mat _redFrameHistogram;
	cv::Mat _blueFrameHistogram;
	cv::Mat _whiteFrameHistogram;
	cv::Mat _blackFrameHistogram;
	cv::Mat _yellowFrameHistogram;
	cv::Mat _artifactFrameHistogram;

	// Card frame average colors
	cv::Scalar _greenFrameMean_CIELAB;
	cv::Scalar _greenFrameMean_BGR;
	cv::Scalar _redFrameMean_CIELAB;
	cv::Scalar _redFrameMean_BGR;
	cv::Scalar _blueFrameMean_CIELAB;
	cv::Scalar _blueFrameMean_BGR;
	cv::Scalar _whiteFrameMean_CIELAB;
	cv::Scalar _whiteFrameMean_BGR;
	cv::Scalar _blackFrameMean_CIELAB;
	cv::Scalar _blackFrameMean_BGR;
	cv::Scalar _yellowFrameMean_CIELAB;
	cv::Scalar _yellowFrameMean_BGR;
	cv::Scalar _artifactFrameMean_CIELAB;
	cv::Scalar _artifactFrameMean_BGR;

	// set indexes
	// TODO think about it.  Map of vector<MagicCard*>'s?
	//std::vector<MagicCard*> _set;

	// functions
	void analyzeMasterCard(MagicCard* cardToAnalyze) const;
	void initializeCardFrameHistograms();
	std::vector<MagicCard*> selectFrameColorFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::FrameColor selectColor) const;
	std::vector<MagicCard*> selectSetFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::CardSet selectSet) const;
	double calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const;
};


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

	// set indexes
	// TODO think about it.  Map of vector<MagicCard*>'s?
	//std::vector<MagicCard*> _set;

	// functions
	void analyzeMasterCard(MagicCard* cardToAnalyze) const;
	void initializeCardFrameHistograms();
	std::vector<MagicCard*> selectFrameColorFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::FrameColor selectColor) const;
	std::vector<MagicCard*> selectSetFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::CardSet selectSet) const;
};


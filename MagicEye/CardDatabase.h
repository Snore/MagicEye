#pragma once
#include <vector>
#include "MagicCard.h"

class CardDatabase
{
public:
	CardDatabase();
	~CardDatabase();

	bool loadSet(const CardDetails::CardSet set);
	MagicCard getCard();

private:
	// Master card list
	std::vector<MagicCard*> _masterList;
	std::vector<MagicCard*>::iterator _masterItr;

	// Card frame histograms
	cv::MatND _greenFrameHistogram;
	cv::MatND _redFrameHistogram;
	cv::MatND _blueFrameHistogram;
	cv::MatND _whiteFrameHistogram;
	cv::MatND _blackFrameHistogram;
	cv::MatND _yellowFrameHistogram;
	cv::MatND _artifactFrameHistogram;

	// set indexes
	// TODO think about it.  Map of vector<MagicCard*>'s?
	//std::vector<MagicCard*> _set;

	// functions
	void analyzeMasterCard(MagicCard* cardToAnalyze) const;
	void initializeCardFrameHistograms();
	CardDetails::FrameColor getCardColor(const MagicCard* card) const;
};


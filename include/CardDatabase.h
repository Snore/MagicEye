#ifndef CARD_DATABASE_H
#define CARD_DATABASE_H

#include <vector>
#include "MagicCard.h"

class CardDatabase
{
public:
	CardDatabase();
	~CardDatabase();

	bool loadSet(const CardDetails::CardSet set);
	std::vector<MagicCard*> returnMostAlike(MagicCard const * const cardToMatch_ptr, const int groupSize) const; //progenator for card matching
	CardDetails::FrameColor getLiveCardColor(const MagicCard* card) const; // In a perfect world, break this out into its own class?  Same with its digital sibling
	void trainLiveCardColor(const MagicCard* card, const CardDetails::FrameColor fcolor);
	std::string toString() const;

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
	cv::Mat _landFrameHistogram;

	// Card frame average colors
	cv::Scalar _greenFrameMean_CIELAB;
	cv::Scalar _redFrameMean_CIELAB;
	cv::Scalar _blueFrameMean_CIELAB;
	cv::Scalar _whiteFrameMean_CIELAB;
	cv::Scalar _blackFrameMean_CIELAB;
	cv::Scalar _yellowFrameMean_CIELAB;
	cv::Scalar _artifactFrameMean_CIELAB;
	cv::Scalar _landFrameMean_CIELAB;

	// Live OLMB K-means frame average colors
	cv::Scalar _greenFrameMean_CIELAB_Live;
	cv::Scalar _redFrameMean_CIELAB_Live;
	cv::Scalar _blueFrameMean_CIELAB_Live;
	cv::Scalar _whiteFrameMean_CIELAB_Live;
	cv::Scalar _blackFrameMean_CIELAB_Live;
	cv::Scalar _yellowFrameMean_CIELAB_Live;
	cv::Scalar _artifactFrameMean_CIELAB_Live;
	cv::Scalar _landFrameMean_CIELAB_Live;
	int _greenFrameMean_CIELAB_Live_Samples;
	int _redFrameMean_CIELAB_Live_Samples;
	int _blueFrameMean_CIELAB_Live_Samples;
	int _whiteFrameMean_CIELAB_Live_Samples;
	int _blackFrameMean_CIELAB_Live_Samples;
	int _yellowFrameMean_CIELAB_Live_Samples;
	int _artifactFrameMean_CIELAB_Live_Samples;
	int _landFrameMean_CIELAB_Live_Samples;

	// set indexes
	// TODO think about it.  Map of vector<MagicCard*>'s?
	//std::vector<MagicCard*> _set;

	// functions
	void analyzeMasterCard(MagicCard* cardToAnalyze) const;
	void initializeCardFrameHistograms();
	std::vector<MagicCard*> selectFrameColorFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::FrameColor selectColor) const;
	std::vector<MagicCard*> selectSetFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::CardSet selectSet) const;
	CardDetails::FrameColor getDigitalCardColor(const MagicCard* card) const;
	double calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const;
	double calcDeltaE_noChroma(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const;
};

#endif // CARD_DATABASE_H

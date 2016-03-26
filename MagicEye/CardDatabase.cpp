#include "CardDatabase.h"
#include "json\json.h"

#include <fstream>
#include <algorithm>
#include <cfloat>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG


//FOR TESTING ONLY
void drawHistogram(cv::MatND & histogram, const int sbins, const int hbins, std::string windowName)
{
	// DEBUG ONLY
	double frameMaxVal = 0;
	cv::minMaxLoc(histogram, 0, &frameMaxVal, 0, 0);

	int scale = 10;
	cv::Mat histImg = cv::Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			float binVal = histogram.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / frameMaxVal);
			rectangle(histImg, cv::Point(h*scale, s*scale),
				cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				cv::Scalar::all(intensity),
				CV_FILLED);
		}
	}
	cv::imshow(windowName, histImg);
}


CardDatabase::CardDatabase()
{
	initializeCardFrameHistograms();

	/* // Break in case of histogram identity crisis
	drawHistogram(_greenFrameHistogram, 4, 16, "green average");
	drawHistogram(_redFrameHistogram, 4, 16, "red average");
	drawHistogram(_blueFrameHistogram, 4, 16, "blue average");
	drawHistogram(_whiteFrameHistogram, 4, 16, "white average");
	drawHistogram(_blackFrameHistogram, 4, 16, "black average");
	drawHistogram(_yellowFrameHistogram, 4, 16, "multi average");
	drawHistogram(_artifactFrameHistogram, 4, 16, "colorless average");
	*/
}


CardDatabase::~CardDatabase()
{
	for (std::vector<MagicCard*>::iterator del_itr = _masterList.begin(); del_itr != _masterList.end(); ++del_itr)
	{
		delete (*del_itr);
	}
	_masterList.clear();
}


bool CardDatabase::loadSet(const CardDetails::CardSet set)
{
	std::ifstream magicJSONFile("Assets\\AllSets.json\\AllSets.json");
	Json::Value magic_json;
	magicJSONFile >> magic_json;
	magicJSONFile.close();

	// get set name
	std::string setName = CardDetails::CardSet_name.at(set);

	Json::Value set_json = magic_json[setName];
	for (Json::Value::iterator card_it = set_json["cards"].begin(); card_it != set_json["cards"].end(); ++card_it)
	{
		MagicCard * newCard = new MagicCard(*card_it, set);
		analyzeMasterCard(newCard);

		_masterList.push_back(newCard);
	}

	_masterItr = _masterList.begin();

	return true;
}


std::vector<MagicCard*> CardDatabase::returnMostAlike(MagicCard const * const cardToMatch_ptr, const int groupSize) const
{
	std::vector<MagicCard*> bestMatches;
	std::vector<MagicCard*> remainingList;

	// Select from only cards that match the unidentified card's frame color
	if (cardToMatch_ptr->getFrameColor() != CardDetails::Unsure)
	{
		// Eliminate all cards that do not share a frame color with the card to match
		remainingList = selectFrameColorFrom(_masterList, cardToMatch_ptr->getFrameColor());
	}
	else
	{
		// If the unidentified card doesn't have its frame color discerned yet, then need to look at all cards
		remainingList = _masterList;
	}

	std::vector<double> matchesValues(remainingList.size());

	// calculate the "Card comparison" value between the passed in card and every remaining card in the query
	for (auto itr = std::make_pair(remainingList.cbegin(), matchesValues.begin()); itr.first != remainingList.cend(); ++itr.first, ++itr.second)
	{
		*itr.second = MagicCard::compareLikeness(cardToMatch_ptr, *itr.first);
	}

	// sort the card pointers from most matching to least matching
	for (int superIndex = 0; superIndex < groupSize; ++superIndex)
	{
		double winningValue = matchesValues[0];
		int winningIndex = 0;
		for (int index = 1; index < matchesValues.size(); ++index)
		{
			if (matchesValues[index] < winningValue)
			{
				winningValue = matchesValues[index];
				winningIndex = index;
			}
		}
		bestMatches.push_back(remainingList[winningIndex]);
		matchesValues[winningIndex] = DBL_MAX;
	}

	return bestMatches;
}


MagicCard CardDatabase::getCard()
{
	// Still testing
	return *(*_masterItr++);
}


MagicCard CardDatabase::getCard(const int index) const
{
	return *_masterList[index];
}


void CardDatabase::analyzeMasterCard(MagicCard * cardToAnalyze) const
{
	// Still testing
	cardToAnalyze->setCardFrameColor(getCardColor(cardToAnalyze));
	cardToAnalyze->deepAnalyze();
}


void CardDatabase::initializeCardFrameHistograms()
{
	const double accumulationRatio = 1.0 / 6.0;

	// Green frame samples
	std::vector<std::string> greenSamplePaths = { "Assets\\FrameTrainingImages\\green\\abundant growth.jpg",
												  "Assets\\FrameTrainingImages\\green\\baloth cage trap.jpg",
												  "Assets\\FrameTrainingImages\\green\\beastmaster ascension.jpg",
												  "Assets\\FrameTrainingImages\\green\\borderland ranger.jpg",
												  "Assets\\FrameTrainingImages\\green\\harrow.jpg",
												  "Assets\\FrameTrainingImages\\green\\pulse of the tangle.jpg" };
	_greenFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator greenSamplePathItr = greenSamplePaths.cbegin(); greenSamplePathItr != greenSamplePaths.cend(); ++greenSamplePathItr)
	{
		MagicCard card(*greenSamplePathItr);
		cv::addWeighted(_greenFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _greenFrameHistogram);
	}

	// Red frame samples
	std::vector<std::string> redSamplePaths = { "Assets\\FrameTrainingImages\\red\\aggravate.jpg",
												"Assets\\FrameTrainingImages\\red\\ash zealot.jpg",
												"Assets\\FrameTrainingImages\\red\\brood birthing.jpg",
												"Assets\\FrameTrainingImages\\red\\curse of bloodletting.jpg",
												"Assets\\FrameTrainingImages\\red\\savage beating.jpg",
												"Assets\\FrameTrainingImages\\red\\slobad, goblin tinkerer.jpg" };
	_redFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator redSamplePathItr = redSamplePaths.cbegin(); redSamplePathItr != redSamplePaths.cend(); ++redSamplePathItr)
	{
		MagicCard card(*redSamplePathItr);
		cv::addWeighted(_redFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _redFrameHistogram);
	}

	// Blue frame samples
	std::vector<std::string> blueSamplePaths = { "Assets\\FrameTrainingImages\\blue\\captain of the mists.jpg",
												 "Assets\\FrameTrainingImages\\blue\\chant of the skifsang.jpg",
												 "Assets\\FrameTrainingImages\\blue\\counterlash.jpg",
												 "Assets\\FrameTrainingImages\\blue\\hisoka, minamo sensei.jpg",
												 "Assets\\FrameTrainingImages\\blue\\neurok prodigy.jpg",
												 "Assets\\FrameTrainingImages\\blue\\psychic overload.jpg" };
	_blueFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator blueSamplePathItr = blueSamplePaths.cbegin(); blueSamplePathItr != blueSamplePaths.cend(); ++blueSamplePathItr)
	{
		MagicCard card(*blueSamplePathItr);
		cv::addWeighted(_blueFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blueFrameHistogram);
	}

	// White frame samples
	std::vector<std::string> whiteSamplePaths = { "Assets\\FrameTrainingImages\\white\\archangel's light.jpg",
												  "Assets\\FrameTrainingImages\\white\\burden of guilt.jpg",
												  "Assets\\FrameTrainingImages\\white\\call to serve.jpg",
												  "Assets\\FrameTrainingImages\\white\\cursebreak.jpg",
												  "Assets\\FrameTrainingImages\\white\\pteron ghost.jpg",
												  "Assets\\FrameTrainingImages\\white\\pulse of the fields.jpg" };
	_whiteFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator whiteSamplePathItr = whiteSamplePaths.cbegin(); whiteSamplePathItr != whiteSamplePaths.cend(); ++whiteSamplePathItr)
	{
		MagicCard card(*whiteSamplePathItr);
		cv::addWeighted(_whiteFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _whiteFrameHistogram);
	}

	// Black frame samples
	std::vector<std::string> blackSamplePaths = { "Assets\\FrameTrainingImages\\black\\burden of greed.jpg",
												  "Assets\\FrameTrainingImages\\black\\deadly allure.jpg",
												  "Assets\\FrameTrainingImages\\black\\essence drain.jpg",
												  "Assets\\FrameTrainingImages\\black\\harvester of souls.jpg",
												  "Assets\\FrameTrainingImages\\black\\homicidal seclusion.jpg",
												  "Assets\\FrameTrainingImages\\black\\markov's servant.jpg" };
	_blackFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator blackSamplePathItr = blackSamplePaths.cbegin(); blackSamplePathItr != blackSamplePaths.cend(); ++blackSamplePathItr)
	{
		MagicCard card(*blackSamplePathItr);
		cv::addWeighted(_blackFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blackFrameHistogram);
	}

	// Multi-color frame samples
	std::vector<std::string> multiSamplePaths = { "Assets\\FrameTrainingImages\\yellow\\bruna, light of alabaster.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\drogskol captain.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\gisela, blade of goldnight.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\havengul lich.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\ravager of the fells.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\wrexial, the risen deep.jpg" };
	_yellowFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator multiSamplePathItr = multiSamplePaths.cbegin(); multiSamplePathItr != multiSamplePaths.cend(); ++multiSamplePathItr)
	{
		MagicCard card(*multiSamplePathItr);
		cv::addWeighted(_yellowFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _yellowFrameHistogram);
	}

	// Colorless frame samples
	std::vector<std::string> colorlessSamplePaths = { "Assets\\FrameTrainingImages\\artifact\\darksteel brute.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\darksteel reactor.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\general's kabuto.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\hankyu.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\kusari-gama.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\muse vessel.jpg" };
	_artifactFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	for (std::vector<std::string>::const_iterator colorlessSamplePathItr = colorlessSamplePaths.cbegin(); colorlessSamplePathItr != colorlessSamplePaths.cend(); ++colorlessSamplePathItr)
	{
		MagicCard card(*colorlessSamplePathItr);
		cv::addWeighted(_artifactFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _artifactFrameHistogram);
	}
}


std::vector<MagicCard*> CardDatabase::selectFrameColorFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::FrameColor selectColor) const
{
	std::vector<MagicCard*> colorOnlyCards(fromCards.size());

	// copy
	std::vector<MagicCard*>::const_iterator current = fromCards.cbegin();
	std::vector<MagicCard*>::const_iterator end = fromCards.cend();
	std::vector<MagicCard*>::iterator result = colorOnlyCards.begin();

	while (current != end)
	{
		if ((*current)->getFrameColor() == selectColor)
		{
			*result = *current;
			++result;
		}
		++current;
	}

	// shrink to fit
	colorOnlyCards.resize(std::distance(colorOnlyCards.begin(), result));

	return colorOnlyCards;
}


std::vector<MagicCard*> CardDatabase::selectSetFrom(const std::vector<MagicCard*> & fromCards, const CardDetails::CardSet selectSet) const
{
	std::vector<MagicCard*> setOnlyCards(fromCards.size());

	// copy
	std::vector<MagicCard*>::const_iterator current = fromCards.cbegin();
	std::vector<MagicCard*>::const_iterator end = fromCards.cend();
	std::vector<MagicCard*>::iterator result = setOnlyCards.begin();

	while (current != end)
	{
		if ((*current)->getCardSet() == selectSet)
		{
			*result = *current;
			++result;
		}
		++current;
	}

	// shrink to fit
	setOnlyCards.resize(std::distance(setOnlyCards.begin(), result));

	return setOnlyCards;
}


CardDetails::FrameColor CardDatabase::getCardColor(const MagicCard* card) const
{
	const int COMP_HIST_METHOD = CV_COMP_CHISQR;
	CardDetails::FrameColor bestMatchingColor = CardDetails::Unsure;
	double bestMatchingDistance = DBL_MAX;

	const cv::Mat cardFrameHistogram = card->getFrameHistogram();

	// Green
	//const double greenFrameDistance = cv::compareHist(cardFrameHistogram, _greenFrameHistogram, COMP_HIST_METHOD);
	const double greenFrameDistance = cv::compareHist(_greenFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (greenFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Green;
		bestMatchingDistance = greenFrameDistance;
	}

	// Red
	//const double redFrameDistance = cv::compareHist(cardFrameHistogram, _redFrameHistogram, COMP_HIST_METHOD);
	const double redFrameDistance = cv::compareHist(_redFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (redFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Red;
		bestMatchingDistance = redFrameDistance;
	}

	// Blue
	//const double blueFrameDistance = cv::compareHist(cardFrameHistogram, _blueFrameHistogram, COMP_HIST_METHOD);
	const double blueFrameDistance = cv::compareHist(_blueFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (blueFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Blue;
		bestMatchingDistance = blueFrameDistance;
	}

	// White
	//const double whiteFrameDistance = cv::compareHist(cardFrameHistogram, _whiteFrameHistogram, COMP_HIST_METHOD);
	const double whiteFrameDistance = cv::compareHist(_whiteFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (whiteFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::White;
		bestMatchingDistance = whiteFrameDistance;
	}

	// Black
	//const double blackFrameDistance = cv::compareHist(cardFrameHistogram, _blackFrameHistogram, COMP_HIST_METHOD);
	const double blackFrameDistance = cv::compareHist(_blackFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (blackFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Black;
		bestMatchingDistance = blackFrameDistance;
	}

	// Multi-color
	//const double yellowFrameDistance = cv::compareHist(cardFrameHistogram, _yellowFrameHistogram, COMP_HIST_METHOD);
	const double yellowFrameDistance = cv::compareHist(_yellowFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (yellowFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Multi;
		bestMatchingDistance = yellowFrameDistance;
	}

	// Colorless
	//const double colorlessFrameDistance = cv::compareHist(cardFrameHistogram, _artifactFrameHistogram, COMP_HIST_METHOD);
	const double colorlessFrameDistance = cv::compareHist(_artifactFrameHistogram, cardFrameHistogram, COMP_HIST_METHOD);
	if (colorlessFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Colorless;
		bestMatchingDistance = colorlessFrameDistance;
	}

	return bestMatchingColor;
}
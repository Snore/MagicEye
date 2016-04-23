#include "CardDatabase.h"
#include "json\json.h"

#include <fstream>
#include <algorithm>
#include <cfloat>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG
#include <iostream> // DEBUG


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

	// Break in case of histogram identity crisis
	/*
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

	//drawHistogram(cardToMatch_ptr->getFrameHistogram(), 4, 16, "incoming card");												// DEBUG
	cv::imshow("INCOMING CARD", cardToMatch_ptr->getBorderlessCardImage());														// DEBUG
	cv::Scalar meanPoint = cardToMatch_ptr->getFrameMeanColor_CIELAB();															// DEBUG
	std::cout << "card frame CIELAB mean point: X:" << meanPoint[0] << " Y:" << meanPoint[1] << " Z:" << meanPoint[2] << "\n";  // DEBUG

	// Select from only cards that match the unidentified card's frame color
	if (cardToMatch_ptr->getFrameColor() != CardDetails::Unsure)
	{
		// Eliminate all cards that do not share a frame color with the card to match
		remainingList = selectFrameColorFrom(_masterList, cardToMatch_ptr->getFrameColor());
		std::cout << "Established card color: " << MagicCard::FrameColorToString(cardToMatch_ptr->getFrameColor()) << "\n";
	}
	else
	{
		// If the unidentified card doesn't have its frame color discerned yet, calculate it
		const CardDetails::FrameColor cardFrameColor = getCardColor(cardToMatch_ptr);
		remainingList = selectFrameColorFrom(_masterList, cardFrameColor);
		//remainingList = _masterList;

		std::cout << "Discerned card color: " << MagicCard::FrameColorToString(cardFrameColor) << "\n";
	}

	std::vector<double> matchesValues(remainingList.size());

	// calculate the "Card comparison" value between the passed in card and every remaining card in the query
	for (auto itr = std::make_pair(remainingList.cbegin(), matchesValues.begin()); itr.first != remainingList.cend(); ++itr.first, ++itr.second)
	{
		//*itr.second = MagicCard::compareLikeness(cardToMatch_ptr, *itr.first); // HSV weak to lighting
		*itr.second = MagicCard::compareDeltaEGrid(cardToMatch_ptr, *itr.first);
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


std::string CardDatabase::toString() const
{
	std::stringstream output;

	output << "The card database currently has ";
	output << _masterList.size();
	output << " cards loaded.\n";

	return output.str();
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
	const int histogramSizes[] = { CardMeasurements::HueBins, CardMeasurements::SaturationBins }; // , CardMeasurements::ValueBins};
	const int numDimens = 2;

	// Green frame samples
	std::vector<std::string> greenSamplePaths = { "Assets\\FrameTrainingImages\\green\\abundant growth.jpg",
												  "Assets\\FrameTrainingImages\\green\\baloth cage trap.jpg",
												  "Assets\\FrameTrainingImages\\green\\beastmaster ascension.jpg",
												  "Assets\\FrameTrainingImages\\green\\borderland ranger.jpg",
												  "Assets\\FrameTrainingImages\\green\\harrow.jpg",
												  "Assets\\FrameTrainingImages\\green\\pulse of the tangle.jpg" };
	//_greenFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_greenFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_greenFrameMean_CIELAB = cv::Scalar(0.0);
	_greenFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator greenSamplePathItr = greenSamplePaths.cbegin(); greenSamplePathItr != greenSamplePaths.cend(); ++greenSamplePathItr)
	{
		MagicCard card(*greenSamplePathItr);
		cv::addWeighted(_greenFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _greenFrameHistogram);
		cv::addWeighted(_greenFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _greenFrameMean_CIELAB);
		cv::addWeighted(_greenFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _greenFrameMean_BGR);
	}
	std::cout << "Green card mean point: X:" << _greenFrameMean_CIELAB[0] << " Y:" << _greenFrameMean_CIELAB[1] << " Z:" << _greenFrameMean_CIELAB[2] << "\n";

	// Red frame samples
	std::vector<std::string> redSamplePaths = { "Assets\\FrameTrainingImages\\red\\aggravate.jpg",
												"Assets\\FrameTrainingImages\\red\\ash zealot.jpg",
												"Assets\\FrameTrainingImages\\red\\brood birthing.jpg",
												"Assets\\FrameTrainingImages\\red\\curse of bloodletting.jpg",
												"Assets\\FrameTrainingImages\\red\\savage beating.jpg",
												"Assets\\FrameTrainingImages\\red\\slobad, goblin tinkerer.jpg" };
	//_redFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_redFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_redFrameMean_CIELAB = cv::Scalar(0.0);
	_redFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator redSamplePathItr = redSamplePaths.cbegin(); redSamplePathItr != redSamplePaths.cend(); ++redSamplePathItr)
	{
		MagicCard card(*redSamplePathItr);
		cv::addWeighted(_redFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _redFrameHistogram);
		cv::addWeighted(_redFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _redFrameMean_CIELAB);
		cv::addWeighted(_redFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _redFrameMean_BGR);
	}
	std::cout << "Red card mean point: X:" << _redFrameMean_CIELAB[0] << " Y:" << _redFrameMean_CIELAB[1] << " Z:" << _redFrameMean_CIELAB[2] << "\n";

	// Blue frame samples
	std::vector<std::string> blueSamplePaths = { "Assets\\FrameTrainingImages\\blue\\captain of the mists.jpg",
												 "Assets\\FrameTrainingImages\\blue\\chant of the skifsang.jpg",
												 "Assets\\FrameTrainingImages\\blue\\counterlash.jpg",
												 "Assets\\FrameTrainingImages\\blue\\hisoka, minamo sensei.jpg",
												 "Assets\\FrameTrainingImages\\blue\\neurok prodigy.jpg",
												 "Assets\\FrameTrainingImages\\blue\\psychic overload.jpg" };
	//_blueFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_blueFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_blueFrameMean_CIELAB = cv::Scalar(0.0);
	_blueFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator blueSamplePathItr = blueSamplePaths.cbegin(); blueSamplePathItr != blueSamplePaths.cend(); ++blueSamplePathItr)
	{
		MagicCard card(*blueSamplePathItr);
		cv::addWeighted(_blueFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blueFrameHistogram);
		cv::addWeighted(_blueFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _blueFrameMean_CIELAB);
		cv::addWeighted(_blueFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _blueFrameMean_BGR);
	}
	std::cout << "Blue card mean point: X:" << _blueFrameMean_CIELAB[0] << " Y:" << _blueFrameMean_CIELAB[1] << " Z:" << _blueFrameMean_CIELAB[2] << "\n";

	// White frame samples
	std::vector<std::string> whiteSamplePaths = { "Assets\\FrameTrainingImages\\white\\archangel's light.jpg",
												  "Assets\\FrameTrainingImages\\white\\burden of guilt.jpg",
												  "Assets\\FrameTrainingImages\\white\\call to serve.jpg",
												  "Assets\\FrameTrainingImages\\white\\cursebreak.jpg",
												  "Assets\\FrameTrainingImages\\white\\pteron ghost.jpg",
												  "Assets\\FrameTrainingImages\\white\\pulse of the fields.jpg" };
	//_whiteFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_whiteFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_whiteFrameMean_CIELAB = cv::Scalar(0.0);
	_whiteFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator whiteSamplePathItr = whiteSamplePaths.cbegin(); whiteSamplePathItr != whiteSamplePaths.cend(); ++whiteSamplePathItr)
	{
		MagicCard card(*whiteSamplePathItr);
		cv::addWeighted(_whiteFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _whiteFrameHistogram);
		cv::addWeighted(_whiteFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _whiteFrameMean_CIELAB);
		cv::addWeighted(_whiteFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _whiteFrameMean_BGR);
	}
	std::cout << "White card mean point: X:" << _whiteFrameMean_CIELAB[0] << " Y:" << _whiteFrameMean_CIELAB[1] << " Z:" << _whiteFrameMean_CIELAB[2] << "\n";

	// Black frame samples
	std::vector<std::string> blackSamplePaths = { "Assets\\FrameTrainingImages\\black\\burden of greed.jpg",
												  "Assets\\FrameTrainingImages\\black\\deadly allure.jpg",
												  "Assets\\FrameTrainingImages\\black\\essence drain.jpg",
												  "Assets\\FrameTrainingImages\\black\\harvester of souls.jpg",
												  "Assets\\FrameTrainingImages\\black\\homicidal seclusion.jpg",
												  "Assets\\FrameTrainingImages\\black\\markov's servant.jpg" };
	//_blackFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_blackFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_blackFrameMean_CIELAB = cv::Scalar(0.0);
	_blackFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator blackSamplePathItr = blackSamplePaths.cbegin(); blackSamplePathItr != blackSamplePaths.cend(); ++blackSamplePathItr)
	{
		MagicCard card(*blackSamplePathItr);
		cv::addWeighted(_blackFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blackFrameHistogram);
		cv::addWeighted(_blackFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _blackFrameMean_CIELAB);
		cv::addWeighted(_blackFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _blackFrameMean_BGR);
	}
	std::cout << "Black card mean point: X:" << _blackFrameMean_CIELAB[0] << " Y:" << _blackFrameMean_CIELAB[1] << " Z:" << _blackFrameMean_CIELAB[2] << "\n";

	// Multi-color frame samples
	std::vector<std::string> multiSamplePaths = { "Assets\\FrameTrainingImages\\yellow\\bruna, light of alabaster.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\drogskol captain.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\gisela, blade of goldnight.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\havengul lich.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\ravager of the fells.jpg",
												  "Assets\\FrameTrainingImages\\yellow\\wrexial, the risen deep.jpg" };
	//_yellowFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_yellowFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_yellowFrameMean_CIELAB = cv::Scalar(0.0);
	_yellowFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator multiSamplePathItr = multiSamplePaths.cbegin(); multiSamplePathItr != multiSamplePaths.cend(); ++multiSamplePathItr)
	{
		MagicCard card(*multiSamplePathItr);
		cv::addWeighted(_yellowFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _yellowFrameHistogram);
		cv::addWeighted(_yellowFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _yellowFrameMean_CIELAB);
		cv::addWeighted(_yellowFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _yellowFrameMean_BGR);
	}
	std::cout << "Multi card mean point: X:" << _yellowFrameMean_CIELAB[0] << " Y:" << _yellowFrameMean_CIELAB[1] << " Z:" << _yellowFrameMean_CIELAB[2] << "\n";

	// Colorless frame samples
	std::vector<std::string> colorlessSamplePaths = { "Assets\\FrameTrainingImages\\artifact\\darksteel brute.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\darksteel reactor.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\general's kabuto.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\hankyu.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\kusari-gama.jpg",
												      "Assets\\FrameTrainingImages\\artifact\\muse vessel.jpg" };
	//_artifactFrameHistogram = cv::Mat::zeros(CardMeasurements::HueBins, CardMeasurements::SaturationBins, CV_32F);
	_artifactFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_artifactFrameMean_CIELAB = cv::Scalar(0.0);
	_artifactFrameMean_BGR = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator colorlessSamplePathItr = colorlessSamplePaths.cbegin(); colorlessSamplePathItr != colorlessSamplePaths.cend(); ++colorlessSamplePathItr)
	{
		MagicCard card(*colorlessSamplePathItr);
		cv::addWeighted(_artifactFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _artifactFrameHistogram);
		cv::addWeighted(_artifactFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _artifactFrameMean_CIELAB);
		cv::addWeighted(_artifactFrameMean_BGR, 1.0, card.getFrameMeanColor_BGR(), accumulationRatio, 0.0, _artifactFrameMean_BGR);
	}
	std::cout << "Colorless card mean point: X:" << _artifactFrameMean_CIELAB[0] << " Y:" << _artifactFrameMean_CIELAB[1] << " Z:" << _artifactFrameMean_CIELAB[2] << "\n";
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
	CardDetails::FrameColor bestMatchingColor = CardDetails::Unsure;
	double bestMatchingDistance = DBL_MAX;

	const cv::Scalar cardFrameMean_CIELAB = card->getFrameMeanColor_CIELAB();

	// Green
	const double greenFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _greenFrameMean_CIELAB);
	if (greenFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Green;
		bestMatchingDistance = greenFrameDistance;
	}

	// Red
	const double redFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _redFrameMean_CIELAB);
	if (redFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Red;
		bestMatchingDistance = redFrameDistance;
	}

	// Blue
	const double blueFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _blueFrameMean_CIELAB);
	if (blueFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Blue;
		bestMatchingDistance = blueFrameDistance;
	}

	// White
	const double whiteFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _whiteFrameMean_CIELAB);
	if (whiteFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::White;
		bestMatchingDistance = whiteFrameDistance;
	}

	// Black
	const double blackFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _blackFrameMean_CIELAB);
	if (blackFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Black;
		bestMatchingDistance = blackFrameDistance;
	}

	// Multi-color
	const double yellowFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _yellowFrameMean_CIELAB);
	if (yellowFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Multi;
		bestMatchingDistance = yellowFrameDistance;
	}

	// Colorless
	const double colorlessFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _artifactFrameMean_CIELAB);
	if (colorlessFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Colorless;
		bestMatchingDistance = colorlessFrameDistance;
	}

	return bestMatchingColor;
}


double CardDatabase::calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const
{
	// CIE76 - not great, but good enough
	cv::Scalar lineSegment = colorPointOne - colorPointTwo;
	lineSegment[0] = cv::pow(lineSegment[0], 2);
	lineSegment[1] = cv::pow(lineSegment[1], 2);
	lineSegment[2] = cv::pow(lineSegment[2], 2);
	const double euclidianDistance = cv::sqrt(lineSegment[0] + lineSegment[1] + lineSegment[2]);

	// CIE94 - better, but not yet perfect

	return euclidianDistance;
}


/*
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
*/
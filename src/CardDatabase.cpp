#include "pch.h"
#include "CardDatabase.h"
#include "json.h"

#include <fstream>
#include <algorithm>
#include <cfloat>
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG
#include <iostream> // DEBUG


//FOR TESTING ONLY
void drawHistogram(cv::Mat & histogram, const int sbins, const int hbins, std::string windowName)
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
	std::ifstream magicJSONFile(ASSETS_PATH + "\\AllSets.json\\AllSets.json");
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
		const CardDetails::FrameColor cardFrameColor = getLiveCardColor(cardToMatch_ptr);
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


void CardDatabase::analyzeMasterCard(MagicCard * cardToAnalyze) const
{
	// Still testing
	cardToAnalyze->setCardFrameColor(getDigitalCardColor(cardToAnalyze));
	cardToAnalyze->deepAnalyze();
}


void CardDatabase::initializeCardFrameHistograms()
{
	const double accumulationRatio = 1.0 / 6.0;
	const int histogramSizes[] = { CardMeasurements::HueBins, CardMeasurements::SaturationBins, CardMeasurements::ValueBins};
	const int numDimens = 3;

	// Green frame samples
	std::vector<std::string> greenSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\green\\abundant growth.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\green\\baloth cage trap.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\green\\beastmaster ascension.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\green\\borderland ranger.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\green\\harrow.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\green\\pulse of the tangle.jpg" };
	_greenFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_greenFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator greenSamplePathItr = greenSamplePaths.cbegin(); greenSamplePathItr != greenSamplePaths.cend(); ++greenSamplePathItr)
	{
		MagicCard card(*greenSamplePathItr);
		cv::addWeighted(_greenFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _greenFrameHistogram);
		cv::addWeighted(_greenFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _greenFrameMean_CIELAB);
	}
	std::cout << "Green card mean point: X:" << _greenFrameMean_CIELAB[0] << " Y:" << _greenFrameMean_CIELAB[1] << " Z:" << _greenFrameMean_CIELAB[2] << "\n";

	// Red frame samples
	std::vector<std::string> redSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\red\\aggravate.jpg",
												ASSETS_PATH + "\\FrameTrainingImages\\red\\ash zealot.jpg",
												ASSETS_PATH + "\\FrameTrainingImages\\red\\brood birthing.jpg",
												ASSETS_PATH + "\\FrameTrainingImages\\red\\curse of bloodletting.jpg",
												ASSETS_PATH + "\\FrameTrainingImages\\red\\savage beating.jpg",
												ASSETS_PATH + "\\FrameTrainingImages\\red\\slobad, goblin tinkerer.jpg" };
	_redFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_redFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator redSamplePathItr = redSamplePaths.cbegin(); redSamplePathItr != redSamplePaths.cend(); ++redSamplePathItr)
	{
		MagicCard card(*redSamplePathItr);
		cv::addWeighted(_redFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _redFrameHistogram);
		cv::addWeighted(_redFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _redFrameMean_CIELAB);
	}
	std::cout << "Red card mean point: X:" << _redFrameMean_CIELAB[0] << " Y:" << _redFrameMean_CIELAB[1] << " Z:" << _redFrameMean_CIELAB[2] << "\n";

	// Blue frame samples
	std::vector<std::string> blueSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\blue\\captain of the mists.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\blue\\chant of the skifsang.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\blue\\counterlash.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\blue\\hisoka, minamo sensei.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\blue\\neurok prodigy.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\blue\\psychic overload.jpg" };
	_blueFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_blueFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator blueSamplePathItr = blueSamplePaths.cbegin(); blueSamplePathItr != blueSamplePaths.cend(); ++blueSamplePathItr)
	{
		MagicCard card(*blueSamplePathItr);
		cv::addWeighted(_blueFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blueFrameHistogram);
		cv::addWeighted(_blueFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _blueFrameMean_CIELAB);
	}
	std::cout << "Blue card mean point: X:" << _blueFrameMean_CIELAB[0] << " Y:" << _blueFrameMean_CIELAB[1] << " Z:" << _blueFrameMean_CIELAB[2] << "\n";

	// White frame samples
	std::vector<std::string> whiteSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\white\\archangel's light.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\white\\burden of guilt.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\white\\call to serve.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\white\\cursebreak.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\white\\pteron ghost.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\white\\pulse of the fields.jpg" };
	_whiteFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_whiteFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator whiteSamplePathItr = whiteSamplePaths.cbegin(); whiteSamplePathItr != whiteSamplePaths.cend(); ++whiteSamplePathItr)
	{
		MagicCard card(*whiteSamplePathItr);
		cv::addWeighted(_whiteFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _whiteFrameHistogram);
		cv::addWeighted(_whiteFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _whiteFrameMean_CIELAB);
	}
	std::cout << "White card mean point: X:" << _whiteFrameMean_CIELAB[0] << " Y:" << _whiteFrameMean_CIELAB[1] << " Z:" << _whiteFrameMean_CIELAB[2] << "\n";

	// Black frame samples
	std::vector<std::string> blackSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\black\\burden of greed.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\black\\deadly allure.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\black\\essence drain.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\black\\harvester of souls.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\black\\homicidal seclusion.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\black\\markov's servant.jpg" };
	_blackFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_blackFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator blackSamplePathItr = blackSamplePaths.cbegin(); blackSamplePathItr != blackSamplePaths.cend(); ++blackSamplePathItr)
	{
		MagicCard card(*blackSamplePathItr);
		cv::addWeighted(_blackFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _blackFrameHistogram);
		cv::addWeighted(_blackFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _blackFrameMean_CIELAB);
	}
	std::cout << "Black card mean point: X:" << _blackFrameMean_CIELAB[0] << " Y:" << _blackFrameMean_CIELAB[1] << " Z:" << _blackFrameMean_CIELAB[2] << "\n";

	// Multi-color frame samples
	std::vector<std::string> multiSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\yellow\\bruna, light of alabaster.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\yellow\\drogskol captain.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\yellow\\gisela, blade of goldnight.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\yellow\\havengul lich.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\yellow\\ravager of the fells.jpg",
												  ASSETS_PATH + "\\FrameTrainingImages\\yellow\\wrexial, the risen deep.jpg" };
	_yellowFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_yellowFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator multiSamplePathItr = multiSamplePaths.cbegin(); multiSamplePathItr != multiSamplePaths.cend(); ++multiSamplePathItr)
	{
		MagicCard card(*multiSamplePathItr);
		cv::addWeighted(_yellowFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _yellowFrameHistogram);
		cv::addWeighted(_yellowFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _yellowFrameMean_CIELAB);
	}
	std::cout << "Multi card mean point: X:" << _yellowFrameMean_CIELAB[0] << " Y:" << _yellowFrameMean_CIELAB[1] << " Z:" << _yellowFrameMean_CIELAB[2] << "\n";

	// Colorless frame samples
	std::vector<std::string> colorlessSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\artifact\\darksteel brute.jpg",
												      ASSETS_PATH + "\\FrameTrainingImages\\artifact\\darksteel reactor.jpg",
												      ASSETS_PATH + "\\FrameTrainingImages\\artifact\\general's kabuto.jpg",
												      ASSETS_PATH + "\\FrameTrainingImages\\artifact\\hankyu.jpg",
												      ASSETS_PATH + "\\FrameTrainingImages\\artifact\\kusari-gama.jpg",
												      ASSETS_PATH + "\\FrameTrainingImages\\artifact\\muse vessel.jpg" };
	_artifactFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_artifactFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator colorlessSamplePathItr = colorlessSamplePaths.cbegin(); colorlessSamplePathItr != colorlessSamplePaths.cend(); ++colorlessSamplePathItr)
	{
		MagicCard card(*colorlessSamplePathItr);
		cv::addWeighted(_artifactFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _artifactFrameHistogram);
		cv::addWeighted(_artifactFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _artifactFrameMean_CIELAB);
	}
	std::cout << "Colorless card mean point: X:" << _artifactFrameMean_CIELAB[0] << " Y:" << _artifactFrameMean_CIELAB[1] << " Z:" << _artifactFrameMean_CIELAB[2] << "\n";

	// Land frame samples
	std::vector<std::string> landSamplePaths = { ASSETS_PATH + "\\FrameTrainingImages\\land\\plains3.jpg",
											     ASSETS_PATH + "\\FrameTrainingImages\\land\\swamp1.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\land\\desolate lighthouse.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\land\\forest3.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\land\\island2.jpg",
												 ASSETS_PATH + "\\FrameTrainingImages\\land\\mountain2.jpg" };
	_landFrameHistogram = cv::Mat::zeros(numDimens, histogramSizes, CV_32F);
	_landFrameMean_CIELAB = cv::Scalar(0.0);
	for (std::vector<std::string>::const_iterator landSamplePathItr = landSamplePaths.cbegin(); landSamplePathItr != landSamplePaths.cend(); ++landSamplePathItr)
	{
		MagicCard card(*landSamplePathItr);
		cv::addWeighted(_landFrameHistogram, 1.0, card.getFrameHistogram(), accumulationRatio, 0.0, _landFrameHistogram);
		cv::addWeighted(_landFrameMean_CIELAB, 1.0, card.getFrameMeanColor_CIELAB(), accumulationRatio, 0.0, _landFrameMean_CIELAB);
	}
	std::cout << "Land card mean point: X:" << _landFrameMean_CIELAB[0] << " Y:" << _landFrameMean_CIELAB[1] << " Z:" << _landFrameMean_CIELAB[2] << "\n";

	// Set the live frame colors to the digital frame colors
	_greenFrameMean_CIELAB_Live = _greenFrameMean_CIELAB;
	_redFrameMean_CIELAB_Live = _redFrameMean_CIELAB;
	_blueFrameMean_CIELAB_Live = _blueFrameMean_CIELAB;
	_whiteFrameMean_CIELAB_Live = _whiteFrameMean_CIELAB;
	_blackFrameMean_CIELAB_Live = _blackFrameMean_CIELAB;
	_yellowFrameMean_CIELAB_Live = _yellowFrameMean_CIELAB;
	_artifactFrameMean_CIELAB_Live = _artifactFrameMean_CIELAB;
	_landFrameMean_CIELAB_Live = _landFrameMean_CIELAB;
	_greenFrameMean_CIELAB_Live_Samples = 1;
	_redFrameMean_CIELAB_Live_Samples = 1;
	_blueFrameMean_CIELAB_Live_Samples = 1;
	_whiteFrameMean_CIELAB_Live_Samples = 1;
	_blackFrameMean_CIELAB_Live_Samples = 1;
	_yellowFrameMean_CIELAB_Live_Samples = 1;
	_artifactFrameMean_CIELAB_Live_Samples = 1;
	_landFrameMean_CIELAB_Live_Samples = 1;
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


CardDetails::FrameColor CardDatabase::getDigitalCardColor(const MagicCard* card) const
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

	// Land
	const double landFrameDistance = calcDeltaE(cardFrameMean_CIELAB, _landFrameMean_CIELAB);
	if (landFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Land_Color;
		bestMatchingDistance = landFrameDistance;
	}

	return bestMatchingColor;
}

/*
CardDetails::FrameColor CardDatabase::getLiveCardColor(const MagicCard* card) const
{
	CardDetails::FrameColor bestMatchingColor = CardDetails::Unsure;
	double bestMatchingDistance = DBL_MAX;

	const cv::Scalar cardFrameMean_CIELAB = card->getFrameMeanColor_CIELAB();

	// Green
	const double greenFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _greenFrameMean_CIELAB_Live);
	if (greenFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Green;
		bestMatchingDistance = greenFrameDistance;
	}

	// Red
	const double redFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _redFrameMean_CIELAB_Live);
	if (redFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Red;
		bestMatchingDistance = redFrameDistance;
	}

	// Blue
	const double blueFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _blueFrameMean_CIELAB_Live);
	if (blueFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Blue;
		bestMatchingDistance = blueFrameDistance;
	}

	// White
	const double whiteFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _whiteFrameMean_CIELAB_Live);
	if (whiteFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::White;
		bestMatchingDistance = whiteFrameDistance;
	}

	// Black
	const double blackFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _blackFrameMean_CIELAB_Live);
	if (blackFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Black;
		bestMatchingDistance = blackFrameDistance;
	}

	// Multi-color
	const double yellowFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _yellowFrameMean_CIELAB_Live);
	if (yellowFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Multi;
		bestMatchingDistance = yellowFrameDistance;
	}

	// Colorless
	const double colorlessFrameDistance = calcDeltaE_noChroma(cardFrameMean_CIELAB, _artifactFrameMean_CIELAB_Live);
	if (colorlessFrameDistance < bestMatchingDistance)
	{
		bestMatchingColor = CardDetails::Colorless;
		bestMatchingDistance = colorlessFrameDistance;
	}


	// We need to check the chroma to differentiatie these cards
	if (bestMatchingColor == CardDetails::Colorless || bestMatchingColor == CardDetails::White || bestMatchingColor == CardDetails::Black)
	{
		const double cardChroma = cardFrameMean_CIELAB[0];

		// White
		bestMatchingColor = CardDetails::White;
		bestMatchingDistance = cv::sqrt(cv::pow((cardChroma - _whiteFrameMean_CIELAB_Live[0]), 2));

		// Black
		const double blackChromaFrameDistance = cv::sqrt(cv::pow((cardChroma - _blackFrameMean_CIELAB_Live[0]), 2));
		if (blackChromaFrameDistance < bestMatchingDistance)
		{
			bestMatchingColor = CardDetails::Black;
			bestMatchingDistance = blackChromaFrameDistance;
		}

		// Colorless
		const double colorlessChromaFrameDistance = cv::sqrt(cv::pow((cardChroma - _artifactFrameMean_CIELAB_Live[0]), 2));
		if (colorlessChromaFrameDistance < bestMatchingDistance)
		{
			bestMatchingColor = CardDetails::Colorless;
			bestMatchingDistance = colorlessChromaFrameDistance;
		}
	}

	return bestMatchingColor;
}


void CardDatabase::trainLiveCardColor(const MagicCard* card, const CardDetails::FrameColor fcolor)
{
	const cv::Scalar cardFrameMean_CIELAB = card->getFrameMeanColor_CIELAB();

	switch (fcolor)
	{
	case CardDetails::White:
		cv::addWeighted(_whiteFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _whiteFrameMean_CIELAB_Live); // Change to this way if the training is too slow
		//cv::addWeighted(_whiteFrameMean_CIELAB_Live, 1.0, cardFrameMean_CIELAB, (1 / ++_whiteFrameMean_CIELAB_Live_Samples), 0.0, _whiteFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live white frame: X:" << _whiteFrameMean_CIELAB_Live[0] << " Y:" << _whiteFrameMean_CIELAB_Live[1] << " Z:" << _whiteFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Black:
		cv::addWeighted(_blackFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _blackFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live black frame: X:" << _blackFrameMean_CIELAB_Live[0] << " Y:" << _blackFrameMean_CIELAB_Live[1] << " Z:" << _blackFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Red:
		cv::addWeighted(_redFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _redFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live red frame: X:" << _redFrameMean_CIELAB_Live[0] << " Y:" << _redFrameMean_CIELAB_Live[1] << " Z:" << _redFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Blue:
		cv::addWeighted(_blueFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _blueFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live blue frame: X:" << _blueFrameMean_CIELAB_Live[0] << " Y:" << _blueFrameMean_CIELAB_Live[1] << " Z:" << _blueFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Green:
		cv::addWeighted(_greenFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _greenFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live green frame: X:" << _greenFrameMean_CIELAB_Live[0] << " Y:" << _greenFrameMean_CIELAB_Live[1] << " Z:" << _greenFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Colorless:
		cv::addWeighted(_artifactFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _artifactFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live colorless frame: X:" << _artifactFrameMean_CIELAB_Live[0] << " Y:" << _artifactFrameMean_CIELAB_Live[1] << " Z:" << _artifactFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	case CardDetails::Multi:
		cv::addWeighted(_yellowFrameMean_CIELAB_Live, 0.5, cardFrameMean_CIELAB, 0.5, 0.0, _yellowFrameMean_CIELAB_Live);
		std::cout << "card frame CIELAB mean point for live multicolor frame: X:" << _yellowFrameMean_CIELAB_Live[0] << " Y:" << _yellowFrameMean_CIELAB_Live[1] << " Z:" << _yellowFrameMean_CIELAB_Live[2] << "\n";  // DEBUG
		break;
	default:
		// Not a valid color?
		assert(false);
		break;
	}
}


double CardDatabase::calcDeltaE_noChroma(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const
{
	// CIE76 - not great, but good enough
	cv::Scalar lineSegment = colorPointOne - colorPointTwo;
	lineSegment[1] = cv::pow(lineSegment[1], 2);
	lineSegment[2] = cv::pow(lineSegment[2], 2);
	const double euclidianDistance = cv::sqrt(lineSegment[1] + lineSegment[2]);

	return euclidianDistance;
}
*/


double CardDatabase::calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const
{
	// CIE76 - not great, but good enough
	cv::Scalar lineSegment = colorPointOne - colorPointTwo;
	lineSegment[0] = cv::pow(lineSegment[0], 2);
	lineSegment[1] = cv::pow(lineSegment[1], 2);
	lineSegment[2] = cv::pow(lineSegment[2], 2);
	const double euclidianDistance = cv::sqrt(lineSegment[0] + lineSegment[1] + lineSegment[2]);

	// CIE94 - better, but not yet perfect  /// can add this back in after video.  Can always omit L_culmination if want to drop luminosity like above
	//const double delta_Lstar = colorPointOne[0] - colorPointTwo[0];
	//const double delta_astar = colorPointOne[1] - colorPointTwo[1];
	//const double delta_bstar = colorPointOne[2] - colorPointTwo[2];
	//const double Cstar_one = cv::sqrt((colorPointOne[1] * colorPointOne[1]) + (colorPointOne[2] * colorPointOne[2]));
	//const double Cstar_two = cv::sqrt((colorPointTwo[1] * colorPointTwo[1]) + (colorPointTwo[2] * colorPointTwo[2]));
	//const double delta_Cstar_ab = Cstar_one - Cstar_two;
	//const double delta_Hstar_ab = cv::sqrt((delta_astar * delta_astar) + (delta_bstar * delta_bstar) - (delta_Cstar_ab * delta_Cstar_ab));
	//const double kL = 1; // graphic arts /// 2.0 // textiles
	//const double K_one = 0.045; // graphic arts /// 0.048 // textiles
	//const double K_two = 0.015; // graphic arts /// 0.014 // textiles
	//const double Sl = 1.0;
	//const double Sc = 1.0 + K_one * Cstar_one;
	//const double Sh = 1.0 + K_two * Cstar_one;
	//
	//const double kC = 1.0; // unity and weighting factors?
	//const double kH = 1.0; // unity and weighting factors?
	//
	//// almost there
	//const double L_culmination = delta_Lstar / (kL * Sl);
	//const double C_culmination = delta_Cstar_ab / (kC * Sc);
	//const double h_culmination = delta_Hstar_ab / (kH * Sh);
	//
	//// coup de grace
	//const double euclidianDistance = cv::sqrt((L_culmination * L_culmination) + (C_culmination * C_culmination) + (h_culmination *h_culmination));

	return euclidianDistance;
}


CardDetails::FrameColor CardDatabase::getLiveCardColor(const MagicCard* card) const
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

	// Land
	// Todo

	return bestMatchingColor;
}


void CardDatabase::trainLiveCardColor(const MagicCard* card, const CardDetails::FrameColor fcolor)
{
	const cv::Mat cardFrameHistogram = card->getFrameHistogram();

	switch (fcolor)
	{
	case CardDetails::White:
		cv::addWeighted(_whiteFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _whiteFrameHistogram); // Change to this way if the training is too slow
		break;
	case CardDetails::Black:
		cv::addWeighted(_blackFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _blackFrameHistogram);
		break;
	case CardDetails::Red:
		cv::addWeighted(_redFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _redFrameHistogram);
		break;
	case CardDetails::Blue:
		cv::addWeighted(_blueFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _blueFrameHistogram);
		break;
	case CardDetails::Green:
		cv::addWeighted(_greenFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _greenFrameHistogram);
		break;
	case CardDetails::Colorless:
		cv::addWeighted(_artifactFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _artifactFrameHistogram);
		break;
	case CardDetails::Multi:
		cv::addWeighted(_yellowFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _yellowFrameHistogram);
		break;
	case CardDetails::Land_Color:
		cv::addWeighted(_landFrameHistogram, 0.5, cardFrameHistogram, 0.5, 0.0, _landFrameHistogram);
		break;
	default:
		// Not a valid color?
		assert(false);
		break;
	}
}
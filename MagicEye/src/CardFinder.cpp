#include "CardFinder.h"
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG
#include <iostream> // DEBUG
#include <climits>
#include <algorithm>



CardFinder::CardFinder(CardDatabase * const cdb)
	:
	_cdb_ptr(cdb)
{
}


CardFinder::~CardFinder()
{
	_cdb_ptr = NULL;
}


std::vector<TableCard>* CardFinder::findAllCards(cv::Mat & scene)
{
	/// Step 1: find table top
	/// - caution, shadows may fool
	/// Step 2: find ROI's of objects that are not part of the table
	/// - caution, exclued pixels outside fo rectangle that is table top
	/// Step 3: find Magic cards inside ROI's
	/// - Try using Generalized Hough Transforms
	/// Step 4: For all full faced cards,
	/// - copy ROI, rotate upright
	/// - send to card database for analysis
	/// Step 5: For all partial cards,
	/// ???
	/// Profit
	//_lastFrameCards = _foundCards;																						// TODO maybe make vector of pointers
	rememberNewCards();
	forgetOldCards();
	_biggestBox = cv::Rect();
	_foundCards.clear();

	/// Step 1: find table top
	/// - caution, shadows may fool
	/// Step 2: find ROI's of objects that are not part of the table
	/// - caution, exclued pixels outside fo rectangle that is table top
	/// HSV floodfill idea
	// Get binary mask of the table in the scene
	cv::Mat tableTopMask = findPlayField(scene);

	// Get a bounding box that included the table; for later exclusing areas not on the table.
	cv::Mat tableTopMaskPoints;
	cv::findNonZero(tableTopMask, tableTopMaskPoints);
	cv::Rect boundingTableRect = cv::boundingRect(tableTopMaskPoints);

	// Debug - show where program thinks table top is in scene
	//cv::rectangle(scene, boundingTableRect, CV_RGB(255, 0, 255));														// Debug
	
	// Create ROIs of only the tabletop
	cv::Mat tableTopScene(scene, boundingTableRect);																	// Is used?
	cv::Mat tableTopBinary(tableTopMask, boundingTableRect);

	// invert mask within tabletop bounding box to get "object on table" mask
	tableTopBinary = 255 - tableTopBinary; // invert mask
	//cv::imshow("trash", tableTopBinary);  // Show the bin mask of all objects that are on the table							// Debug

	// find the edges of all object on top of the field
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(tableTopBinary.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// encase all of the unkown object edges in bounding rectangles
	for (auto cardEdgeItr = contours.begin(); cardEdgeItr != contours.end(); ++cardEdgeItr)
	{
		const cv::Rect ROI = cv::boundingRect(*cardEdgeItr) + boundingTableRect.tl();
		//cv::rectangle(scene, ROI, CV_RGB(0, 0, 255));																	// Debug

		if (ROI.area() > _biggestBox.area())
		{
			// For omitting cards under player's hand from timing out and being forgotten
			_biggestBox = ROI;
		}

		// eliminate noise
		if (ROI.area() > MIN_AREA_ELIMINATION_THRESHOLD)
		{
			identifyCardsInRegion(scene(ROI), ROI.tl(), boundingTableRect, _foundCards);
		}
	}

	// Debug, draw rectangles of memory cards
	for (auto MemC_itr = _cardMemory.cbegin(); MemC_itr != _cardMemory.cend(); ++MemC_itr)
	{
		// full cards are megenta, all other cards are yellow
		outlineRotatedRectangle(scene, MemC_itr->getMinimumBoundingRect(), CV_RGB(255, 255, 0));
	}

	// Draw rectangles around identified Magic cards
	for (auto MC_itr = _foundCards.cbegin(); MC_itr != _foundCards.cend(); ++MC_itr)
	{
		// full cards are megenta, all other cards are yellow
		outlineRotatedRectangle(scene, MC_itr->getMinimumBoundingRect(), (MC_itr->getCardVisibility() == TableCard::Visible ? CV_RGB(255, 0, 255) : CV_RGB(0, 255, 255)));
	}
	
	evaluateCardsColors(_foundCards);
	return &_foundCards;
}


void CardFinder::identifyCardsInRegion(const cv::Mat & ROI, const cv::Point ROIOffset, const cv::Rect & tableBB, std::vector<TableCard>& runningList) const
{
	// Save a copy of the scene so we can turn it black to sort out stacked cards.
	cv::Mat originalImage = ROI.clone();

	// threshold away anything that is not dark black
	cv::Mat bwScene;
	cv::cvtColor(ROI, bwScene, CV_BGR2GRAY);  // OR: Now filter the black regions by filtering the HSV Range V- 100 to 255. The result will look like this.
	//const double meanSceneChroma = cv::mean(bwScene)[0];
	//const int borderThreshold = static_cast<int>(meanSceneChroma * 0.2) + 30; // was 0.165  57 for the red card
	cv::threshold(bwScene, bwScene, 80, 255, CV_THRESH_BINARY_INV); // | CV_THRESH_OTSU); // was 42 // 70 for mid day shots
	//cv::Canny(bwScene, bwScene, 20, 50);
	
	// reduce noise with erosion
	cv::erode(bwScene, bwScene, cv::Mat());
	cv::dilate(bwScene, bwScene, cv::Mat()); // need this or sometimes borders get cut by erosion
	//cv::imshow("binScene", bwScene);
	//cv::waitKey();

	// find edges
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(bwScene.clone(), contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);

	// Only process the contours at level 1
	// Those are the ones inside the outer most contours and represent the card faces without the black borders.
	assert(hierarchy.size() == contours.size());
	auto hierarchy_itr = hierarchy.begin();
	auto contours_itr = contours.begin();
	while (hierarchy_itr != hierarchy.end())
	{
		// erase all contours that are not on the first level (0, since -1 is ground floor) of the contour hierarchy
		if ((*hierarchy_itr)[3] != 0)
		{
			hierarchy_itr = hierarchy.erase(hierarchy_itr);
			contours_itr = contours.erase(contours_itr);
		}
		else
		{
			// move along
			++hierarchy_itr;
			++contours_itr;
		}
	}

	// Now that only lvl1 contours remain...
	bool isFirstContour = true;
	while (!contours.empty()) // While there are still contours to process
	{
		//If there is only ONE contour; this is a single card
		//If there is more than one contour; find the contour that is most rectangular: that is the most faceing card
		// - note: we're going to assume that the contour with the biggest area is probably a card
	
		// Find the biggest (area) remaining contour
		double biggestAreaYet = 0.0;
		auto biggestContourFound = contours.begin();
		for (auto contour_itr = contours.begin(); contour_itr != contours.end(); ++contour_itr)
		{
			double currentArea = cv::contourArea(*contour_itr);
			if (currentArea > biggestAreaYet)
			{
				biggestAreaYet = currentArea;
				biggestContourFound = contour_itr;
			}
		}
	
		// Find minimum bounding rectangle
		const cv::RotatedRect cardBoundingBox = cv::minAreaRect(*biggestContourFound);
		std::vector<cv::Point2f> bbContours(4); // number of points in a rectangle
		cardBoundingBox.points(&bbContours.front());
	
		// Determine if it is a full card or is partially blocked
		/// TODO: Explore if have time
		/*
		double epsilon = cv::arcLength(*biggestContourFound, true) * 0.1;
		cv::approxPolyDP(*biggestContourFound, *biggestContourFound, epsilon, true);
		const bool isFullCard = cv::isContourConvex(*biggestContourFound);
		*/
	
		// remove all contours that are completely enclosed by the bounding rectangle enclosing the largest contour, including the largest contour
		// this will remove noise that is part of a card
		contours.erase(biggestContourFound); // remove this now because once we remove one object this iterator is invalidated.
		if (biggestAreaYet < MIN_AREA_ELIMINATION_THRESHOLD)
		{
			continue;
		}
	
		auto smallContour_itr = contours.begin();
		while ( smallContour_itr != contours.end())
		{
			// Take care as to not remove the winning contour prematurely
			if (isContourConsumedByAnother(*smallContour_itr, bbContours))
			{
				smallContour_itr = contours.erase(smallContour_itr);
			}
			else
			{
				// move along
				++smallContour_itr;
			}
		}
	
		// For each card face found; copy, rotate, and add black border. Then ship it to TableCard class
		runningList.emplace_back(extractCardImage(originalImage, cardBoundingBox, ROIOffset, tableBB, isFirstContour));
		isFirstContour = false;
	
		// Blackout where card used to be for later processing on same region
		// OPTIMIZATION: Do not have to do if no contours remain at this point
		blackoutRotatedRectangle(originalImage, cardBoundingBox);
	}

	// DEBUG
	/*
	cv::Mat contoursImage = cv::Mat::zeros(bwScene.size(), CV_8UC3);
	cv::RNG rng(12345);
	for (int i = 0; i< contours.size(); i++)
	{
		//if (hierarchy[i][3] < 1) // Only draw the outer contours and one level down.
		{
			cv::Scalar color = cv::Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
			drawContours(contoursImage, contours, i, color, 2, 8, hierarchy, 0, cv::Point());
		}
	}

	static std::string title = "s";
	title += "s";
	cv::imshow(title, contoursImage);
	cv::waitKey();
	*/
}


TableCard CardFinder::extractCardImage(const cv::Mat & fromScene, const cv::RotatedRect boundingRect, const cv::Point worldPosition, const cv::Rect & tableBB, const bool isFullCard) const
{
	cv::Mat rotatedUpright;

	float boxAngle = boundingRect.angle;
	cv::Size boxSize = boundingRect.size;
	cv::Size canvasSize = fromScene.size();

	// using wrong orientation, correct
	if (boxAngle < -45.f)
	{
		boxAngle += 90.f;
		cv::swap(boxSize.width, boxSize.height);
	}

	/// Cards should be right side up.  If it is wider than it is tall, it's sideways and needs to be rotated 90 degrees
	if (boxSize.width > boxSize.height)  // I don't know if I can trust boxSize; if not, trust canvasSize
	{
		boxAngle -= 90.f;
		
		cv::swap(boxSize.width, boxSize.height);
		cv::swap(canvasSize.width, canvasSize.height);
	}

	cv::Mat rotationMat = cv::getRotationMatrix2D(boundingRect.center, boxAngle, 1.0);

	// adjust transformation matrix, Makes image dead center
	rotationMat.at<double>(0, 2) += canvasSize.width / 2.0 - boundingRect.center.x;
	rotationMat.at<double>(1, 2) += canvasSize.height / 2.0 - boundingRect.center.y;

	cv::warpAffine(fromScene, rotatedUpright, rotationMat, canvasSize, CV_INTER_CUBIC);

	// Need to crop out extra information after rotation
	cv::Mat croppedToFit;
	cv::Point cardCenter = cv::Point(canvasSize.width / 2, canvasSize.height / 2);
	//croppedToFit = rotatedUpright(cv::Rect(cardCenter.x - (boxSize.width / 2), cardCenter.y - (boxSize.height / 2), boxSize.width, boxSize.height));
	cv::getRectSubPix(rotatedUpright, boxSize, cardCenter/*boundingRect.center*/, croppedToFit);

	// return 
	const cv::Point2f newCenter(boundingRect.center.x + static_cast<float>(worldPosition.x), boundingRect.center.y + static_cast<float>(worldPosition.y));
	const cv::RotatedRect cardBoundingBox(newCenter, boundingRect.size, boundingRect.angle);
	const TableCard::VisibilityState cardVisibility = (isFullCard ? TableCard::Visible : TableCard::PartialBlockedUnidentified);
	return TableCard(cardBoundingBox, croppedToFit, cardVisibility);
}


bool CardFinder::isContourConsumedByAnother(const std::vector<cv::Point> contour, const std::vector<cv::Point2f> consumedBy) const
{
	bool isCompletelyEnclosed = true;

	for (auto point_itr = contour.cbegin(); point_itr != contour.cend(); ++point_itr)
	{
		if (cv::pointPolygonTest(consumedBy, *point_itr, false) == -1) // if at least one point lies outside of the enclosing contour...
		{
			isCompletelyEnclosed = false;
			break;
		}
	}

	return isCompletelyEnclosed;
}


void CardFinder::outlineRotatedRectangle(cv::Mat & scene, const cv::RotatedRect RR, const cv::Scalar & color) const
{
	const int pointsInARectangle = 4;
	cv::Point2f vertices[pointsInARectangle];
	RR.points(vertices);
	for (int index = 0; index < pointsInARectangle; ++index)
	{
		cv::line(scene, vertices[index], vertices[(index + 1) % pointsInARectangle], color);
	}
}


void CardFinder::blackoutRotatedRectangle(cv::Mat & scene, const cv::RotatedRect RR) const
{
	const int pointsInARectangle = 4;
	cv::Point2f vertices2f_array[pointsInARectangle];
	RR.points(vertices2f_array);

	std::vector<cv::Point> vertices2f(std::begin(vertices2f_array), std::end(vertices2f_array));

	cv::fillConvexPoly(scene, vertices2f, CV_RGB(0, 0, 0));
}


cv::Mat CardFinder::findPlayField(const cv::Mat & scene) const
{
	/// Ideas if needed:
	// Check all three channels of HSV.
	// Or, convert to CIELAB, then floodfill
	// Or, once table corners are established. use those as floodfill seeds too? (not sure on this one)
	cv::Mat hsvScene;
	cv::cvtColor(scene, hsvScene, CV_BGR2HSV);

	std::vector<cv::Mat> _channels;
	cv::split(hsvScene, _channels);

	cv::Mat colorizedSceneHue;
	cv::applyColorMap(_channels[0], colorizedSceneHue, cv::COLORMAP_JET);

	cv::Mat backgroundMask = cv::Mat::zeros(scene.rows + 2, scene.cols + 2, CV_8UC1);
	cv::Point centerPoint((backgroundMask.cols / 2) + 1, (backgroundMask.rows / 2) + 1);
	cv::floodFill(_channels[1], backgroundMask, centerPoint, CV_RGB(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX), NULL, cv::Scalar(4.0), cv::Scalar(3.0), 4 | CV_FLOODFILL_MASK_ONLY | (255 << 8));

	// Take out the extra pixel boundrey floodfill needed.
	backgroundMask = backgroundMask(cv::Rect(1, 1, scene.cols, scene.rows));

	/// Why do I need this?  Does Floodfill do this?  Fix floodfill so it doesn't change my zero pixels
	cv::threshold(backgroundMask, backgroundMask, 20, UCHAR_MAX, CV_THRESH_BINARY);

	/// Idea: dilate -> erode to remove noise?
	cv::dilate(backgroundMask, backgroundMask, cv::Mat());
	cv::erode(backgroundMask, backgroundMask, cv::Mat());

	//cv::imshow("sceneHue", colorizedSceneHue);
	//cv::imshow("sceneSaturation", colorizedSceneSaturation);
	//cv::imshow("sceneValue", colorizedSceneValue);
	//cv::imshow("background scene", backgroundMask);

	return backgroundMask;
}


// Partial card finder functions try 1
void CardFinder::rememberNewCards()
{
	for (auto seenCards_itr = _foundCards.begin(); seenCards_itr != _foundCards.end(); ++seenCards_itr)
	{
		if ((seenCards_itr->getCardVisibility() == TableCard::Visible) && (!recallCard(*seenCards_itr)))
		{
			// New card, remember it
			_cardMemory.push_back(*seenCards_itr);
		}
	}
}


void CardFinder::forgetOldCards()
{
	const double timeToForgetThreshold = 8.0; // was 8.0
	auto memoryCards_itr = _cardMemory.begin();
	while ( memoryCards_itr != _cardMemory.end() )
	{
		/*
		if (_biggestBox.contains(memoryCards_itr->getMinimumBoundingRect().center))
		{
			// if the card is contained under the biggest box, then it should not be forgotten yet since it's obstructed by the payer's hand
			// If the biggest box is not the player's hand, then those cards should not be forgotten anyways.
			// hack fix though, should use temporal data when have time
			memoryCards_itr->resetTimedReferenceCheck();
			++memoryCards_itr;
		}
		else // let it be forgotten if it times out
		{
		*/
			if (memoryCards_itr->checkIfXSecondsSinceLastReference(timeToForgetThreshold))
			{
				memoryCards_itr = _cardMemory.erase(memoryCards_itr);
				// I don't care if this skips a card, it'll get it on the next pass.
			}
			else
			{
				++memoryCards_itr;
			}
		//}
	}
}


bool CardFinder::recallCard(const TableCard& cardToRecall)
{
	// Go through all of the remembered cards and see if this card exists
	for (auto recallCard_itr = _cardMemory.begin(); recallCard_itr != _cardMemory.end(); ++recallCard_itr)
	{
		if (recallCard_itr->isProbablySameTableCard(cardToRecall))
		{
			recallCard_itr->resetTimedReferenceCheck();
			return true;
		}
	}

	return false;
}


void CardFinder::reevaluateMemoryCards()
{
	for (auto memoryCard_itr = _cardMemory.begin(); memoryCard_itr != _cardMemory.end(); ++memoryCard_itr)
	{
		const CardDetails::FrameColor descernedFrameColor = _cdb_ptr->getLiveCardColor(memoryCard_itr->getMagicCard());
		memoryCard_itr->setCardFrameColor(descernedFrameColor);
	}
}


void CardFinder::evaluateCardsColors(std::vector<TableCard> & cards)
{
	for (auto tableCard_itr = cards.begin(); tableCard_itr != cards.end(); ++tableCard_itr)
	{
		const CardDetails::FrameColor descernedFrameColor = _cdb_ptr->getLiveCardColor(tableCard_itr->getMagicCard());
		tableCard_itr->setCardFrameColor(descernedFrameColor);
	}
}


// Try 3 for the function I hate
void CardFinder::discernPartialCards()
{
	// Step 1) split found card list into visible cards and partial cards
	// Step 2) use found visible cards to make list of potential cards from memory (by excluding visible cards)
	// step 3) For each partialUnidentified card:
	/// Get sublist of potential cards by matching by color
	/// rank each potential card by position to last known location?
	/// Keep list of all potential cards for learning?

	std::vector<int> potentialCardIndicies;
	//for (auto ptrMaker_itr = _cardMemory.begin(); ptrMaker_itr != _cardMemory.end(); ++ptrMaker_itr)
	for (int index = 0; index < _cardMemory.size(); index++)
	{
		_cardMemory[index].hasBeenIdentified = false; // reset has been identified
		potentialCardIndicies.push_back(index);
	}

	// step 1 & 2
	//std::vector<TableCard> potentialCards = _cardMemory;
	// need to remove cards from the old list that are accounted for
	for (auto foundCard_itr = _foundCards.begin(); foundCard_itr != _foundCards.end(); ++foundCard_itr)
	{
		// for each visible car on the new list, delete that card from the old list
		if (foundCard_itr->getCardVisibility() == TableCard::Visible)
		{
			//std::remove_if(potentialCards.begin(), potentialCards.end(), [foundCard_itr](TableCard pot_tableCard) { return foundCard_itr->isProbablySameTableCard(pot_tableCard); });
			auto indexRemover_itr = potentialCardIndicies.begin();
			while (indexRemover_itr != potentialCardIndicies.end())
			{
				if (foundCard_itr->isProbablySameTableCard(_cardMemory[(*indexRemover_itr)]))
				{
					_cardMemory[(*indexRemover_itr)].hasBeenIdentified = true;
					indexRemover_itr = potentialCardIndicies.erase(indexRemover_itr);
				}
				else
				{
					++indexRemover_itr;
				}
			}
		}
	}

	// step 3
	for (auto foundCardUnknown_itr = _foundCards.begin(); foundCardUnknown_itr != _foundCards.end(); ++foundCardUnknown_itr)
	{
		if (foundCardUnknown_itr->getCardVisibility() == TableCard::PartialBlockedUnidentified)
		{
			// Go through all of the remembered cards and see if this card exists
			for (auto recallCard_itr = _cardMemory.begin(); recallCard_itr != _cardMemory.end(); ++recallCard_itr)
			{
				if (recallCard_itr->getCardVisibility() == TableCard::PartialBlocked && (recallCard_itr->hasBeenIdentified == false) && recallCard_itr->isProbablySameTableCard(*foundCardUnknown_itr))
				{
					recallCard_itr->resetTimedReferenceCheck();
					foundCardUnknown_itr->setToAssumedCard(*recallCard_itr);
					recallCard_itr->hasBeenIdentified = true;
					break; // bounce out
				}
			}
		}
	}

	for (auto foundCardUnknown_itr = _foundCards.begin(); foundCardUnknown_itr != _foundCards.end(); ++foundCardUnknown_itr)
	{
		if (foundCardUnknown_itr->getCardVisibility() == TableCard::PartialBlockedUnidentified)
		{
			//////  Doesn't have a partial match yet, let's find it.
			std::vector<int> potentialColorMatchIndicies;
			for (auto colorMatcher_itr = potentialCardIndicies.begin(); colorMatcher_itr != potentialCardIndicies.end(); ++colorMatcher_itr)
			{
				if (foundCardUnknown_itr->getCardFrameColor() == _cardMemory[(*colorMatcher_itr)].getCardFrameColor() && (_cardMemory[(*colorMatcher_itr)].hasBeenIdentified == false))
				{
					potentialColorMatchIndicies.push_back(*colorMatcher_itr);
				}
			}
			//std::copy_if(potentialCards.begin(), potentialCards.end(), potentialColorMatches.begin(), [foundCardUnknown_itr](TableCard pot_tableCard) { return foundCardUnknown_itr->getCardFrameColor() == pot_tableCard.getCardFrameColor(); });

			int bestMatch_index = -1;
			double closestDistance = DBL_MAX;
			// TODO: could add weight to already partially blocked cards? Probably not
			for (auto potentialCards_itr = potentialColorMatchIndicies.begin(); potentialCards_itr != potentialColorMatchIndicies.end(); ++potentialCards_itr)
			{
				double newDistance = _cardMemory[(*potentialCards_itr)].distanceFrom(*foundCardUnknown_itr);
				if (newDistance < closestDistance)
				{
					closestDistance = newDistance;
					bestMatch_index = *potentialCards_itr;
				}
			}

			if (bestMatch_index != -1)
			{
				foundCardUnknown_itr->setToAssumedCard(_cardMemory[bestMatch_index]);
				_cardMemory[bestMatch_index] = *foundCardUnknown_itr;
				_cardMemory[bestMatch_index].hasBeenIdentified = true;
			}
		}
	}
}
#include "CardFinder.h"
#include <opencv2\imgproc.hpp>
#include <opencv2\highgui.hpp> // DEBUG
#include <iostream> // DEBUG
#include <climits>



CardFinder::CardFinder()
{
}


CardFinder::~CardFinder()
{
}


std::vector<TableCard> CardFinder::findAllCards(cv::Mat & scene)
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
	std::vector<cv::Rect> cardROIsInScene;
	_foundCards.clear();																								// TODO - temp, testing only

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
	cv::rectangle(scene, boundingTableRect, CV_RGB(255, 0, 255));														// Debug
	
	// Create ROIs of only the tabletop
	cv::Mat tableTopScene(scene, boundingTableRect);																	// Is used?
	cv::Mat tableTopBinary(tableTopMask, boundingTableRect);

	// invert mask within tabletop bounding box to get "object on table" mask
	tableTopBinary = 255 - tableTopBinary; // invert mask
	//cv::imshow("trash", tableTop);  // Show the bin mask of all objects that are on the table							// Debug

	// find the edges of all object on top of the field
	std::vector<std::vector<cv::Point>> contours;
	cv::findContours(tableTopBinary.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

	// encase all of the unkown object edges in bounding rectangles
	for (auto cardEdgeItr = contours.begin(); cardEdgeItr != contours.end(); ++cardEdgeItr)
	{
		const cv::Rect ROI = cv::boundingRect(*cardEdgeItr) + boundingTableRect.tl();

		// eliminate noise
		if (ROI.area() > MIN_AREA_ELIMINATION_THRESHOLD)
		{
			identifyCardsInRegion(scene(ROI), ROI.tl(), boundingTableRect, _foundCards);
		}
	}

	// Draw rectangles around identified Magic cards
	for (auto MC_itr = _foundCards.cbegin(); MC_itr != _foundCards.cend(); ++MC_itr)
	{
		outlineRotatedRectangle(scene, MC_itr->getMinimumBoundingRect());
	}
	
	//return cardROIsInScene;
	return _foundCards;
}


void CardFinder::identifyCardsInRegion(const cv::Mat & ROI, const cv::Point ROIOffset, const cv::Rect & tableBB, std::vector<TableCard>& runningList) const
{
	// Save a copy of the scene so we can turn it black to sort out stacked cards.
	cv::Mat originalImage = ROI.clone();

	// threshold away anything that is not dark black
	cv::Mat bwScene;
	cv::cvtColor(ROI, bwScene, CV_BGR2GRAY);  // OR: Now filter the black regions by filtering the HSV Range V- 100 to 255. The result will look like this.
	cv::threshold(bwScene, bwScene, 42, 255, CV_THRESH_BINARY_INV);// | CV_THRESH_OTSU); // was 42
	
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

		// remove all contours that are completely enclosed by the bounding rectangle enclosing the largest contour, including the largest contour
		// this will remove noise that is part of a card
		contours.erase(biggestContourFound); // remove this now because once we remove one object this iterator is invalidated.
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
		runningList.emplace_back(extractCardImage(originalImage, cardBoundingBox, ROIOffset, tableBB));

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


TableCard CardFinder::extractCardImage(const cv::Mat & fromScene, const cv::RotatedRect boundingRect, const cv::Point worldPosition, const cv::Rect & tableBB) const
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
	return TableCard(cardBoundingBox, croppedToFit);
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


void CardFinder::outlineRotatedRectangle(cv::Mat & scene, const cv::RotatedRect RR) const
{
	const int pointsInARectangle = 4;
	cv::Point2f vertices[pointsInARectangle];
	RR.points(vertices);
	for (int index = 0; index < pointsInARectangle; ++index)
	{
		cv::line(scene, vertices[index], vertices[(index + 1) % pointsInARectangle], CV_RGB(255, 0, 0));
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
	cv::floodFill(_channels[1], backgroundMask, centerPoint, CV_RGB(UCHAR_MAX, UCHAR_MAX, UCHAR_MAX), NULL, cv::Scalar(5.0), cv::Scalar(5.0), 4 | CV_FLOODFILL_MASK_ONLY | (255 << 8));

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

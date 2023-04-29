#include "TableCard.h"
#include <opencv2/imgproc.hpp>

/**
 * @brief Construct a new TableCard::TableCard object.
 *
 */
TableCard::TableCard()
    : hasBeenIdentified(false),
      _assumedCard_ptr(nullptr),
      _boundingBoxInScene(),
      _cardFrameColor(CardDetails::Unsure),
      _visibilityState(VisibilityState::Missing),
      _lastReferenced(std::chrono::system_clock::now()),
      _forceExpire(false)
{
}

/**
 * @brief Construct a new TableCard::TableCard object.
 *
 * @param boundingBox The minimum bounding rectangle of the card to track.
 * @param cardImage The lifted image from the scene of the card to track.
 * @param vstate The current visibility state of the card to track.
 */
TableCard::TableCard(const cv::RotatedRect boundingBox, const cv::Mat &cardImage, const VisibilityState vstate)
    : hasBeenIdentified(false),
      _boundingBoxInScene(boundingBox),
      _cardFrameColor(CardDetails::Unsure),
      _visibilityState(vstate),
      _lastReferenced(std::chrono::system_clock::now()),
      _forceExpire(false)
{
    // add a black border
    cv::Mat cardImageWithBorder;
    cv::copyMakeBorder(cardImage, cardImageWithBorder, 10, 10, 10, 10, cv::BORDER_CONSTANT, CV_RGB(0, 0, 0));

    // check to see if we need to flip, will do nothing if nothing needs to be done
    makeRightsideUp(cardImageWithBorder);

    /// Make a new magic card and pass in ROI of rotated image
    _assumedCard_ptr = std::make_shared<MagicCard>(cardImageWithBorder);
    _assumedCard_ptr->deepAnalyze();
}

/**
 * @brief Construct a new TableCard::TableCard object.
 *
 * @param obj The TableCard object to copy from.
 */
TableCard::TableCard(const TableCard &obj)
    : hasBeenIdentified(obj.hasBeenIdentified),
      _assumedCard_ptr(obj._assumedCard_ptr),
      _boundingBoxInScene(obj._boundingBoxInScene),
      _cardFrameColor(obj._cardFrameColor),
      _visibilityState(obj._visibilityState),
      _lastReferenced(obj._lastReferenced),
      _forceExpire(obj._forceExpire)
{
    // Everything else is covered
}

/**
 * @brief Destroy the TableCard::TableCard object.
 *
 */
TableCard::~TableCard()
{
    // Let the shared_ptr object dereference itself
    _assumedCard_ptr = nullptr;
}

/**
 * @brief Returns axis aligned bounding rectangle of this card.
 *
 * @return cv::Rect Axis aligned bounding rectangle of this TableCard.
 */
cv::Rect TableCard::getBoundingRect() const
{
    return _boundingBoxInScene.boundingRect();
}

/**
 * @brief Returns the minimum bounding rectangle of this card.
 *
 * @return cv::RotatedRect The minimum bounding rectangle of this card.
 */
cv::RotatedRect TableCard::getMinimumBoundingRect() const
{
    return _boundingBoxInScene;
}

/**
 * @brief Returns the Magic card that this TableCard is tracking.
 *
 * @return MagicCard* Pointer to the Magic card that this TableCard is tracking.
 */
MagicCard *TableCard::getMagicCard() const
{
    // Maybe should return a weak pointer
    return _assumedCard_ptr.get();
}

/**
 * @brief Checks to see if the given point falls within this card.
 *
 * @param point The point to check against the area of this card.
 * @return true The point does lie within this card.
 * @return false The point is not ontop of this card.
 */
bool TableCard::isPointInside([[maybe_unused]] cv::Point point) const
{
    // TODO
    return true;
}

/**
 * @brief Returns this card's visibility state. Whether it is fully visible, partially occluded, etc.
 *
 * @return TableCard::VisibilityState This card's current visibility state.
 */
TableCard::VisibilityState TableCard::getCardVisibility() const
{
    return _visibilityState;
}

/**
 * @brief Returns this card's frame color.
 *
 * @return CardDetails::FrameColor Returns ( Red, Blue, Green, White, Black, etc )
 */
CardDetails::FrameColor TableCard::getCardFrameColor() const
{
    return _cardFrameColor;
}

/**
 * @brief Sets this card's color.
 *
 * @param fcolor The color to set this card to for identification.
 */
void TableCard::setCardFrameColor(const CardDetails::FrameColor fcolor)
{
    _cardFrameColor = fcolor;
    _assumedCard_ptr->setCardFrameColor(fcolor);
}

/**
 * @brief Checks to see if another card identification is probably the same as this card.
 *
 * @param other The other card identification to check against.
 * @return true The other TableCard instance is representing the same card as this one.
 * @return false The other TableCard instance is representing a different card.
 */
bool TableCard::isProbablySameTableCard(const TableCard &other) const
{
    // get positions
    cv::Point distanceVector = this->_boundingBoxInScene.center - other._boundingBoxInScene.center;
    const double distanceBetween =
        cv::sqrt((distanceVector.x * distanceVector.x) + (distanceVector.y * distanceVector.y));

    // get distance threshold
    const double smallestDimension =
        cv::min(this->_boundingBoxInScene.size.height, this->_boundingBoxInScene.size.width);
    const double distanceThreshold = smallestDimension * 0.2;

    // get volumes
    const double areaDifference =
        cv::abs(other._boundingBoxInScene.size.area() - this->_boundingBoxInScene.size.area());

    // get area threshold
    const double smallestArea = cv::min(other._boundingBoxInScene.size.area(), this->_boundingBoxInScene.size.area());
    const double areaThreshold = smallestArea * 0.1;

    // Do we need more?

    return (distanceBetween < distanceThreshold) && (areaDifference < areaThreshold);
}

/**
 * @brief Given another TableCard, returns a score depicting the distance from this card to the other
 * so long as these two TableCards area are similar.
 *
 * @param other The other TableCard to comapre this TableCard against.
 * @return double Distance from this TableCard to another, or DBL_MAX if the areas of these two TableCards is too
 * different.
 */
double TableCard::distanceFrom(const TableCard &other) const
{
    cv::Point distanceVector = this->_boundingBoxInScene.center - other._boundingBoxInScene.center;
    const double distanceBetween =
        cv::sqrt((distanceVector.x * distanceVector.x) + (distanceVector.y * distanceVector.y));

    // get volumes
    const double areaDifference =
        cv::abs(other._boundingBoxInScene.size.area() - this->_boundingBoxInScene.size.area());

    // get area threshold
    const double smallestArea = cv::min(other._boundingBoxInScene.size.area(), this->_boundingBoxInScene.size.area());
    const double areaThreshold = smallestArea * 0.1;

    // Cards are the same size, invalidate distance if the size between two cards is vastly different
    return (areaDifference < areaThreshold) ? distanceBetween : std::numeric_limits<double>::max();
}

/**
 * @brief Sets this card's assumed card data to another TableCard's since they are probaly tracking the same card.
 *
 * @param isProbablyThis The other TableCard that this TableCard will set it's assumed card to.
 */
void TableCard::setToAssumedCard(const TableCard &isProbablyThis)
{
    assert(this->_visibilityState == VisibilityState::PartialBlockedUnidentified);

    this->_visibilityState = VisibilityState::PartialBlocked;
    this->_assumedCard_ptr = isProbablyThis._assumedCard_ptr;
}

/**
 * @brief Sets this card's spatial properties to another card's.
 *
 * @param isProbablyHere Other TableCard whose spatial data we will copy.
 */
void TableCard::setToAssumedBoundingBox(const TableCard &isProbablyHere)
{
    this->_boundingBoxInScene.center = isProbablyHere._boundingBoxInScene.center;
    this->_boundingBoxInScene.angle = isProbablyHere._boundingBoxInScene.angle;
}

/**
 * @brief Checks to see if this card was referenced within the last (seconds) seconds
 *
 * @param seconds The upper limit, in seconds, since this card was last referenced
 * @return true This card was last referenced within the passed in seconds
 * @return false This card was not referenced within the time allotted
 */
bool TableCard::checkIfXSecondsSinceLastReference(const double seconds) const
{
    // seconds since last referenced
    const std::chrono::duration<double> lapsedTime = std::chrono::system_clock::now() - _lastReferenced;
    const std::chrono::duration<double> secondsThreshold(seconds);
    return _forceExpire || (lapsedTime > secondsThreshold);
}

/**
 * @brief Resets this TableCard's last referenced timer to now.
 *
 */
void TableCard::resetTimedReferenceCheck()
{
    _lastReferenced = std::chrono::system_clock::now();
}

/**
 * @brief Forces this TableCard's reference check to expire.
 *
 */
void TableCard::expireTimedReferenceCheck()
{
    _forceExpire = true;
}

/**
 * @brief Checks to see if we need to flip the image of this TableCard if it is upside down,
 * then flips the image if it needs to.
 *
 * @param cardImage The card image the flip rightside up if needed.
 */
void TableCard::makeRightsideUp(cv::Mat &cardImage) const
{
    const double inletRatio = 0.05; // 23 pixel in from 460 px width

    /// Could probably optimize with floodfill?
    // cut out the black border
    cv::Mat card = cardImage.clone();
    cv::cvtColor(card, card, cv::COLOR_BGR2GRAY);
    cv::threshold(card, card, 40, UCHAR_MAX, cv::THRESH_BINARY);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(card, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // Merge all contours points into one
    std::vector<cv::Point> allPoints;
    for (std::size_t outer = 0; outer < contours.size(); ++outer)
    {
        for (std::vector<cv::Point>::iterator pointItr = contours[outer].begin(); pointItr != contours[outer].end();
             ++pointItr)
        {
            allPoints.push_back(*pointItr);
        }
    }

    cv::Rect cardROI = cv::boundingRect(allPoints);

    if (cardROI.area() == 0)
        return;

    // Sample the two vertical trace lines
    const int leftColumn = static_cast<int>(cardROI.width * inletRatio);
    const int rightColumn = cv::min((cardROI.width - leftColumn), (cardROI.width - 1));

    cv::Mat cardAlone = cardImage(cardROI);
    cv::Rect leftColumnROI = cv::Rect(leftColumn, 0, 1, cardAlone.rows);
    cv::Rect rightColumnROI = cv::Rect(rightColumn, 0, 1, cardAlone.rows);
    cv::Mat leftColumnBW, rightColumnBW;

    cv::cvtColor(cardAlone(leftColumnROI), leftColumnBW, cv::COLOR_BGR2GRAY);
    cv::cvtColor(cardAlone(rightColumnROI), rightColumnBW, cv::COLOR_BGR2GRAY);

    // make ranges for upper and lower halves
    const double rangePercent = 0.4;
    const int topUpperRangeIndex = 0;
    const int bottomUpperRangeIndex = static_cast<int>(leftColumnBW.rows * rangePercent);
    const int topLowerRangeIndex = leftColumnBW.rows - bottomUpperRangeIndex;
    const int bottomLowerRangeIndex = leftColumnBW.rows;

    // Find means ans standard deviations
    cv::Scalar topMeanLeft, topStdvLeft, bottomMeanLeft, bottomStdvLeft;
    cv::meanStdDev(leftColumnBW.rowRange(topUpperRangeIndex, topLowerRangeIndex), topMeanLeft, topStdvLeft);
    cv::meanStdDev(leftColumnBW.rowRange(topLowerRangeIndex, bottomLowerRangeIndex), bottomMeanLeft, bottomStdvLeft);

    cv::Scalar topMeanRight, topStdvRight, bottomMeanRight, bottomStdvRight;
    cv::meanStdDev(leftColumnBW.rowRange(topUpperRangeIndex, topLowerRangeIndex), topMeanRight, topStdvRight);
    cv::meanStdDev(leftColumnBW.rowRange(topLowerRangeIndex, bottomLowerRangeIndex), bottomMeanRight, bottomStdvRight);

    // DECIDE if these fetures tell us that the card is rightside up or not
    // const double topDelta = topMeanLeft[0] - topStdvLeft[0];
    // const double bottomDelta = bottomMeanLeft[0] - bottomStdvLeft[0];
    const double topDelta = ((topMeanLeft[0] + topMeanRight[0]) / 2) - topStdvLeft[0] - topStdvRight[0];
    const double bottomDelta = ((bottomMeanLeft[0] + bottomMeanRight[0]) / 2) - bottomStdvLeft[0] - bottomStdvRight[0];

    //
    const bool isUpsideDown = topDelta > bottomDelta;
    // const bool isUpsideDown = topDelta < bottomDelta;
    if (isUpsideDown)
    {
        cv::flip(cardImage, cardImage, -1);
    }
}
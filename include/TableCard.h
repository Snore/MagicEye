#ifndef TABLE_CARD_H
#define TABLE_CARD_H

#include "MagicCard.h"
#include <chrono>
#include <memory>
#include <opencv2/core/core.hpp>

class TableCard
{
  public:
    enum VisibilityState
    {
        Visible,
        PartialBlocked,
        PartialBlockedUnidentified,
        Missing // might not need this one;  just make a vector for all TableCards that do not match the new list.
    };

    TableCard();
    TableCard(const cv::RotatedRect boundingBox, const cv::Mat &cardImage, const VisibilityState vstate);
    TableCard(const TableCard &obj);
    ~TableCard();

    TableCard &operator=(const TableCard &rhs) = default;

    cv::Rect getBoundingRect() const;
    cv::RotatedRect getMinimumBoundingRect() const;
    MagicCard *getMagicCard() const;
    bool isPointInside(cv::Point point) const;
    VisibilityState getCardVisibility() const;
    CardDetails::FrameColor getCardFrameColor() const;
    void setCardFrameColor(const CardDetails::FrameColor fcolor);
    bool isProbablySameTableCard(const TableCard &other) const;
    double distanceFrom(const TableCard &other) const;
    void setToAssumedCard(const TableCard &isProbablyThis);
    [[deprecated("Not used.")]] void setToAssumedBoundingBox(const TableCard &isProbablyHere);
    bool checkIfXSecondsSinceLastReference(const double seconds) const;
    void resetTimedReferenceCheck();
    void expireTimedReferenceCheck();

    // temp
    bool hasBeenIdentified;

  private:
    std::shared_ptr<MagicCard> _assumedCard_ptr;
    cv::RotatedRect _boundingBoxInScene;
    CardDetails::FrameColor _cardFrameColor;
    VisibilityState _visibilityState;
    std::chrono::time_point<std::chrono::system_clock> _lastReferenced;
    bool _forceExpire;

    // functions
    void makeRightsideUp(cv::Mat &cardImage) const;
};

#endif // TABLE_CARD_H

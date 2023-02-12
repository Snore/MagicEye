#ifndef CARD_FINDER_H
#define CARD_FINDER_H

#include "CardDatabase.h"
#include "TableCard.h"
#include <opencv2/core/core.hpp>
#include <vector>

class CardFinder
{
  public:
    CardFinder(CardDatabase *const cdb);
    ~CardFinder();

    std::vector<TableCard> *findAllCards(cv::Mat &scene);
    void discernPartialCards(); /// TODO: could live in board manager class // needs to be called after frame colors are
                                /// assigned.  more reason for new class
    void reevaluateMemoryCards();

  private:
    CardDatabase *_cdb_ptr;
    std::vector<TableCard> _foundCards; /// TODO: could live in board manager class
    std::vector<TableCard> _cardMemory; /// TODO: could live in board manager class
    cv::Rect _biggestBox;               // usually a player's hand

    // constants
    static const int MIN_AREA_ELIMINATION_THRESHOLD = 100;

    // functions for finding individual cards
    cv::Mat findPlayField(const cv::Mat &scene) const;
    void identifyCardsInRegion(const cv::Mat &ROI,
                               const cv::Point ROIOffset,
                               const cv::Rect &tableBB,
                               std::vector<TableCard> &runningList) const;
    TableCard extractCardImage(const cv::Mat &fromScene,
                               const cv::RotatedRect boundingRect,
                               const cv::Point worldPosition,
                               const cv::Rect &tableBB,
                               const bool isFullCard) const;
    bool isContourConsumedByAnother(const std::vector<cv::Point> contour,
                                    const std::vector<cv::Point2f> consumedBy) const;
    void outlineRotatedRectangle(cv::Mat &scene, const cv::RotatedRect RR, const cv::Scalar &color) const;
    void blackoutRotatedRectangle(cv::Mat &scene, const cv::RotatedRect RR) const;

    // Trying things out part 1
    void rememberNewCards();
    void forgetOldCards();
    bool recallCard(const TableCard &cardToRecall);
    void evaluateCardsColors(std::vector<TableCard> &cards);
};

#endif // CARD_FINDER_H

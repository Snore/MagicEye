#ifndef MAGIC_CARD_H
#define MAGIC_CARD_H

#include <opencv2/core/core.hpp>
#include <string>

#include "CardDetails.h"
#include "json.h"

// testing
#include "DeltaEGrid.h"
#include "HistoGrid.h"

namespace CardMeasurements
{
constexpr int HueBins = 16;
constexpr int SaturationBins = 4;
constexpr int ValueBins = 3;
} // namespace CardMeasurements

class MagicCard
{
  public:
    MagicCard(const Json::Value json_card, const CardDetails::CardSet set);
    MagicCard(cv::Mat &cardImage);
    MagicCard(const std::string imagePath);
    MagicCard(const std::string name,
              const std::string imagePath,
              const CardDetails::CardSet set,
              const CardDetails::Type type);
    ~MagicCard() = default;

    cv::Mat loadCardImage() const;
    CardDetails::FrameColor getFrameColor() const;
    CardDetails::CardSet getCardSet() const;
    void deepAnalyze(); // TODO make this private?

    // discern frame histogram color
    cv::Mat getFrameHistogram() const;
    cv::Scalar getFrameMeanColor_CIELAB() const;
    cv::Scalar getFrameMeanColor_BGR() const;
    void setCardFrameColor(const CardDetails::FrameColor fcolor);

    std::string toString() const;
    static double compareLikeness(MagicCard const *const cardOne, MagicCard const *const cardTwo);
    static double compareDeltaEGrid(MagicCard const *const cardOne, MagicCard const *const cardTwo);
    static double compareHSVGrid(MagicCard const *const cardOne, MagicCard const *const cardTwo);
    static std::string FrameColorToString(const CardDetails::FrameColor fcolor);

    // DEBUG
    cv::Mat getBorderlessCardImage() const; // DEBUG

    bool operator==(const MagicCard &other) const;

  private:
    // card properties
    std::string _name;
    std::string _imageFilePath;
    CardDetails::CardSet _set;
    CardDetails::Type _type;
    cv::Mat _ROIImage;

    // Card image properties
    CardDetails::FrameColor _fcolor;
    double _perceivedTextVerbosity;        // analyzeTextBox
    HistoGrid _artHistoGrid;               // analyzeArt
    DeltaEGrid _artDeltaEGrid;             // analyzeArt
    std::vector<cv::Point> _featurePoints; // analyzeFeatures
    cv::Rect _artROI;
    cv::Rect _textROI;
    cv::Rect _borderlessROI;

    // functions
    void locateCardRegions(); // Needs to be called before other functions
    cv::Rect findBorderlessROI(cv::Mat &wholeCardImage) const;
    cv::Mat getFrameOnlyMask() const;
    // cv::Mat getBorderlessCardImage() const;

    void analyzeTextBox();
    void analyzeArt();
    void analyzeFeatures();
};

#endif // MAGIC_CARD_H

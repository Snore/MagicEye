#pragma once
#include <opencv2/core/core.hpp>
#include <opencv2\highgui.hpp>
#include <vector>

class MagicEyeGUI
{
public:
	MagicEyeGUI(const std::string windowName);
	~MagicEyeGUI();

	void drawWindow() const;

	void setMainDisplayFrame(const cv::Mat & image);
	void setSecondaryDisplayFrame(const cv::Mat & image);
	void setResultText(const std::string & textToDisplay);
	void setResultCardImage(const cv::Mat & image, const int position);

private:
	static const int WIN_HEIGHT = 1010; //1000 might be more optimal, but will have to resize RESULT_HEIGHT and RESULT_WIDTH
	static const int WIN_WIDTH = 1920;
	static const int NUM_OF_RESULTS = 7;
	static const int RESULT_WIDTH = 142;
	static const int RESULT_HEIGHT = 200;
	static const int TEXT_HEIGHT = 40;
	cv::Mat _window;

	// ROI boundss of the gui
	cv::Rect _mainDisplayROIBounds;
	cv::Rect _secondaryDisplayROIBounds;
	cv::Rect _resultsTextROIBounds;
	cv::Rect _resultsCardsLayoutBounds;
	std::vector<cv::Rect> _resultCardsROIBounds;

	// ROI's of the gui
	cv::Mat _mainDisplayROI;
	cv::Mat _secondaryDisplayROI;
	cv::Mat _resultsTextROI;
	std::vector<cv::Mat> _resultCardsROI;

	cv::Point _resultsTextPoint;
	std::string _resultsText;

	std::string _windowName;

	// functions
	void initializeROIs();
	cv::Mat scaleImageToDimensions(const cv::Mat & image, const cv::Size fitToSize) const;
};


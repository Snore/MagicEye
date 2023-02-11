#include "MagicEyeGUI.h"
#include <opencv2/imgproc.hpp>



MagicEyeGUI::MagicEyeGUI(const std::string windowName)
{
	_windowName = windowName;
	_window = cv::Mat::zeros(WIN_HEIGHT, WIN_WIDTH, CV_8UC3);
	initializeROIs();

	cv::namedWindow(_windowName, cv::WINDOW_AUTOSIZE);
}


MagicEyeGUI::~MagicEyeGUI()
{
}


void MagicEyeGUI::drawWindow() const
{
	cv::imshow(_windowName, _window);
}


void MagicEyeGUI::setMainDisplayFrame(const cv::Mat & image)
{
	if (_mainDisplayROI.size() != image.size())
	{
		// not worth figuring out until we get the video
		// if image ratio == _mainDisplayROI ratio
		//then

		// Assuming we always shrink
		cv::resize(image, _mainDisplayROI, _mainDisplayROI.size(), 0.0, 0.0, cv::INTER_AREA);

		// else
		// could figure it out each time, or can assume dest is always wider than source image
	}
	else
	{
		_mainDisplayROI = image.clone();
	}
}


void MagicEyeGUI::setSecondaryDisplayFrame(const cv::Mat & image)
{
	if (_secondaryDisplayROI.size() != image.size())
	{
		// Assuming we always Grow
		cv::resize(image, _secondaryDisplayROI, _secondaryDisplayROI.size(), 0.0, 0.0, cv::INTER_LINEAR);
	}
	else
	{
		_secondaryDisplayROI = image.clone();
	}
}


void MagicEyeGUI::setResultCardImage(const cv::Mat & image, const int position)
{
	assert(position >= 0);
	assert(position < NUM_OF_RESULTS);

	if (_resultCardsROI[position].size() != image.size())
	{
		// Assuming we always shrink
		cv::resize(image, _resultCardsROI[position], _resultCardsROI[position].size(), 0.0, 0.0, cv::INTER_AREA);
	}
	else
	{
		_resultCardsROI[position] = image.clone();
	}
}


int MagicEyeGUI::returnResultSelectionIndex(const cv::Point & clickCoord) const
{
	int currentIndex = 0;
	for (auto resultsBB_itr = _resultCardsROIBounds.cbegin(); resultsBB_itr != _resultCardsROIBounds.cend(); ++resultsBB_itr)
	{
		if (resultsBB_itr->contains(clickCoord))
		{
			return currentIndex;
		}

		++currentIndex;
	}

	return -1;
}


// cv::Mat MagicEyeGUI::scaleImageToDimensions(const cv::Mat & image, const cv::Size fitToSize) const
// {
// 	// TODO: delete?
// 	return cv::Mat::zeros(2, 2, CV_8UC3);
// }


void MagicEyeGUI::initializeROIs()
{
	// Main display
	_mainDisplayROIBounds = cv::Rect(0, 0, 1440, 810);
	_mainDisplayROI = _window(_mainDisplayROIBounds);

	// Secondary display
	_secondaryDisplayROIBounds = cv::Rect(_mainDisplayROIBounds.width, 
										  0,
										  (WIN_WIDTH - _mainDisplayROIBounds.width), 
										  _mainDisplayROIBounds.height);
	_secondaryDisplayROI = _window(_secondaryDisplayROIBounds);

	// Results text display
	_resultsTextROIBounds = cv::Rect(0, _mainDisplayROIBounds.height - TEXT_HEIGHT, _mainDisplayROIBounds.width, TEXT_HEIGHT);

	// Results images displays
	_resultsCardsLayoutBounds = cv::Rect(0,
										_mainDisplayROIBounds.height,
										_mainDisplayROIBounds.width,
										(WIN_HEIGHT - _mainDisplayROIBounds.height));

	const int emptyWidth = _resultsCardsLayoutBounds.width - (RESULT_WIDTH * NUM_OF_RESULTS);
	const int emptyHeight = _resultsCardsLayoutBounds.height - RESULT_HEIGHT;
	const int widthSpacing = (emptyWidth / NUM_OF_RESULTS) / 2;
	const int heightSpacing = emptyHeight / 2;
	assert(emptyWidth >= 0);
	assert(emptyHeight >= 0);

	int x = _resultsCardsLayoutBounds.x;
	int y = _resultsCardsLayoutBounds.y + heightSpacing;
	for (int i = 0; i < NUM_OF_RESULTS; ++i)
	{
		x += widthSpacing;

		_resultCardsROIBounds.emplace_back(cv::Rect(x, y, RESULT_WIDTH, RESULT_HEIGHT));
		_resultCardsROI.emplace_back(_window(_resultCardsROIBounds.back()));

		x += RESULT_WIDTH + widthSpacing;
	}
}

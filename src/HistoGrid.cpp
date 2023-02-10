#include "pch.h"
#include "HistoGrid.h"
#include <opencv2/imgproc.hpp>


HistoGrid::HistoGrid()
{
	// Delete later?
}


HistoGrid::HistoGrid(const cv::Mat image)
{
	const double ratio = 1.0 / 3.0;
	const int ratioLengthSteps = static_cast<int>(image.cols * ratio);
	const int ratioHeightSteps = static_cast<int>(image.rows * ratio);
	const cv::Size roiDimensions(ratioLengthSteps, ratioHeightSteps);

	// make nine ROIs for all histograms
	for (int lengthStep = 0; lengthStep < CELL_LENGTH; lengthStep++)
	{
		for (int heightStep = 0; heightStep < CELL_LENGTH; heightStep++)
		{
			const cv::Rect miniROI(cv::Point(ratioLengthSteps * lengthStep,
											 ratioHeightSteps * heightStep),
											 roiDimensions);
			_histGridCells.push_back(calcSubHistogram(image(miniROI)));
		}
	}
}


HistoGrid::~HistoGrid()
{
	_histGridCells.clear();
}


cv::Mat HistoGrid::visualRepresentation() const
{
	return cv::Mat();
}


double HistoGrid::compare(const HistoGrid & gridOne, const HistoGrid & gridTwo)
{
	assert(gridOne._histGridCells.size() == gridTwo._histGridCells.size());

	//These grid cells should be in the same order relative to each other
	double distance = 0.0;
	const size_t numOfCells = gridOne._histGridCells.size();
	for (size_t index = 0; index < numOfCells; ++index)
	{
		distance += cv::compareHist(gridOne._histGridCells[index], gridTwo._histGridCells[index], cv::HISTCMP_CHISQR);
	}

	return distance;
}


cv::Mat HistoGrid::calcSubHistogram(const cv::Mat & image)
{
	cv::Mat imageHLS;
	cv::cvtColor(image, imageHLS, cv::COLOR_BGR2HLS);

	const int histSize[] = { HUE_BINS, SAT_BINS };
	const float hranges[] = { 0, 180 };
	const float sranges[] = { 0, 256 };
	const float * ranges[] = { hranges, sranges };
	cv::Mat histogram;
	int channels[] = { 0, 1 };

	cv::calcHist(&imageHLS, 1, channels, cv::Mat(), histogram, 2, histSize, ranges, true, false);
	cv::normalize(histogram, histogram);
	return histogram;
}
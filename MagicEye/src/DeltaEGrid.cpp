#include "pch.h"
#include "DeltaEGrid.h"
#include <opencv2\imgproc.hpp>


DeltaEGrid::DeltaEGrid()
{
}


DeltaEGrid::DeltaEGrid(const cv::Mat image)
{
	const double ratio = 1.0 / 3.0;
	const int ratioLengthSteps = static_cast<int>(image.cols * ratio);
	const int ratioHeightSteps = static_cast<int>(image.rows * ratio);
	const cv::Size roiDimensions(ratioLengthSteps, ratioHeightSteps);

	cv::Mat imageLAB;
	cv::cvtColor(image, imageLAB, CV_BGR2Lab);

	// Get image average
	_averageImageColor = cv::mean(imageLAB); /// Does this work in LAB color space?  Or should it be done in BGR?

	// make nine ROIs for all image color euclidian distances
	for (int lengthStep = 0; lengthStep < CELL_LENGTH; lengthStep++)
	{
		for (int heightStep = 0; heightStep < CELL_LENGTH; heightStep++)
		{
			const cv::Rect miniROI(cv::Point(ratioLengthSteps * lengthStep,
											 ratioHeightSteps * heightStep),
											 roiDimensions);

			_cellColorDistances.push_back(calcDeltaE(cv::mean(imageLAB(miniROI)), _averageImageColor));
		}
	}
}


DeltaEGrid::~DeltaEGrid()
{
	_cellColorDistances.clear();
}


cv::Mat DeltaEGrid::visualRepresentation() const
{
	return cv::Mat();
}


double DeltaEGrid::compare(const DeltaEGrid & gridOne, const DeltaEGrid & gridTwo)
{
	assert(gridOne._cellColorDistances.size() == gridTwo._cellColorDistances.size());

	double distance = 0.0;
	const size_t numOfCells = gridOne._cellColorDistances.size();
	for (size_t index = 0; index < numOfCells; ++index)
	{
		distance += cv::abs(gridOne._cellColorDistances[index] - gridTwo._cellColorDistances[index]);
	}

	return distance;
}


double DeltaEGrid::calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const
{
	// CIE76 - not great, but good enough
	cv::Scalar lineSegment = colorPointOne - colorPointTwo;
	lineSegment[0] = cv::pow(lineSegment[0], 2);
	lineSegment[1] = cv::pow(lineSegment[1], 2);
	lineSegment[2] = cv::pow(lineSegment[2], 2);
	const double euclidianDistance = cv::sqrt(lineSegment[0] + lineSegment[1] + lineSegment[2]);

	return euclidianDistance;
}
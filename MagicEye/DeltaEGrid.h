#pragma once
#include <opencv2/core/core.hpp>
#include <vector>


class DeltaEGrid
{
public:
	DeltaEGrid();
	DeltaEGrid(const cv::Mat image);
	~DeltaEGrid();

	cv::Mat visualRepresentation() const;
	static double compare(const DeltaEGrid & gridOne, const DeltaEGrid & gridTwo);

private:
	static const int CELL_LENGTH = 3;

	//variables
	cv::Scalar _averageImageColor;
	std::vector<double> _cellColorDistances;

	//functions
	double calcDeltaE(const cv::Scalar colorPointOne, const cv::Scalar colorPointTwo) const;
};


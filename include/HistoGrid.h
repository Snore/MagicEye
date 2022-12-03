#ifndef HISTO_GRID_H
#define HISTO_GRID_H

#include <opencv2/core/core.hpp>
#include <vector>

class HistoGrid
{
public:
	HistoGrid();
	HistoGrid(const cv::Mat image);
	~HistoGrid();

	cv::Mat visualRepresentation() const;
	static double compare(const HistoGrid & gridOne, const HistoGrid & gridTwo);

private:
	const static int CELL_LENGTH = 3;
	const static int HUE_BINS = 18;
	const static int SAT_BINS = 3;
	const static int LUM_BINS = 3;

	// variables
	std::vector<cv::Mat> _histGridCells;

	// functions
	cv::Mat calcSubHistogram(cv::Mat & image);
};

#endif // HISTO_GRID_H

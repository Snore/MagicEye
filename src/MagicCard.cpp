#include "pch.h"
#include "MagicCard.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <iostream>
#include <climits>
#include <sstream>


MagicCard::MagicCard(const Json::Value json_card, const CardDetails::CardSet set)
{
	// get card name, image path, and set
	_name = json_card["name"].asString();
	_imageFilePath = ASSETS_PATH + "\\AllSets\\" + CardDetails::CardSet_name.at(set) + "\\" + json_card["imageName"].asString() + ".jpg";
	_set = set;

	// discern if card is creature or not
	Json::Value card_types = json_card["types"];
	Json::ValueIterator type_itr = std::find(card_types.begin(), card_types.end(), "Creature");
	if (type_itr == card_types.end())
	{
		_type = CardDetails::Instant;
	}
	else
	{
		_type = CardDetails::Creature;
	}

	// set card border color
	_fcolor = CardDetails::Unsure;

	// Analyze card
	locateCardRegions();
}


MagicCard::MagicCard(cv::Mat & cardImage)
	:
	_name(""),
	_imageFilePath(""),
	_set(CardDetails::ALA),
	_type(CardDetails::Unidentified),
	_fcolor(CardDetails::Unsure),
	_perceivedTextVerbosity(0.0),
	_ROIImage(cardImage.clone())
{
	// Analyze card
	locateCardRegions();
}


MagicCard::MagicCard(const std::string imagePath)
	:
	_name(""),
	_imageFilePath(imagePath),
	_set(CardDetails::ALA),
	_type(CardDetails::Unidentified),
	_fcolor(CardDetails::Unsure),
	_perceivedTextVerbosity(0.0)
{
	// Analyze card
	locateCardRegions();
}


MagicCard::MagicCard(const std::string name, const std::string imagePath, const CardDetails::CardSet set, const CardDetails::Type type)
	:
	_name(name),
	_imageFilePath(imagePath),
	_set(set),
	_type(type),
	_fcolor(CardDetails::Unsure)
{
	// Analyze card
	locateCardRegions();
}


MagicCard::~MagicCard()
{
}


#ifdef UWP_VERSION
#include <ppltasks.h>
#include <codecvt>
#include <sstream>

cv::Mat readImageFromImageFile(std::string pathFromInstallDirectory /*Windows::Storage::StorageFile^ imageFile*/)
{
	/*
	cv::Mat decodedImage;

	std::wstring_convert<std::codecvt_utf8_utf16<wchar_t>> converter;
	Platform::String^ relativePath = ref new Platform::String(converter.from_bytes(pathFromInstallDirectory).c_str());
	Platform::String^ imageFilePath = Windows::ApplicationModel::Package::Current->InstalledLocation->Path + relativePath;

	Concurrency::create_task(Windows::Storage::StorageFile::GetFileFromPathAsync(imageFilePath)).then([](Windows::Storage::StorageFile^ imageFile) 
	{
		return Windows::Storage::FileIO::ReadBufferAsync(imageFile);
	}).then([&decodedImage](Concurrency::task<Windows::Storage::Streams::IBuffer^> task) //[this, imageFile]
	{
		try
		{
			Windows::Storage::Streams::IBuffer^ buffer = task.get();
			Windows::Storage::Streams::DataReader^ dataReader = Windows::Storage::Streams::DataReader::FromBuffer(buffer);
				
			std::vector<unsigned char> fileContent(dataReader->UnconsumedBufferLength);
			dataReader->ReadBytes(Platform::ArrayReference<unsigned char>(fileContent.data(), fileContent.size()));
			delete dataReader;

			decodedImage = cv::imdecode(fileContent, CV_LOAD_IMAGE_COLOR);
		}
		catch (Platform::COMException^ ex)
		{
			assert(false);
		}
	}).wait();
	*/

	Platform::String^ localfolder = Windows::ApplicationModel::Package::Current->InstalledLocation->Path;
	std::wstring folderNameW(localfolder->Begin());
	std::string folderNameA(folderNameW.begin(), folderNameW.end());
	std::string path = folderNameA + pathFromInstallDirectory;
	cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_COLOR);

	return image;
}
#endif // UWP_VERSION

cv::Mat MagicCard::loadCardImage() const
{
	if (_imageFilePath.length() > 0)
	{
#ifdef UWP_VERSION
		//cv::Mat cardImage = readImageFromImageFile(getFileFromPath(_imageFilePath));
		cv::Mat cardImage = readImageFromImageFile(_imageFilePath);
#else
		cv::Mat cardImage = cv::imread(_imageFilePath, cv::IMREAD_COLOR);
#endif // UWP_VERSION
		if (!cardImage.data)
		{
			std::cerr << "Failed to load image for " << _name << " located at: " << _imageFilePath << "\n";
			assert(false);
		}

		return cardImage;
	}
	else
	{
		// If not a card for the HDD, return local image
		return _ROIImage;
	}
}


CardDetails::FrameColor MagicCard::getFrameColor() const
{
	return _fcolor;
}


CardDetails::CardSet MagicCard::getCardSet() const
{
	return _set;
}


void MagicCard::deepAnalyze()
{
	analyzeTextBox();
	analyzeArt();
	analyzeFeatures();
}


void MagicCard::locateCardRegions()
{
	// find the outer boarder of the card
	cv::Mat card = loadCardImage();
	//cv::imshow("Card", card);
	_borderlessROI = findBorderlessROI(card);
	card = card(_borderlessROI);
	//cv::imshow("Now borderless", card);

	// discern the card art from the card's inner border
	const int cardHeight = _borderlessROI.height;
	const int cardWidth = _borderlessROI.width;
	const double topRatio = 60.0 / 660.0;
	const double leftRatio = 20.0 / 440.0;
	const double bottomRatio = 290.0 / 660.0;
	const double cardArtWidthRatio = 420.0 / 460.0;
	const double cardArtHeightRatio = 310.0 / 660.0;
	cv::Point ulArtPoint(static_cast<int>(cardWidth * leftRatio), static_cast<int>(cardHeight * topRatio));
	cv::Point brArtPoint(ulArtPoint.x + static_cast<int>(cardWidth*cardArtWidthRatio), ulArtPoint.y + static_cast<int>(cardHeight * cardArtHeightRatio));
	_artROI = cv::Rect(ulArtPoint, brArtPoint);

	// MORE TESTING
	//cv::imshow("Just Art", histoCard(_artROI));

	// discern the card text box from the card's inner border
	const double textTopRatio = 0.6378787878787879;
	const double textHeightRatio = 0.2969;
	_textROI = _artROI;
	_textROI.y = static_cast<int>(_borderlessROI.height * textTopRatio);
	_textROI.height = static_cast<int>(_borderlessROI.height * textHeightRatio);

	// MORE TESTING
	//cv::imshow("Just Text", histoCard(_textROI));

	// MORE TESTING
	//cv::Mat test = card(_borderlessROI).clone();
	//test.setTo(0, getFrameOnlyMask());
	//cv::imshow("Just frame", test);
}


cv::Mat MagicCard::getFrameHistogram() const
{
	cv::Mat cardHLS, imageGray, imageColor;
	imageColor = getBorderlessCardImage();
	cv::cvtColor(imageColor, imageGray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageColor, cardHLS, cv::COLOR_BGR2HLS);

	//int histSize[] = { CardMeasurements::HueBins, CardMeasurements::SaturationBins };
	int histSize[] = { CardMeasurements::HueBins, CardMeasurements::SaturationBins, CardMeasurements::ValueBins };
	float hranges[] = { 0, 180 };
	float sranges[] = { 0, 256 };
	float vranges[] = { 0, 256 };
	//const float * ranges[] = { hranges, sranges };
	const float * ranges[] = { hranges, sranges, vranges };
	cv::Mat frameHist;
	//int channels[] = { 0, 1 };
	int channels[] = { 0, 1, 2};

	// Filter out partial card shadows
	// the filter should set all the pixels in the shadow to exactly 0; should be rare that this pixel color exists in real life
	cv::Mat mask;
	cv::Mat partialShadowMask;
	cv::threshold(imageGray, partialShadowMask, 0.0, 255, cv::THRESH_BINARY);
	mask = cv::Mat(partialShadowMask.size(), CV_8UC1);
	cv::bitwise_and(getFrameOnlyMask(), partialShadowMask, mask);

	//cv::calcHist(&cardHLS, 1, channels, getFrameOnlyMask(), frameHist, 2, histSize, ranges, true, false);
	cv::calcHist(&cardHLS, 1, channels, mask, frameHist, 3, histSize, ranges, true, false);

	// normalize for all resolutions
	cv::normalize(frameHist, frameHist);

	return frameHist;
}


cv::Scalar MagicCard::getFrameMeanColor_CIELAB() const
{
	cv::Mat imageLAB, imageGray, imageColor;
	imageColor = getBorderlessCardImage();
	cv::cvtColor(imageColor, imageGray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(imageColor, imageLAB, cv::COLOR_BGR2Lab);
	///cv::cvtColor(getBorderlessCardImage(), imageLAB, CV_BGR2Luv);

	// Filter out partial card shadows
	// the filter should set all the pixels in the shadow to exactly 0; should be rare that this pixel color exists in real life
	cv::Mat mask;
	cv::Mat partialShadowMask;
	cv::threshold(imageGray, partialShadowMask, 0.0, 255, cv::THRESH_BINARY);
	mask = cv::Mat(partialShadowMask.size(), CV_8UC1);
	cv::bitwise_and(getFrameOnlyMask(), partialShadowMask, mask);

	return cv::mean(imageLAB, mask);
}


cv::Scalar MagicCard::getFrameMeanColor_BGR() const
{
	return cv::mean(getBorderlessCardImage(), getFrameOnlyMask());
}


void MagicCard::setCardFrameColor(const CardDetails::FrameColor fcolor)
{
	_fcolor = fcolor;
}


std::string MagicCard::toString() const
{
	std::stringstream description;

	//print card name
	description << "Name: " << _name << "\n";

	// print frame color
	description << "Frame color: ";
	description << FrameColorToString(_fcolor) << "\n";

	// print text box verbosity
	description << "Perceived text amout: ";
	description << _perceivedTextVerbosity << std::endl;

	return description.str();
}


double MagicCard::compareLikeness(MagicCard const * const cardOne, MagicCard const * const cardTwo)
{
	const double deltaE_comp = DeltaEGrid::compare(cardOne->_artDeltaEGrid, cardTwo->_artDeltaEGrid);
	const double histGrid_comp = HistoGrid::compare(cardOne->_artHistoGrid, cardTwo->_artHistoGrid);

	return deltaE_comp + histGrid_comp;
}


double MagicCard::compareDeltaEGrid(MagicCard const * const cardOne, MagicCard const * const cardTwo)
{
	return DeltaEGrid::compare(cardOne->_artDeltaEGrid, cardTwo->_artDeltaEGrid);
}


double MagicCard::compareHSVGrid(MagicCard const * const cardOne, MagicCard const * const cardTwo)
{
	return HistoGrid::compare(cardOne->_artHistoGrid, cardTwo->_artHistoGrid);
}


std::string MagicCard::FrameColorToString(const CardDetails::FrameColor fcolor)
{
	switch (fcolor)
	{
	case CardDetails::Red:
		return "Red";
		break;
	case CardDetails::Blue:
		return "Blue";
		break;
	case CardDetails::White:
		return "White";
		break;
	case CardDetails::Black:
		return "Black";
		break;
	case CardDetails::Green:
		return "Green";
		break;
	case CardDetails::Multi:
		return "Multi";
		break;
	case CardDetails::Colorless:
		return "Colorless";
		break;
	case CardDetails::Land_Color:
		return "Land";
		break;
	default:
		return "Who Knows?";
		break;
	}
}


cv::Rect MagicCard::findBorderlessROI(cv::Mat & wholeCardImage) const
{
	cv::Mat card = wholeCardImage.clone();
	cv::cvtColor(card, card, cv::COLOR_BGR2GRAY);
	cv::threshold(card, card, 40, UCHAR_MAX, cv::THRESH_BINARY);
	//cv::imshow("thresh", card);  // DEBUG

	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
	cv::findContours(card, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

	// Draw contours // DEBUG
	/*
	cv::Mat contourCanvas = cv::Mat::zeros(card.size(), card.type());
	for (int i = 0; i < contours.size(); ++i)
	{
		cv::drawContours(contourCanvas, contours, i, CV_RGB(255, 255, 255));
	}
	cv::imshow("contours", contourCanvas);
	*/

	// Merge all contours points into one
	std::vector<cv::Point> allPoints;
	for (int outer = 0; outer < contours.size(); ++outer)
	{
		for (std::vector<cv::Point>::iterator pointItr = contours[outer].begin(); pointItr != contours[outer].end(); ++pointItr)
		{
			allPoints.push_back(*pointItr);
		}
	}

	return cv::boundingRect(allPoints);
}


cv::Mat MagicCard::getFrameOnlyMask() const
{
	// Inverse mask for testing
	//cv::Mat frameOnlyMask = cv::Mat::zeros(_borderlessROI.size(), CV_8UC1);
	//cv::rectangle(frameOnlyMask, _artROI, CV_RGB(1, 1, 1), CV_FILLED);
	//cv::rectangle(frameOnlyMask, _textROI, CV_RGB(1, 1, 1), CV_FILLED);

	cv::Mat frameOnlyMask = cv::Mat::ones(_borderlessROI.size(), CV_8UC1);
	cv::rectangle(frameOnlyMask, _artROI, CV_RGB(0, 0, 0), cv::FILLED);
	cv::rectangle(frameOnlyMask, _textROI, CV_RGB(0, 0, 0), cv::FILLED);
	return frameOnlyMask;
}


cv::Mat MagicCard::getBorderlessCardImage() const
{
	cv::Mat cardImage = loadCardImage();
	return cardImage(_borderlessROI);
}


void MagicCard::analyzeTextBox()
{
	// START FROM HERE.
	_perceivedTextVerbosity = 0.0;
}


void MagicCard::analyzeArt()
{
	// Find HSL histogram of entire art
	cv::Mat cardArt = getBorderlessCardImage()(_artROI);
	_artHistoGrid = HistoGrid(cardArt);
	_artDeltaEGrid = DeltaEGrid(cardArt);
}


void MagicCard::analyzeFeatures()
{
	_featurePoints.clear();
	// do more
}


bool MagicCard::operator==(const MagicCard& other) const
{
	// Never have I ever seen two cards with the same name in the same set
	return ((this->_name == other._name) && (this->_set == other._set));
}
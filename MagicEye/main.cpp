#include <opencv2/core/core.hpp>
#include <opencv2\highgui.hpp>
#include <opencv2\imgproc.hpp>

#include <iostream>
#include <chrono>
#include <fstream>

#include "CardFinder.h"
#include "CardDatabase.h"
#include "MagicEyeGUI.h"

// Images we use often in the GUI
const std::string MAIN_WINDOW_NAME = "Welcome to MagicEye";
const cv::Mat NoGoodResults = cv::imread("Assets\\AllSets\\NoShow.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat WhiteFrameImage = cv::imread("Assets\\AllSets\\ZEN\\plains4.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat BlackFrameImage = cv::imread("Assets\\AllSets\\ZEN\\swamp2.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat RedFrameImage = cv::imread("Assets\\AllSets\\ZEN\\mountain3.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat BlueFrameImage = cv::imread("Assets\\AllSets\\ZEN\\island3.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat GreenFrameImage = cv::imread("Assets\\AllSets\\ZEN\\forest1.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat ColorlessFrameImage = cv::imread("Assets\\AllSets\\DST\\darksteel citadel.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat MultiColorFrameImage = cv::imread("Assets\\AllSets\\AVR\\cavern of souls.jpg", CV_LOAD_IMAGE_COLOR);
const cv::Mat clearResults = cv::Mat::zeros(NoGoodResults.size(), CV_8UC3);

enum GUIState
{
	InFrameSelect,
	ResultSelect,
	FrameColorSelect
};

CardDatabase cdb;
CardFinder cardFinder(&cdb);
std::vector<TableCard>* foundCards;
MagicEyeGUI gui(MAIN_WINDOW_NAME);
GUIState guiState = GUIState::InFrameSelect;
TableCard LastCardClicked;  // For color training

// for metrics gathering
int fullCardsQueried = 0;
int cardsCorrect_FirstGuess = 0;
int cardsCorrect_TopSeven = 0;
int cardsIncorrect_WrongColor = 0;
int cardsIncorrect_RightColor = 0;
int partialCardsQueried = 0;
int cardsIncorrect_RightPartial = 0;
int cardsIncorrect_ConfusedPartial = 0;

//FOR TESTING ONLY
void drawHistogram(cv::MatND & histogram, const int sbins, const int hbins)
{
	// DEBUG ONLY
	double frameMaxVal = 0;
	cv::minMaxLoc(histogram, 0, &frameMaxVal, 0, 0);

	int scale = 10;
	cv::Mat histImg = cv::Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

	for (int h = 0; h < hbins; h++)
	{
		for (int s = 0; s < sbins; s++)
		{
			float binVal = histogram.at<float>(h, s);
			int intensity = cvRound(binVal * 255 / frameMaxVal);
			rectangle(histImg, cv::Point(h*scale, s*scale),
				cv::Point((h + 1)*scale - 1, (s + 1)*scale - 1),
				cv::Scalar::all(intensity),
				CV_FILLED);
		}
	}
	cv::imshow("H-S Histogram", histImg);
}


//FOR TESTING ONLY
void jumpXFrames(cv::VideoCapture & cap, const int framesToJump)
{
	for (int index = 0; index < framesToJump; ++index)
	{
		//throw away
		cap.grab();
	}

}

void compileMetrics()
{
	std::stringstream metrics;
	metrics << "Total number of queries on full cards: " << fullCardsQueried << std::endl;
	double totalFullPoints = cardsCorrect_FirstGuess + (cardsCorrect_TopSeven * 0.5);
	metrics << "Total full card score: " << totalFullPoints << std::endl;
	metrics << cardsCorrect_FirstGuess << " completely correct cards and " << cardsCorrect_TopSeven << " top seven correct cards.\n";

	// Dont forget to manually do card locator

	metrics << "Total number of queries on partial cards: " << partialCardsQueried << std::endl;
	double totalPartialPoints = cardsIncorrect_RightPartial + (cardsIncorrect_ConfusedPartial * 0.5);
	metrics << "Total partial card score: " << totalPartialPoints << std::endl;
	metrics << cardsIncorrect_RightPartial << " completely correct partial cards and " << cardsIncorrect_ConfusedPartial << " color confused cards.\n";
	// Don't forget to manually do partial card locator

	std::ofstream myMetrics("metrics.txt", std::ios::trunc);
	if (myMetrics.is_open())
	{
		//
		myMetrics << metrics.str();
		myMetrics.close();
	}
	else
	{
		assert(false);
	}
}

// Mouse click controls for the GUI
void mouseEventCallback(int mouseEvent, int x, int y, int flags, void* userData)
{
	if (mouseEvent == CV_EVENT_LBUTTONDOWN)
	{
		std::cout << "Mouse pointer X: " << x << " Y: " << y << "\n";

		cv::Point mousePoint(x, y);
		int selectedResultIndex = -1;

		switch (guiState)
		{
		case InFrameSelect:
			// convert mousePoint from gui coord to world coord : because of resizing
			mousePoint *= 0.89;

			for (auto tableCardItr = foundCards->cbegin(); tableCardItr != foundCards->cend(); ++tableCardItr)
			{
				// find which rectangle contains the mouse pointer if any
				if (tableCardItr->getBoundingRect().contains(mousePoint))
				{
					std::chrono::time_point<std::chrono::system_clock> start, end;
					start = std::chrono::system_clock::now();

					LastCardClicked = *tableCardItr;
					std::vector<MagicCard*> matches = cdb.returnMostAlike(tableCardItr->getMagicCard(), 6);

					end = std::chrono::system_clock::now();
					std::chrono::duration<double> elapsed_seconds = end - start;
					gui.setSecondaryDisplayFrame(tableCardItr->getMagicCard()->loadCardImage());
					gui.setResultCardImage(matches[0]->loadCardImage(), 0);
					gui.setResultCardImage(matches[1]->loadCardImage(), 1);
					gui.setResultCardImage(matches[2]->loadCardImage(), 2);
					gui.setResultCardImage(matches[3]->loadCardImage(), 3);
					gui.setResultCardImage(matches[4]->loadCardImage(), 4);
					gui.setResultCardImage(matches[5]->loadCardImage(), 5);
					gui.setResultCardImage(NoGoodResults, 6);
					gui.drawWindow();
					std::cout << "Done finding best match.\nElapsed time: " << elapsed_seconds.count() << " seconds\n";

					if (tableCardItr->getCardVisibility() == TableCard::Visible)
					{
						fullCardsQueried++;
					}
					else
					{
						partialCardsQueried++;
					}

					guiState = GUIState::ResultSelect;
					break;
				}
			}
			break;

		case ResultSelect:
			selectedResultIndex = gui.returnResultSelectionIndex(mousePoint);

			if (selectedResultIndex == -1)
			{
				// User picked nothing
				break;
			}
			else if (selectedResultIndex >= 0 && selectedResultIndex < 6)
			{
				// User picked one of the 6 possible entries
				if (selectedResultIndex == 0)
				{
					cardsCorrect_FirstGuess++;
				}
				else
				{
					cardsCorrect_TopSeven++;
				}
				// TODO: add logic

				gui.setResultCardImage(clearResults, 0);
				gui.setResultCardImage(clearResults, 1);
				gui.setResultCardImage(clearResults, 2);
				gui.setResultCardImage(clearResults, 3);
				gui.setResultCardImage(clearResults, 4);
				gui.setResultCardImage(clearResults, 5);
				gui.setResultCardImage(clearResults, 6);
				gui.drawWindow();

				guiState = GUIState::InFrameSelect;
			}
			else if (selectedResultIndex == 6)
			{
				// User picked "no good result" go to frame color selector

				gui.setResultCardImage(WhiteFrameImage, 0);
				gui.setResultCardImage(BlackFrameImage, 1);
				gui.setResultCardImage(RedFrameImage, 2);
				gui.setResultCardImage(BlueFrameImage, 3);
				gui.setResultCardImage(GreenFrameImage, 4);
				gui.setResultCardImage(ColorlessFrameImage, 5);
				gui.setResultCardImage(MultiColorFrameImage, 6);
				gui.drawWindow();

				guiState = GUIState::FrameColorSelect;
			}
			else
			{
				// User picked a result that we cannot account for
				// ex: we think we only show 7 results and they pick the 11th.
				assert(false);
			}

			break;

		case FrameColorSelect:
			selectedResultIndex = gui.returnResultSelectionIndex(mousePoint);

			if (selectedResultIndex == -1)
			{
				// User picked nothing
				break;
			}
			else if (selectedResultIndex >= 0 && selectedResultIndex < 7)
			{
				// User picked one of the 7 possible entries
				bool sameColorAsCard = false;
				switch (selectedResultIndex)
				{
				case 0:
					// selection 1: white
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::White;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::White);
					break;
				case 1:
					// selection 1: black
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Black;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Black);
					break;
				case 2:
					// selection 1: red
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Red;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Red);
					break;
				case 3:
					// selection 1: blue
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Blue;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Blue);
					break;
				case 4:
					// selection 1: green
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Green;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Green);
					break;
				case 5:
					// selection 1: colorless
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Colorless;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Colorless);
					break;
				case 6:
					// selection 1: multicolor
					sameColorAsCard = LastCardClicked.getMagicCard()->getFrameColor() == CardDetails::Multi;
					cdb.trainLiveCardColor(LastCardClicked.getMagicCard(), CardDetails::Multi);
					break;
				default:
					// invalid selection
					assert(false);
					break;
				}

				// Metrics
				if (!sameColorAsCard)
				{
					cardsIncorrect_WrongColor++;
				}
				else
				{
					cardsIncorrect_RightColor++;
				}

				// Clear results selections for future queries
				gui.setResultCardImage(clearResults, 0);
				gui.setResultCardImage(clearResults, 1);
				gui.setResultCardImage(clearResults, 2);
				gui.setResultCardImage(clearResults, 3);
				gui.setResultCardImage(clearResults, 4);
				gui.setResultCardImage(clearResults, 5);
				gui.setResultCardImage(clearResults, 6);
				gui.drawWindow();

				// Set state back to default state
				guiState = GUIState::InFrameSelect;

				cardFinder.reevaluateMemoryCards();
			}
			else
			{
				assert(false);
			}

			break;

		default:
			assert(false);
			break;
		}
	}
}


int main(int argc, char** argv)
{
	/// Database / real loop
	///CardDatabase cdb; TODO FOR DEMO ONLY
	std::chrono::time_point<std::chrono::system_clock> start, end;
	start = std::chrono::system_clock::now();
	//cdb.loadSet(CardDetails::RTR); // Return to Ravnica block
	//cdb.loadSet(CardDetails::GTC);
	//cdb.loadSet(CardDetails::DGM);
	//cdb.loadSet(CardDetails::ISD); // Innistrad block
	//cdb.loadSet(CardDetails::DKA);
	cdb.loadSet(CardDetails::AVR);
	//cdb.loadSet(CardDetails::ZEN); // Zendikar block
	//cdb.loadSet(CardDetails::WWK);
	//cdb.loadSet(CardDetails::ROE);
	//cdb.loadSet(CardDetails::MRD); // Mirrodin block
	//cdb.loadSet(CardDetails::DST);
	//cdb.loadSet(CardDetails::_5DN);
	//cdb.loadSet(CardDetails::CHK); // Kamigawa block
	//cdb.loadSet(CardDetails::BOK);
	//cdb.loadSet(CardDetails::SOK);
	//cdb.loadSet(CardDetails::RAV); // Ravnica block
	//cdb.loadSet(CardDetails::GPT);
	//cdb.loadSet(CardDetails::DIS);
	//cdb.loadSet(CardDetails::ALA); // Alara block
	//// Where is Conflux? - CON
	//cdb.loadSet(CardDetails::ARB);
	//cdb.loadSet(CardDetails::SOM); // Scars of Mirrodin block
	//cdb.loadSet(CardDetails::MBS);
	//cdb.loadSet(CardDetails::NPH);
	//cdb.loadSet(CardDetails::LRW); // Lorwyn-Shadowmoor block
	//cdb.loadSet(CardDetails::MOR);
	//cdb.loadSet(CardDetails::SHM);
	//cdb.loadSet(CardDetails::EVE);
	//cdb.loadSet(CardDetails::THS); // Theros block
	//cdb.loadSet(CardDetails::BNG);
	//cdb.loadSet(CardDetails::JOU);
	////////  Above new style, below old style
	//cdb.loadSet(CardDetails::ODY); // Odyssey block
	//cdb.loadSet(CardDetails::TOR);
	//cdb.loadSet(CardDetails::JUD);
	//cdb.loadSet(CardDetails::ONS); // Onslaught block
	//cdb.loadSet(CardDetails::LGN);
	//cdb.loadSet(CardDetails::SCG);
	//cdb.loadSet(CardDetails::TSP); // Time spiral block
	//cdb.loadSet(CardDetails::PLC);
	//cdb.loadSet(CardDetails::FUT);
	//cdb.loadSet(CardDetails::MMQ); // Masques block
	//cdb.loadSet(CardDetails::NMS);
	//cdb.loadSet(CardDetails::PCY);
	//cdb.loadSet(CardDetails::USG); // Urza block
	//cdb.loadSet(CardDetails::ULG);
	//cdb.loadSet(CardDetails::UDS);
	//cdb.loadSet(CardDetails::TMP); // Tempest block
	//cdb.loadSet(CardDetails::STH);
	//cdb.loadSet(CardDetails::EXO);
	//cdb.loadSet(CardDetails::MIR); // Mirage block
	//cdb.loadSet(CardDetails::VIS);
	//cdb.loadSet(CardDetails::WTH);
	//cdb.loadSet(CardDetails::INV); // Invasion block
	//cdb.loadSet(CardDetails::PLS);
	//cdb.loadSet(CardDetails::APC);
	end = std::chrono::system_clock::now();
	std::chrono::duration<double> elapsed_seconds = end - start;
	std::cout << "Done loading card database.\nElapsed time: " << elapsed_seconds.count() << " seconds\n";
	std::cout << "loaded : " << cdb.toString() << "\n";
	
	/// Video test
	//cv::VideoCapture cap("Assets\\videos\\Pro Tour Return to Ravnica- Finals.mp4"); //1251
	//cv::VideoCapture cap(ASSETS_PATH + "\\videos\\Pro Tour Avacyn Restored Top 8 Finals.mp4"); //13940
	//cv::VideoCapture cap("Assets\\videos\\testvideo2.mp4"); // 42

	// game 4 for another reason against auto gain
	//cv::VideoCapture cap("Assets\\videos\\swaptest2.mp4"); // 80 // dont even use this; lame
	//cv::VideoCapture cap("Assets\\videos\\swaptest6.mp4"); // 80 // better than swaptest2, shows movement of stacks
	//cv::VideoCapture cap("Assets\\videos\\game1.mp4"); // 80 // shows one stolen memory
	//cv::VideoCapture cap("Assets\\videos\\scattertest2.mp4"); // 80
	//cv::VideoCapture cap("Assets\\videos\\scattertest3.mp4"); // 80 // best?
	cv::VideoCapture cap("Assets\\videos\\swaptest7.mp4"); //80 // best?
	//cv::VideoCapture cap("Assets\\videos\\game9.mp4"); // 80 // show that lighting, autogain hurts
	if (!cap.isOpened())
	{
		std::cout << "Cannot open the video file.\n";
		return -1;
	}

	/// Display
	cv::namedWindow("Magic eye", CV_WINDOW_AUTOSIZE);
	cv::setMouseCallback(MAIN_WINDOW_NAME, mouseEventCallback, NULL);
	//cv::imshow("Magic eye", image);

	//jumpXFrames(cap, 2850);
	bool pause = false;
	cv::Mat frame;
	unsigned int frameNumber = 0; //4208 for multi cards and stacked cards // 1251 for beginning
	while (true)
	{
		if (!pause)
		{
			bool success = cap.read(frame);
			if (!success)
			{
				std::cout << "Cannot read frame.\n";
				break;
			}
			else
			{
				//frame = cv::imread("Assets\\videos\\reasonable.jpg", CV_LOAD_IMAGE_COLOR);
				foundCards = cardFinder.findAllCards(frame);
				// cardFinder.discernPartialCards(); // needs to be called after frame colors are assigned.  TODO start here

				//std::cout << "Frame: ";
				//std::cout << frameNumber++;
				//std::cout << std::endl;
			}
		}
		cardFinder.discernPartialCards(); // needs to be called after frame colors are assigned.  TODO start here
		gui.setMainDisplayFrame(frame);
		gui.drawWindow();
		cv::imshow("Magic eye", frame);
		char keypress = cv::waitKey(5);
		if (keypress == 'p')
		{
			pause = !pause;
		}
		else if (keypress == 'q')
		{
			break;
		}
	}

	compileMetrics();
	return 0;
}
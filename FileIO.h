#pragma once
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <random>
#include <chrono>
//used from part 1
void collectImgData(std::vector<std::vector<float>>& images, std::vector<int>& labels) {
	std::string imgPath = R"(C:\Users\Moez_\source\repos\ahmed2mz_HW4_task2\ahmed2mz_HW4_task2\MNISTnumImages5000_balanced.txt)";
	std::string lblPath = R"(C:\Users\Moez_\source\repos\ahmed2mz_HW4_task2\ahmed2mz_HW4_task2\MNISTnumLabels5000_balanced.txt)";
	std::ifstream imgFile(imgPath);
	std::ifstream lblFile(lblPath);

	if (!imgFile.is_open() || !lblFile.is_open()) {
		std::cerr << "Error: Unable to open file \n";
		return;
	}

	std::string imgLine;
	std::string lblLine;

	while (std::getline(imgFile, imgLine) && std::getline(lblFile, lblLine)) {
		std::vector<float> image;
		std::stringstream ss(imgLine);
		std::string pixelValue;
		int pixelCount = 0;

		while (std::getline(ss, pixelValue, '\t')) {
			float value = std::stof(pixelValue);
			image.push_back(value);
			pixelCount++;
		}
		if (pixelCount != 784) {
			std::cerr << "Error: image with incorrect pixel cnt \n";
			return;
		}

		images.push_back(image);
		labels.push_back(std::stoi(lblLine));
	}

	imgFile.close();
	lblFile.close();
}
void splitData(const std::vector<std::vector<float>>& images, const std::vector<int>& labels, std::vector<std::vector<float>>& trainImages, std::vector<int>& trainLabels, std::vector<std::vector<float>>& testImages, std::vector<int>& testLabels) {
	if (images.size() != labels.size()) {
		std::cerr << "Error, Label and img mismatch \n";
		return;
	}
	std::vector<std::vector<std::pair<std::vector<float>, int>>> newData(10); //-> cool article https://www.educative.io/answers/how-to-use-stdpair-in-cpp
	for (int i = 0; i < images.size(); i++) {
		int label = labels[i];
		if (label < 0 || label > 9) {
			std::cerr << "Error, Label mismatch \n";
			return;
		}
		newData[label].emplace_back(images[i], labels[i]);
	}

	//shuffle the data
	unsigned seed = std::chrono::system_clock::now().time_since_epoch().count(); //-> https://stackoverflow.com/questions/23884866/c-random-number-generation-not-random
	std::shuffle(newData.begin(), newData.end(), std::default_random_engine(seed));

	trainImages.clear();
	trainLabels.clear();
	testImages.clear();
	testLabels.clear();

	for (int j = 0; j <= 4; j++) {
		auto& data = newData[j];//may need to make auto
		//error check
		if (data.size() < 500) {
			std::cout << "Error" << std::endl;
			return;
		}

		std::shuffle(data.begin(), data.end(), std::default_random_engine(seed));
		//training data
		for (int i = 0; i < 400; i++) {
			trainImages.push_back(data[i].first);
			trainLabels.push_back(data[i].second);
		}
		for (int i = 400; i < 500; i++) {
			testImages.push_back(data[i].first);
			testLabels.push_back(data[i].second);
		}
		for (int j = 5; j <= 9; j++) {
			auto& data = newData[j];
			if (data.size() < 500) {
				std::cerr << "Error: Not enough data for digit " << j << std::endl;
				return;
			}

			// Test data: first 100 samples
			for (int i = 0; i < 100; i++) {
				testImages.push_back(data[i].first);
				testLabels.push_back(data[i].second);
			}
		}
	}
}

//need a new function to store all values of the test data in seperate boxes for each digit
std::vector<std::vector<std::vector<float>>> sortTestData(const std::vector<std::vector<float>>& testImages, std::vector<int>& testLabels) {
	//create vector. outmost ->digit; middle ->testImage; inside ->"weights"
	int index;
	std::vector<std::vector<std::vector<float>>> digitData(10, std::vector<std::vector<float>>());
	for (int i = 0; i < testImages.size(); i++) {
		index = testLabels[i];
		digitData[index].push_back(testImages[i]);
	}
	return digitData;
}

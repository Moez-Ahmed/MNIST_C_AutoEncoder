#pragma once
#include <vector>
#include <string>
#include "Perceptron.h"

class PerceptronLayer {
public:
	//hold neurons in vector
	std::vector<Perceptron> numPerceptronsLayer;

	//construtor
	PerceptronLayer(int neuronCount, int inputCount) {
		for (int i = 0; i < neuronCount; i++) {
			numPerceptronsLayer.emplace_back(inputCount);
		}
	}

};
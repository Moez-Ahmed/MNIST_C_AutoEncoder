#pragma once
#include <iostream>
#include <vector>
#include <random>
#include <chrono>

//setting up the random seed for the normal distribution that will be used to set the weights
//need to static cast to get the time_since_epoch() to work
std::default_random_engine seed(static_cast<long unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));
class Perceptron {
public:
	//important values of perceptron
	float biasWeight;
	std::vector<float> weights;
	float changeWeight;
	float sumOutput;

	//constructor:
	Perceptron(int inputCount) {
		biasWeight = 0.0f;
		changeWeight = 0.0f;
		sumOutput = 0.0f;
		std::normal_distribution<float> normalDistribution(0.0, 0.01);
		for (int i = 0; i < inputCount; i++) {
			weights.push_back(normalDistribution(seed));
		}
	}

};
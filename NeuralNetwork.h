#pragma once
#include "Perceptron.h"
#include "PerceptronLayer.h"
#include <vector>
#include <string>
#include <cmath>
#include <numeric>
#include <fstream>
std::default_random_engine seed1(static_cast<long unsigned int>(std::chrono::system_clock::now().time_since_epoch().count()));

class NeuralNetwork {
public:
	//need to follow parameters of part 1 thus
	//1 hidden layer
	PerceptronLayer hiddenLayer;
	std::vector<std::vector<float>> hiddenWeights; //double vector to hold weights per neuron
	std::vector<float> hiddenBias; //vector of every neuron bias
	//1 output layer
	PerceptronLayer outputLayer;
	std::vector<std::vector<float>> outputWeights; 
	std::vector<float> outputBias;
	//hyper paramaters
	float learningRate;
	float momentum;

	//sigmoid
	float sigmoid(float input) {
		return (1.0f / (1.0f + exp(-input)));
	}
	float sigmoidChange(float input) {
		float temp = sigmoid(input);
		return (temp * (1 - temp));
	}
	//constructor
	NeuralNetwork(int inputNeurons, int hiddenNeurons, int outPutNeurons, float learningrate, float moment) : hiddenLayer(hiddenNeurons, inputNeurons), outputLayer(outPutNeurons, hiddenNeurons) {
		learningRate = learningrate;
		momentum = moment;
		hiddenWeights.resize(hiddenNeurons, std::vector<float>(inputNeurons, 0.0f));
		outputWeights.resize(outPutNeurons, std::vector<float>(hiddenNeurons, 0.0f));
		hiddenBias.resize(hiddenNeurons, 0.0f);
		outputBias.resize(outPutNeurons, 0.0f);
	}
	//forward run
	std::vector<float> forwardProp(const std::vector<float>& input) {
		std::vector<float> hiddenOutputs(hiddenLayer.numPerceptronsLayer.size(), 0.0f);
		//for hidden layer
		for (int i = 0; i < hiddenLayer.numPerceptronsLayer.size(); i++) {
			float sum = hiddenLayer.numPerceptronsLayer[i].biasWeight;
			for (int j = 0; j < input.size(); j++) {
				sum = sum + hiddenLayer.numPerceptronsLayer[i].weights[j] * input[j];
			}
			//send it to sigmoid to see if activate
			hiddenLayer.numPerceptronsLayer[i].sumOutput = sigmoid(sum);
		}
		//for outer layer
		std::vector<float> results(outputLayer.numPerceptronsLayer.size());
		for (int i = 0; i < outputLayer.numPerceptronsLayer.size(); i++) {
			float sum = outputLayer.numPerceptronsLayer[i].biasWeight;
			for (int j = 0; j < hiddenOutputs.size(); j++) {
				sum = sum + outputLayer.numPerceptronsLayer[i].weights[j] * hiddenOutputs[j];
			}
			//send it to sigmoid to see if activate
			outputLayer.numPerceptronsLayer[i].sumOutput = sigmoid(sum);
			results[i] = outputLayer.numPerceptronsLayer[i].sumOutput;
		}
		return results;
	}
	//backprop
	void backwardProp(const std::vector<float>& input, const std::vector<float>& temp) {
		//first find the output layer changes
		for (int i = 0; i < outputLayer.numPerceptronsLayer.size(); i++) {
			float n1 = outputLayer.numPerceptronsLayer[i].sumOutput;
			float n2 = temp[i];
			outputLayer.numPerceptronsLayer[i].changeWeight = (n1 - n2) * n1 * (1.0f - n1) / 784.0f;
		}
		//then find the hidden layer changes
		for (int i = 0; i < hiddenLayer.numPerceptronsLayer.size(); i++) {
			float temp = hiddenLayer.numPerceptronsLayer[i].sumOutput;
			float sum = 0.0f;
			for (int j = 0; j < outputLayer.numPerceptronsLayer.size(); j++) {
				sum = sum + outputLayer.numPerceptronsLayer[j].weights[i] * outputLayer.numPerceptronsLayer[j].changeWeight;
			}
			hiddenLayer.numPerceptronsLayer[i].changeWeight = sum * temp * (1.0f - temp);
		}
		//change outputLayer weights
		for (int i = 0; i < outputLayer.numPerceptronsLayer.size(); i++) {
			for (int j = 0; j < hiddenLayer.numPerceptronsLayer.size(); j++) {
				float change = hiddenLayer.numPerceptronsLayer[j].sumOutput * outputLayer.numPerceptronsLayer[i].changeWeight * -learningRate;
				outputWeights[i][j] = outputWeights[i][j] * momentum + change;
			}
			float changeBias = outputLayer.numPerceptronsLayer[i].changeWeight * -learningRate;
			outputBias[i] = outputBias[i] * momentum + changeBias;
			outputLayer.numPerceptronsLayer[i].biasWeight = outputLayer.numPerceptronsLayer[i].biasWeight + outputBias[i];
		}
		//change hiddenLayer weights
		for (int i = 0; i < hiddenLayer.numPerceptronsLayer.size(); i++) {
			for (int j = 0; j < input.size(); j++) {
				float change = input[j] * hiddenLayer.numPerceptronsLayer[i].changeWeight * -learningRate;
				hiddenWeights[i][j] = hiddenWeights[i][j] * momentum + change;
			}
			float changeBias = hiddenLayer.numPerceptronsLayer[i].changeWeight * -learningRate;
			hiddenBias[i] = hiddenBias[i] * momentum + changeBias;
			hiddenLayer.numPerceptronsLayer[i].biasWeight = hiddenLayer.numPerceptronsLayer[i].biasWeight + hiddenBias[i];
		}
	}
	//train in 1 run
	float oneTrain(const std::vector<float>& temp) {
		std::vector<float> outputs = forwardProp(temp);
		float loss = 0.0f;
		for (int i = 0; i < outputs.size(); i++) {
			float delta = temp[i] - outputs[i];
			loss = loss + (delta * delta * 0.5f);
		}
		loss = loss / 784.0f;
		backwardProp(temp, temp);
		return loss;
	}
	//train overload
	void train(const std::vector<std::vector<float>>& trainData) {
		int sampleSize = trainData.size();
		std::vector<float> MREs;
		float totloss = 0.0f;
		float loss = 0.0f;
		for (int epoch = 0; epoch < 500; epoch++) {
			totloss = 0.0f;
			loss = 0.0f;
			//shuffle training data per epoch; i think this was an error that my previous model suffered from
			std::vector<int> indices(sampleSize);
			std::iota(indices.begin(), indices.end(), 0);
			std::shuffle(indices.begin(), indices.end(), seed);

			for (int i = 0; i < sampleSize; i++) {
				int shuffIndex = indices[i];
				const std::vector<float>& temp = trainData[shuffIndex];
				float sampleLoss = oneTrain(temp);
				loss = loss + sampleLoss;
				totloss = totloss + sampleLoss;
			}
			float MRE = totloss / sampleSize;
			if (epoch == 0 || (epoch + 1) % 10 == 0 || epoch ==  499) {
				MREs.push_back(MRE);
				std::cout << "Epoch: " << epoch + 1 << " MRE: " << MRE << std::endl;
			}
			if (MRE < 0.11f) {
				std::cout << "MRE is lower than 0.11, program will stop" << std::endl;
				break;
			}
			std::ofstream MREFILE("epochMRE.txt", std::ios::app);
			if (MREFILE.is_open()) {
				MREFILE << MRE << "\n";
				MREFILE.close();
			}
			else {
				std::cerr << "Error";
			}
			
		}
	}
	//find network values
	void values(const std::vector<std::vector<std::vector<float>>>& temp) {
		std::vector<float> MREs(10, 0.0);
		std::vector<float> variances(10, 0.0);

		for (int i = 0; i < 10; i++) {
			//input vector only holds the training data per digit
			const std::vector<std::vector<float>>& input = temp[i];
			std::vector<float> errors(input.size(), 0.0);
			//run forward prop
			for (int j = 0; j < input.size(); j++) {
				const std::vector<float>& digData = input[j];
				std::vector<float> forwardDigData = forwardProp(digData);
				//find the error
				float changeError = 0.0;
				for (int k = 0; k < forwardDigData.size(); k++) {
					float error = digData[k] - forwardDigData[k];
					changeError = changeError + (0.5 * error * error);
				}
				changeError = changeError / 784.0f;
				errors[j] = changeError;
			}
			float errorTot = 0.0f;
				for (int k = 0; k < errors.size(); k++) {
					errorTot = errorTot + errors[k];
				}
				//mean error per digit
				float digitMRE = errorTot / errors.size();
				//sum of squared errors
				float sumSQE = 0.0f;
				for (int k = 0; k < errors.size(); k++) {
					sumSQE = sumSQE + (errors[k] * errors[k]);
				}
				float variance = (sumSQE / input.size()) - (digitMRE * digitMRE);
				float stdev = std::sqrt(variance);
				MREs[i] = digitMRE;
				variances[i] = stdev;
				std::cout << "For Integer: " << i << " : MRE = " << MREs[i] << ", Standard deviation = " << variances[i] << std::endl;
		}
		float totalMRE = 0.0f;
		for (int l = 0; l < 10; l++) {
			totalMRE = totalMRE + MREs[l];
		}
		totalMRE = totalMRE / 10.0f;
		std::cout << "Total test MRE = " << totalMRE << std::endl;
	}
	//output hidden neurons -> 150 total

	//save weights
	void saveWeights(const std::string& filename) {
		std::ofstream weightsName(filename);
		//output hidden layer stuff
		weightsName << "Hidden Layer: " << std::endl;
		for (int i = 0; i < hiddenLayer.numPerceptronsLayer.size(); i++) {
			const Perceptron& perceptron = hiddenLayer.numPerceptronsLayer[i];
			weightsName << "Neuron " << i + 1 << ":" << std::endl;
			weightsName << "Bias: " << perceptron.biasWeight << std::endl;
			weightsName << "Weights: ";
			for (int k = 0; k < perceptron.weights.size(); k++) {
				weightsName << " " << perceptron.weights[k];
			}
			weightsName << std::endl;
		}
		weightsName << std::endl;
		//output output layer stuff
		weightsName << "Output Layer: " << std::endl;
		for (int i = 0; i < outputLayer.numPerceptronsLayer.size(); i++) {
			const Perceptron& perceptron = outputLayer.numPerceptronsLayer[i];
			weightsName << "Neuron " << i + 1 << ":" << std::endl;
			weightsName << "Bias: " << perceptron.biasWeight << std::endl;
			weightsName << "Weights: ";
			for (int k = 0; k < perceptron.weights.size(); k++) {
				weightsName << " " << perceptron.weights[k];
			}
			weightsName << std::endl;
		}
		weightsName.close();
	}
	//output 20 neurons


};
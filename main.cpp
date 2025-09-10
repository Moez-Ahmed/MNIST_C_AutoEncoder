#include <iostream>
#include <vector>
#include "NeuralNetwork.h"
#include "Perceptron.h"
#include "PerceptronLayer.h"
#include "FileIO.h"
void saveImageData(const std::string& filename, const std::vector<float>& imageData) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < imageData.size(); ++i) {
        outFile << imageData[i];
        if ((i + 1) % 28 == 0) {
            outFile << "\n"; // Newline after every 28 pixels (one row)
        }
        else {
            outFile << " "; // Space between pixels
        }
    }
    outFile.close();
}
void saveReconstructionsForVisualization(
    const std::vector<std::vector<std::vector<float>>>& testData,
    NeuralNetwork& autoencoder
) {
    std::default_random_engine rng(static_cast<unsigned>(std::chrono::system_clock::now().time_since_epoch().count()));

    for (int digit = 5; digit <= 9; ++digit) {
        const auto& digitData = testData[digit];

        // Check if there are enough samples
        if (digitData.size() < 5) {
            std::cerr << "Not enough samples for digit " << digit << std::endl;
            continue;
        }

        // Randomly select 5 samples
        std::vector<int> indices(digitData.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::shuffle(indices.begin(), indices.end(), rng);
        indices.resize(5); // Keep only the first 5 indices

        for (size_t i = 0; i < indices.size(); ++i) {
            int idx = indices[i];
            const auto& originalImage = digitData[idx];
            auto reconstructedImage = autoencoder.forwardProp(originalImage);

            // Save original image
            std::string originalFilename = "digit_" + std::to_string(digit) + "_original_" + std::to_string(i) + ".txt";
            saveImageData(originalFilename, originalImage);

            // Save reconstructed image
            std::string reconstructedFilename = "digit_" + std::to_string(digit) + "_reconstructed_" + std::to_string(i) + ".txt";
            saveImageData(reconstructedFilename, reconstructedImage);
        }
    }
}
int main()  {
    //set up parameters
    int inputNeurons = 784;
    int hiddenNeurons = 150;
    int outputNeurons = 784;
    float learningRate = 0.0f;
    float momentum = 0.0f;
    int epochs = 100;
    //create neural networl
    
    //grab data
    //create values to send through data functions
    std::vector<std::vector<float>> images;
    std::vector<int> labels;
    collectImgData(images, labels);

    //create test and training data
    std::vector<std::vector<float>> trainImages;
    std::vector<int> trainLabels;
    std::vector<std::vector<float>> testImages;
    std::vector<int> testLabels;
    splitData(images, labels, trainImages, trainLabels, testImages, testLabels);
    //train neural network ->hardcoded values to network for now
    NeuralNetwork PerceptronNetwork(784, 150, 784, 0.05f, 0.09f);

    //test neural network ->first make test set
    std::vector<std::vector<std::vector<float>>> sortedData = sortTestData(testImages, testLabels);
    PerceptronNetwork.train(trainImages);

    //save weights
    std::string neuronWeights = "AutoEncoder_Weights.txt";
    //output hidden neurons
    //work on that
    PerceptronNetwork.values(sortedData);
    saveReconstructionsForVisualization(sortedData, PerceptronNetwork);
    PerceptronNetwork.saveWeights(neuronWeights);



    
}


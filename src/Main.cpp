#include "DatasetLoader.hpp"
#include "Logger.hpp"
#include "Network.hpp"
#include "Util.hpp"

//TODO: CNNs

int main(){
    stbi_set_flip_vertically_on_load(false);
    nc::random::seed(std::chrono::system_clock::now().time_since_epoch().count());

    std::vector<nc::NdArray<double>> inputs, targets;
    load_dataset(inputs, targets, "mnist_dataset", 3000); // Last parameter (3000) means how much data per number.
    shuffle_dataset(inputs, targets);
    Logger::log("Dataset Loaded: %d", inputs.size());

    Layer testlayer(28*28, 200); //784
    Layer oho(200, 10, ActivationFunction::mSoftmax);
    
    NeuralNetwork nn({testlayer, oho});

    Logger::log("Training.");

    nn.Train(inputs, targets, 5, 0.01, 0.05); // TODO: increase the generalization.

    for(;;){
        std::string input;
        std::cout << "> ";
        std::getline(std::cin, input);

        if(input == "" || input == "quit" || input == "exit") break;

        try{
            auto b = LoadGrayScaleImage(input.c_str()) / 255.0;
            b = b.flatten();
            auto a = nn.Predict(b).flatten();
        
            Logger::log(a);
            Logger::log("Which means, Neural network thinks this is a %d with %.0f%% confidence.", a.argmax()[0], a.max()[0]*100);
        }catch(const std::exception& e){
            Logger::log(Logger::LogLevel::Error, e.what());
        }
    }
    return 0;
}
#pragma once

#include "Util.hpp"
#include "Logger.hpp"

class Neuron
{
public:
    double bias;
    nc::NdArray<double> weights;
    double output;
    double delta;

    Neuron(nc::uint32 input_size)
    {
        double stddev = std::sqrt(2.0 / input_size);
        weights = stddev * nc::random::randN<double>({1, input_size});
        bias = stddev * nc::random::randN<double>();
    }

    double Activate(const nc::NdArray<double> &inputs)
    {
        output = nc::dot(weights, inputs).item() + bias;
        return output;
    }
};

class Layer
{
public:
    std::vector<Neuron> neurons;
    nc::NdArray<double> outputs;
    nc::NdArray<double> deltas;
    Activation activation;
    Activation activation_derivative;
    ActivationFunction activationFunctionEnum;

    Layer(int input_count, nc::uint32 neuron_count, ActivationFunction activationFunction = ActivationFunction::mReLU)
    {
        outputs = nc::zeros<double>({1, neuron_count});
        deltas = nc::zeros<double>({1, neuron_count});
        activationFunctionEnum = activationFunction;
        if (activationFunction == ActivationFunction::mReLU)
        {
            activation = Activation(ReLU);
            activation_derivative = Activation(d_ReLU);
        }
        else if (activationFunction == ActivationFunction::mSigmoid)
        {
            activation = Activation(Sigmoid);
            activation_derivative = Activation(d_Sigmoid);
        }
        else if (activationFunction == ActivationFunction::mSoftmax)
        {
            activation = Activation(nullptr, Softmax);
            activation_derivative = Activation(); // no need
        }

        for (int i = 0; i < neuron_count; i++)
        {
            Neuron neuron(input_count);
            neurons.push_back(neuron);
        }
    }

    nc::NdArray<double> Forward(const nc::NdArray<double> &inputs)
    {
        if (activationFunctionEnum == ActivationFunction::mSoftmax)
        {
            nc::NdArray<double> zs(1, neurons.size());
            for (int i = 0; i < neurons.size(); i++)
            {
                zs(0, i) = neurons[i].Activate(inputs);
            }
            outputs = activation.vector(zs);
        }
        else
        {
            for (int i = 0; i < neurons.size(); i++)
            {
                outputs(0, i) = activation.scalar(neurons[i].Activate(inputs));
            }
        }
        return outputs;
    }
};

class NeuralNetwork
{
public:
    std::vector<Layer> layers;

    NeuralNetwork(std::vector<Layer> hiddenlayers)
    {
        layers = hiddenlayers;
    }

    void Train(std::vector<nc::NdArray<double>> inputs, std::vector<nc::NdArray<double>> targets, int epoch = 10, double learning_rate = 0.01, double cross_validate = 0)
    {
        if (inputs.size() != targets.size())
            throw std::runtime_error("Datasets must have same length!");

        std::vector<nc::NdArray<double>> test_input, test_target;
        int validation_amount = inputs.size() * cross_validate;

        test_input.assign(inputs.end() - validation_amount, inputs.end());
        test_target.assign(targets.end() - validation_amount, targets.end());

        inputs.resize(inputs.size() - validation_amount);
        targets.resize(targets.size() - validation_amount);

        for (int e = 1; e < epoch + 1; e++)
        {
            double train_accuracy = 0;
            double total_loss = 0;
            double test_accuracy = 0;
            for (int i = 0; i < inputs.size(); i++)
            {
                nc::NdArray<double> predicted = Predict(inputs[i]);

                if (predicted.argmax()[0] == targets[i].argmax()[0])
                    train_accuracy++;

                double loss = cross_entropy(predicted, targets[i]); // Changed from MSE to CE
                total_loss += loss;

                nc::NdArray<double> dl_da = predicted - targets[i]; // ∂L/∂A | the rate of change in L by A (activation)

                // OUTPUT LAYER (cuz its different from other layers. They dont have loss thingy)
                Layer &output_layer = layers.back();
                if (output_layer.activationFunctionEnum == ActivationFunction::mSoftmax)
                {                                // Further optimization is possible
                    output_layer.deltas = dl_da; // predicted - targets will be equal to ∂L/∂z which means delta, not the ∂L/∂A
                }
                else
                {
                    for (int j = 0; j < output_layer.neurons.size(); j++)
                    {
                        output_layer.deltas(0, j) = dl_da(0, j) * output_layer.activation_derivative.scalar(output_layer.outputs(0, j)); // j th delta = dL/da * da/dz = dL/dz
                    }
                }

                for (int j = layers.size() - 2; j >= 0; j--)
                {
                    Layer &current = layers[j];
                    Layer &next = layers[j + 1];
                    for (int k = 0; k < current.neurons.size(); k++)
                    {
                        double sum = 0;
                        for (int l = 0; l < next.neurons.size(); l++)
                        {
                            sum += next.neurons[l].weights(0, k) * next.deltas(0, l);
                        }
                        /*current.neurons[k].delta*/ current.deltas(0, k) = sum * current.activation_derivative.scalar(current.neurons[k].output);
                    }
                }

                // real learning process
                for (int l = 0; l < layers.size(); l++)
                {
                    auto prev = l == 0 ? inputs[i] : layers[l - 1].outputs;
                    for (int j = 0; j < layers[l].neurons.size(); j++)
                    {
                        for (int k = 0; k < layers[l].neurons[j].weights.shape().cols; k++)
                        {
                            layers[l].neurons[j].weights(0, k) -= learning_rate * layers[l].deltas(0, j) * prev(0, k);
                        }
                        layers[l].neurons[j].bias -= learning_rate * layers[l].deltas(0, j);
                    }
                }
            }
            for (int i = 0; i < test_input.size(); i++)
            {
                nc::NdArray<double> predicted = Predict(test_input[i]);
                if (predicted.argmax()[0] == test_target[i].argmax()[0])
                    test_accuracy++;
            }

            Logger::log("Epoch %d done. Train Accuracy: %.2f%% | Test Accuracy: %.2f%% | Loss: %.2f", e, 100 * train_accuracy / inputs.size(), 100 * test_accuracy / test_input.size(), total_loss / inputs.size());
        }
    }

    nc::NdArray<double> Predict(nc::NdArray<double> inputs)
    {
        for (int i = 0; i < layers.size(); i++)
        {
            inputs = layers[i].Forward(inputs);
        }
        return inputs;
    }
};
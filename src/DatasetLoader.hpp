#pragma once

#define STB_IMAGE_IMPLEMENTATION
#define NUMCPP_NO_USE_BOOST

#include <vector>
#include <stb_image.h>
#include <NumCpp.hpp>
#include <system_error>
#include "Logger.hpp"

nc::NdArray<double> LoadGrayScaleImage(const char *path)
{
    int width, height, channels;
    unsigned char *buf = stbi_load(path, &width, &height, &channels, 1);
    if (!buf)
        throw std::runtime_error("Couldn't read file.");

    nc::NdArray<double> ret(height, width);
    std::transform(buf, buf + (height * width), ret.begin(), [](unsigned char v)
                   { return static_cast<double>(v); });

    stbi_image_free(buf);
    return ret;
}

void load_dataset(std::vector<nc::NdArray<double>> &inputs, std::vector<nc::NdArray<double>> &targets, std::string dataset_path, int count = 500)
{
    inputs.clear();
    targets.clear();

    inputs.reserve(count * 10);
    targets.reserve(count * 10);

    for (int i = 0; i < 10; i++)
    {
        std::string path = dataset_path +"/"+ std::to_string(i);
        int sum = 0;
        try{
            for (const auto &entry : std::filesystem::directory_iterator(path)) {
                sum++;
                if (sum > count) break;
    
                if (entry.is_regular_file()){
                    nc::NdArray<double> input = LoadGrayScaleImage(entry.path().string().c_str()).flatten() / 255.0;
                    inputs.push_back(std::move(input));
    
                    nc::NdArray<double> target = nc::zeros<double>({1, 10});
                    target(0, i) = 1.0f;
                    targets.push_back(std::move(target));
                }
            }
        }
        catch (const std::filesystem::filesystem_error &e){
            Logger::log(Logger::LogLevel::Critical, "Path '%s' does not exist or unreadable!\n%s", path.c_str(), e.what());
            exit(-1);
        }
        catch (const std::exception &e){
            Logger::log(Logger::LogLevel::Error, e.what());
            exit(-1);
        }
    }
}

void shuffle_dataset(std::vector<nc::NdArray<double>> &inputs, std::vector<nc::NdArray<double>> &targets)
{
    if (inputs.size() != targets.size())
        throw std::runtime_error("Mismatch in dataset sizes!");

    std::vector<size_t> indices(inputs.size());
    std::iota(indices.begin(), indices.end(), 0);

    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(indices.begin(), indices.end(), g);

    std::vector<nc::NdArray<double>> inputs_shuffled, targets_shuffled;
    inputs_shuffled.reserve(inputs.size());
    targets_shuffled.reserve(targets.size());

    for (size_t i : indices)
    {
        inputs_shuffled.push_back(std::move(inputs[i]));
        targets_shuffled.push_back(std::move(targets[i]));
    }

    inputs = std::move(inputs_shuffled);
    targets = std::move(targets_shuffled);
}
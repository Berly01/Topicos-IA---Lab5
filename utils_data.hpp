
#pragma once
#include <cmath>
#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <string>
#include <tuple>
#include "Matrix.hpp"

using uchar = unsigned char;

static auto read_csv(const std::string& file_name, const size_t& _rows = 0, const bool& _header = false) {
    auto split = [](const std::string& _line) {
        std::istringstream iss(_line);
        std::vector<std::string> tokens;
        std::string token;

        while (std::getline(iss, token, ','))
            tokens.emplace_back(std::move(token));

        return tokens;
        };

    std::ifstream in_file(file_name + ".csv");

    if (!in_file.is_open())
        throw std::invalid_argument("Error al intentar abrir " + file_name + ".csv");

    std::vector<std::vector<std::string>> splited_lines;
    std::string line;

    std::getline(in_file, line);

    if (_header)
        splited_lines.emplace_back(split(line));

    if (_rows == 0) {
        while (std::getline(in_file, line))
            splited_lines.emplace_back(split(line));
    }
    else {
        int max_rows = 1;
        while (std::getline(in_file, line) && max_rows < _rows) {
            splited_lines.emplace_back(split(line));
            ++max_rows;
        }
    }

    return splited_lines;
}

static auto get_minist_raw_data(const std::string& _data, const size_t& _rows = 0) {

    if (_data != "test" && _data != "train" && _data != "extra")
        throw std::invalid_argument(R"(El parametro tiene que que ser "train" o "test" o "extra")");

    std::string file_name;

    if (_data == "train")
      file_name = "mnist_train";
    else if (_data == "test")
      file_name = "mnist_test";
    else
      file_name = "mnist_extra";

    const auto tokens_per_line = read_csv(file_name, _rows, true);

    std::vector<int> label;
    std::vector<std::vector<uchar>> image;

    for (const auto& tokens : tokens_per_line) {
        bool is_label = true;
        std::vector<uchar> pixels{};
        for (const auto& token : tokens) {
            if (is_label) {
                label.emplace_back(std::stoi(token));
                is_label = false;
            }
            else {
                pixels.emplace_back(static_cast<uchar>(std::stoi(token)));
            }
        }
        image.emplace_back(pixels);
    }

    return std::make_tuple(image, label);
}


static auto get_mmist_processed_data(const std::string& _data, const size_t& _rows = 0) {
    auto tuple = get_minist_raw_data(_data, _rows);

    auto& image = std::get<0>(tuple);
    auto& label = std::get<1>(tuple);

    if (label.size() != image.size())
        throw std::invalid_argument("E");

    std::vector<Matrix<double>> x;
    std::vector<Matrix<double>> y;

    x.reserve(label.size());
    y.reserve(label.size());

    auto label_b = label.begin();
    auto label_e = label.end();
    auto image_b = image.begin();

    for (; label_b != label_e; ++label_b, ++image_b) {
        Matrix<double> label(10, 1, 0.0);
        label[*label_b][0] = 1.0;

        std::vector<double> norma_pixel;
        for (const auto& v : *image_b)
            norma_pixel.emplace_back(static_cast<double>(v) / 255.0);

        x.emplace_back(norma_pixel);
        y.emplace_back(label);
    }

    return std::make_tuple(x, y);
}

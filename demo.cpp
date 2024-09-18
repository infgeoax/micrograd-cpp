//
// Created by Xiaofeng Li on 18/9/2024.
//

#include <iostream>
#include <fstream>
#include <filesystem>
#include <vector>
#include <matplot/matplot.h>
#include "engine.h"
#include "neuronet.h"

Vector2D read2d(const std::filesystem::path &filename) {
    Vector2D ret;
    std::ifstream in(filename, std::ios_base::in);
    while (in.good()) {
        Vector row;
        row.reserve(2);

        int idx = 0;
        in >> idx;
        if (in.peek() == ',') in.ignore();
        for (DataType x; in >> x;) {
            row.emplace_back(x);
            if (in.peek() == ',') {
                in.ignore();
            } else if (in.peek() == '\n') {
                break;
            }
        }

        ret.emplace_back(std::move(row));
    }
    return ret;
}

Vector read1d(const std::filesystem::path &filename) {
    Vector ret;
    std::ifstream in(filename, std::ios_base::in);
    while (in.good()) {
        int idx = 0;
        DataType x = 0;
        in >> idx;
        if (in.peek() == ',') in.ignore();
        in >> x;
        ret.emplace_back(x);
    }
    return ret;
}

void plot(const Vector2D &X, const Vector &Y) {
    using namespace matplot;
    std::vector<DataType> x0, x1, y;
    x0.reserve(X.size());
    x1.reserve(X.size());
    y.reserve(Y.size());
    for (auto &x: X) {
        x0.push_back(x[0]->data());
        x1.push_back(x[1]->data());
    }
    for (auto &v: Y) {
        y.push_back((v->data() + 1) / 2);
    }

    colormap(palette::jet());
    scatter(x0, x1, std::vector<DataType>{}, y);
    save("demo.png");
}

void decisionBoundary(const Vector2D &X, const Vector &Y, MLP &model) {
    using namespace matplot;
    std::vector<DataType> x0, x1, y;
    x0.reserve(X.size());
    x1.reserve(X.size());
    y.reserve(Y.size());
    for (auto &x: X) {
        x0.push_back(x[0]->data());
        x1.push_back(x[1]->data());
    }
    for (auto &v: Y) {
        y.push_back((v->data() + 1) / 2);
    }

    colormap(palette::spectral());
    {
        DataType x0Min = *std::min_element(x0.begin(), x0.end()) - 1,
                x0Max = *std::max_element(x0.begin(), x0.end()) + 1;
        DataType x1Min = *std::min_element(x1.begin(), x1.end()) - 1,
                x1Max = *std::max_element(x1.begin(), x1.end()) + 1;
        auto xx = linspace(x0Min, x0Max);
        auto yy = linspace(x1Min, x1Max);
        auto [X, Y] = meshgrid(xx, yy);
        auto Z = transform(X, Y, [&](double x, double y) {
            return model({Value(x), Value(y)})[0]->data() > 0;
        });
        contour(X, Y, Z);
        hold(on);
        scatter(x0, x1, std::vector<DataType>{}, y);
        hold(off);
        show();
    }
}

int main() {
    auto X = read2d(std::filesystem::path("data/moonX.csv"));
    auto y = read1d(std::filesystem::path("data/moonY.csv"));
    if (X.size() != y.size()) {
        std::cout << "X and y have different sizes.";
        exit(1);
    }
    std::cout << "X: " << X.size() << "; y: " << y.size() << "\n";
    int N = X.size();

    // plot(X, y);

    MLP model(2, {16, 16, 1});
    std::cout << "Number of parameters: " << model.parameters().size() << "\n";

    int n = 200;

    for (int k = 0; k < n; ++k) {
        Vector losses;
        losses.reserve(N);
        double accuracy = 0;
        for (int i = 0; i < N; ++i) {
            Value score = model(X[i])[0];
            losses.push_back((1 - y[i] * score).relu());
            accuracy += (y[i]->data() > 0) == (score->data() > 0);
        }
        accuracy /= N;
        Value dataLoss = std::accumulate(losses.begin(), losses.end(), Value(0)) / N;

        auto alpha = 1e-4;
        auto params = model.parameters();
        auto regLoss =
                alpha * std::accumulate(params.begin(), params.end(), Value(0), [](const Value &l, const Value &r) {
                    return l + r * r;
                });
        Value totalLoss = dataLoss + regLoss;

        totalLoss.backward();
        double learningRate = 1.0 - 0.9 * k / n;
        for (auto &p: params) {
            p->data() -= learningRate * p->grad();
            p->grad() = 0;
        }

        std::cout << "step " << k << " loss " << totalLoss->data() << ", accuracy " << accuracy * 100 << "%\n";
    }

    decisionBoundary(X, y, model);
}
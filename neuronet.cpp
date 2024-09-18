//
// Created by Xiaofeng Li on 18/9/2024.
//

#include "neuronet.h"


/// Generate uniform random value in range [left, right]
/// \param left
/// \param right
/// \return
DataType uniform(DataType left, DataType right) {
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(left, right);
    return dis(gen);
}


/**
 * A neuron has n inputs and one output.
 */
Neuron::Neuron(int nIn) : _w(nIn), _b(uniform(-1, 1)) {
    for (int i = 0; i < nIn; ++i) {
        _w[i] = Value(uniform(-1, 1));
    }
}

Value Neuron::operator()(const std::vector<Value> &x) {
    // w * x + b
    Value r = _b;
    for (int i = 0; i < x.size(); ++i) {
        r += x[i] * _w[i];
    }
    //r = r.tanh();
    return r.tanh();
}

std::vector<Value> Neuron::parameters() {
    std::vector<Value> ret(_w);
    ret.push_back(_b);
    return ret;
}

Layer::Layer(int nIn, int nOut) {
    for (int i = 0; i < nOut; ++i) {
        _neurons.emplace_back(nIn);
    }
}

std::vector<Value> Layer::operator()(const std::vector<Value> &x) {
    std::vector<Value> out;
    out.reserve(_neurons.size());
    for (auto &_neuron: _neurons) {
        out.emplace_back(_neuron(x));
    }
    return out;
}

std::vector<Value> Layer::parameters() {
    std::vector<Value> ret;
    for (auto &n: _neurons) {
        ret.append_range(n.parameters());
    }
    return ret;
}

MLP::MLP(int nIn, const std::vector<int> &nOuts) {
    std::vector<int> sz;
    sz.reserve(1 + nOuts.size());
    sz.push_back(nIn);
    sz.insert(sz.end(), nOuts.begin(), nOuts.end());
    for (int i = 0; i < nOuts.size(); ++i) {
        _layers.emplace_back(sz[i], sz[i + 1]);
    }
}

std::vector<Value> MLP::operator()(const std::vector<Value> &x) {
    auto t = x;
    for (auto &_layer: _layers) {
        t = _layer(t);
    }
    return t;
}

std::vector<Value> MLP::parameters() {
    std::vector<Value> ret;
    for (auto &l: _layers) {
        ret.append_range(l.parameters());
    }
    return ret;
}

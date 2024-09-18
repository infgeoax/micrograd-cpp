//
// Created by Xiaofeng Li on 18/9/2024.
//

#ifndef MICROGRAD_NEURONET_H
#define MICROGRAD_NEURONET_H

#include "engine.h"

/// Generate uniform random value in range [left, right]
/// \param left
/// \param right
/// \return
DataType uniform(DataType left, DataType right);

/**
 * A neuron has n inputs and one output.
 */
class Neuron {
public:
    explicit Neuron(int nIn);

    Value operator()(const Vector &x);

    Vector parameters();

private:
    Vector _w;
    Value _b;
};

class Layer {
public:
    Layer(int nIn, int nOut);

    Vector operator()(const Vector &x);

    Vector parameters();

private:
    std::vector<Neuron> _neurons;
};

class MLP {
public:
    MLP(int nIn, const std::vector<int> &nOuts);

    Vector operator()(const Vector &x);

    Vector parameters();

private:
    std::vector<Layer> _layers;
};

template<typename V>
std::ostream &operator<<(std::ostream &out, const std::vector<V> &vec) {
    out << "[";
    bool first = true;
    for (auto &val: vec) {
        if (!first) out << ", "; else first = false;
        out << val;
    }
    out << "]";
    return out;
}

#endif //MICROGRAD_NEURONET_H

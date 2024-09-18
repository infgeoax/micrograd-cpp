//
// Created by Xiaofeng Li on 18/9/2024.
//

#ifndef MICROGRAD_ENGINE_H
#define MICROGRAD_ENGINE_H

#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <random>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>


/// Type of data in each node: double
using DataType = double;
/// Backward function signature;
using BackwardFunc = std::function<void()>;

/// Default backward function: does nothing.
void empty_backward();

/**
 * Represents a value data object that can be referenced via shared_ptr.
 */
class ValueData {
public:
    using Pointer = std::shared_ptr<ValueData>;

    ValueData();

    explicit ValueData(DataType data);

    ValueData(DataType data, std::string op, std::vector<Pointer> prev, std::string label);

    /// Reference to the current value, can be used to update.
    DataType &data();

    /// Reference to the current grad, can be used to update.
    DataType &grad();

    /// Reference to parents, could be empty if the value is created from scratch.
    const std::vector<Pointer> &prev() const;

    /// Operation used to generate the value from its parents.
    std::string op() const;

    /// Optional label on the value.
    std::string label() const;

    /// Update label
    void label(std::string label);

    /// Update backward function
    ValueData &operator<<(BackwardFunc backward);

    /// Call backward function on the current node.
    void backward();

private:
    DataType _data;
    DataType _grad;
    std::string _op;
    std::vector<Pointer> _prev;
    std::string _label;
    BackwardFunc _backwardFunction;
};

/// Alias for value data pointer type.
using ValueDataPtr = ValueData::Pointer;


/// A value is just a holder of a shared pointer to ValueData
/// Calculations are done on Value, this enables us to build an expression tree from C++ expression.
class Value {
public:
    Value();

    explicit Value(DataType data, const std::string &label = "");

    Value(DataType data, const std::string &op, const std::vector<ValueDataPtr> &prev);

    Value(const Value &other) = default;

    ValueData *operator->() const;

    ValueDataPtr pointer();

    Value &operator<<(BackwardFunc backward);

    const ValueDataPtr &pointer() const;

    ValueData *raw_pointer() const;

    Value exp();

    Value pow(DataType a) const;

    Value relu();

    Value tanh();

    Value &operator|(const std::string &label);

    /// The actual backward function:
    /// - Grad on the current node is always 1: as it's just identity function: y = x
    /// - Then we update grads in topological order
    void backward();

private:
    ValueDataPtr _valueData;
};

Value &operator^=(Value &lh, DataType rh);

Value operator+(const Value &lh, const Value &rh);

Value operator+(const Value &lh, DataType rh);

Value operator+(DataType lh, const Value &rh);

template<typename RH>
Value &operator+=(Value &lh, RH rh) {
    lh = lh + rh;
    return lh;
}

template<typename RH>
Value &operator-=(Value &lh, RH rh) {
    lh = lh - rh;
    return lh;
}

template<typename RH>
Value &operator*=(Value &lh, const RH &rh) {
    lh = lh * rh;
    return lh;
}

template<typename RH>
Value &operator/=(Value &lh, RH rh) {
    lh = lh / rh;
    return lh;
}

Value operator-(const Value &lh, const Value &rh);

Value operator-(const Value &lh, DataType rh);

Value operator-(DataType lh, const Value &rh);

Value operator-(const Value &v);

Value operator*(const Value &lh, const Value &rh);

Value operator*(const Value &lh, DataType rh);

Value operator*(DataType lh, const Value &rh);


Value operator/(const Value &lh, const Value &rh);

Value operator/(const Value &lh, DataType rh);

Value operator/(DataType lh, const Value &rh);

std::ostream &operator<<(std::ostream &out, const Value &val);
#endif //MICROGRAD_ENGINE_H

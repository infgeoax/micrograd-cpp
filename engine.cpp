//
// Created by Xiaofeng Li on 18/9/2024.
//

#include "engine.h"

void empty_backward() {}

ValueData::ValueData() : ValueData(0, "val", {}, "") {}

ValueData::ValueData(DataType data) : ValueData(data, "val", {}, "") {}

ValueData::ValueData(DataType data, std::string op, std::vector<Pointer> prev, std::string label)
        : _data(data), _grad(0.0), _op(std::move(op)), _prev(std::move(prev)), _label(std::move(label)),
          _backwardFunction(empty_backward) {}

DataType &ValueData::data() {
    return _data;
}

DataType &ValueData::grad() {
    return _grad;
}

const std::vector<ValueDataPtr> &ValueData::prev() const {
    return _prev;
}

std::string ValueData::op() const {
    return _op;
}

std::string ValueData::label() const {
    return _label;
}

void ValueData::label(std::string label) {
    _label = std::move(label);
}

ValueData &ValueData::operator<<(BackwardFunc backward) {
    _backwardFunction = std::move(backward);
    return *this;
}

void ValueData::backward() {
    _backwardFunction();
}

Value::Value() : Value(0) {}

Value::Value(DataType data, const std::string &label) : _valueData(std::make_shared<ValueData>(data)) {
_valueData->label(label);
}

Value::Value(DataType data, const std::string &op, const std::vector<ValueDataPtr> &prev) : _valueData(
        std::make_shared<ValueData>(data, op, prev, "")) {}

ValueData *Value::operator->() const {
    return _valueData.get();
}

ValueDataPtr Value::pointer() {
    return _valueData;
}

Value &Value::operator<<(BackwardFunc backward) {
    *_valueData << std::move(backward);
    return *this;
}

const ValueDataPtr &Value::pointer() const {
    return _valueData;
}

ValueData *Value::raw_pointer() const {
    return _valueData.get();
}

Value Value::exp() {
    auto x = Value(std::exp(_valueData->data()), "exp", {pointer()});
    auto valueData = _valueData;
    x << [=] {
        valueData->grad() += x->grad() * x->data();
    };

    return x;
}

Value Value::pow(DataType a) const {
    auto x = Value(std::pow(_valueData->data(), a), "power", {pointer()});
    auto lh = _valueData;
    x << [=] {
        lh->grad() += x->grad() * a * std::pow(lh->data(), a - 1);
    };

    return x;
}

Value Value::relu() {
    auto x = Value(std::max(0.0, _valueData->data()), "relu", {pointer()});
    auto valueData = _valueData;
    x << [=, this] {
        valueData->grad() += x->grad() * (x->data() > 0 ? 1 : 0);
    };

    return x;
}

Value Value::tanh() {
    auto x = Value(std::tanh(_valueData->data()), "tanh", {pointer()});
    auto valueData = _valueData;
    x << [=] {
        valueData->grad() += x->grad() * (1 - x->data() * x->data());
    };
    return x;
}

Value &Value::operator|(const std::string &label) {
    _valueData->label(label);
    return *this;
}

/// Topological sort an expression graph, used to determine the order in which to run backward functions.
class Topo : public std::vector<ValueData *> {
public:
    explicit Topo(ValueData *root) {
        dfs(root);
    }

    auto order() {
        return std::ranges::reverse_view(*this);
    }

private:
    void dfs(ValueData *n) {
        if (!_visited.contains(n)) {
            _visited.insert(n);
            for (auto &ptr: n->prev()) {
                dfs(ptr.get());
            }
            push_back(n);
        }
    }

    std::unordered_set<ValueData *> _visited;
};

/// The actual backward function:
/// - Grad on the current node is always 1: as it's just identity function: y = x
/// - Then we update grads in topological order
void Value::backward() {
    _valueData->grad() = 1;
    Topo topo(raw_pointer());
    for (auto p: topo.order()) {
        p->backward();
    }
}

Value &operator^=(Value &lh, DataType rh) {
    lh = lh.pow(rh);
    return lh;
}

Value operator+(const Value &lh, const Value &rh) {
    auto x = Value(lh->data() + rh->data(), "+", {lh.pointer(), rh.pointer()});
    x << [=] {
        lh->grad() += x->grad();
        rh->grad() += x->grad();
    };
    return x;
}

Value operator+(const Value &lh, DataType rh) {
    return lh + Value(rh);
}

Value operator+(DataType lh, const Value &rh) {
    return rh + lh;
}

Value operator-(const Value &lh, const Value &rh) {
    auto x = Value(lh->data() - rh->data(), "-", {lh.pointer(), rh.pointer()});
    x << [=] {
        lh->grad() += x->grad();
        rh->grad() -= x->grad();
    };
    return x;
}

Value operator-(const Value &lh, DataType rh) {
    return lh - Value(rh);
}

Value operator-(DataType lh, const Value &rh) {
    return Value(lh) - rh;
}

Value operator-(const Value &v) {
    auto x = Value(-v->data(), "neg", {v.pointer()});
    x << [=] {
        v->grad() -= x->grad();
    };
    return x;
}


Value operator*(const Value &lh, const Value &rh) {
    auto x = Value(lh->data() * rh->data(), "*", {lh.pointer(), rh.pointer()});
    x << [=] {
        lh->grad() += x->grad() * rh->data();
        rh->grad() += x->grad() * lh->data();
    };
    return x;
}

Value operator*(const Value &lh, DataType rh) {
    return lh * Value(rh);
}

Value operator*(DataType lh, const Value &rh) {
    return Value(lh) * rh;
}

Value operator/(const Value &lh, const Value &rh) {
    return lh * rh.pow(-1);
}

Value operator/(const Value &lh, DataType rh) {
    return lh / Value(rh);
}

Value operator/(DataType lh, const Value &rh) {
    return Value(lh) / rh;
}

std::ostream &operator<<(std::ostream &out, const Value &val) {
    out << "Value(data=" << val->data() << ")";
    return out;
}
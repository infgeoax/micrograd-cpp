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
void empty_backward() {}

/**
 * Represents a value data object that can be referenced via shared_ptr.
 */
class ValueData {
public:
    using Pointer = std::shared_ptr<ValueData>;

    ValueData() : ValueData(0, "val", {}, "") {}

    explicit ValueData(DataType data) : ValueData(data, "val", {}, "") {}

    ValueData(DataType data, std::string op, std::vector<Pointer> prev, std::string label)
            : _data(data), _grad(0.0), _op(std::move(op)), _prev(std::move(prev)), _label(std::move(label)),
              _backwardFunction(empty_backward) {}

    /// Referece to the current value, can be used to update.
    DataType &data() {
        return _data;
    }

    /// Reference to the current grad, can be used to update.
    DataType &grad() {
        return _grad;
    }

    /// Reference to parents, could be empty if the value is created from scratch.
    const std::vector<Pointer> &prev() const {
        return _prev;
    }

    /// Operation used to generate the value from its parents.
    std::string op() const {
        return _op;
    }

    /// Optional label on the value.
    std::string label() const {
        return _label;
    }

    /// Update label
    void label(std::string label) {
        _label = std::move(label);
    }

    /// Update backward function
    ValueData &operator<<(BackwardFunc backward) {
        _backwardFunction = std::move(backward);
        return *this;
    }

    /// Call backward function on the current node.
    void backward() {
        _backwardFunction();
    }

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


/// A value is just a holder of a shared pointer to ValueData
/// Calculations are done on Value, this enables us to build an expression tree from C++ expression.
class Value {
public:
    Value() : Value(0) {}

    explicit Value(DataType data, const std::string &label = "") : _valueData(std::make_shared<ValueData>(data)) {
        _valueData->label(label);
    }

    Value(DataType data, const std::string &op, const std::vector<ValueDataPtr> &prev) : _valueData(
            std::make_shared<ValueData>(data, op, prev, "")) {}

    Value(const Value &other) = default;

    ValueData *operator->() const {
        return _valueData.get();
    }

    ValueDataPtr pointer() {
        return _valueData;
    }

    Value &operator<<(BackwardFunc backward) {
        *_valueData << std::move(backward);
        return *this;
    }

    [[nodiscard]] const ValueDataPtr &pointer() const {
        return _valueData;
    }

    [[nodiscard]] ValueData *raw_pointer() const {
        return _valueData.get();
    }

    Value exp() {
        auto x = Value(std::exp(_valueData->data()), "exp", {pointer()});
        auto valueData = _valueData;
        x << [=] {
            valueData->grad() += x->grad() * x->data();
        };

        return x;
    }

    Value pow(DataType a) const {
        auto x = Value(std::pow(_valueData->data(), a), "power", {pointer()});
        auto lh = _valueData;
        x << [=] {
            lh->grad() += x->grad() * a * std::pow(lh->data(), a - 1);
        };

        return x;
    }

    Value relu() {
        auto x = Value(std::max(0.0, _valueData->data()), "relu", {pointer()});
        auto valueData = _valueData;
        x << [=, this] {
            valueData->grad() += x->grad() * (x->data() > 0 ? 1 : 0);
        };

        return x;
    }

    Value tanh() {
        auto x = Value(std::tanh(_valueData->data()), "tanh", {pointer()});
        auto valueData = _valueData;
        x << [=] {
            valueData->grad() += x->grad() * (1 - x->data() * x->data());
        };
        return x;
    }

    Value &operator|(const std::string &label) {
        _valueData->label(label);
        return *this;
    }

    /// The actual backward function:
    /// - Grad on the current node is always 1: as it's just identity function: y = x
    /// - Then we update grads in topological order
    void backward() {
        _valueData->grad() = 1;
        Topo topo(raw_pointer());
        for (auto p: topo.order()) {
            p->backward();
        }
    }

private:
    ValueDataPtr _valueData;
};

/// Util to visualize a expression graph.
class Dot {
public:
    explicit Dot(ValueData *root) {
        _buildGraph(root);
    }

private:
    void _buildGraph(ValueData *root) {
        if (!_visited.contains(root)) {
            _visited.insert(root);
            auto nodeId = _valueNodeId(root);
            _nodes.insert(std::make_pair(nodeId, _getValueNodeLabel(root)));

            if (!root->prev().empty()) {
                auto opId = _opNodeId(root);
                _nodes.insert(std::make_pair(opId, _getOpNodeAttrs(root)));
                _edges.push_back(std::make_tuple(opId, nodeId));
                for (auto &p: root->prev()) {
                    _edges.push_back(std::make_tuple(_valueNodeId(p.get()), opId));
                    _buildGraph(p.get());
                }
            }
        }
    }

    std::string _getValueNodeLabel(ValueData *vd) {
        std::ostringstream oss;
        oss << "shape=rectangle,";
        oss << "label=\"";
        if (!vd->label().empty()) { oss << vd->label() << "|"; }
        oss << "val=" << vd->data() << ", grad=" << vd->grad() << "\"";
        return oss.str();
    }

    std::string _getOpNodeAttrs(ValueData *vd) {
        std::ostringstream oss;
        oss << "label=\"" << vd->op() << "\"";
        return oss.str();
    }

    std::string _valueNodeId(ValueData *vd) {
        std::ostringstream oss;
        oss << "value_node_" << vd;
        return oss.str();
    }

    std::string _opNodeId(ValueData *vd) {
        std::ostringstream oss;
        oss << "op_node_" << vd;
        return oss.str();
    }

    // node id -> node attributes (label, shape etc.)
    std::unordered_map<std::string, std::string> _nodes;
    // edges (node id 1 -> node id 2)
    std::vector<std::tuple<std::string, std::string>> _edges;

    std::unordered_set<ValueData *> _visited;

    friend std::ostream &operator<<(std::ostream &out, const Dot &dot);
};

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

template<typename RH>
Value &operator+=(Value &lh, RH rh) {
    lh = lh + rh;
    return lh;
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

template<typename RH>
Value &operator-=(Value &lh, RH rh) {
    lh = lh - rh;
    return lh;
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

template<typename RH>
Value &operator*=(Value &lh, const RH &rh) {
    lh = lh * rh;
    return lh;
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

template<typename RH>
Value &operator/=(Value &lh, RH rh) {
    lh = lh / rh;
    return lh;
}

std::ostream &operator<<(std::ostream &out, const Value &val) {
    out << "Value(data=" << val->data() << ")";
    return out;
}

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
class Neuron {
public:
    explicit Neuron(int nIn) : _w(nIn), _b(uniform(-1, 1)) {
        for (int i = 0; i < nIn; ++i) {
            _w[i] = Value(uniform(-1, 1));
        }
    }

    Value operator()(const std::vector<Value> &x) {
        // w * x + b
        Value r = _b;
        for (int i = 0; i < x.size(); ++i) {
            r += x[i] * _w[i];
        }
        //r = r.tanh();
        return r.tanh();
    }

    std::vector<Value> parameters() {
        std::vector<Value> ret(_w);
        ret.push_back(_b);
        return ret;
    }

private:
    std::vector<Value> _w;
    Value _b;
};

class Layer {
public:
    Layer(int nIn, int nOut) {
        for (int i = 0; i < nOut; ++i) {
            _neurons.emplace_back(nIn);
        }
    }

    std::vector<Value> operator()(const std::vector<Value> &x) {
        std::vector<Value> out;
        out.reserve(_neurons.size());
        for (auto &_neuron: _neurons) {
            out.emplace_back(_neuron(x));
        }
        return out;
    }

    std::vector<Value> parameters() {
        std::vector<Value> ret;
        for (auto &n: _neurons) {
            ret.append_range(n.parameters());
        }
        return ret;
    }

private:
    std::vector<Neuron> _neurons;
};

class MLP {
public:
    MLP(int nIn, const std::vector<int> &nOuts) {
        std::vector<int> sz;
        sz.reserve(1 + nOuts.size());
        sz.push_back(nIn);
        sz.insert(sz.end(), nOuts.begin(), nOuts.end());
        for (int i = 0; i < nOuts.size(); ++i) {
            _layers.emplace_back(sz[i], sz[i + 1]);
        }
    }

    std::vector<Value> operator()(const std::vector<Value> &x) {
        auto t = x;
        for (auto &_layer: _layers) {
            t = _layer(t);
        }
        return t;
    }

    std::vector<Value> parameters() {
        std::vector<Value> ret;
        for (auto &l: _layers) {
            ret.append_range(l.parameters());
        }
        return ret;
    }

private:
    std::vector<Layer> _layers;
};

template<typename Val>
std::ostream &operator<<(std::ostream &out, const std::vector<Val> &vec) {
    out << "[";
    bool first = true;
    for (auto &val: vec) {
        if (!first) out << ", "; else first = false;
        out << val;
    }
    out << "]";
    return out;
}


/// Write DOT to an output stream, in a format that can be used by tools to generate a visualization.
std::ostream &operator<<(std::ostream &out, const Dot &dot) {
    out << "digraph G {\n";
    for (auto &vn: dot._nodes) {
        out << vn.first << "[" << vn.second << "];\n";
    }
    for (auto &edge: dot._edges) {
        out << std::get<0>(edge) << " -> " << std::get<1>(edge) << ";\n";
    }
    out << "}\n";
    return out;
}

int main() {
    Value a(0.1);
    a.backward();
    std::vector<std::vector<Value>> xs = {
            {Value(2.0), Value(3.0),  Value(-1.0)},
            {Value(3.0), Value(-1.0), Value(0.5)},
            {Value(0.5), Value(1.0),  Value(1.0)},
            {Value(1.0), Value(1.0),  Value(-1.0)},
    };
    std::vector<Value> ys = {
            Value(1.0), Value(-1.0), Value(-1.0), Value(1.0)
    };

    MLP mlp(3, {4, 4, 1});
    for (int i = 0; i < 120; ++i) {
        std::vector<Value> ypreds;
        for (auto &x: xs) {
            ypreds.push_back(mlp(x)[0]);
        }

        auto loss = Value(0);
        for (int j = 0; j < xs.size(); ++j) {
            loss += (ypreds[j] - ys[j]).pow(2);
        }
        std::cout << "Preds: " << ypreds << "\n";
        std::cout << "Loss: " << loss->data() << "\n";
        loss.backward();

        auto parameters = mlp.parameters();
        for (auto &p: parameters) {
            p->data() += -0.01 * p->grad();
            p->grad() = 0;
        }
    }

    return 0;
}

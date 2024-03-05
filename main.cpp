#include <functional>
#include <iostream>
#include <memory>
#include <ranges>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include <sstream>

using DataType = double;
using BackwardFunc = std::function<void()>;

class ValueData {
public:
    using Pointer = std::shared_ptr<ValueData>;

    ValueData() : ValueData(0, "val", {}, "") {}

    explicit ValueData(DataType data) : ValueData(data, "val", {}, "") {}

    ValueData(DataType data, std::string op, std::vector<Pointer> prev, std::string label)
            : _data(data), _grad(0.0), _op(std::move(op)), _prev(std::move(prev)), _label(std::move(label)),
              _backwardFunctions() {}

    [[nodiscard]] DataType data() const {
        return _data;
    }

    [[nodiscard]] const std::vector<Pointer> &prev() const {
        return _prev;
    }

    [[nodiscard]] DataType grad() const {
        return _grad;
    }

    [[nodiscard]] std::string op() const {
        return _op;
    }

    [[nodiscard]] std::string label() const {
        return _label;
    }

    DataType &grad() {
        return _grad;
    }

    void label(std::string label) {
        _label = std::move(label);
    }

    ValueData &operator<<(BackwardFunc backward) {
        _backwardFunctions.emplace_back(std::move(backward));
        return *this;
    }

    void backward() {
        for (const auto &backward_func: _backwardFunctions) {
            backward_func();
        }
    }

private:
    DataType _data;
    DataType _grad;
    std::string _op;
    std::vector<Pointer> _prev;
    std::string _label;
    std::vector<BackwardFunc> _backwardFunctions;
};

using ValueDataPtr = ValueData::Pointer;

class Topo : public std::vector<ValueData *> {
public:
    explicit Topo(ValueData *root) {
        dfs(root);
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


class Dot {
public:
    Dot(ValueData *root) {
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
                for (auto &p : root->prev()) {
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
        if (!vd->label().empty()) {oss << vd->label() << "|";}
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

    std::unordered_set<ValueData*> _visited;
    friend std::ostream& operator << (std::ostream &out, const Dot &dot);
};

std::ostream &operator << (std::ostream &out, const Dot &dot) {
    out << "digraph G {\n";
    for (auto &vn : dot._nodes) {
        out << vn.first << "[" << vn.second << "];\n";
    }
    for (auto &edge : dot._edges) {
        out << std::get<0>(edge) << " -> " << std::get<1>(edge) << ";\n";
    }
    out << "}\n";
    return out;
}


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

        x << [=, this] {
            _valueData->grad() += x->grad() * x->data();
        };

        return x;
    }

    Value relu() {
        auto x = Value(std::max(0.0, _valueData->data()), "relu", {pointer()});

        x << [=, this] {
            _valueData->grad() += x->grad() * (x->data() > 0 ? 1 : 0);
        };

        return x;
    }

    Value &operator|(const std::string &label) {
        _valueData->label(label);
        return *this;
    }

    void backward() {
        _valueData->grad() = 1;
        Topo topo(raw_pointer());
        for (auto &p: std::ranges::reverse_view(topo)) {
            p->backward();
        }
    }

private:
    ValueDataPtr _valueData;
};

/// lh to the power of rh.
/// \param lh
/// \param rh
/// \return
Value operator^(const Value &lh, DataType rh) {
    std::ostringstream oss;
    oss << "pow(" << rh << ")";
    auto x = Value(std::pow(lh->data(), rh), oss.str(), {lh.pointer()});
    x << [=] {
        lh->grad() += x->grad() * rh * std::pow(lh->data(), rh - 1);
    };
    return x;
}

Value &operator^=(Value &lh, DataType rh) {
    lh = lh ^ rh;
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
    return lh * (rh ^ -1);
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

int main() {
    auto t1 = Value(10);
    auto t2 = Value(20);
    auto t3 = 14 * t2 + 17 * t1 ^ 3;
    t3.backward();

    auto a = Value(-4.0) | "a";
    auto b = Value(2.0) | "b";
    auto c = (a + b) | "c1";
    auto d = (a * b + (b ^ 3)) | "d";
    c += (c + 1) | "c2";
    c += (1 + c + (-a)) | "c3";
    d += (d * 2 + (b + a).relu()) | "d2";
    d += (3 * d + (b - a).relu()) | "d3";
    auto e = (c - d) | "e";
    auto f = e ^ 2;
    auto g = f / 2.0;
    g += 10.0 / f;
    g.backward();

    Dot dot1(g.raw_pointer());
    std::cout << dot1 << "\n";

    return 0;
}

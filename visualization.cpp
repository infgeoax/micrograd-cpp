//
// Created by Xiaofeng Li on 18/9/2024.
//

#include "visualization.h"

ValueGraph::ValueGraph(ValueData *root) {
    _buildGraph(root);
}

ValueGraph::ValueGraph(Value &&root) : ValueGraph(root.raw_pointer()) {}

void ValueGraph::_buildGraph(ValueData *root) {
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

std::string ValueGraph::_getValueNodeLabel(ValueData *vd) {
    std::ostringstream oss;
    oss << "shape=rectangle,";
    oss << "label=\"";
    if (!vd->label().empty()) { oss << vd->label() << "|"; }
    oss << "val=" << vd->data() << ", grad=" << vd->grad() << "\"";
    return oss.str();
}

std::string ValueGraph::_getOpNodeAttrs(ValueData *vd) {
    std::ostringstream oss;
    oss << "label=\"" << vd->op() << "\"";
    return oss.str();
}

std::string ValueGraph::_valueNodeId(ValueData *vd) {
    std::ostringstream oss;
    oss << "value_node_" << vd;
    return oss.str();
}

std::string ValueGraph::_opNodeId(ValueData *vd) {
    std::ostringstream oss;
    oss << "op_node_" << vd;
    return oss.str();
}

/// Write DOT to an output stream, in a format that can be used by tools to generate a visualization.
std::ostream &operator<<(std::ostream &out, const ValueGraph &dot) {
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
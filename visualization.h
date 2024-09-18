//
// Created by Xiaofeng Li on 18/9/2024.
//

#ifndef MICROGRAD_VISUALIZATION_H
#define MICROGRAD_VISUALIZATION_H

#include "engine.h"

/// Util to visualize a expression graph.
class ValueGraph {
public:
    explicit ValueGraph(ValueData *root);
    explicit ValueGraph(Value &&root);

private:
    void _buildGraph(ValueData *root);

    static std::string _getValueNodeLabel(ValueData *vd);

    std::string _getOpNodeAttrs(ValueData *vd);

    std::string _valueNodeId(ValueData *vd);

    std::string _opNodeId(ValueData *vd);

    // node id -> node attributes (label, shape etc.)
    std::unordered_map<std::string, std::string> _nodes;
    // edges (node id 1 -> node id 2)
    std::vector<std::tuple<std::string, std::string>> _edges;

    std::unordered_set<ValueData *> _visited;

    friend std::ostream &operator<<(std::ostream &out, const ValueGraph &dot);
};

#endif //MICROGRAD_VISUALIZATION_H

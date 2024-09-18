#include <iostream>
#include <vector>
#include "engine.h"
#include "neuronet.h"

int main() {
    Vector2D xs = {
            {Value(2.0), Value(3.0),  Value(-1.0)},
            {Value(3.0), Value(-1.0), Value(0.5)},
            {Value(0.5), Value(1.0),  Value(1.0)},
            {Value(1.0), Value(1.0),  Value(-1.0)},
    };
    Vector ys = {
            Value(1.0),
            Value(-1.0),
            Value(-1.0),
            Value(1.0)
    };

    MLP mlp(3, {4, 4, 1});
    for (int i = 0; i < 500; ++i) {
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

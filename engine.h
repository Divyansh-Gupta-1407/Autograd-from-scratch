#pragma once
#include <vector>
#include <memory>
#include <functional>
#include <set>
#include <cmath>
#include <algorithm>

class Value {
public:
    double data;
    double grad;

    std::vector<std::shared_ptr<Value>> prev;
    std::function<void()> backward_fn;

    Value(double data);

    static std::shared_ptr<Value> add(
        std::shared_ptr<Value> a,
        std::shared_ptr<Value> b);

    static std::shared_ptr<Value> mul(
        std::shared_ptr<Value> a,
        std::shared_ptr<Value> b);

    static std::shared_ptr<Value> power(
        std::shared_ptr<Value> a,
        double exp);

    static std::shared_ptr<Value> relu(
        std::shared_ptr<Value> a);

    void backward();
};

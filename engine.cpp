#include "engine.h"

Value::Value(double data)
    : data(data), grad(0.0) {}

std::shared_ptr<Value> Value::add(
    std::shared_ptr<Value> a,
    std::shared_ptr<Value> b)
{
    auto out = std::make_shared<Value>(a->data + b->data);

    out->prev = {a, b};

    out->backward_fn = [a, b, out]() {
        a->grad += out->grad;
        b->grad += out->grad;
    };

    return out;
}

std::shared_ptr<Value> Value::mul(
    std::shared_ptr<Value> a,
    std::shared_ptr<Value> b)
{
    auto out = std::make_shared<Value>(a->data * b->data);

    out->prev = {a, b};

    out->backward_fn = [a, b, out]() {
        a->grad += b->data * out->grad;
        b->grad += a->data * out->grad;
    };

    return out;
}

std::shared_ptr<Value> Value::power(
    std::shared_ptr<Value> a,
    double exp)
{
    auto out = std::make_shared<Value>(
        std::pow(a->data, exp));

    out->prev = {a};

    out->backward_fn = [a, out, exp]() {
        a->grad += exp *
                   std::pow(a->data, exp - 1) *
                   out->grad;
    };

    return out;
}

std::shared_ptr<Value> Value::relu(
    std::shared_ptr<Value> a)
{
    auto out = std::make_shared<Value>(
        a->data < 0 ? 0 : a->data);

    out->prev = {a};

    out->backward_fn = [a, out]() {
        a->grad += (out->data > 0) * out->grad;
    };

    return out;
}

void Value::backward()
{
    std::vector<std::shared_ptr<Value>> topo;
    std::set<Value*> visited;

    std::function<void(std::shared_ptr<Value>)> build =
        [&](std::shared_ptr<Value> v)
    {
        if (visited.find(v.get()) == visited.end())
        {
            visited.insert(v.get());

            for (auto child : v->prev)
                build(child);

            topo.push_back(v);
        }
    };

    build(std::shared_ptr<Value>(this, [](Value*){}));

    grad = 1.0;

    for (auto it = topo.rbegin();
         it != topo.rend();
         ++it)
    {
        if ((*it)->backward_fn)
            (*it)->backward_fn();
    }
}

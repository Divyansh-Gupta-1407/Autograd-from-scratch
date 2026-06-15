#include "nn.h"

std::vector<std::shared_ptr<Value>>
Module::parameters()
{
    return {};
}

void Module::zero_grad()
{
    for (auto p : parameters())
        p->grad = 0.0;
}

Neuron::Neuron(int nin, bool nonlin)
{
    this->nonlin = nonlin;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(-1.0, 1.0);

    for (int i = 0; i < nin; i++)
        w.push_back(
            std::make_shared<Value>(dist(gen)));

    b = std::make_shared<Value>(0.0);
}

std::shared_ptr<Value> Neuron::forward(
    const std::vector<std::shared_ptr<Value>>& x)
{
    auto act = b;

    for (size_t i = 0; i < w.size(); i++)
    {
        act = Value::add(
            act,
            Value::mul(w[i], x[i]));
    }

    return nonlin ? Value::relu(act) : act;
}

std::vector<std::shared_ptr<Value>>
Neuron::parameters()
{
    auto params = w;
    params.push_back(b);
    return params;
}

Layer::Layer(
    int nin,
    int nout,
    bool nonlin)
{
    for (int i = 0; i < nout; i++)
        neurons.emplace_back(nin, nonlin);
}

std::vector<std::shared_ptr<Value>>
Layer::forward(
    const std::vector<std::shared_ptr<Value>>& x)
{
    std::vector<std::shared_ptr<Value>> out;

    for (auto& neuron : neurons)
        out.push_back(neuron.forward(x));

    return out;
}

std::vector<std::shared_ptr<Value>>
Layer::parameters()
{
    std::vector<std::shared_ptr<Value>> params;

    for (auto& neuron : neurons)
    {
        auto p = neuron.parameters();
        params.insert(
            params.end(),
            p.begin(),
            p.end());
    }

    return params;
}

MLP::MLP(
    int nin,
    const std::vector<int>& nouts)
{
    std::vector<int> sz;
    sz.push_back(nin);

    for (int n : nouts)
        sz.push_back(n);

    for (size_t i = 0; i < nouts.size(); i++)
    {
        layers.emplace_back(
            sz[i],
            sz[i + 1],
            i != nouts.size() - 1);
    }
}

std::vector<std::shared_ptr<Value>>
MLP::forward(
    std::vector<std::shared_ptr<Value>> x)
{
    for (auto& layer : layers)
        x = layer.forward(x);

    return x;
}

std::vector<std::shared_ptr<Value>>
MLP::parameters()
{
    std::vector<std::shared_ptr<Value>> params;

    for (auto& layer : layers)
    {
        auto p = layer.parameters();
        params.insert(
            params.end(),
            p.begin(),
            p.end());
    }

    return params;
}

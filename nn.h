#pragma once
#include "engine.h"
#include <vector>
#include <memory>
#include <random>

class Module {
public:
    virtual std::vector<std::shared_ptr<Value>>
    parameters();

    void zero_grad();
};

class Neuron : public Module {
private:
    std::vector<std::shared_ptr<Value>> w;
    std::shared_ptr<Value> b;
    bool nonlin;

public:
    Neuron(int nin, bool nonlin = true);

    std::shared_ptr<Value> forward(
        const std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>>
    parameters() override;
};

class Layer : public Module {
private:
    std::vector<Neuron> neurons;

public:
    Layer(int nin,
          int nout,
          bool nonlin = true);

    std::vector<std::shared_ptr<Value>> forward(
        const std::vector<std::shared_ptr<Value>>& x);

    std::vector<std::shared_ptr<Value>>
    parameters() override;
};

class MLP : public Module {
private:
    std::vector<Layer> layers;

public:
    MLP(int nin,
        const std::vector<int>& nouts);

    std::vector<std::shared_ptr<Value>> forward(
        std::vector<std::shared_ptr<Value>> x);

    std::vector<std::shared_ptr<Value>>
    parameters() override;
};

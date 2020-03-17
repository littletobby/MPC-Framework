//
// Created by tangdingyi on 2019/12/26.
//

#include "NN.h"

NN::NN() {
    tot = 0;
    cur = 0;
    adj.resize(MAX_NODE_NUM, vector<int>(0));
    neuron.resize(MAX_NODE_NUM);
    vst.resize(MAX_NODE_NUM);
    q.resize(MAX_NODE_NUM);
    to.resize(MAX_NODE_NUM);
}

NN& NN::operator=(NN &a) {
    for (int i = 1; i <= tot; i++) {
        *getNeuron(i)->getForward() = *a.getNeuron(i)->getForward();
        *getNeuron(i)->getGrad() = *a.getNeuron(i)->getGrad();
    }
    return *this;
}

void NN::global_variables_initializer() {
    for (int i = 1; i <= tot; i++) {
        if (neuron[i]->getForward() == nullptr) {
            neuron[i]->initForward();
        }
        if (neuron[i]->getGrad() == nullptr) {
            neuron[i]->initGrad();
        }
        if (neuron[i]->getIsNet()) {
            neuron[i]->setOpUpdate(new MathOp::Mul_Const_Trunc(getNeuron(i)->getGrad(), getNeuron(i)->getGrad(), 10));
        }
    }
    curForward = 1;
    curGrad = tot;
}

void NN::epoch_init() {
    for (int i = 1; i <= tot; i++) {
        if (!neuron[i]->getIsBack()) {
            neuron[i]->getGrad()->clear();
        }
    }
    for (int i = 1; i <= tot; i++) {
        neuron[i]->resetOp();
    }
    curForward = 1;
    curGrad = tot;
}

void NN::reveal_init(int u) {
    neuron[u]->setOpReveal(new MathOp::Reveal(getNeuron(u)->getForward(), getNeuron(u)->getForward()));
}

void NN::reveal_init() {
    for (int i = 1; i <= tot; i++) {
            neuron[i]->resetOp();
    }
}

void NN::addedge(int u, int v) {
    adj[u].push_back(v);
    to[v]++;
}

int NN::addnode(int r, int c, int k) {
    neuron[++tot] = new NeuronMat(r, c, k);
    return tot;
}

NeuronMat* NN::getNeuron(int u) {
    return neuron[u];
}

void NN::setOp(int u, Op *op) {
    neuron[u]->setOp(op);
}

void NN::addOpAdd_Mat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Add_Mat(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpMul_Mat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Mul_Mat(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpMeanSquaredLoss(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::MeanSquaredLoss(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpSimilar(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Similar(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpConcat(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Concat(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpHstack(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Hstack(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpVstack(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Vstack(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::addOpVia(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Via(getNeuron(res), getNeuron(a)));
}

void NN::addOpArgmax(int res, int a) {
    addedge(a, res);
    setOp(res, new MathOp::Argmax(getNeuron(res), getNeuron(a)));
}

void NN::addOpEqual(int res, int a, int b) {
    addedge(a, res);
    addedge(b, res);
    setOp(res, new MathOp::Equal(getNeuron(res), getNeuron(a), getNeuron(b)));
}

void NN::toposort() {
    int l, r;
    vst = vector<bool>(MAX_NODE_NUM, 0);
    l = r = 0;
    for (int i = 1; i <= tot; i++) {
        if (!to[i]) {
            q[++r] = i;
            vst[i] = 1;
        }
    }
    while (l < r) {
        int u = q[++l];
        int len = adj[u].size();
        for (int i = 0; i < len; i++) {
            int j = adj[u][i];
            to[j]--;
            if (!to[j]) {
                vst[j] = 1;
                q[++r] = j;
            }
        }
    }
    for (int i = 1; i <= tot; i++) {
        DBGprint("%d ", q[i]);
    }
    DBGprint("\n");
}

void NN::gradUpdate() {
    for (int i = 1; i <= tot; i++)
        neuron[i]->update();
}

bool NN::forwardHasNext() {
    return curForward < tot || neuron[q[curForward]]->forwardHasNext();
}

void NN::forwardNext() {
    while (!neuron[q[curForward]]->forwardHasNext()) {
        curForward++;
        if (curForward > tot)
            return;
    }
    neuron[q[curForward]]->forward();
}

bool NN::backHasNext() {
    return curGrad > 1 || neuron[q[curGrad]]->backHasNext();
}

void NN::backNext() {
    if (!backHasNext())
        return;
    if (neuron[q[curGrad]]->backHasNext()) {
        neuron[q[curGrad]]->back();
    }
    else {
        neuron[q[--curGrad]]->back();
    }
}

bool NN::updateHasNext() {
    for (int i = 1; i <= tot; i++) {
        if (neuron[i]->updateGradHasNext())
            return 1;
    }
    return 0;
}

void NN::update() {
    for (int i = 1; i <= tot; i++) {
        neuron[i]->update_grad();
    }
}

bool NN::revealHasNext() {
    for (int i = 1; i <= tot; i++) {
        if (neuron[i]->revealHasNext())
            return 1;
    }
    return 0;
}

void NN::reveal() {
    for (int i = 1; i <= tot; i++) {
            neuron[i]->reveal();
    }
}

int NN::getTot() {
    return tot;
}
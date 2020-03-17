//
// Created by tangdingyi on 2019/12/26.
//

#include "BPGraph.h"

BPGraph::LR::LR() {}

BPGraph::LR::LR(Mat *train_data, Mat *train_label, Mat *test_data, Mat *test_label) {
    DBGprint("LR constructor\n");
    nn = new NN();
    this->train_data = train_data;
    this->train_label = train_label;
    this->test_data = test_data;
    this->test_label = test_label;
}

void BPGraph::LR::graph() {
    int hidden_num = 1;
    int output_num = 1;
    input = nn->addnode(D+1, B, NeuronMat::NODE_INPUT);
    output = nn->addnode(output_num, B, NeuronMat::NODE_INPUT);
    st_w = nn->addnode(hidden_num, D+1, NeuronMat::NODE_NET);
    st_mul = nn->addnode(hidden_num, B, NeuronMat::NODE_OP);
    out_sig = nn->addnode(hidden_num, B, NeuronMat::NODE_OP);
    int sd = nn->addnode(1, 1, NeuronMat::NODE_OP);
    argmax = nn->addnode(1, 1, NeuronMat::NODE_OP);

    nn->global_variables_initializer();

    nn->addOpMul_Mat(st_mul, st_w, input);
    nn->addOpVia(out_sig, st_mul);
    nn->addOpMeanSquaredLoss(sd, output, out_sig);
    nn->addOpSimilar(argmax, output, out_sig);
    nn->toposort();

    nn->reveal_init(out_sig);
}

void BPGraph::LR::train() {
    Mat x_batch(D + 1, B), y_batch(1, B);
    clock_train = new Constant::Clock(CLOCK_TRAIN);
    globalRound = 0;
    for (int i = 0; i < 110000 && i < TRAIN_ITE; i++) {
        globalRound++;
        next_batch(x_batch, i * B, train_data, N);
        next_batch(y_batch, i * B, train_label, N);
        feed(nn, x_batch, y_batch, input, output);
        {
            nn->epoch_init();
            while (nn->forwardHasNext()) {
                nn->forwardNext();
            }
            while (nn->backHasNext()) {
                nn->backNext();
            }
            while (nn->updateHasNext()) {
                nn->update();
            }
            nn->gradUpdate();
        }
        if ((i+1)%PRINT_PRE_ITE == 0) {
            test();
            print_perd(i+1);
        }
    }
}

void BPGraph::LR::test() {
    Mat x_batch(D + 1, B), y_batch(1, B);
    int total = 0;
    for (int i = 0; i < NM / B; i++) {
        globalRound++;
        next_batch(x_batch, i * B, test_data);
        next_batch(y_batch, i * B, test_label);
        feed(nn, x_batch, y_batch, input, output);
        nn->epoch_init();
        while (nn->forwardHasNext()) {
            nn->forwardNext();
        }

        nn->reveal_init();
        while (nn->revealHasNext()) {
            nn->reveal();
        }

        total += nn->getNeuron(output)->getForward()->equal(*nn->getNeuron(out_sig)->getForward()).count();\
    }
    DBGprint("accuracy: %f\n", total * 1.0 / (NM / B * B));
}

void BPGraph::LR::feed(NN* nn, Mat &x_batch, Mat &y_batch, int input, int output) {
    *nn->getNeuron(input)->getForward() = x_batch;
    *nn->getNeuron(output)->getForward() = y_batch;
}

void BPGraph::LR::next_batch(Mat &batch, int start, Mat *A, int mod) {
    A->col(start%mod, start%mod + B, batch);
}

void BPGraph::LR::print_perd(int round) {
    ll tot_send = 0, tot_recv = 0;
    for (int i = 0; i < M; i++) {
        if (node_type != i) {
            tot_send += socket_io[node_type][i]->send_num;
            tot_recv += socket_io[node_type][i]->recv_num;
        }
    }
    DBGprint("round: %d tot_time: %.3f ",
             round, clock_train->get());
    DBGprint("tot_send: %lld tot_recv: %lld\n", tot_send, tot_recv);
}
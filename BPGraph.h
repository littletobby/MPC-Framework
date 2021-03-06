//
// Created by tangdingyi on 2019/12/26.
//

#ifndef MPC_ML_BPGRAPH_H
#define MPC_ML_BPGRAPH_H

#include "Constant.h"
#include "NN.h"
#include "Player.h"

extern int node_type;

class BPGraph {
public:
    class LR {
        NN* nn;
        Constant::Clock *clock_train;
        Mat *train_data, *train_label;
        Mat *test_data, *test_label;
        int input, output;
        int argmax, st_con, st_w, st_b, st_mul;
        int sd;
        int re_st_add, re_output;
        int out_sig, re_out_sig;
        int id;
    public:
        LR();
        LR(Mat* train_data, Mat* train_label, Mat* test_data, Mat* test_label);
        void train();
        void test();
        void feed(NN* nn, Mat& x_batch, Mat& y_batch, int input, int output);
        void next_batch(Mat& batch, int start, Mat* A, int mod = NM);
        void graph();
        void print_perd(int round);
    };
};


#endif //MPC_ML_BPGRAPH_H

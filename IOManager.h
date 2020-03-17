//
// Created by tangdingyi on 2019/12/26.
//

#ifndef MPC_ML_IOMANAGER_H
#define MPC_ML_IOMANAGER_H

#include "Mat.h"

class IOManager {
public:
    static Mat train_data, train_label;
    static Mat test_data, test_label;
    static void load(ifstream& in, Mat& data, Mat& label, int size);
    static void init();
};


#endif //MPC_ML_IOMANAGER_H

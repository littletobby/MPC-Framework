//
// Created by tangdingyi on 2019/12/26.
//

#include "IOManager.h"

//Mat train_data(N,D), train_label(N,1);
//Mat test_data(NM,D), test_label(NM,1);

Mat IOManager::train_data = Mat(D+1, N + B - 1);
Mat IOManager::train_label = Mat(1, N + B - 1);
Mat IOManager::test_data = Mat(D+1, NM + B - 1);
Mat IOManager::test_label = Mat(1, NM + B - 1);


void IOManager::load(ifstream &in, Mat &data, Mat &label, int size) {
    int i=0;
    while(in){

        string s;
        if (!getline(in,s))
            break;
        char* ch;
        ch = const_cast<char *>(s.c_str());
        int temp;
        char c;

        temp = Constant::Util::getint(ch);
        if (temp > 1)
            temp = 1;
        label(0, i) = temp * IE;

//        if (i == 3034)
//            printf("%d: %d\n", i, temp);

        int nd = min(D, ND);
        data(nd, i) = IE;
        for(int j=0;j<nd;j++){
            temp = Constant::Util::getint(ch);
            data(j, i) = temp * IE / 256;
//            if (!i) {
//                printf("%d ", temp);
//                data(i, j).print();
//                DBGprint(" ");
//            }
        }
//        if (!i)
//            printf("\n");

        i++;
        if (i >= size)
            break;
//        if (i >= 5)
//            break;
//            printf("%d ", i);
//        DBGprint("%d ", i);
    }
//    cout<<"n= "<<i<<endl;
    for (i; i < size + B - 1; i++) {
        int tmp_r;
        tmp_r = data.rows();
        for (int j = 0; j < tmp_r; j++) {
            data(j, i) = data(j, i - size);
        }
        tmp_r = label.rows();
        for (int j = 0; j < tmp_r; j++) {
            label(j, i) = label(j, i - size);
        }
    }
    DBGprint("n=%d\n", i);
}

void IOManager::init() {
    ifstream infile( "mnist/mnist_train.csv" );

//    cout<<"load training data.......\n";
    DBGprint("load training data......\n");

    load(infile, train_data, train_label, N);
    infile.close();
    ifstream intest( "mnist/mnist_test.csv" );
    load(intest, test_data, test_label, NM);
    intest.close();
//    train_data.reoeder();
//    train_label.reoeder();
//    test_data.reoeder();
//    test_label.reoeder();
}
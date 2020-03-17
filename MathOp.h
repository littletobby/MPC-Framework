//
// Created by tangdingyi on 2019/12/26.
//

#ifndef MPC_ML_MATHOP_H
#define MPC_ML_MATHOP_H

#include "Op.h"
#include "NeuronMat.h"
#include "SocketOnline.h"
#include "Player.h"

extern int node_type;
extern SocketOnline *socket_io[M][M];
extern Player player[M];
extern Mat metadata;

class MathOp {
public:
    class PRandFld;
    class MulPub;
    class PRandBit;
    class PRandM;
    class Reveal;
    class Div2mP;
    class DegRed;
    class RevealD;
    class Add_Mat: public Op {
        NeuronMat *res, *a, *b;
    public:
        Add_Mat();
        Add_Mat(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Mul_Mat: public Op {
        NeuronMat *res, *a, *b;
        Div2mP *div2mP_f;
        Div2mP *div2mP_b_a, *div2mP_b_b;
    public:
        Mul_Mat();
        Mul_Mat(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Mul_Const_Trunc: public Op {
        Mat *res, *a;
        ll128 b;
        Div2mP *div2mP;
    public:
        Mul_Const_Trunc();
        Mul_Const_Trunc(Mat* res, Mat* a, ll128 b);
        void forward();
        void back();
    };
    class Via: public Op {
        NeuronMat *res, *a;
    public:
        Via();
        Via(NeuronMat* res, NeuronMat* a);
        void forward();
        void back();
    };
    class MeanSquaredLoss: public Op {
        NeuronMat *res, *a, *b;
    public:
        MeanSquaredLoss();
        MeanSquaredLoss(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Similar: public Op {
        NeuronMat *res, *a, *b;
    public:
        Similar();
        Similar(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Concat: public Op {
        NeuronMat *res, *a, *b;
    public:
        Concat();
        Concat(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Hstack: public Op {
        NeuronMat *res, *a, *b;
    public:
        Hstack();
        Hstack(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Vstack: public Op {
        NeuronMat *res, *a, *b;
    public:
        Vstack();
        Vstack(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    class Div2mP: public Op {
        Mat *res, *a;
        Mat *r_nd, *r_st, *r_B;
        Mat *r;
        Mat *b;
        int k, m;
        PRandM *pRandM;
        RevealD *pRevealD;
    public:
        Div2mP();
        Div2mP(Mat* res, Mat* a, int k, int m);
        void forward();
        void back();
    };
    class Reveal: public Op {
        Mat *res, *a;
        Mat *b;
    public:
        Reveal();
        Reveal(Mat* res, Mat* a);
        void forward();
        void back();
    };
    class PRandM: public Op {
        Mat *r_nd, *r_st, *b_B;
        int k, m;
        PRandFld *pRandFld;
        PRandBit **pRandBit;
    public:
        PRandM();
        PRandM(Mat* r_nd, Mat* r_st, Mat* b_B, int k, int m);
        PRandM(Mat* r_nd, Mat* r_st, int k, int m);
        PRandM(int r, int c, int k, int m);
        void PRandM_init(Mat* r_nd, Mat* r_st, Mat* b_B, int k, int m);
        void forward();
        void back();
    };
    class PRandBit: public Op {
        Mat *res;
        PRandFld *pRandFld;
        MulPub *mulPub;
        Mat *a, *a2;
        Mat *a_r, *a2_r;
    public:
        PRandBit();
        PRandBit(Mat* res);
        void forward();
        void back();
    };
    class MulPub: public Op {
        Mat *res, *a, *b;
    public:
        MulPub();
        MulPub(Mat* res, Mat* a, Mat* b);
        void forward();
        void back();
    };
    class PRandFld: public Op {
        Mat *res;
        ll128 range;
    public:
        PRandFld();
        PRandFld(Mat* res, ll range);
        void forward();
        void back();
    };
    class DegRed: public Op {
        Mat *res, *a;
        Mat *tmp;
    public:
        DegRed();
        DegRed(Mat* res, Mat* a);
        void forward();
        void back();
    };
    class RevealD: public Op {
        Mat *res, *a, *b;
        Reveal *pReveal;
        DegRed *pDegRed;
    public:
        RevealD();
        RevealD(Mat* res, Mat* a, Mat* b);
        void forward();
        void back();
    };
    class Argmax: public Op {
        NeuronMat *res, *a;
    public:
        Argmax();
        Argmax(NeuronMat* res, NeuronMat* a);
        void forward();
        void back();
    };
    class Equal: public Op {
        NeuronMat *res, *a, *b;
    public:
        Equal();
        Equal(NeuronMat* res, NeuronMat* a, NeuronMat* b);
        void forward();
        void back();
    };
    static void broadcast(Mat* a);
    static void broadcase_rep(Mat* a);
    static void receive(Mat* a);
    static void receive_add(Mat* a);
    static void receive_rep(Mat* a);
    static void random(Mat* a, ll range);
};


#endif //MPC_ML_MATHOP_H

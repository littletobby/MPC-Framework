//
// Created by tangdingyi on 2019/12/26.
//

#include "MathOp.h"

MathOp::Add_Mat::Add_Mat() {}

MathOp::Add_Mat::Add_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Add_Mat::forward() {
    reinit();
    *res->getForward() = (*a->getForward()) + (*b->getForward());
}

void MathOp::Add_Mat::back() {
    backRound++;
    if (!a->getIsBack())
        *a->getGrad() = (*a->getGrad()) + (*res->getGrad());
    if (!b->getIsBack())
        *b->getGrad() = (*b->getGrad()) + (*res->getGrad());
}

MathOp::Mul_Mat::Mul_Mat() {}

MathOp::Mul_Mat::Mul_Mat(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    div2mP_f = new Div2mP(res->getForward(), res->getForward(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_a = new Div2mP(a->getGrad(), a->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    div2mP_b_b = new Div2mP(b->getGrad(), b->getGrad(), BIT_P_LEN, DECIMAL_PLACES);
    init(2, 2);
}

void MathOp::Mul_Mat::forward() {
    reinit();
    switch (forwardRound) {
        case 1: {
            *res->getForward() = (*a->getForward()) * (*b->getForward());
        }
            break;
        case 2: {
            div2mP_f->forward();
            if (div2mP_f->forwardHasNext()) {
                forwardRound--;
            }
        }
            break;
    }
}

void MathOp::Mul_Mat::back() {
    backRound++;
    switch (backRound) {
        case 1: {
            if (!a->getIsBack()) {
                b->getForward()->transorder();
                *a->getGrad() += (*res->getGrad()) * (*b->getForward());
                b->getForward()->transorder();
            }
            if (!b->getIsBack()) {
                a->getForward()->transorder();
                *b->getGrad() += (*a->getForward()) * (*res->getGrad());
                a->getForward()->transorder();
            }
        }
            break;
        case 2:
            if (!a->getIsBack()) {
                div2mP_b_a->forward();
            }
            if (!b->getIsBack()) {
                div2mP_b_b->forward();
            }
            if ((!a->getIsBack() && div2mP_b_a->forwardHasNext()) ||
                (!b->getIsBack() && div2mP_b_b->forwardHasNext())) {
                backRound--;
            }
            break;
    }
}

MathOp::Mul_Const_Trunc::Mul_Const_Trunc() {}

MathOp::Mul_Const_Trunc::Mul_Const_Trunc(Mat *res, Mat *a, ll128 b) {
    this->res = res;
    this->a = a;
    this->b = b;
    div2mP = new Div2mP(res, res, BIT_P_LEN, DECIMAL_PLACES);
    init(2, 0);
}

void MathOp::Mul_Const_Trunc::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            *res = (*a) * b;
            break;
        case 2:
            div2mP->forward();
            if (div2mP->forwardHasNext()) {
                forwardRound--;
            }
            break;
    }
}

void MathOp::Mul_Const_Trunc::back() {}

MathOp::Via::Via() {}

MathOp::Via::Via(NeuronMat *res, NeuronMat *a) {
    this->res = res;
    this->a = a;
    init();
}

void MathOp::Via::forward() {
    reinit();
    *res->getForward() = *a->getForward();
}

void MathOp::Via::back() {
    backRound++;
    if (!a->getIsBack())
        *a->getGrad() = (*a->getGrad()) + (*res->getGrad());
}

MathOp::MeanSquaredLoss::MeanSquaredLoss() {}

MathOp::MeanSquaredLoss::MeanSquaredLoss(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init(0, 1);
}

void MathOp::MeanSquaredLoss::forward() {}

void MathOp::MeanSquaredLoss::back() {
    backRound++;
    *b->getGrad() = (*b->getGrad()) + (*a->getForward()) - (*b->getForward());
}

MathOp::Similar::Similar() {}

MathOp::Similar::Similar(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init(1, 0);
}

void MathOp::Similar::forward() {
    reinit();
    res->getForward()->operator()(0, 0) = a->getForward()->equal(*b->getForward()).count();
}

void MathOp::Similar::back() {}

MathOp::Concat::Concat() {}

MathOp::Concat::Concat(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Concat::forward() {
    reinit();
    Mat::concat(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Concat::back() {
    backRound++;
    Mat::reconcat(res->getGrad(), a->getGrad(), !a->getIsBack(), b->getGrad(), !b->getIsBack());
}

MathOp::Hstack::Hstack() {}

MathOp::Hstack::Hstack(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Hstack::forward() {
    reinit();
    Mat::hstack(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Hstack::back() {
    backRound++;
    Mat::re_hstack(res->getGrad(), a->getGrad(), !a->getIsBack(), b->getGrad(), !b->getIsBack());
}

MathOp::Vstack::Vstack() {}

MathOp::Vstack::Vstack(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init();
}

void MathOp::Vstack::forward() {
    reinit();
    Mat::concat(res->getForward(), a->getForward(), b->getForward());
}

void MathOp::Vstack::back() {
    backRound++;
    Mat::reconcat(res->getBack(), a->getBack(), !a->getIsBack(), b->getBack(), !b->getIsBack());
}

MathOp::Div2mP::Div2mP() {}

MathOp::Div2mP::Div2mP(Mat *res, Mat *a, int k, int m) {
    this->res = res;
    this->a = a;
    this->k = k;
    this->m = m;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    r_nd = new Mat(tmp_r, tmp_c);
    r_st = new Mat(tmp_r, tmp_c);
    r_B = new Mat[m];
    for (int i = 0; i < m; i++) {
        r_B[i].init(tmp_r, tmp_c);
    }
    r = new Mat(tmp_r, tmp_c);
    b = new Mat(tmp_r, tmp_c);
    pRandM = new PRandM(r_nd, r_st, r_B, k, m);
    pRevealD = new RevealD(b, a, r);
    init(4, 0);
}

void MathOp::Div2mP::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
//            pRandM->forward();
//            if (pRandM->forwardHasNext()) {
//                forwardRound--;
//            }
            break;
        case 2:
            *r = *r_nd * (1ll << m) + *r_st;
            *r = *a + (1ll << BIT_P_LEN - 1) + *r;
            break;
        case 3:
            pRevealD->forward();
            if (pRevealD->forwardHasNext()) {
                forwardRound--;
            }
            break;
        case 4:
            *b = b->mod(1ll << m);
            *res = (*a - (*b - *r_st)) * Constant::Util::inverse(1ll << m, MOD);
            break;
    }
}

void MathOp::Div2mP::back() {}

MathOp::Reveal::Reveal() {}

MathOp::Reveal::Reveal(Mat *res, Mat *a) {
    this->res = res;
    this->a = a;
    int tmp_r, tmp_c;
    tmp_r = a->rows();
    tmp_c = a->cols();
    b = new Mat(tmp_r, tmp_c);
    init(2, 0);
}

void MathOp::Reveal::forward() {
    reinit();
    Mat tmp[M];
    switch (forwardRound) {
        case 1:
            *b = *a * player[node_type].lagrange;
            broadcase_rep(b);
            break;
        case 2:
            *b = *a * player[node_type].lagrange;
            receive_add(b);
            *res = *b;
            break;
    }
}

void MathOp::Reveal::back() {}

MathOp::PRandM::PRandM() {}

MathOp::PRandM::PRandM(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m) {
    PRandM_init(r_nd, r_st, b_B, k, m);
}

MathOp::PRandM::PRandM(Mat *r_nd, Mat *r_st, int k, int m) {
    int tmp_r, tmp_c;
    tmp_r = r_nd->rows();
    tmp_c = r_nd->cols();
    b_B = new Mat[m];
    for (int i = 0; i < m; i++) {
        b_B[i].init(tmp_r, tmp_c);
    }
    PRandM_init(r_nd, r_st, b_B, k, m);
}

MathOp::PRandM::PRandM(int r, int c, int k, int m) {
    r_nd = new Mat(r, c);
    r_st = new Mat(r, c);
    b_B = new Mat[m];
    for (int i = 0; i < m; i++) {
        b_B[i].init(r, c);
    }
    PRandM_init(r_nd, r_st, b_B, k, m);
}

void MathOp::PRandM::PRandM_init(Mat *r_nd, Mat *r_st, Mat *b_B, int k, int m) {
    this->r_nd = r_nd;
    this->r_st = r_st;
    this->b_B = b_B;
    this->k = k;
    this->m = m;
    pRandFld = new PRandFld(r_nd, 1ll << k - m);
    pRandBit = new PRandBit *[m];
    for (int i = 0; i < m; i++) {
        pRandBit[i] = new PRandBit(b_B+i);
    }
    init(3, 0);
}

void MathOp::PRandM::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            pRandFld->forward();
            if (pRandFld->forwardHasNext()) {
                forwardRound--;
            }
            break;
        case 2:
            for (int i = 0; i < m; i++) {
                pRandBit[i]->forward();
            }
            for (int i = 0; i < m; i++) {
                if (pRandBit[i]->forwardHasNext()) {
                    forwardRound--;
                    break;
                }
            }
            break;
        case 3:
            for (int i = 0; i < m; i++) {
                *r_st = *r_st + b_B[i] * (1ll << i);
            }
            break;
    }
}

void MathOp::PRandM::back() {}

MathOp::PRandBit::PRandBit() {}

MathOp::PRandBit::PRandBit(Mat *res) {
    this->res = res;
    int tmp_r, tmp_c;
    tmp_r = res->rows();
    tmp_c = res->cols();
    a = new Mat(tmp_r, tmp_c);
    a_r = new Mat(1, tmp_r * tmp_c + REDUNDANCY);
    a2 = new Mat(tmp_r, tmp_c);
    a2_r = new Mat(1, tmp_r * tmp_c + REDUNDANCY);
    pRandFld = new PRandFld(a_r, MOD);
    mulPub = new MulPub(a2_r, a_r, a_r);
    init(3, 0);
}

void MathOp::PRandBit::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            pRandFld->forward();
            if (pRandFld->forwardHasNext()) {
                forwardRound--;
            }
            break;
        case 2:
            mulPub->forward();
            if (mulPub->forwardHasNext()) {
                forwardRound--;
            }
            break;
        case 3:
            if (a2_r->count() > REDUNDANCY) {
                forwardRound = 0;
                break;
            }
            Mat::fill(a2, a2_r, a, a_r);
            *a2 = a2->sqrt_inv();
            *a2 = a2->dot(*a) + 1;
            *res = a2->divideBy2();
            break;
    }
}

void MathOp::PRandBit::back() {}

MathOp::MulPub::MulPub() {}

MathOp::MulPub::MulPub(Mat *res, Mat *a, Mat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init(2, 0);
}

void MathOp::MulPub::forward() {
    reinit();
    Mat tmp[M];
    switch (forwardRound) {
        case 1:
            *res = a->dot(*b);
            *res = *res * player[node_type].lagrange;
            for (int i = 0; i < M; i++) {
                if (i != node_type) {
                    tmp[i] = *res;
                }
            }
            broadcast(tmp);
            break;
        case 2:
            receive(tmp);
            for (int i = 0; i < M; i++) {
                if (i != node_type) {
                    *res = *res + tmp[i];
                }
            }
            break;
    }
}

void MathOp::MulPub::back() {}

MathOp::PRandFld::PRandFld() {}

MathOp::PRandFld::PRandFld(Mat *res, ll range) {
    this->res = res;
    this->range = range;
    init(2, 0);
}

void MathOp::PRandFld::forward() {
    reinit();
    Mat a[M];
    switch (forwardRound) {
        case 1:
            int r, c;
            r = res->rows();
            c = res->cols();
            for (int i = 0; i < M; i++) {
                a[i].init(r, c);
            }
            random(a, range);
            *res = a[node_type];
            broadcast(a);
            break;
        case 2:
            receive(a);
            for (int i = 0; i < M; i++) {
                if (i != node_type) {
                    *res = *res + a[i];
                }
            }
            break;
    }
}

void MathOp::PRandFld::back() {}

MathOp::DegRed::DegRed() {}

MathOp::DegRed::DegRed(Mat *res, Mat *a) {
    this->res = res;
    this->a = a;
    tmp = new Mat[M];
    init(2, 0);
}

void MathOp::DegRed::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            for (int i = 0; i < M; i++) {
                if (i == node_type) {
                    continue;
                }
                tmp[i] = *a * metadata(node_type, i);
            }
            broadcast(tmp);
            break;
        case 2:
            *res = *a * metadata(node_type, node_type);
            receive(tmp);
            for (int i = 0; i < M; i++) {
                if (i != node_type) {
                    *res = *res + tmp[i];
                }
            }
    }
}

void MathOp::DegRed::back() {}

MathOp::RevealD::RevealD() {}

MathOp::RevealD::RevealD(Mat *res, Mat *a, Mat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    pReveal = new Reveal(res, b);
    pDegRed = new DegRed(a, a);
    init(1, 0);
}

void MathOp::RevealD::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            pReveal->forward();
            pDegRed->forward();
            if (pReveal->forwardHasNext() || pDegRed->forwardHasNext()) {
                forwardRound--;
            }
            break;
    }
}

void MathOp::RevealD::back() {}

MathOp::Argmax::Argmax() {}

MathOp::Argmax::Argmax(NeuronMat *res, NeuronMat *a) {
    this->res = res;
    this->a = a;
    init(1, 0);
}

void MathOp::Argmax::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            *res->getForward() = a->getForward()->argmax();
            break;
    }
}

void MathOp::Argmax::back() {}

MathOp::Equal::Equal() {}

MathOp::Equal::Equal(NeuronMat *res, NeuronMat *a, NeuronMat *b) {
    this->res = res;
    this->a = a;
    this->b = b;
    init(1, 0);
}

void MathOp::Equal::forward() {
    reinit();
    switch (forwardRound) {
        case 1:
            (*res->getForward())(0, 0) = a->getForward()->eq(*b->getForward()).count();
            break;
    }
}

void MathOp::Equal::back() {}

void MathOp::broadcast(Mat *a) {
    for (int i = 0; i < M; i++) {
        if (i != node_type) {
            socket_io[node_type][i]->send_message(a[i]);
        }
    }
}

void MathOp::broadcase_rep(Mat *a) {
    for (int i = 0; i < M; i++) {
        if (i != node_type) {
            socket_io[node_type][i]->send_message(a);
        }
    }
}

void MathOp::receive(Mat* a) {
    for (int i = 0; i < M; i++) {
        if (i != node_type) {
            a[i] = socket_io[node_type][i]->recv_message();
        }
    }
}

void MathOp::receive_add(Mat *a) {
    for (int i = 0; i < M; i++) {
        if (i != node_type) {
            socket_io[node_type][i]->recv_message(a);
        }
    }
}

void MathOp::receive_rep(Mat *a) {
    for (int i = 0; i < M; i++) {
        if (i != node_type) {
            socket_io[node_type][i]->recv_message(*(a+i));
        }
    }
}

void MathOp::random(Mat *a, ll range) {
    int len = a[0].rows() * a[0].cols();
    ll128 coefficient[TN];
    for (int i = 0; i < len; i++) {
        coefficient[0] = (Constant::Util::randomlong() % range);
        for (int j = 1; j < TN; j++) {
            coefficient[j] = Constant::Util::randomlong();
        }
        for (int j = 0; j < M; j++) {
            ll128 tmp = coefficient[0];
            ll128 key = player[j].key;
            for (int k = 1; k < TN; k++) {
                tmp += coefficient[k] * key;
                key *= player[j].key;
                key = Constant::Util::get_residual(key);
            }
            a[j].getVal(i) = tmp;
        }
    }
}
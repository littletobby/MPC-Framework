//
// Created by tangdingyi on 2019/12/25.
//

#include "Mat.h"

double generateGaussianNoise(double mu, double sigma)
{
    static const double epsilon = std::numeric_limits<double>::min();
    static const double two_pi = 2.0*3.14159265358979323846;

    thread_local double z1;
    thread_local bool generate;
    generate = !generate;

    if (!generate)
        return z1 * sigma + mu;

    double u1, u2;
    do
    {
        u1 = rand() * (1.0 / RAND_MAX);
        u2 = rand() * (1.0 / RAND_MAX);
    }
    while ( u1 <= epsilon );

    double z0;
    z0 = sqrt(-2.0 * log(u1)) * cos(two_pi * u2);
    z1 = sqrt(-2.0 * log(u1)) * sin(two_pi * u2);
    return z0 * sigma + mu;
}

Mat::Mat(int r, int c) {
    this->r = r;
    this->c = c;
    order = 0;
    val.resize(r * c);
}

Mat::Mat(int r, int c, ll128 b) {
    this->r = r;
    this->c = c;
    order = 0;
    val.resize(r * c, b);
}

Mat::Mat() {
    order = 0;
}

Mat::Mat(const Mat &a) {
    r = a.rows();
    c = a.cols();
    order = a.order;
    val.resize(r * c);
    val.assign(a.val.begin(), a.val.end());
}

Mat::~Mat() {
    order = 0;
}

void Mat::init(int r, int c) {
    this->r = r;
    this->c = c;
    order = 0;
    val.resize(r * c);
}

ll128& Mat::operator()(int a, int b) {
    return val[b * r + a];
}

ll128 Mat::get(int a, int b) const{
    return val[b * r + a];
}

ll128& Mat::getVal(int a) {
    return val[a];
}

int Mat::rows() const{
    return r;
}

int Mat::cols() const{
    return c;
}

int Mat::size() const {
    return r * c;
}

Mat& Mat::operator=(const Mat &a) {
    r = a.rows();
    c = a.cols();
    order = a.order;
    val.resize(r * c);
    int l = r * c;
    val.assign(a.val.begin(), a.val.end());
    return *this;
}

Mat& Mat::operator=(vector<ll128> &a) {
    int l = a.size();
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            this->operator()(i, j) = a[l];
    return *this;
}

Mat& Mat::operator=(char* &p) {
    int pr = Constant::Util::char_to_int(p);
    int pc = Constant::Util::char_to_int(p);
    this->r = pr;
    this->c = pc;
    order = 0;
    val.resize(r * c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = Constant::Util::char_to_ll(p);
    }
    return *this;
}

Mat Mat::transpose() const{
    Mat ret(c, r);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            ret.val[i * c + j] = val[j * r + i];
        }
    }
    return ret;
}

Mat Mat::operator+(const Mat &a) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] + a.val[i];
        ret.val[i] = ret.val[i] >= MOD ? ret.val[i] - MOD : ret.val[i];
        ret.val[i] = ret.val[i] <= -MOD ? ret.val[i] + MOD : ret.val[i];
    }
    return ret;
}

void Mat::operator+=(const Mat &a) {
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] += a.val[i];
        val[i] = val[i] >= MOD ? val[i] - MOD : val[i];
        val[i] = val[i] <= -MOD ? val[i] + MOD : val[i];
    }
}

Mat Mat::operator+(const ll128 &b) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] + b;
        ret.val[i] = ret.val[i] >= MOD ? ret.val[i] - MOD : ret.val[i];
        ret.val[i] = ret.val[i] <= -MOD ? ret.val[i] + MOD : ret.val[i];
    }
    return ret;
}


Mat Mat::operator-(const Mat &a) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] - a.val[i];
        ret.val[i] = ret.val[i] > MOD ? ret.val[i] - MOD : ret.val[i];
        ret.val[i] = ret.val[i] < -MOD ? ret.val[i] + MOD : ret.val[i];
    }
    return ret;
}

Mat Mat::operator-(ll128 b) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] - b;
        ret.val[i] = ret.val[i] > MOD ? ret.val[i] - MOD : ret.val[i];
        ret.val[i] = ret.val[i] < -MOD ? ret.val[i] + MOD : ret.val[i];
    }
    return ret;
}

Mat Mat::operator*(const Mat &a) {
    int tmp_c;
    tmp_c = a.cols();
    Mat ret(r, a.cols());
    int i, j;
    for (int j = 0; j < tmp_c; j++) {
        for (int i = 0; i < r; i++) {
            switch (pair_order_type(this, &a)) {
                case MM_NN:
                    for (int k = 0; k < c; k++) {
                        ret.val[j * r + i] += val[k * r + i] * a.val[j * c + k];
                    }
                    break;
                case MM_NT:
                    for (int k = 0; k < c; k++) {
                        ret.val[j * r + i] += val[k * r + i] * a.val[k * tmp_c + j];
                    }
                    break;
                case MM_TN:
                    for (int k = 0; k < c; k++) {
                        ret.val[j * r + i] += val[i * c + k] * a.val[j * c + k];
                    }
                    break;
                case MM_TT:
                    for (int k = 0; k < c; k++) {
                        ret.val[j * r + i] += val[i * c + k] * a.val[k * tmp_c + j];
                    }
                    break;
            }
        }
    }
    ret.residual();
    return ret;
}

Mat Mat::operator*(const ll128 &b) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] * b;
    }
    ret.residual();
    return ret;
}

Mat Mat::operator/(const ll128 &b) {
    Mat ret;
    ret = *this;
    ret.sign();
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            ret(i, j) = ret(i, j) / b;
        }
    ret.residual();
    return ret;
}

Mat Mat::operator<<(int b) const{
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] << b;
    }
    ret.residual();
    return ret;
}

Mat Mat::operator>>(int b) const{
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] > MOD / 2 ? val[i] - MOD >> b : val[i] >> b;
    }
    return ret;
}

Mat Mat::operator&(int b) const {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] & b;
    }
    return ret;
}

Mat Mat::operator&(const Mat &a) const {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] & a.val[i];
    }
    return ret;
}

Mat Mat::oneMinus() {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = 1 - val[i];
    }
    ret.residual();
    return ret;
}

Mat Mat::oneMinus_IE() {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = IE - val[i];
    }
    ret.residual();
    return ret;
}

Mat Mat::dot(const Mat &a) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] * a.val[i];
    }
    ret.residual();
    return ret;
}

vector<ll128> Mat::row(int a) {
    vector<ll128> ret(c);
    for (int i = 0; i < c; i++) {
        ret[i] = get(a, i);
    }
    return ret;
}

void Mat::init_row(vector<ll128> a, int b) {
    for (int i = 0; i < c; i++) {
        operator()(b, i) = a[i];
    }
}

Mat Mat::resize(int a, int b) {
    Mat ret(a, b);
    vector<ll128> temp;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            temp.push_back(get(i, j));
    int l = 0;
    for (int i = 0; i < a; i++)
        for (int j = 0; j < b; j++) {
            ret(i, j) = temp[l++];
        }
    return ret;
}

Mat Mat::argmax() {
    Mat ret(1, c);
    for (int i = 0; i < c; i++) {
        int k = 0;
        for (int j = 1; j < r; j++)
            if (Constant::Util::get_sign(get(j, i)) > Constant::Util::get_sign(get(k, i)))
                k = j;
        ret(0, i) = k;
    }
    return ret;
}

Mat Mat::equal(const Mat &a) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ll128 tmp = val[i] - a.val[i];
        tmp = (tmp % MOD + MOD) % MOD;
        tmp = tmp > MOD / 2 ? tmp - MOD : tmp;
        tmp = tmp < 0 ? -tmp : tmp;
        ret.val[i] = tmp < IE / 2 ? 1 : 0;
    }
    return ret;
}

Mat Mat::eq(const Mat &a) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = (val[i] == a.val[i]) ? 1 : 0;
    }
    return ret;
}

Mat Mat::getSign() {
    Mat ret;
    ret = *this;
    ret.sign();
    return ret;
}

Mat Mat::sqrt() {
    Mat ret(r, c);
    int l = r * c;
        for (int i = 0; i < l; i++) {
        ret.val[i] = Constant::Util::sqrt(val[i]);
    }
    return ret;
}

Mat Mat::inverse() {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = Constant::Util::inverse(val[i], MOD);
    }
    return ret;
}

Mat Mat::sqrt_inv() {
    Mat ret(r, c, 1);
    Mat a(*this);
    int l = r * c;
    ll b = Constant::SQRTINV;
    while (b != 0) {
        if (b & 1) {
            for (int i = 0; i < l; i++) {
                ret.val[i] = ret.val[i] * a.val[i];
                ret.val[i] %= MOD;
            }
        }
        for (int i = 0; i < l; i++) {
            a.val[i] *= a.val[i];
            a.val[i] %= MOD;
        }
        b >>= 1;
    }
    ret.residual();
    return ret;
}

Mat Mat::divideBy2() {
    Mat ret(r, c);
    ll128 inv2 = Constant::inv2;
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] * inv2;
    }
    ret.residual();
    return ret;
}

Mat Mat::relu() {
    Mat ret(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            ret(i, j) = Constant::Util::get_sign(get(i, j)) > 0 ? this->operator()(i, j) : 0;
    return ret;
}

Mat Mat::d_relu() {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = Constant::Util::get_sign(val[i]) > 0 ? 1 : 0;
    }
    return ret;
}

Mat Mat::sigmoid() const {
    Mat ret(r, c);
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            ll128 u = this->get(i, j);
            u = Constant::Util::get_residual(u);
            ll128 u_add = u + IE / 2;
            ll128 u_sub = u - IE / 2;
            bool p, q;
            p = u < MOD / 2;
            q = u > MOD - IE / 2;
            ret(i, j) = u_add * (!(!p & !q) ? 1 : 0);
            q = u > IE / 2;
            ret(i, j) -= u_sub * ((p & q) ? 1 : 0);
            ret(i, j) = Constant::Util::get_residual(ret(i, j));
        }
    return ret;
}

Mat Mat::not_eqz() const {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] != 0 ? 1 : 0;
    }
    return ret;
}

Mat Mat::row(int st, int ed) const {
    Mat ret((ed - st + r) % r, c);
    if (order == MatColMajor || 1) {
        if (st < ed) {
            int tmp_r = ed - st;
            for (int i = 0; i < c; i++) {
                copy(val.begin() + i * r + st, val.begin() + i * r + st + tmp_r, ret.val.begin() + i * tmp_r);
            }
        } else {
            int ret_r = (ed - st + r) % r;
            int tmp_r = r - st;
            for (int i = 0; i < c; i++) {
                copy(val.begin() + i * r + st, val.begin() + i * r + st + tmp_r, ret.val.begin() + i * ret_r);
                copy(val.begin() + i * r, val.begin() + i * r + (ret_r - tmp_r), ret.val.begin() + i * ret_r + tmp_r);
            }
        }
    }
    else {
        if (st < ed) {
            ret.val.assign(val.begin() + st * c, val.begin() + ed * c);
        } else {
            ret.val.assign(val.begin() + st * c, val.end());
            ret.val.insert(ret.val.end(), val.begin(), val.begin() + ed * c);
        }
        ret.order = order;
        ret.reoeder();
    }
    return ret;
}

Mat Mat::col(int st, int ed) const {
    Mat ret(r, (ed - st + c) % c);
    ret.val.assign(val.begin() + st * r, val.begin() + ed * r);
    return ret;
}

Mat Mat::get_bit(int b) const {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] >> b & 1;
    }
    return ret;
}

Mat Mat::opposite() const {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = MOD - val[i];
    }
    return ret;
}

Mat Mat::toOneHot() const {
    Mat ret(10, c);
    for (int i = 0; i < c; i++) {
        int y = val[i] >> DECIMAL_PLACES;
        ret(y, i) = IE;
    }
    return ret;
}

void Mat::col(int st, int ed, Mat &a) const {
    a.val.assign(val.begin() + st * r, val.begin() + ed * r);
}

Mat Mat::mod(ll b) {
    Mat ret(r, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret.val[i] = val[i] % b;
    }
    return ret;
}

int Mat::count() {
    int ret = 0;
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret += val[i] == 1;
    }
    return ret;
}

int Mat::count(const ll128 &b) {
    int ret = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (get(i, j) == b)
                ret++;
    return ret;
}

int Mat::countNe(const ll128 &b) const {
    int ret = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            if (get(i, j) != b)
                ret++;
    return ret;
}

int Mat::count_not_eqz() const {
    int ret = 0;
    int l = r * c;
    for (int i = 0; i < l; i++) {
        ret += val[i] != 0 ? 1 : 0;
    }
    return ret;
}

bool Mat::is_zero() {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            if (this->operator()(i, j) == 0)
                return 1;
        }
    return 0;
}

bool Mat::fill(const Mat &a) {
    int k = 0;
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++) {
            while (a.get(0, k) == 0)
                k++;
            this->operator()(i, j) = a.get(0, k);
            k++;
        }
    return 1;
}

void Mat::init(const ll128 &b) {
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = b;
    }
}

void Mat::cp(const Mat &a, int st, int len) {
    copy(a.val.begin() + st, a.val.begin() + st + len, val.begin());
}

void Mat::cp(const Mat &a, const Mat &mask) {
    int l = r * c;
    int j = 0;
    for (int i = 0; i < l; i++) {
        if (mask.val[i] != 0) {
            val[j++] = a.val[i];
        }
    }
}

void Mat::residual() {
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = (val[i] % MOD + MOD) % MOD;
    }
}

void Mat::AddDot(int k, ll128 *x, int incx, ll128 *y, ll128 *gamma) {
    int p;
    for (p = 0; p < k; p++) {
        *gamma += x[p * incx] * y[p];
    }
}

void Mat::sign() {
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = (val[i] > MOD / 2) ? val[i] - MOD : val[i];
    }
}

void Mat::reoeder() {
    vector<ll128> v(r * c);
    int tmp_r = r;
    int tmp_c = c;
    if (order == MatRowMajor) {
        swap(tmp_r, tmp_c);
    }
    for (int i = 0; i < tmp_r; i++) {
        for (int j = 0; j < tmp_c; j++) {
            v[i * tmp_c + j] = val[j * tmp_r + i];
        }
    }
    val.assign(v.begin(), v.end());
    order ^= 1;
}

void Mat::transorder() {
    swap(r, c);
    order ^= 1;
}

void Mat::truncated_normal(double mean, double stddev) {
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            this->operator()(i, j) = floor(generateGaussianNoise(mean, stddev) * IE);
}

void Mat::constant(double b) {
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = b;
    }
}

void Mat::col(int u, vector<ll128> &a) {
    for (int i = 0; i < r; i++)
        this->operator()(i, u) = a[i];
}

void Mat::clear() {
    val = vector<ll128>(r * c);
}

void Mat::print() const {
    if (c == 1) {
        for (int i = 0; i < r; i++) {
            DBGprint("%lld ", (ll)get(i, 0));
        }
        DBGprint("\n");
        return;
    }
    DBGprint("r: %d c: %d\n", r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            DBGprint("%lld ", (ll)get(i, j));
        }
        DBGprint("\n");
    }
}

void Mat::printSign() {
    if (c == 1) {
        for (int i = 0; i < r; i++) {
            DBGprint("%lld ", (ll)Constant::Util::get_sign(get(i, 0)));
        }
        DBGprint("\n");
        return;
    }
    DBGprint("r: %d c: %d\n", r, c);
    for (int i = 0; i < r; i++) {
        for (int j = 0; j < c; j++) {
            DBGprint("%lld ", (ll)Constant::Util::get_sign(get(i, j)));
        }
        DBGprint("\n");
    }
}

void Mat::toString(char *p) {
    Constant::Util::ll_to_char(p, r);
    *p++ = ' ';
    Constant::Util::ll_to_char(p, c);
    *p++ = ' ';
    int l = r * c;
    for (int i = 0; i < l; i++) {
        Constant::Util::ll_to_char(p, val[i]);
        *p++ = ' ';
    }
    *p = 0;
}

int Mat::toString_pos(char *p) const {
    Constant::Util::int_to_char(p, r);
    Constant::Util::int_to_char(p, c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        Constant::Util::ll_to_char(p, val[i]);
    }
    *p = 0;
    return 2 * 4 + r * c * 8;
}

int Mat::getStringLen() {
    return 2 * 4 + r * c * 8;
}

void Mat::getFrom_pos(char* &p) {
    r = Constant::Util::char_to_int(p);
    c = Constant::Util::char_to_int(p);
    val.resize(r * c);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] = Constant::Util::char_to_ll(p);
    }
}

void Mat::addFrom_pos(char *&p) {
    int tmp_r, tmp_c;
    tmp_r = Constant::Util::char_to_int(p);
    tmp_c = Constant::Util::char_to_int(p);
    int l = r * c;
    for (int i = 0; i < l; i++) {
        val[i] += Constant::Util::char_to_ll(p);
        val[i] = val[i] >= MOD ? val[i] - MOD : val[i];
        val[i] = val[i] <= -MOD ? val[i] + MOD : val[i];
    }
}

bool Mat::fill(Mat* a, Mat* a_r, Mat* b, Mat* b_r) {
    int l = a->rows() * a->cols();
    int l_r = a_r->rows() * a_r->cols();
    int k = 0;
    for (int i = 0; i < l; i++) {
        while (k < l_r && a_r->val[k] == 0) {
            k++;
        }
        if (k >= l_r) {
            return 0;
        }
        a->val[i] = a_r->val[k];
        b->val[i] = b_r->val[k];
        k++;
    }
    return 1;
}

void Mat::concat(Mat *res, Mat *a, Mat *b) {
    for (int i = 0; i < a->rows(); i++)
        for (int j = 0; j < a->cols(); j++)
            res->operator()(i, j) = a->operator()(i, j);
    for (int i = 0; i < b->rows(); i++)
        for (int j = 0; j < b->cols(); j++)
            res->operator()(i+a->rows(), j) = b->operator()(i, j);
}

void Mat::reconcat(Mat *res, Mat *a, bool fa, Mat *b, bool fb) {
    if (fa) {
        for (int i = 0; i < a->rows(); i++)
            for (int j = 0; j < a->cols(); j++)
                a->operator()(i, j) += res->operator()(i, j);
    }
    if (fb) {
        for (int i = 0; i < b->rows(); i++)
            for (int j = 0; j < b->cols(); j++)
                b->operator()(i, j) += res->operator()(i + a->rows(), j);
    }
}

void Mat::vstack(Mat *res, Mat *a, Mat *b) {}

void Mat::re_vstack(Mat *res, Mat *a, bool fa, Mat *b, bool fb) {}

void Mat::hstack(Mat *res, Mat *a, Mat *b) {
    res->val.assign(a->val.begin(), a->val.end());
    res->val.insert(res->val.end(), b->val.begin(), b->val.end());
}

void Mat::re_hstack(Mat *res, Mat *a, bool fa, Mat *b, bool fb) {
    int len_a = a->rows() * a->cols();
    int len_b = a->rows() * a->cols();
    if (fa) {
        a->val.assign(res->val.begin(), res->val.begin() + len_a);
    }
    if (fb) {
        b->val.assign(res->val.begin() + len_a, res->val.end());
    }
}

int Mat::pair_order_type(Mat *a, const Mat *b) {
    return (a->order << 1) + b->order;
}
// STB header only style library.
#ifndef NN_H_
#define NN_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#ifndef NN_MALLOC
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_FREE
#define NN_FREE free
#endif // NN_FREE

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
    size_t rows;
    size_t cols;
    size_t stride;
    float* es;
} Mat;

typedef struct {
    size_t count;
    Mat* ws; // weights
    Mat* bs; // biases
    Mat* as; // activations - size of as is count + 1
} NN;

#define MAT_AT(m, i, j) m.es[(i) * ((m).cols + (m).stride) + (j)]

float rand_float();
float sigmoidf(float x);

Mat mat_alloc(size_t rows, size_t cols);
Mat mat_submatrix(Mat m, size_t row, size_t col, size_t h, size_t w);
void mat_fill(Mat m, float v);
void mat_copy(Mat res, Mat a);
void mat_rand(Mat m, float low, float high);
void mat_sig(Mat m);
void mat_dot(Mat res, Mat a, Mat b);
void mat_mul(Mat res, float v);
void mat_sum(Mat res, Mat a);
void mat_diff(Mat res, Mat a);
void mat_print_indent(Mat m, char const* mat_name, int indent);
void mat_print(Mat m, char const* mat_name);

NN nn_alloc(size_t* arch, size_t arch_count);
void nn_rand(NN nn, float low, float high);
Mat nn_forward(NN nn, Mat X);
float nn_cost(NN nn, Mat X, Mat Y);
void nn_finite_diff(NN nn, NN d, Mat X, Mat Y, float eps);
void nn_learn(NN nn, Mat X, Mat Y, size_t train_count, float learning_rate, float eps);
void nn_print(NN nn, char const* nn_name);

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float rand_float()
{
    return rand() / (float)RAND_MAX;
}

float sigmoidf(float x)
{
    return 1.0f / (1.0f + expf(-x));
}

Mat mat_alloc(size_t rows, size_t cols)
{
    Mat m = { 0 };

    m.rows = rows;
    m.cols = cols;
    m.es = (float*)NN_MALLOC(sizeof(*m.es) * rows * cols);
    NN_ASSERT(m.es != NULL);

    return m;
}

Mat mat_submatrix(Mat m, size_t row, size_t col, size_t h, size_t w)
{
    NN_ASSERT(row + h <= m.rows);
    NN_ASSERT(col + w <= m.cols);

    Mat res;

    res.rows = h;
    res.cols = w;
    res.stride = col + m.cols - (col + w);
    res.es = m.es + row * m.cols + col;

    return res;
}

void mat_fill(Mat m, float v)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = v;
        }
    }
}

void mat_copy(Mat res, Mat a)
{
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == a.cols);

    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.cols; ++j) {
            MAT_AT(res, i, j) = MAT_AT(a, i, j);
        }
    }
}

void mat_rand(Mat m, float low, float high)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = rand_float() * (high - low) + low;
        }
    }
}

void mat_sig(Mat m)
{
    for (size_t i = 0; i < m.rows; ++i) {
        for (size_t j = 0; j < m.cols; ++j) {
            MAT_AT(m, i, j) = sigmoidf(MAT_AT(m, i, j));
        }
    }
}

void mat_dot(Mat res, Mat a, Mat b)
{
    NN_ASSERT(a.cols == b.rows);
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == b.cols);

    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.cols; ++j) {
            MAT_AT(res, i, j) = 0.0f;
            for (size_t k = 0; k < a.cols; ++k) {
                MAT_AT(res, i, j) += MAT_AT(a, i, k) * MAT_AT(b, k, j);
            }
        }
    }
}

void mat_mul(Mat res, float v)
{
    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.cols; ++j) {
            MAT_AT(res, i, j) *= v;
        }
    }
}

void mat_sum(Mat res, Mat a)
{
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == a.cols);

    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.cols; ++j) {
            MAT_AT(res, i, j) += MAT_AT(a, i, j);
        }
    }
}

void mat_diff(Mat res, Mat a)
{
    NN_ASSERT(res.rows == a.rows);
    NN_ASSERT(res.cols == a.cols);

    for (size_t i = 0; i < res.rows; ++i) {
        for (size_t j = 0; j < res.cols; ++j) {
            MAT_AT(res, i, j) -= MAT_AT(a, i, j);
        }
    }
}

void mat_print_indent(Mat m, char const* mat_name, int indent)
{
    printf("%*s%s (%zux%zu): [\n", indent, "", mat_name, m.rows, m.cols);
    for (size_t i = 0; i < m.rows; ++i) {
        printf("%*s    ", indent, "");
        for (size_t j = 0; j < m.cols; ++j) {
            printf("%f ", MAT_AT(m, i, j));
        }
        printf("\n");
    }
    printf("%*s]\n", indent, "");
}

void mat_print(Mat m, char const* mat_name)
{
    mat_print_indent(m, mat_name, 0);
}

#define mat_print(m) mat_print(m, #m)

NN nn_alloc(size_t* arch, size_t arch_count)
{
    NN_ASSERT(arch_count > 0);

    NN nn = {
        .count = arch_count - 1,
        .ws = NN_MALLOC(sizeof(*nn.ws) * nn.count),
        .bs = NN_MALLOC(sizeof(*nn.bs) * nn.count),
        .as = NN_MALLOC(sizeof(*nn.as) * arch_count)
    };

    NN_ASSERT(nn.ws != NULL);
    NN_ASSERT(nn.bs != NULL);
    NN_ASSERT(nn.as != NULL);

    for (size_t i = 0; i < nn.count; ++i) {
        nn.ws[i] = mat_alloc(arch[i + 1], arch[i]);
        nn.bs[i] = mat_alloc(arch[i + 1], 1);
        nn.as[i] = mat_alloc(arch[i], 1);
    }

    nn.as[nn.count] = mat_alloc(arch[nn.count], 1);

    return nn;
}

void nn_rand(NN nn, float low, float high)
{
    for (size_t i = 0; i < nn.count; ++i) {
        mat_rand(nn.ws[i], low, high);
        mat_rand(nn.bs[i], low, high);
        mat_rand(nn.as[i], low, high);
    }

    mat_rand(nn.as[nn.count], low, high);
}

Mat nn_forward(NN nn, Mat X)
{
    mat_copy(nn.as[0], X);

    for (size_t i = 0; i < nn.count; ++i) {
        mat_dot(nn.as[i + 1], nn.ws[i], nn.as[i]);
        mat_sum(nn.as[i + 1], nn.bs[i]);
        mat_sig(nn.as[i + 1]);
    }

    return nn.as[nn.count];
}

float nn_cost(NN nn, Mat X, Mat Y)
{
    float error = 0.0f;

    for (size_t i = 0; i < X.cols; ++i) {
        Mat XCol = mat_submatrix(X, 0, i, X.rows, 1);
        Mat out = nn_forward(nn, XCol);

        float expected = MAT_AT(Y, 0, i);
        float actual = MAT_AT(out, int(expected), 0);
        float d = actual - expected;

        error += d * d;
    }

    return error / X.cols;
}

void nn_finite_diff(NN nn, NN d, Mat X, Mat Y, float eps)
{
    float c = nn_cost(nn, X, Y);
    float saved;

    for (size_t i = 0; i < nn.count; ++i) {
        for (size_t j = 0; j < nn.ws[i].rows; ++j) {
            for (size_t k = 0; k < nn.ws[i].cols; ++k) {
                saved = MAT_AT(nn.ws[i], j, k);
                MAT_AT(nn.ws[i], j, k) += eps;
                MAT_AT(d.ws[i], j, k) = (nn_cost(nn, X, Y) - c) / eps;
                MAT_AT(nn.ws[i], j, k) = saved;
            }
        }

        for (size_t j = 0; j < nn.bs[i].rows; ++j) {
            for (size_t k = 0; k < nn.bs[i].cols; ++k) {
                saved = MAT_AT(nn.bs[i], j, k);
                MAT_AT(nn.bs[i], j, k) += eps;
                MAT_AT(d.bs[i], j, k) = (nn_cost(nn, X, Y) - c) / eps;
                MAT_AT(nn.bs[i], j, k) = saved;
            }
        }
    }
}

void nn_learn(NN nn, Mat X, Mat Y, size_t train_count, float learning_rate, float eps)
{
    // Buffer neural network used for intermediate calculations in nn_finite_diff().
    // TODO: Maybe nn_learn() should accept d as a parameter?
    printf("[nn_learn] Allocating additional resources\n");
    NN d = { 0 };

    d.count = nn.count;
    d.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
    NN_ASSERT(d.ws != NULL);
    d.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
    NN_ASSERT(d.bs != NULL);

    for (size_t i = 0; i < d.count; ++i) {
        d.ws[i] = mat_alloc(nn.ws[i].rows, nn.ws[i].cols);
        d.bs[i] = mat_alloc(nn.bs[i].rows, nn.bs[i].cols);
    }

    printf("[nn_learn] Entering learning\n");

    for (size_t i = 0; i < train_count; ++i) {
        nn_finite_diff(nn, d, X, Y, eps);

        for (size_t i = 0; i < d.count; ++i) {
            mat_mul(d.ws[i], learning_rate);
            mat_diff(nn.ws[i], d.ws[i]);
            mat_mul(d.bs[i], learning_rate);
            mat_diff(nn.bs[i], d.bs[i]);
        }

        printf("[nn_learn] Cost: %f\r", nn_cost(nn, X, Y));
    }

    printf("[nn_learn] Deallocating resources\n");
    for (size_t i = 0; i < d.count; ++i) {
        free(d.ws[i].es);
        free(d.bs[i].es);
    }

    free(d.ws);
    free(d.bs);
}

void nn_print(NN nn, char const* nn_name)
{
    static char matrix_name[256];
    printf("%s: [\n", nn_name);

    for (size_t i = 0; i < nn.count; ++i) {
        sprintf(matrix_name, "W%zu", i + 1);
        mat_print_indent(nn.ws[i], matrix_name, 4);
    }

    for (size_t i = 0; i < nn.count; ++i) {
        sprintf(matrix_name, "B%zu", i + 1);
        mat_print_indent(nn.bs[i], matrix_name, 4);
    }

    printf("]\n");
}

#define nn_print(nn) nn_print(nn, #nn)

#endif // NN_IMPLEMENTATION

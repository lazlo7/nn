#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define ARRAY_SIZE(a) sizeof((a)) / sizeof(*(a))

int main()
{
    srand(time(NULL));

    // 2 input neurons, 2 hidden neurons, 1 output neuron.
    size_t arch[] = { 2, 2, 1 };
    NN nn = nn_alloc(arch, ARRAY_SIZE(arch));
    nn_rand(nn, 0, 1);

    // XOR gate.
    float Xdata[] = { 0, 0, 1, 1, 0, 1, 0, 1 };
    Mat X = { .rows = 2, .cols = 4, .es = Xdata };

    float Ydata[] = { 1, 1, 0, 1 };
    Mat Y = { .rows = 1, .cols = 4, .es = Ydata };

    nn_learn(nn, X, Y, 1e6, 1e-1, 1e-1);

    printf("\n------------------------------\n");
    nn_print(nn);

    printf("Testing:\n");
    for (size_t i = 0; i < X.cols; ++i) {
        Mat XCol = mat_submatrix(X, 0, i, 2, 1);
        Mat out = nn_forward(nn, XCol);
        float x1 = MAT_AT(XCol, 0, 0);
        float x2 = MAT_AT(XCol, 1, 0);
        float expected = MAT_AT(Y, 0, i);
        float actual = MAT_AT(out, 0, 0);
        printf("x1: %f | x2: %f | expected: %f | actual: %f\n", x1, x2, expected, actual);
    }

    return 0;
}

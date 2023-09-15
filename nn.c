#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define ARRAY_SIZE(a) sizeof((a)) / sizeof(*(a))

int main()
{
    srand(time(NULL));

    // Standard XOR's model architecture.
    size_t arch[] = { 2, 2, 1 };
    NN nn = nn_alloc(arch, ARRAY_SIZE(arch));
    nn_print(nn);

    return 0;
}

#include <assert.h>
#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define ARRAY_SIZE(a) (sizeof(a) / sizeof(*(a)))

typedef uint8_t u8;
typedef uint32_t u32;

typedef struct {
    u32 count;
    u32 rows;
    u32 cols;
} ImagesInfo;

#define LABEL_MAGIC 0x00000801
#define IMAGE_MAGIC 0x00000803

void read_u8_many(FILE* file, u8* buffer, u32 count)
{
    size_t read_bytes = fread(buffer, sizeof(u8), count, file);
    if (read_bytes != count) {
        fprintf(stderr, "[ERROR] Could not read another byte: %s\n", strerror(errno));
        exit(1);
    }
}

u8 read_u8(FILE* file)
{
    u8 buffer;
    read_u8_many(file, &buffer, 1);
    return buffer;
}

u32 read_u32(FILE* file)
{
    static u8 buffer[4];
    size_t read_bytes = fread(buffer, sizeof(u8), 4, file);
    if (read_bytes != 4) {
        fprintf(stderr, "[ERROR] Could not read another 4 bytes: %s\n", strerror(errno));
        exit(1);
    }
    return (u32)buffer[3] | (u32)buffer[2] << 8 | (u32)buffer[1] << 16 | (u32)buffer[0] << 24;
}

FILE* open_train_images(char const* path)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Could not read training images file '%s': %s\n",
            path, strerror(errno));
        exit(1);
    }
    return f;
}

FILE* open_train_labels(char const* path)
{
    FILE* f = fopen(path, "rb");
    if (f == NULL) {
        fprintf(stderr, "[ERROR] Could not read training labels file '%s': %s\n",
            path, strerror(errno));
        exit(1);
    }
    return f;
}

ImagesInfo parse_train_images(FILE* file)
{
    u32 magic = read_u32(file);
    if (magic != IMAGE_MAGIC) {
        fprintf(stderr, "[ERROR] Magic numbers for train images do not match: expected = %d, but got = %d\n",
            IMAGE_MAGIC, magic);
        exit(1);
    }

    ImagesInfo info = {
        .count = read_u32(file),
        .rows = read_u32(file),
        .cols = read_u32(file)
    };

    return info;
}

u32 parse_train_labels(FILE* file)
{
    u32 magic = read_u32(file);
    if (magic != LABEL_MAGIC) {
        fprintf(stderr, "[ERROR] Magic numbers for train labels do not match: expected = %d, but got = %d\n",
            IMAGE_MAGIC, magic);
        exit(1);
    }

    return read_u32(file);
}

void print_number(u8* buffer, u32 rows, u32 cols)
{
    static char mask[] = ".-=@";
    for (u32 i = 0; i < rows; ++i) {
        for (u32 j = 0; j < cols; ++j) {
            char c = mask[buffer[i * cols + j] / (256 / (sizeof(mask) - 1))];
            printf("%c", c);
        }
        printf("\n");
    }
}

int main()
{
    srand(time(NULL));

    FILE* f_train_images = open_train_images("./data/train-images-idx3-ubyte");
    FILE* f_train_labels = open_train_labels("./data/train-labels-idx1-ubyte");

    ImagesInfo images_info = parse_train_images(f_train_images);
    u32 labels_count = parse_train_labels(f_train_labels);

    // FIXME: For now, only use the first 100 images.
    images_info.count = 1;
    labels_count = 1; 

    assert(images_info.count == labels_count);
    printf("Images Info | count: %u | rows: %u | cols: %u\n", images_info.count, images_info.rows, images_info.cols);

    u32 buffer_size = images_info.rows * images_info.cols;

    // 28*28 input neurons, 16 hidden neurons, 16 hidden neurons, 10 output neurons.
    size_t arch[] = { buffer_size, 16, 16, 10 };
    NN nn = nn_alloc(arch, ARRAY_SIZE(arch));
    nn_rand(nn, 0, 1);

    // MNIST.
    printf("Preparing X Matrix...\n");
    Mat X = mat_alloc(buffer_size, images_info.count);
    for (u32 i = 0; i < images_info.count; ++i) {
        for (u32 j = 0; j < buffer_size; ++j) {
            MAT_AT(X, j, i) = read_u8(f_train_images);
        }
    }

    printf("Preparing Y Matrix...\n");
    Mat Y = mat_alloc(1, images_info.count);
    for (u32 i = 0; i < images_info.count; ++i) {
        MAT_AT(Y, 0, i) = read_u8(f_train_labels);
    }

    printf("Started nn_learn\n");
    nn_learn(nn, X, Y, 1e6, 1e-3, 1e-3);

    free(X.es);
    free(Y.es);
    free(nn.ws);
    free(nn.bs);
    free(nn.as);

    return 0;
}

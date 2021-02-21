#include <stdio.h>
#include <stdlib.h>

#define KERNEL_STRING "    Kernel"
#define TRANSFER_STRING "    Transfer"
#define ASCII_MAX 127

typedef struct {
    const char * filename;
    unsigned int offset;
} Options;

typedef struct {
    char * output;
    char * input;
    unsigned int length;
} Results;

typedef struct {
    char * block;
    char * encrypted;
    char * decrypted;
} Device;
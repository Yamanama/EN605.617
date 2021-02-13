#include <stdio.h>
#include <unistd.h>

void parseArgs(unsigned int argc, char** argv, Options* options);
void allocate(Results* results, Device* device, Options options);
void clean(Device device);
void generate(Results* results, Device device, Options options);
void calc(Results* results, Device device, Options options);
void print(Results results, Options options);


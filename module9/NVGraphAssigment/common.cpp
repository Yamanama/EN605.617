/*
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
 *
 * Please refer to the NVIDIA end user license agreement (EULA) associated
 * with this source code for terms and conditions that govern your use of
 * this software. Any use, reproduction, disclosure, or distribution of
 * this software and related documentation outside the terms of the EULA
 * is strictly prohibited.
 *
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <helper_cuda.h>
#include "nvgraph.h"

/* PageRank
 *  Find PageRank for a graph with a given transition probabilities, a bookmark vector of dangling vertices, and the damping factor.
 *  This is equivalent to an eigenvalue problem where we want the eigenvector corresponding to the maximum eigenvalue.
 *  By construction, the maximum eigenvalue is 1.
 *  The eigenvalue problem is solved with the power method.

Initially :
V = 6 
E = 10

Edges       W
0 -> 1    0.50
0 -> 2    0.50
2 -> 0    0.33
2 -> 1    0.33
2 -> 4    0.33
3 -> 4    0.50
3 -> 5    0.50
4 -> 3    0.50
4 -> 5    0.50
5 -> 3    1.00

bookmark (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)^T note: 1.0 if i is a dangling node, 0.0 otherwise

Source oriented representation (CSC):
destination_offsets {0, 1, 3, 4, 6, 8, 10}
source_indices {2, 0, 2, 0, 4, 5, 2, 3, 3, 4}
W0 = {0.33, 0.50, 0.33, 0.50, 0.50, 1.00, 0.33, 0.50, 0.50, 1.00}

----------------------------------

Operation : Pagerank with various damping factor 
----------------------------------

Expected output for alpha= 0.9 (result stored in pr_2) : (0.037210, 0.053960, 0.041510, 0.37510, 0.206000, 0.28620)^T 
From "Google's PageRank and Beyond: The Science of Search Engine Rankings" Amy N. Langville & Carl D. Meyer
*/

/**
 * Setup Timer
 *
 * start - start marker
 * stop - stop marker
 */
void setupTimer(cudaEvent_t* start, cudaEvent_t* stop){
    cudaEventCreate(start);
    cudaEventCreate(stop);
}
/**
 * Log the computed time
 * 
 * start - start marker
 * stop - stop marker
 * message - log message
 */
void logTime(cudaEvent_t start, cudaEvent_t stop, const char* message){
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float elapsed;
    cudaEventElapsedTime(&elapsed, start, stop);
    printf("  %8s: %f\n", message, elapsed);
}
/**
 *  Clean up memory for timers
 *
 * start - start marker
 * stop - stop marker
 */
void cleanTimer(cudaEvent_t start, cudaEvent_t stop){
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

/**
 * Check the status of the calls to nvgraph
 */
void check_status(nvgraphStatus_t status)
{
    if ((int)status != 0)
    {
        printf("ERROR : %d\n",status);
        exit(0);
    }
}
/**
 * The lists necessary for the program
 */
typedef struct {
    float* weights_h;
    float* bookmark_h;
    float* pr_1;
    float* pr_2;
    int* destination_offsets_h;
    int* source_indices_h;
    void** vertex_dim;
} Lists;

/**
 * Initialize
 *
 * lists - the lists to init
 * vertex_dimT - the dim to init
 */
void initialize(Lists lists, cudaDataType_t* vertex_dimT) {
    lists.vertex_dim[0] = (void*)lists.bookmark_h; 
    lists.vertex_dim[1] = (void*)lists.pr_1;
    lists.vertex_dim[2] = (void*)lists.pr_2;
    vertex_dimT[0]= CUDA_R_32F; 
    vertex_dimT[1]= CUDA_R_32F;
    vertex_dimT[2]= CUDA_R_32F;
}
/**
 * Fill the weights array
 * weights - the weights array
 */
void fillWeights(float* weights) {
    weights[0] = 0.333333f;
    weights[1] = 0.500000f;
    weights[2] = 0.333333f;
    weights[3] = 0.500000f;
    weights[4] = 0.500000f;
    weights[5] = 1.000000f;
    weights[6] = 0.333333f;
    weights[7] = 0.500000f;
    weights[8] = 0.500000f;
    weights[9] = 0.500000f;
}
/**
 * Fill the bookmark array
 * bookmark - the bookmark array
 */
void fillBookmark(float * bookmark) {
    bookmark[0] = 0.0f;
    bookmark[1] = 1.0f;
    bookmark[2] = 0.0f;
    bookmark[3] = 0.0f;
    bookmark[4] = 0.0f;
    bookmark[5] = 0.0f;
}
/**
 * Fill the destination array
 * destination - the destination array
 */
void fillDestination(int * destination) {
    destination[0] = 0;
    destination[1] = 1;
    destination[2] = 3;
    destination[3] = 4;
    destination[4] = 6;
    destination[5] = 8;
    destination[6] = 10;   
}
/**
 * Fill the source array
 * source - the source pointer
 */
void fillSource(int * source) {
    source[0] = 2;
    source[1] = 0;
    source[2] = 2;
    source[3] = 0;
    source[4] = 4;
    source[5] = 5;
    source[6] = 2;
    source[7] = 3;
    source[8] = 3;
    source[9] = 4;
}
/**
 * Fill the lists
 *
 * lists - the lists
 */
void fill (Lists lists) {
    // markers
    cudaEvent_t start, stop;
    // setup timer
    setupTimer(&start, &stop);
    // mark start
    cudaEventRecord(start, 0);
    // fill the weights
    fillWeights(lists.weights_h);
    // fill destination
    fillDestination(lists.destination_offsets_h);
    // fill source
    fillSource(lists.source_indices_h);
    // fill bookmarks
    fillBookmark(lists.bookmark_h);
    logTime(start, stop, "Fill");
}
/**
 * Create CSC
 *
 * lists - the lists
 * n - the n
 * nnz - the nnz
 * input - the CSC
 */
void createCSC(Lists lists, int n, int nnz, 
                 nvgraphCSCTopology32I_t CSC_input) {
    CSC_input->nvertices = n;
    CSC_input->nedges = nnz;
    CSC_input->destination_offsets = lists.destination_offsets_h;
    CSC_input->source_indices = lists.source_indices_h;
}
/**
 * Set the data
 *
 * handle - the handler
 * graph - the graph description
 * lists - the lists
 */
void setData(nvgraphHandle_t handle, nvgraphGraphDescr_t graph, Lists lists) {
    // markers
    cudaEvent_t start, stop;
    // setup timer
    setupTimer(&start, &stop);
    // mark start
    cudaEventRecord(start, 0);
    for (int i = 0; i < 2; ++i)
        check_status(nvgraphSetVertexData(handle, graph, lists.vertex_dim[i], i));
    check_status(nvgraphSetEdgeData(handle, graph, (void*)lists.weights_h, 0));
    logTime(start, stop, "Set Data");
}
/**
 * Print the computed data
 */
void print(Lists lists, int n) {
    for (int i = 0; i<n; i++)
        printf("%f\n",lists.pr_1[i]); printf("\n");
}
/**
 * Page Rank
 *
 * handle - the handler
 * graph - the graph description
 * lists - the lists
 * alpha_p - the alpha pointer
 */
void pageRank(nvgraphHandle_t handle, nvgraphGraphDescr_t graph, Lists lists, const void* alpha1_p) {
     // markers
    cudaEvent_t start, stop;
    // setup timer
    setupTimer(&start, &stop);
    // mark start
    cudaEventRecord(start, 0);
    check_status(nvgraphPagerank(handle, graph, 0, alpha1_p, 0, 0, 1, 0.0f, 0));
    nvgraphGetVertexData(handle, graph, lists.vertex_dim[1], 1);
    logTime(start, stop, "Page Rank");
}
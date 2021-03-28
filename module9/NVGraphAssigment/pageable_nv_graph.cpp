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
#include "common.cpp"
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
 * Allocate memory
 */
void allocate(Lists* lists, size_t n, size_t nnz, size_t vertex, size_t edge) {
    lists->destination_offsets_h = (int*) malloc((n+1)*sizeof(int));
    lists->source_indices_h = (int*) malloc(nnz*sizeof(int));
    lists->weights_h = (float*)calloc(sizeof(float), nnz);
    lists->bookmark_h = (float*)malloc(n*sizeof(float));
    lists->pr_1 = (float*)malloc(n*sizeof(float));
}
/**
 * Free memory
 */
void cleanUp(nvgraphHandle_t handle, nvgraphGraphDescr_t graph, Lists lists) {
    check_status(nvgraphDestroyGraphDescr(handle, graph));
    check_status(nvgraphDestroy(handle));

    free(lists.destination_offsets_h);
    free(lists.source_indices_h);
    free(lists.weights_h);
    free(lists.bookmark_h);
    free(lists.pr_1);
}
/**
 * Main
 */
int main(int argc, char **argv)
{
    // constants
    const size_t  n = 6, nnz = 10, vertex_numsets = 3, edge_numsets = 1;
    const float alpha = 0.85;
    const void *alpha_p = (const void *) &alpha;
    void** vertex_dim;
    // lists
    Lists lists;
    // nvgraph variables
    nvgraphStatus_t status;
    nvgraphHandle_t handle;
    nvgraphGraphDescr_t graph;
    nvgraphCSCTopology32I_t CSC_input;
    cudaDataType_t edge_dimT = CUDA_R_32F;
    cudaDataType_t* vertex_dimT;
    // allocate
    allocate(&lists, n, nnz, vertex_numsets, edge_numsets);
    vertex_dim = (void**)malloc(vertex_numsets*sizeof(void*));
    vertex_dimT = (cudaDataType_t*)malloc(vertex_numsets*sizeof(cudaDataType_t));
    CSC_input = (nvgraphCSCTopology32I_t) malloc(sizeof(struct nvgraphCSCTopology32I_st));
    // init
    initialize(lists, vertex_dimT);
    vertex_dim[0] = (void*)lists.bookmark_h; vertex_dim[1]= (void*)lists.pr_1;
    vertex_dimT[0] = CUDA_R_32F; vertex_dimT[1]= CUDA_R_32F; vertex_dimT[2]= CUDA_R_32F;
    // fill
    fill(lists);
    // create
    check_status(nvgraphCreate (&handle));
    check_status(nvgraphCreateGraphDescr (handle, &graph));
    // create CSC
    createCSC(lists, n, nnz, CSC_input);
    // allocate
    check_status(nvgraphSetGraphStructure(handle, graph, (void*)CSC_input, NVGRAPH_CSC_32));
    check_status(nvgraphAllocateVertexData(handle, graph, vertex_numsets, vertex_dimT));
    check_status(nvgraphAllocateEdgeData  (handle, graph, edge_numsets, &edge_dimT));
    // set data
    setData(handle, graph, lists);
    // page rank
    pageRank(handle, graph, lists, alpha_p);
    // print
    print(lists, n);
    // clean up
    cleanUp(handle, graph, lists);
    return EXIT_SUCCESS;
}

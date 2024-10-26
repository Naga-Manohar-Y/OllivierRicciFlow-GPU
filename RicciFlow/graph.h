#ifndef _GRAPH_H_
#define _GRAPH_H_
#include "common.h"

class Graph
{

public:
    std::string file; // input graph directory
    ui n;            // number of nodes of the graph
    ept m;           // number of edges of the graph

    ept *offset; // offset of neighbors of nodes
    ui *neighbors;         // adjacent ids of edges
    ui *degree;            // degree of each node
    ui *edges;
    float* weights;
    ~Graph(){
        delete[] offset;
        delete[] neighbors;
        delete[] degree;
    }
    void readBinaryFile(const char* _file)
    {
        file = string(_file);
        FILE *f = fopen(_file, "rb");
        if (f == NULL)
        {
            std::cerr << "Error: Could not open file for reading." << std::endl;
            return;
        }

        // Read size of ui
        ui tt;
        fread(&tt, sizeof(ui), 1, f);
        // Ensure that sizeof(ui) matches the value read
        if (tt != sizeof(ui))
        {
            std::cerr << "Error: ui size mismatch." << std::endl;
            fclose(f);
            return;
        }

        // Read number of nodes and edges
        fread(&n, sizeof(ui), 1, f);
        fread(&m, sizeof(ui), 1, f);
        std::cout << "Read n: " << n << ", m: " << m << std::endl;

        // Allocate memory for degree array
        degree = new ui[n];
        fread(degree, sizeof(ui), n, f);

        // Allocate memory for offset and neighbors arrays
        offset = new ept[n + 1];
        neighbors = new ui[m];

        offset[0] = 0;
        partial_sum(degree, degree + n, offset + 1);
        fread(neighbors, sizeof(ui), m, f);

        edges=new ui[m*2];
        for(ui i=0;i<n;i++)
            for(ui j=offset[i];j<offset[i+1];j++)
                edges[2*j]=i, edges[2*j+1]=neighbors[j];
        weights=new float[m];
        fill(weights, weights+m, 1);
    }
};


class GPUGraph: public Graph{
public:
    ui *d_degree;
    ept *d_offset; // offset of neighbors of nodes
    ui *d_neighbors;         // adjacent ids of edges
    ui* d_edges;
    float* d_weights;
    void copyToGPU(){
        chkerr(cudaMalloc(&(d_degree), (n) * sizeof(ui)));
        chkerr(cudaMemcpy(d_degree, degree, (n) * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc(&(d_offset), (n+1) * sizeof(ept)));
        chkerr(cudaMemcpy(d_offset, offset, (n+1) * sizeof(ept), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc(&(d_neighbors), m * sizeof(ui)));
        chkerr(cudaMemcpy(d_neighbors, neighbors, m * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc(&(d_edges), 2*m * sizeof(ui)));
        chkerr(cudaMemcpy(d_edges, edges, 2*m * sizeof(ui), cudaMemcpyHostToDevice));
        chkerr(cudaMalloc(&(d_weights), m * sizeof(float)));
        fill<<<BLK_NUMS, BLK_DIM>>>(d_weights, 1, m);
    }
};
#endif
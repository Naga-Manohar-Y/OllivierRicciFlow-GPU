#ifndef _GRAPH_H_
#define _GRAPH_H_

#include <string>
#include <vector>

typedef unsigned int ui;
typedef long long ept;

class Graph {
private:
    std::string dir; // input graph directory
    ui n; // number of nodes of the graph
    ept m; // number of edges of the graph

    ept *neighbors_offset; // offset of neighbors of nodes
    ui *neighbors; // adjacent ids of edges
    ui *degree; // degree of each node
    ui *reverse;

    void readDIMACS2Text(const char* filepath);
    void readRawSNAPText(const char* filepath);

public:
    Graph(const char *_dir);
    ~Graph();

    void readTextFile(const char* filepath);
    void writeBinaryFile(const char* filepath);
    void readBinaryFile(const char* filepath);
};

#endif
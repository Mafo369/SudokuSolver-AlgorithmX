#include <bits/stdc++.h> 

typedef struct s_Node Node;

int getRight(int i, int nCol);

int getLeft(int i, int nCol);

int getUp(int i, int nRow);

int getDown(int i, int nRow);

Node *createToridolMatrix(Node **Matrix, int nRow, int nCol, bool **ProbMat, Node *header);

void cover(Node *targetNode, Node **Matrix);

void uncover(Node *targetNode, Node **Matrix);

Node *getMinColumn(Node *header);

void printSolutions(vector <Node*> solutions); 


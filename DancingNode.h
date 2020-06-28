#include <bits/stdc++.h> 

typedef struct s_DancingNode 
{ 
public: 
    struct s_DancingNode *left; 
    struct s_DancingNode *right; 
    struct s_DancingNode *up; 
    struct s_DancingNode *down; 
    struct s_DancingNode *column; 
    int row; 
    int col; 
    int nodeCount; 
}DancingNode;

int getRight(int i, int nCol);

int getLeft(int i, int nCol);

int getUp(int i, int nRow);

int getDown(int i, int nRow);

DancingNode *createToridolMatrix(DancingNode **Matrix, int nRow, int nCol, bool **ProbMat, DancingNode *header);

void cover(DancingNode *targetDancingNode, DancingNode **Matrix);

void uncover(DancingNode *targetDancingNode, DancingNode **Matrix);

DancingNode *getMinColumn(DancingNode *header);

void printSolutions(std::vector<DancingNode*> solutions); 

void search(int k, DancingNode *header, std::vector<DancingNode*> solutions, DancingNode **Matrix) 

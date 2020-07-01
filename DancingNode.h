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

DancingNode *createToridolMatrix(std::vector<std::vector<DancingNode>> &Matrix, int nRow, int nCol, std::vector<std::vector<bool>> &ProbMat, DancingNode *header);

void cover(DancingNode *targetDancingNode, std::vector<std::vector<DancingNode>> &Matrix);

void uncover(DancingNode *targetDancingNode, std::vector<std::vector<DancingNode>> &Matrix);

DancingNode *getMinColumn(DancingNode *header);

void printSolutions(std::vector<DancingNode*> &solutions, std::vector<std::vector<int>> &grid); 

void search(int k, DancingNode *header, std::vector<DancingNode*> &solutions, std::vector<std::vector<DancingNode>> &Matrix, std::vector<std::vector<int>> &grid); 

#include "Node.h"

using namespace std; 
  
struct s_Node 
{ 
public: 
    struct s_Node *left; 
    struct s_Node *right; 
    struct s_Node *up; 
    struct s_Node *down; 
    struct s_Node *column; 
    int row; 
    int col; 
    int nodeCount; 
}; 

// Functions to get next index in any direction 
// for given index (circular in nature)  
int getRight(int i, int nCol){
    return (i+1) % nCol; 
} 
int getLeft(int i, int nCol){
    return (i-1 < 0) ? nCol-1 : i-1 ; 
} 
int getUp(int i, int nRow){
    return (i-1 < 0) ? nRow : i-1 ; 
}   
int getDown(int i, int nRow){
    return (i+1) % (nRow+1); 
} 
  
// Create 4 way linked matrix of nodes 
// called Toroidal due to resemblance to 
// toroid 
Node *createToridolMatrix(Node **Matrix, int nRow, int nCol, bool **ProbMat, Node *header) 
{ 
    // One extra row for list header nodes 
    // for each column 
    for(int i = 0; i <= nRow; i++) 
    { 
        for(int j = 0; j < nCol; j++) 
        { 
            // If it's 1 in the problem matrix then  
            // only create a node  
            if(ProbMat[i][j]) 
            { 
                int a, b; 
  
                // If it's 1, other than 1 in 0th row 
                // then count it as node of column  
                // and increment node count in column header 
                if(i) Matrix[0][j].nodeCount += 1; 
  
                // Add pointer to column header for this  
                // column node 
                Matrix[i][j].column = &Matrix[0][j]; 
  
                // set row and column id of this node 
                Matrix[i][j].row = i; 
                Matrix[i][j].col = j; 
  
                // Link the node with neighbors 
  
                // Left pointer 
                a = i; b = j; 
                do{ b = getLeft(b, nCol); } while(!ProbMat[a][b] && b != j); 
                Matrix[i][j].left = &Matrix[i][b]; 
  
                // Right pointer 
                a = i; b = j; 
                do { b = getRight(b, nCol); } while(!ProbMat[a][b] && b != j); 
                Matrix[i][j].right = &Matrix[i][b]; 
  
                // Up pointer 
                a = i; b = j; 
                do { a = getUp(a, nRow); } while(!ProbMat[a][b] && a != i); 
                Matrix[i][j].up = &Matrix[a][j]; 
  
                // Down pointer 
                a = i; b = j; 
                do { a = getDown(a, nRow); } while(!ProbMat[a][b] && a != i); 
                Matrix[i][j].down = &Matrix[a][j]; 
            } 
        } 
    } 
  
    // link header right pointer to column  
    // header of first column  
    header->right = &Matrix[0][0]; 
  
    // link header left pointer to column  
    // header of last column  
    header->left = &Matrix[0][nCol-1]; 
  
    Matrix[0][0].left = header; 
    Matrix[0][nCol-1].right = header; 
    return header; 
} 
  
// Cover the given node completely 
void cover(Node *targetNode, Node **Matrix) 
{ 
    Node *row, *rightNode; 
  
    // get the pointer to the header of column 
    // to which this node belong  
    Node *colNode = targetNode->column; 
  
    // unlink column header from it's neighbors 
    colNode->left->right = colNode->right; 
    colNode->right->left = colNode->left; 
  
    // Move down the column and remove each row 
    // by traversing right 
    for(row = colNode->down; row != colNode; row = row->down) 
    { 
        for(rightNode = row->right; rightNode != row; 
            rightNode = rightNode->right) 
        { 
            rightNode->up->down = rightNode->down; 
            rightNode->down->up = rightNode->up; 
  
            // after unlinking row node, decrement the 
            // node count in column header 
            Matrix[0][rightNode->col].nodeCount -= 1; 
        } 
    } 
} 
  
// Uncover the given node completely 
void uncover(Node *targetNode, Node **Matrix) 
{ 
    Node *rowNode, *leftNode; 
  
    // get the pointer to the header of column 
    // to which this node belong  
    Node *colNode = targetNode->column; 
  
    // Move down the column and link back 
    // each row by traversing left 
    for(rowNode = colNode->up; rowNode != colNode; rowNode = rowNode->up) 
    { 
        for(leftNode = rowNode->left; leftNode != rowNode; 
            leftNode = leftNode->left) 
        { 
            leftNode->up->down = leftNode; 
            leftNode->down->up = leftNode; 
  
            // after linking row node, increment the 
            // node count in column header 
            Matrix[0][leftNode->col].nodeCount += 1; 
        } 
    } 
  
    // link the column header from it's neighbors 
    colNode->left->right = colNode; 
    colNode->right->left = colNode; 
} 
  
// Traverse column headers right and  
// return the column having minimum  
// node count 
Node *getMinColumn(Node *header) 
{ 
    Node *h = header; 
    Node *min_col = h->right; 
    h = h->right->right; 
    do
    { 
        if(h->nodeCount < min_col->nodeCount) 
        { 
            min_col = h; 
        } 
        h = h->right; 
    }while(h != header); 
  
    return min_col; 
} 
  
  
void printSolutions(vector <Node*> solutions) 
{ 
    cout<<"Printing Solutions: "; 
    vector<Node*>::iterator i; 
  
    for(i = solutions.begin(); i!=solutions.end(); i++) 
        cout<<(*i)->row<<" "; 
    cout<<"\n"; 
} 

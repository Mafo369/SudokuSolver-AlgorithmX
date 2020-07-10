#include "DancingNode.h"

using namespace std; 

// Functions to get next index in any direction 
// for given index (circular in nature)  
int getRight(int i, int nCol){
    return (i+1) % nCol; 
} 
int getLeft(int i, int nCol){
    return (i-1 < 0) ? nCol-1 : i-1 ; 
} 
int getUp(int i, int nRow){
    return (i-1 < 0) ? nRow-1 : i-1 ; 
}   
int getDown(int i, int nRow){
    return (i+1) % (nRow); 
} 
  
// Create 4 way linked matrix of nodes 
// called Toroidal due to resemblance to 
// toroid 
DancingNode *createToridolMatrix(vector<vector<DancingNode>> &Matrix, int nRow, int nCol, vector<vector<bool>> &ProbMat, DancingNode *header) 
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
                //Matrix[i].push_back(DancingNode()); 
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
void cover(DancingNode *targetDancingNode, vector<vector<DancingNode>> &Matrix) 
{ 
    DancingNode *row, *rightDancingNode; 
  
    // get the pointer to the header of column 
    // to which this node belong  
    DancingNode *colDancingNode = targetDancingNode->column; 
  
    // unlink column header from it's neighbors 
    colDancingNode->left->right = colDancingNode->right; 
    colDancingNode->right->left = colDancingNode->left; 
  
    // Move down the column and remove each row 
    // by traversing right 
    for(row = colDancingNode->down; row != colDancingNode; row = row->down) 
    { 
        for(rightDancingNode = row->right; rightDancingNode != row; 
            rightDancingNode = rightDancingNode->right) 
        { 
            rightDancingNode->up->down = rightDancingNode->down; 
            rightDancingNode->down->up = rightDancingNode->up; 
  
            // after unlinking row node, decrement the 
            // node count in column header 
            Matrix[0][rightDancingNode->col].nodeCount -= 1; 
        } 
    } 
} 
  
// Uncover the given node completely 
void uncover(DancingNode *targetDancingNode, vector<vector<DancingNode>> &Matrix) 
{ 
    DancingNode *rowDancingNode, *leftDancingNode; 
  
    // get the pointer to the header of column 
    // to which this node belong  
    DancingNode *colDancingNode = targetDancingNode->column; 
  
    // Move down the column and link back 
    // each row by traversing left 
    for(rowDancingNode = colDancingNode->up; rowDancingNode != colDancingNode; rowDancingNode = rowDancingNode->up) 
    { 
        for(leftDancingNode = rowDancingNode->left; leftDancingNode != rowDancingNode; 
            leftDancingNode = leftDancingNode->left) 
        { 
            leftDancingNode->up->down = leftDancingNode; 
            leftDancingNode->down->up = leftDancingNode; 
  
            // after linking row node, increment the 
            // node count in column header 
            Matrix[0][leftDancingNode->col].nodeCount += 1; 
        } 
    } 
  
    // link the column header from it's neighbors 
    colDancingNode->left->right = colDancingNode; 
    colDancingNode->right->left = colDancingNode; 
} 
  
// Traverse column headers right and  
// return the column having minimum  
// node count 
DancingNode *getMinColumn(DancingNode *header) 
{ 
    DancingNode *h = header;
    DancingNode *min_col = h->right; 
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
  

  
void printSolutions(vector<DancingNode*> &solutions, vector<vector<int>> &grid) 
{ 
    
    cout<<"\nPrinting Solution: \n"<< endl;
    vector<DancingNode*>::iterator i; 
  
    for(i = solutions.begin(); i!=solutions.end(); i++){
        //cout<<(*i)->row<<endl;
        int coverRow = (*i)->row - 1;
        int r = coverRow / 81 ;
        int c = (coverRow / 9 ) % 9;
        int answer = (coverRow % 9) + 1;
        if(answer == 0){
            answer = 9;
        }
        
        grid[r][c] = answer;
    }
}



// Search for exact covers 
void search(int k, DancingNode *header, vector<DancingNode*> &solutions, vector<vector<DancingNode>> &Matrix, vector<vector<int>> &grid) 
{ 
    DancingNode *rowNode; 
    DancingNode *rightNode; 
    DancingNode *leftNode; 
    DancingNode *column; 
    //cout<<"IN"<<endl; 
    // if no column left, then we must 
    // have found the solution 
    if(header->right == header) 
    {
        //cout << "OUT" << endl; 
        printSolutions(solutions, grid); 
        return; 
    } 
  
    // choose column deterministically 
    column = getMinColumn(header); 
    //cout << "COVERING" <<endl; 
    // cover chosen column 
    cover(column, Matrix); 
  
    for(rowNode = column->down; rowNode != column;  
        rowNode = rowNode->down ) 
    {
        //cout << "hey u" << endl; 
        solutions.push_back(rowNode); 
  
        for(rightNode = rowNode->right; rightNode != rowNode; 
            rightNode = rightNode->right) 
            cover(rightNode, Matrix); 
  
        // move to level k+1 (recursively) 
        search(k+1, header, solutions, Matrix, grid); 
  
        // if solution in not possible, backtrack (uncover) 
        // and remove the selected row (set) from solution 
        solutions.pop_back(); 
  
        column = rowNode->column; 
        for(leftNode = rowNode->left; leftNode != rowNode; 
            leftNode = leftNode->left) 
            uncover(leftNode, Matrix); 
    } 
  
    uncover(column, Matrix); 
} 

#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <vector>

#include "DancingNode.h"
#include "ImageProcessing.h"

using namespace ml;
using namespace std;

#define SIZE 9
#define BOX_SIZE 3
#define EMPTY_CELL 0
#define CONSTRAINTS 4
#define MIN_VALUE 1
#define MAX_VALUE 9
#define COVER_START_INDEX 1


/********************************* Exact Cover Problem Functions ***********************************/

int indexInCoverMatrix(int row, int col, int num);
int createBoxConstraints(vector<vector<bool> > &coverMatrix, int header);
int createColumnConstraints(vector<vector<bool>> &coverMatrix, int header);
int createRowConstraints(vector<vector<bool>> &coverMatrix, int header);
int createCellConstraints(vector<vector<bool>> &coverMatrix, int header);
void createCoverMatrix(vector<vector<bool>> &coverMatrix); 
void convertInCoverMatrix(vector<vector<int>> &grid, vector<vector<bool>> &coverMatrix);

/*********************************************** MAIN **************************************************/

int main( int argc, char* argv[] ){


    /******************************** Read original image ********************************/
    if(argc != 2){
        printf("Usage : %s sudokuPath", argv[0]);
        exit(1);
    } 
    char *sudokuName = argv[1];
	Mat src = imread(sudokuName, IMREAD_GRAYSCALE );

	//if fail to read the image
	if (!src.data){
		cout << "Error loading the image" << endl;
		exit(2);
	}
	
    Mat original = src.clone();
	imshow("original image",src);
   

    /*********************************** Preprocessing Image **************************************/ 

	Mat thresholded;

    Mat outerBox = Mat(src.size(), CV_8UC1);
    // Blur
	GaussianBlur(src, src, Size(11, 11), 0); //removing noises
    // Threshold
	adaptiveThreshold(src, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);// thresholding the image
    // Invert Image to have white borders
    bitwise_not(outerBox, outerBox);    
    // Fill up small cracks
    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(outerBox, outerBox, kernel);


    /************************************ Borders ****************************************/
    
    // Find the biggest bounding box in the image
    int max=-1;
    Point maxPt;
    for(int y=0;y<outerBox.size().height;y++){
        uchar *row = outerBox.ptr(y);
        for(int x=0;x<outerBox.size().width;x++){
            if(row[x]>=128){

                 int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,64));
                 if(area>max){
                     maxPt = Point(x,y);
                     max = area;
                 }
            }
        }

    }
    // Floodfill the biggest box with white
    floodFill(outerBox, maxPt, CV_RGB(255,255,255));
    // Turn the others boxes black
    for(int y=0;y<outerBox.size().height;y++){
        uchar *row = outerBox.ptr(y);
        for(int x=0;x<outerBox.size().width;x++){
            if(row[x]==64 && x!=maxPt.x && y!=maxPt.y){
                floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
    // Restore the dilated image by eroding
    erode(outerBox, outerBox, kernel);
    //imshow("outer Box",outerBox);
    
    /************************************* Hough Lines *********************************/

    vector<Vec2f> lines;
    HoughLines(outerBox, lines, 1, CV_PI/180, 200);
 
    /*************************************** Merge Lines *********************************/

    mergeRelatedLines(&lines, src);
    for(unsigned int i=0;i<lines.size();i++){
        drawLine(lines[i], outerBox, CV_RGB(0,0,128));
    }
    //imshow("HoughLines merged", outerBox);        

    /************************************** Extreme Lines *********************************/
   
    // Now detect the lines on extremes
    Vec2f topEdge = Vec2f(1000,1000);    
    Vec2f bottomEdge = Vec2f(-1000,-1000);        
    Vec2f leftEdge = Vec2f(1000,1000);    double leftXIntercept=100000;
    Vec2f rightEdge = Vec2f(-1000,-1000);        double rightXIntercept=0;
    // Loop over all lines
    for(unsigned int i=0;i<lines.size();i++){
        Vec2f current = lines[i];

        float p=current[0];// rho

        float theta=current[1];
        
        // If Merged Line we skip
        if(p==0 && theta==-100)
            continue;

        // Normal form of line to calculate where the lines intersects
        double xIntercept;
        xIntercept = p/cos(theta);
        
        // If line vertical
        if(theta>CV_PI*80/180 && theta<CV_PI*100/180){
            if(p<topEdge[0])
                topEdge = current;

            if(p>bottomEdge[0])
                bottomEdge = current;
        }
        //Otherwise if horizontal
        else if(theta<CV_PI*10/180 || theta>CV_PI*170/180){
            if(xIntercept>rightXIntercept){
                rightEdge = current;
                rightXIntercept = xIntercept;
            }
            else if(xIntercept<=leftXIntercept){
                leftEdge = current;
                leftXIntercept = xIntercept;
            }
        }
    }

    drawLine(topEdge, src, CV_RGB(0,0,0));
    drawLine(bottomEdge, src, CV_RGB(0,0,0));
    drawLine(leftEdge, src, CV_RGB(0,0,0));
    drawLine(rightEdge, src, CV_RGB(0,0,0));

    // Find two points in each line
    Point left1, left2, right1, right2, bottom1, bottom2, top1, top2;

    int height=outerBox.size().height;

    int width=outerBox.size().width;

    if(leftEdge[1]!=0){
        left1.x=0;        left1.y=leftEdge[0]/sin(leftEdge[1]);
        left2.x=width;    left2.y=-left2.x/tan(leftEdge[1]) + left1.y;
    }
    else{
        left1.y=0;        left1.x=leftEdge[0]/cos(leftEdge[1]);
        left2.y=height;    left2.x=left1.x - height*tan(leftEdge[1]);

    }
    if(rightEdge[1]!=0){
        right1.x=0;        right1.y=rightEdge[0]/sin(rightEdge[1]);
        right2.x=width;    right2.y=-right2.x/tan(rightEdge[1]) + right1.y;
    }
    else{
        right1.y=0;        right1.x=rightEdge[0]/cos(rightEdge[1]);
        right2.y=height;    right2.x=right1.x - height*tan(rightEdge[1]);

    }
    bottom1.x=0;    bottom1.y=bottomEdge[0]/sin(bottomEdge[1]);

    bottom2.x=width;bottom2.y=-bottom2.x/tan(bottomEdge[1]) + bottom1.y;

    top1.x=0;        top1.y=topEdge[0]/sin(topEdge[1]);
    top2.x=width;    top2.y=-top2.x/tan(topEdge[1]) + top1.y;

    // Next, we find the intersection of  these four lines
    double leftA = left2.y-left1.y;
    double leftB = left1.x-left2.x;

    double leftC = leftA*left1.x + leftB*left1.y;

    double rightA = right2.y-right1.y;
    double rightB = right1.x-right2.x;

    double rightC = rightA*right1.x + rightB*right1.y;

    double topA = top2.y-top1.y;
    double topB = top1.x-top2.x;

    double topC = topA*top1.x + topB*top1.y;

    double bottomA = bottom2.y-bottom1.y;
    double bottomB = bottom1.x-bottom2.x;

    double bottomC = bottomA*bottom1.x + bottomB*bottom1.y;

    // Intersection of left and top
    double detTopLeft = leftA*topB - leftB*topA;

    Point ptTopLeft = Point((topB*leftC - leftB*topC)/detTopLeft, (leftA*topC - topA*leftC)/detTopLeft);

    // Intersection of top and right
    double detTopRight = rightA*topB - rightB*topA;

    Point ptTopRight = Point((topB*rightC-rightB*topC)/detTopRight, (rightA*topC-topA*rightC)/detTopRight);

    // Intersection of right and bottom
    double detBottomRight = rightA*bottomB - rightB*bottomA;
    Point ptBottomRight = Point((bottomB*rightC-rightB*bottomC)/detBottomRight, (rightA*bottomC-bottomA*rightC)/detBottomRight);// Intersection of bottom and left
    double detBottomLeft = leftA*bottomB-leftB*bottomA;
    Point ptBottomLeft = Point((bottomB*leftC-leftB*bottomC)/detBottomLeft, (leftA*bottomC-bottomA*leftC)/detBottomLeft);
    
    //FInd longest edge of the puzzle
    int maxLength = (ptBottomLeft.x-ptBottomRight.x)*(ptBottomLeft.x-ptBottomRight.x) + (ptBottomLeft.y-ptBottomRight.y)*(ptBottomLeft.y-ptBottomRight.y);
    int temp = (ptTopRight.x-ptBottomRight.x)*(ptTopRight.x-ptBottomRight.x) + (ptTopRight.y-ptBottomRight.y)*(ptTopRight.y-ptBottomRight.y);

    if(temp>maxLength) maxLength = temp;

    temp = (ptTopRight.x-ptTopLeft.x)*(ptTopRight.x-ptTopLeft.x) + (ptTopRight.y-ptTopLeft.y)*(ptTopRight.y-ptTopLeft.y);

    if(temp>maxLength) maxLength = temp;

    temp = (ptBottomLeft.x-ptTopLeft.x)*(ptBottomLeft.x-ptTopLeft.x) + (ptBottomLeft.y-ptTopLeft.y)*(ptBottomLeft.y-ptTopLeft.y);

    if(temp>maxLength) maxLength = temp;

    maxLength = sqrt((double)maxLength);
    Point2f source[4], dst[4];
    source[0] = ptTopLeft;            dst[0] = Point2f(0,0);
    source[1] = ptTopRight;        dst[1] = Point2f(maxLength-1, 0);
    source[2] = ptBottomRight;        dst[2] = Point2f(maxLength-1, maxLength-1);
    source[3] = ptBottomLeft;        dst[3] = Point2f(0, maxLength-1);
    
    /************************************* Warp perspective with extreme lines *****************************/

    Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    
    warpPerspective(original, undistorted, getPerspectiveTransform(source, dst), Size(maxLength, maxLength));

    //imshow("undistorted", undistorted);


    /*************************************** Number Classification *************************************/

    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 1);
    //imshow("new threshold", undistortedThreshed);
    //waitKey(0);
    
    // Create KNearest neighbors model
    Ptr<KNearest> knn;
    knn = KNearest::create();

    int numRows, numCols, numImages;
    Mat trainingVectors, trainingClasses;

    // Read the dataset images and respective labels, and put it into vectors
    loadDigitsDataset(trainingVectors, trainingClasses, numRows, numCols, numImages  );
    
    knn->setDefaultK(7);
    
    // Train model
    knn->train(trainingVectors, ml::ROW_SAMPLE,trainingClasses);
    if(knn->isTrained())
        std::cout << "Training DONE\n" << std::endl;

    /******************************************** Isolate Numbers from grid **********************************/

    int dist = ceil((double)maxLength/9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);

    Mat horizontal = undistortedThreshed.clone();
    Mat vertical = undistortedThreshed.clone();

    vector<vector<Point>> cnts;
    
    Mat thresh = undistortedThreshed.clone();
    
    // Find Sudoku Grid
    findContours(thresh, cnts, RETR_TREE, CHAIN_APPROX_SIMPLE); 
     
    for(unsigned int i = 0; i<cnts.size(); i++){ 
        double contour = contourArea(cnts[i]);
        if(contour < 500){
            drawContours(thresh, cnts, i, Scalar(0,0,0), -1, LINE_8, noArray(), 0);
        }
    }
    // Substract grid from original image to isolate numbers
    undistortedThreshed = undistortedThreshed - thresh;
    
    // Specify size on vertical axis 
    int vertical_size = 5;

    // Create structure element for extracting vertical lines through morphology operations 
    Mat vertical_kernel = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    
    // Apply morphology operations
    morphologyEx(thresh, thresh ,MORPH_CLOSE, vertical_kernel, Point(-1,-1), 2);
    
    // Specify size on horizontal axis 
    int horizontal_size = vertical_size;
    
    // Create structure element for extracting horizontal lines through morphology operations 
    Mat horizontal_kernel = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    
    // Apply morphology operations
    morphologyEx(thresh, thresh ,MORPH_CLOSE, horizontal_kernel, Point(-1,-1), 2);

    // Substract grid again from original image to better isolate numbers
    undistortedThreshed = undistortedThreshed - thresh;
    imshow("no lines", undistortedThreshed);
    //waitKey();

    /************************************* Sudoku Cells ********************************/

    /* List Creation */
    // Header node, contains pointer to the list header node of first column
    DancingNode *header = new DancingNode();

    vector<vector <int> > grid(9, vector<int>(9))  ;
    cout << "Reading Sudoku...\n" << endl;    

    // Iterate through sudoku cells
    for(int j=0;j<9;j++){
        for(int i=0;i<9;i++){
            for(int y=0;y<dist && j*dist+y<undistortedThreshed.cols;y++){

                uchar* ptr = currentCell.ptr(y);

                for(int x=0;x<dist && i*dist+x<undistortedThreshed.rows;x++){
                    ptr[x] = undistortedThreshed.at<uchar>(j*dist+y, i*dist+x);
                }
            }
            
            // Get the number of white pixels on the curent cell 
            Moments m = cv::moments(currentCell, true);
            int area = m.m00;
            
            //imshow("currentCell" , currentCell);
            //waitKey(0);
            
            // If enough white then its probably a number
            if(area > 20){
                Mat cloneImg = preprocessImage(currentCell, numRows, numCols);
                Mat response;
                float number = knn->findNearest(cloneImg, knn->getDefaultK(), response, noArray(), noArray());
                printf("%d ", (int)number);
                grid[j][i] = (int)number;
            }
            else{ // Otherwise its an empty cell
                grid[j][i] = 0;
                printf("0 ");
            }
        }
        printf("\n");
    }
    waitKey();
    
    int rowsCover = SIZE * SIZE * MAX_VALUE ;
    int colsCover = SIZE * SIZE * CONSTRAINTS;
    vector<vector<bool>> coverMatrix(rowsCover, vector<bool>(colsCover)) ;
    
    // Convert sudoku grid into cover matrix (Exact Cover Problem)
    convertInCoverMatrix(grid, coverMatrix);  
    
    // Matrix to contain nodes of linked mesh
    vector<vector<DancingNode>> Matrix(rowsCover+1, vector<DancingNode>(colsCover));

    // vector containing solutions
    vector <DancingNode*> solutions;
    
    // Add one header's row at the beginning of cover matrix to be able to convert it into Toroidal Matrix
    vector<bool> headers(colsCover);
    fill(headers.begin(), headers.end(), true);
    coverMatrix.push_back(headers);
    rotate(coverMatrix.rbegin(), coverMatrix.rbegin() + 1, coverMatrix.rend());

    // Create Toroidal Matrix of cover matrix
    createToridolMatrix(Matrix, rowsCover, colsCover, coverMatrix, header);
    
    // Solve Exact Cover Problem
    search(0, header, solutions, Matrix, grid ); 
    
    // Print resolved grid
    for(int i=0;i<9;i++){
        for(int j=0;j<9;j++){
            printf("%d ", grid[i][j]);
        }
        printf("\n");
    }

	return 0;
}


/********************************* Exact Cover Problem Functions ***********************************/

int indexInCoverMatrix(int row, int col, int num){
    return (row - 1) * SIZE * SIZE + (col- 1) * SIZE + (num - 1);
}

int createBoxConstraints(vector<vector<bool> > &coverMatrix, int header){
    for (int row = COVER_START_INDEX; row <= SIZE; row += BOX_SIZE) {
      for (int column = COVER_START_INDEX; column <= SIZE; column += BOX_SIZE) {
        for (int n = COVER_START_INDEX; n <= SIZE; n++, header++) {
          for (int rowDelta = 0; rowDelta < BOX_SIZE; rowDelta++) {
            for (int columnDelta = 0; columnDelta < BOX_SIZE; columnDelta++) {
              int index = indexInCoverMatrix(row + rowDelta, column + columnDelta, n);
              //matrix[index][header] = 1;
              coverMatrix[index][header] = true;
            }
          }
        }
      }
    }
    return header;
}

int createColumnConstraints(vector<vector<bool>> &coverMatrix, int header){
    for (int column = COVER_START_INDEX; column <= SIZE; column++) {
      for (int n = COVER_START_INDEX; n <= SIZE; n++, header++) {
        for (int row = COVER_START_INDEX; row <= SIZE; row++) {
          int index = indexInCoverMatrix(row, column, n);
            coverMatrix[index][header] = true;

        }
      }
    }

    return header;
}

int createRowConstraints(vector<vector<bool>> &coverMatrix, int header){
    for (int row = COVER_START_INDEX; row <= SIZE; row++) {
      for (int n = COVER_START_INDEX; n <= SIZE; n++, header++) {
        for (int column = COVER_START_INDEX; column <= SIZE; column++) {
            int index = indexInCoverMatrix(row, column, n);
            
            coverMatrix[index][header] = true;
        }
      }
    }

    return header;
}

int createCellConstraints(vector<vector<bool>> &coverMatrix, int header){
    for (int row = COVER_START_INDEX; row <= SIZE; row++) {
        for (int column = COVER_START_INDEX; column <= SIZE; column++, header++) {
            for (int n = COVER_START_INDEX; n <= SIZE; n++) {
                int index = indexInCoverMatrix(row, column, n);
                
            coverMatrix[index][header] = true;

            }
        }
    }
    return header;
}

void createCoverMatrix(vector<vector<bool>> &coverMatrix){ 
    //coverMatrix[SIZE*SIZE*MAX_VALUE][SIZE * SIZE * CONSTRAINTS];

    int header = 0;
    header = createCellConstraints(coverMatrix, header);
    header = createRowConstraints(coverMatrix, header);
    header = createColumnConstraints(coverMatrix, header);
    createBoxConstraints(coverMatrix, header);

}

void convertInCoverMatrix(vector<vector<int>> &grid, vector<vector<bool>> &coverMatrix){
    //createCoverMatrix(coverMatrix);
    
    int header = 0;
    header = createCellConstraints(coverMatrix, header);
    header = createRowConstraints(coverMatrix, header);
    header = createColumnConstraints(coverMatrix, header);
    createBoxConstraints(coverMatrix, header);

    // Taking into account the values already entered in Sudoku's grid instance
    for (int row = COVER_START_INDEX; row <= SIZE; row++) {
      for (int column = COVER_START_INDEX; column <= SIZE; column++) {
        int n = grid[row - 1][column - 1];
        if (n != EMPTY_CELL) {
          for (int num = MIN_VALUE; num <= MAX_VALUE; num++) {
            if (num != n) {
                std::fill(coverMatrix[indexInCoverMatrix(row, column, num)].begin(), coverMatrix[indexInCoverMatrix(row, column, num)].end(), false);
            }
          }
        }
      }
    }
}

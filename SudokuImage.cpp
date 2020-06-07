#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "digitrecognizer.h"
#include <stdlib.h>
#include <stdio.h>
#include "iostream"
#include <dirent.h>
using namespace cv;
using namespace std;
using namespace ml;

#include <stdio.h>
#include <stdint.h>

// UNASSIGNED is used for empty cells in sudoku
#define UNASSIGNED 0

// N is used for size of Sudoku grid. Size will be NxN
#define N 9

typedef unsigned char BYTE;

bool FindUnassignedLocation(int grid[N][N], int &row, int &col);

// Checks whether it will be legal to assign num to the given row,col
bool isSafe(int grid[N][N], int row, int col, int num);

//method for solving
bool SolveSudoku(int grid[N][N]){
	int row, col;

	// If there is no unassigned location,sudoku is solved
	if (!FindUnassignedLocation(grid, row, col))
	    return true;
    //std::cout << "solving .. " << std::endl;
	// numbers 1 to 9 will be tested
	for (int num = 1; num <= 9; num++){

		if (isSafe(grid, row, col, num)){
			 // make first possible legal  assignment
			grid[row][col] = num;

			if (SolveSudoku(grid))
				return true;

			grid[row][col] = UNASSIGNED;
		}
	}
	return false; // // this triggers backtracking.If no number satisfies the square this means we went wrong and it starts going back.
    // Though a genuine problem is if some num is wrong somewhere it starts filling the peviously filled square bby starting off with 1 again.Think!!!
}



bool FindUnassignedLocation(int grid[N][N], int &row, int &col){
	for (row = 0; row < N; row++)
		for (col = 0; col < N; col++)
			if (grid[row][col] == UNASSIGNED)
				return true;
	return false;
}

//checks whether the entry is in some row or column
bool UsedInRow(int grid[N][N], int row, int num){
	for (int col = 0; col < N; col++)
		if (grid[row][col] == num)
			return true;
	return false;
}


bool UsedInCol(int grid[N][N], int col, int num){
	for (int row = 0; row < N; row++)
		if (grid[row][col] == num)
			return true;
	return false;
}


bool UsedInBox(int grid[N][N], int boxStartRow, int boxStartCol, int num){
	for (int row = 0; row < 3; row++)
		for (int col = 0; col < 3; col++)
			if (grid[row+boxStartRow][col+boxStartCol] == num)
				return true;
	return false;
}
// whether it will be legal to assign num

bool isSafe(int grid[N][N], int row, int col, int num){
	/* Check if 'num' is not already placed in current row,
	current column and current 3x3 box */
	return !UsedInRow(grid, row, num) &&
		!UsedInCol(grid, col, num) &&
		!UsedInBox(grid, row - row%3 , col - col%3, num);
}


void printGrid(int grid[N][N]){
	for (int row = 0; row < N; row++){
	    for (int col = 0; col < N; col++)
			printf("%2d", grid[row][col]);
		printf("\n");
	}
}


void drawLine(Vec2f line, Mat &img, Scalar rgb = CV_RGB(0,0,255)){
    if(line[1]!=0){
        float m = -1/tan(line[1]);

        float c = line[0]/sin(line[1]);

        cv::line(img, Point(0, c), Point(img.size().width, m*img.size().width+c), rgb);
    }
    else{
        cv::line(img, Point(line[0], 0), Point(line[0], img.size().height), rgb);
    }

}

void mergeRelatedLines(vector<Vec2f> *lines, Mat &img){
    vector<Vec2f>::iterator current;
    for(current=lines->begin();current!=lines->end();current++){
        if((*current)[0]==0 && (*current)[1]==-100) continue;
        float p1 = (*current)[0];
        float theta1 = (*current)[1];
        Point pt1current, pt2current;
        if(theta1>CV_PI*45/180 && theta1<CV_PI*135/180){
            pt1current.x=0;

            pt1current.y = p1/sin(theta1);

            pt2current.x=img.size().width;
            pt2current.y=-pt2current.x/tan(theta1) + p1/sin(theta1);
        }
        else{
            pt1current.y=0;

            pt1current.x=p1/cos(theta1);

            pt2current.y=img.size().height;
            pt2current.x=-pt2current.y/tan(theta1) + p1/cos(theta1);

        }
        vector<Vec2f>::iterator    pos;
        for(pos=lines->begin();pos!=lines->end();pos++){
            if(*current==*pos) continue;
            if(fabs((*pos)[0]-(*current)[0])<20 && fabs((*pos)[1]-(*current)[1])<CV_PI*10/180){
                float p = (*pos)[0];
                float theta = (*pos)[1];
                Point pt1, pt2;
                if((*pos)[1]>CV_PI*45/180 && (*pos)[1]<CV_PI*135/180){
                    pt1.x=0;
                    pt1.y = p/sin(theta);
                    pt2.x=img.size().width;
                    pt2.y=-pt2.x/tan(theta) + p/sin(theta);
                }
                else{
                    pt1.y=0;
                    pt1.x=p/cos(theta);
                    pt2.y=img.size().height;
                    pt2.x=-pt2.y/tan(theta) + p/cos(theta);
                }
                if(((double)(pt1.x-pt1current.x)*(pt1.x-pt1current.x) + (pt1.y-pt1current.y)*(pt1.y-pt1current.y)<64*64) &&
                    ((double)(pt2.x-pt2current.x)*(pt2.x-pt2current.x) + (pt2.y-pt2current.y)*(pt2.y-pt2current.y)<64*64))
                {
                    // Merge the two
                    (*current)[0] = ((*current)[0]+(*pos)[0])/2;

                    (*current)[1] = ((*current)[1]+(*pos)[1])/2;

                    (*pos)[0]=0;
                    (*pos)[1]=-100;
                }
            }
        }
    }
}

int readFlippedInteger(FILE *fp)
{
    int ret = 0;

    BYTE *temp;

    temp = (BYTE*)(&ret);
    fread(&temp[3], sizeof(BYTE), 1, fp);
    fread(&temp[2], sizeof(BYTE), 1, fp);
    fread(&temp[1], sizeof(BYTE), 1, fp);

    fread(&temp[0], sizeof(BYTE), 1, fp);

    return ret;

}

inline uint32_t EndianSwap (uint32_t a)
{
    return (a<<24) | ((a<<8) & 0x00ff0000) |
           ((a>>8) & 0x0000ff00) | (a>>24);
}

Mat preprocessImage(Mat img, int numRows, int numCols)
{

    int rowTop=-1, rowBottom=-1, colLeft=-1, colRight=-1;

    Mat temp;
    int thresholdBottom = 50;
    int thresholdTop = 50;
    int thresholdLeft = 50;
    int thresholdRight = 50;
    int center = img.rows/2;
    for(int i=center;i<img.rows;i++){
        if(rowBottom==-1)
        {
            temp = img.row(i);
            Mat stub = temp;
            if(sum(stub).val[0] < thresholdBottom || i==img.rows-1)
                rowBottom = i;

        }

        if(rowTop==-1)
        {
            temp = img.row(img.rows-i);
            Mat stub = temp;
            if(sum(stub).val[0] < thresholdTop || i==img.rows-1)
                rowTop = img.rows-i;

        }
        if(colRight==-1)
        {
            temp = img.col(i);
            Mat stub = temp;
            if(sum(stub).val[0] < thresholdRight|| i==img.cols-1)
                colRight = i;

        }

        if(colLeft==-1)
        {
            temp = img.col(img.cols-i);
            Mat stub = temp;
            if(sum(stub).val[0] < thresholdLeft|| i==img.cols-1)
                colLeft = img.cols-i;
        }
    }
    Mat newImg;

    newImg = newImg.zeros(img.rows, img.cols, CV_8UC1);

    int startAtX = (newImg.cols/2)-(colRight-colLeft)/2;

    int startAtY = (newImg.rows/2)-(rowBottom-rowTop)/2;

    for(int y=startAtY;y<(newImg.rows/2)+(rowBottom-rowTop)/2;y++)
    {
        uchar *ptr = newImg.ptr<uchar>(y);
        for(int x=startAtX;x<(newImg.cols/2)+(colRight-colLeft)/2;x++)
        {
            ptr[x] = img.at<uchar>(rowTop+(y-startAtY),colLeft+(x-startAtX));
        }
    }

    Mat cloneImg = Mat(numRows, numCols, CV_8UC1);
    resize(newImg, cloneImg, Size(numRows, numCols), 0, 0 , INTER_NEAREST);

    // Now fill along the borders
    for(int i=0;i<cloneImg.rows;i++)
    {
        floodFill(cloneImg, Point(0, i), Scalar(0,0,0));

        floodFill(cloneImg, Point(cloneImg.cols-1, i), Scalar(0,0,0));

        floodFill(cloneImg, Point(i, 0), Scalar(0));
        floodFill(cloneImg, Point(i, cloneImg.rows-1), Scalar(0));
    }
    imshow("clone img", cloneImg);
    waitKey(0);
    cloneImg = cloneImg.reshape(1,1);
    return cloneImg;
}

bool loadDataset(string trainPath, string labelsPath, int *numImages, int *numRows, int *numCols){
    int n = trainPath.length();
    char trainName[n+1];
    strcpy(trainName, trainPath.c_str());    
    n = labelsPath.length();
    char labelsName[n+1];
    strcpy(labelsName, labelsPath.c_str());
    
    FILE *fp = fopen(trainName, "rb");
    FILE *fp2 = fopen(labelsName, "rb");

    if(!fp || !fp2){
        cout << "Could not open training files" <<endl;
        return false;
    }
    // Read bytes in flipped order
    /*int magicNumber = readFlippedInteger(fp);
    *numImages = readFlippedInteger(fp);
    *numRows = readFlippedInteger(fp);

    *numCols = readFlippedInteger(fp);
    cout << *numImages <<" " << *numRows <<" " << *numCols << endl;

    fseek(fp2, 0x08, SEEK_SET);


    if(*numImages > MAX_NUM_IMAGES) *numImages = MAX_NUM_IMAGES;
    */





    //////////////////////////////////////////////////////////////////
    // Go through each training data entry and save a

    // label for each digit

    int size = (*numRows) * (*numCols);
    Mat trainingVectors = Mat(*numImages, size, CV_32FC1);

    Mat trainingClasses = Mat(*numImages, 1, CV_32FC1);

    memset(trainingClasses.data, 0, sizeof(float)*(*numImages));

    BYTE *tempo = new BYTE[size];
    BYTE tempClass=0;
    for(int i=0;i<*numImages;i++)
    {

        fread((void*)tempo, size, 1, fp);

        fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

        trainingClasses.data[i] = tempClass;

        for(int k=0;k<size;k++)
            trainingVectors.data[i*size+k] = tempo[k]; ///sumofsquares;

    }

    fclose(fp);

    fclose(fp2);
}


int main( int argc, char* argv[] ){
	// Read original image 
	Mat src = imread("sudoku.jpg", IMREAD_GRAYSCALE );

	//if fail to read the image
	if (!src.data){
		cout << "Error loading the image" << endl;
		return -1;
	}
	
    Mat original = src.clone();
	imshow("original image",src);
    
    /**************************** Thresholding ****************************/ 

	Mat thresholded;

    Mat outerBox = Mat(src.size(), CV_8UC1);

	GaussianBlur(src, src, Size(11, 11), 0); //removing noises
	
	adaptiveThreshold(src, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);// thresholding the image
    
    bitwise_not(outerBox, outerBox);    

    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(outerBox, outerBox, kernel);
    imshow("outer box", outerBox);
	
	
    /***************************** Borders *******************************/

    int count=0;
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

    floodFill(outerBox, maxPt, CV_RGB(255,255,255));
    for(int y=0;y<outerBox.size().height;y++){
        uchar *row = outerBox.ptr(y);
        for(int x=0;x<outerBox.size().width;x++){
            if(row[x]==64 && x!=maxPt.x && y!=maxPt.y){
                floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
    erode(outerBox, outerBox, kernel);
    imshow("borders", outerBox);
    
    /************************** Hough Lines ************************/

    vector<Vec2f> lines;
    HoughLines(outerBox, lines, 1, CV_PI/180, 200);
 
    /**************************** Merge Lines *************************/

    mergeRelatedLines(&lines, src);
    for(unsigned int i=0;i<lines.size();i++){
        drawLine(lines[i], outerBox, CV_RGB(0,0,128));
    }
    
    imshow("HoughLines merged", outerBox);        

    /****************************** Extreme Lines ***************************/
   
    // Now detect the lines on extremes
    Vec2f topEdge = Vec2f(1000,1000);    double topYIntercept=100000, topXIntercept=0;
    Vec2f bottomEdge = Vec2f(-1000,-1000);        double bottomYIntercept=0, bottomXIntercept=0;
    Vec2f leftEdge = Vec2f(1000,1000);    double leftXIntercept=100000, leftYIntercept=0;
    Vec2f rightEdge = Vec2f(-1000,-1000);        double rightXIntercept=0, rightYIntercept=0;

    for(unsigned int i=0;i<lines.size();i++){
        Vec2f current = lines[i];

        float p=current[0];

        float theta=current[1];

        if(p==0 && theta==-100)
            continue;
        double xIntercept;
        double yIntercept;
        xIntercept = p/cos(theta);
        yIntercept = p/(cos(theta)*sin(theta));
        if(theta>CV_PI*80/180 && theta<CV_PI*100/180){
            if(p<topEdge[0])
                topEdge = current;

            if(p>bottomEdge[0])
                bottomEdge = current;
        }
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
    Mat undistorted = Mat(Size(maxLength, maxLength), CV_8UC1);
    
    warpPerspective(original, undistorted, getPerspectiveTransform(source, dst), Size(maxLength, maxLength));

    imshow("undistorted", undistorted);


    /******************************* Number Classification *************************/

    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 1);
    imshow("new threshold", undistortedThreshed);
    //waitKey(0);

    Ptr<KNearest> knn;
    knn = KNearest::create();

    string trainPath = "test_train/train-images.idx3-ubyte";
    string labelsPath = "test_train/train-labels.idx1-ubyte";

    
    
    knn->train(trainingVectors, ml::ROW_SAMPLE,trainingClasses);
    if(knn->isTrained())
        std::cout << "training succ" << std::endl;


    int dist = ceil((double)maxLength/9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);

    for(int j=0;j<9;j++){
        for(int i=0;i<9;i++){
            for(int y=0;y<dist && j*dist+y<undistortedThreshed.cols;y++){

                uchar* ptr = currentCell.ptr(y);

                for(int x=0;x<dist && i*dist+x<undistortedThreshed.rows;x++){
                ptr[x] = undistortedThreshed.at<uchar>(j*dist+y, i*dist+x);
                }
            }
            Moments m = cv::moments(currentCell, true);
            int area = m.m00;
            if(area > currentCell.rows*currentCell.cols/5){
                Mat cloneImg = preprocessImage(currentCell, numRows, numCols);
                imshow("after process" , cloneImg);
                waitKey(0);
                Mat response, dist;
                float number = knn->findNearest(Mat_<float>(cloneImg), 1, response, noArray(), noArray());
                cout << "P : " << number << endl;
            }
            else{
                printf("  ");
            }
        }
        printf(" ");
    }

    
	waitKey(0);
	return 0;
}
		

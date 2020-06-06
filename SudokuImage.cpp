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

// UNASSIGNED is used for empty cells in sudoku
#define UNASSIGNED 0

// N is used for size of Sudoku grid. Size will be NxN
#define N 9


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
    resize(newImg, cloneImg, Size(numRows, numCols), 1, 1 , INTER_LINEAR);

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


int main( int argc, char* argv[] ){
	// Read original image 
	Mat src = imread("sudoku.jpg", IMREAD_GRAYSCALE );
	//resize(src,src,Size(540,540),0,0,INTER_NEAREST);

	//if fail to read the image
	if (!src.data){
		cout << "Error loading the image" << endl;
		return -1;
	}
	
	//Mat srcb; // Copy of original image but in grey scale
	//cvtColor(src, srcb, COLOR_BGR2GRAY);
    Mat original = src.clone();
	imshow("original image",src);
    
    /**************************** Thresholding ****************************/ 

	Mat thresholded;

    Mat outerBox = Mat(src.size(), CV_8UC1);
    //Mat outerBox1 = Mat(src.size(), CV_8UC1);

	GaussianBlur(src, src, Size(11, 11), 0); //removing noises
	
	adaptiveThreshold(src, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);// thresholding the image
    
    bitwise_not(outerBox, outerBox);    

    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(outerBox, outerBox, kernel);
    imshow("outer box", outerBox);

	//adaptiveThreshold(src, outerBox1, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 5, 2);// thresholding the image
    //imshow("smooth image", outerBox1);	
	
    /***************************** Borders *******************************/

    vector<vector<Point>> contours; 
	vector<Vec4i> hierarchy;
	
	//findContours(outerBox, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);//FINDING CONTOUR
	
	//finding the sudoku with max area which will be our main grid
	double area; double maxarea = 0; int p;
	/*for (int i = 0; i < contours.size(); i++){
		area = contourArea(contours[i], false);
		if (area > 16){
			if (area > maxarea){
				maxarea = area;
				p = i;
			}
		}
	}

	double perimeter = arcLength(contours[p], true);
	
	approxPolyDP(contours[p], contours[p], 0.01*perimeter, true);

	drawContours(src, contours, p, Scalar(255, 0, 0), 1, 8);*/

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
                int area = floodFill(outerBox, Point(x,y), CV_RGB(0,0,0));
            }
        }
    }
    erode(outerBox, outerBox, kernel);
    imshow("borders", outerBox);

	//imshow("countour image",src);
    
    /************************** Hough Lines ************************/

    vector<Vec2f> lines;
    HoughLines(outerBox, lines, 1, CV_PI/180, 200);
 
    /**************************** Merge Lines *************************/

    mergeRelatedLines(&lines, src);
    for(int i=0;i<lines.size();i++){
        drawLine(lines[i], outerBox, CV_RGB(0,0,128));
    }
    
    imshow("HoughLines merged", outerBox);        

    /****************************** Extreme Lines ***************************/
   
    // Now detect the lines on extremes
    Vec2f topEdge = Vec2f(1000,1000);    double topYIntercept=100000, topXIntercept=0;
    Vec2f bottomEdge = Vec2f(-1000,-1000);        double bottomYIntercept=0, bottomXIntercept=0;
    Vec2f leftEdge = Vec2f(1000,1000);    double leftXIntercept=100000, leftYIntercept=0;
    Vec2f rightEdge = Vec2f(-1000,-1000);        double rightXIntercept=0, rightYIntercept=0;

    for(int i=0;i<lines.size();i++){
        Vec2f current = lines[i];

        float p=current[0];

        float theta=current[1];

        if(p==0 && theta==-100)
            continue;
        double xIntercept, yIntercept;
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
    //waitKey(0);
    //Point2f entry[4];
	//Point2f out[4];
	//double sum = 0; double prevsum = 0; int a; int b; double diff1; double diff2;  double diffprev2 = 0; double diffprev=0;double prevsum2=contours[p][0].x + contours[p][0].y;
	
	//int c; int d;
    //cout << "hey" << endl;
    /*
	for (int i = 0; i < 4; i++){
		sum = contours[p][i].x + contours[p][i].y;
		diff1 = contours[p][i].x - contours[p][i].y;
		diff2= contours[p][i].y - contours[p][i].x;
		if (diff1 > diffprev){
			diffprev = diff1;
			c = i;
		}
		if (diff2 > diffprev2){
			diffprev2 = diff2;
			d= i;
		}

		if (sum > prevsum){
			prevsum = sum; a = i;
		}
		
		if (sum < prevsum2){
		    prevsum2 = sum;
			b = i;
		}
	}
	
	entry[0] = contours[p][a];
	entry[1] = contours[p][b];
	entry[2] = contours[p][c];
	entry[3] = contours[p][d];

	out[0] = Point2f(450, 450);
	out[1] = Point2f(0, 0);
	out[2] = Point2f(450, 0);
	out[3] = Point(0, 450);

    */

	Mat wrap; Mat mat; 

	/*mat = Mat::zeros(src.size(), src.type());
	
	wrap = getPerspectiveTransform(entry, out);
	
	warpPerspective(src, mat, wrap, Size(450, 450));

	imshow("sudoku part",mat);

	Mat ch; Mat thresholded2;

	cvtColor(mat,ch,COLOR_BGR2GRAY);

	GaussianBlur(ch, ch, Size(11, 11), 0, 0);

	adaptiveThreshold(ch, thresholded2, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	bitwise_not(thresholded2, thresholded2);

	//Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    //dilate(thresholded2, thresholded2, kernel,Point(-1,-1),1);

	erode(thresholded2,thresholded2,2);

    int p2=0;int p3=0;
	
    while(p3<450){
        for(int i=p3;i<p3+10;i++){
	        for(int j=0;j<450;j++){
		        thresholded2.at<uchar>(j,i)=0;
	        }
        }
        p3=p3+50;
    }

    while(p2<450){
        for( int i=0;i<450;i++){
	        for(int j=p2;j<p2+10;j++){
		        thresholded2.at<uchar>(j,i)=0;
	        }
        }
        p2=p2+50;
    }

    for(int i=440;i<450;i++){
	    for(int j=0;j<450;j++){
		    thresholded2.at<uchar>(j,i)=0;
	    }
    }

    for(int i=0;i<450;i++){
	for(int j=440;j<450;j++){
		thresholded2.at<uchar>(j,i)=0;
	    }
    }

    for(int i=0;i<450;i++){
	    for(int j=150;j<160;j++){
		    thresholded2.at<uchar>(j,i)=0;
	    }

    }

	imshow("thresholded new",thresholded2);
	*/

    /******************************* Number Classification *************************/

    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 1);
    imshow("new threshold", undistortedThreshed);
    //waitKey(0);

    /*DigitRecognizer *dr = new DigitRecognizer();
    bool b = dr->train("test_train/train-images.idx3-ubyte", "test_train/train-labels.idx1-ubyte");
    if(!b)
        cout << "NOT TRAINING" << endl;
    */

    int num = 797;
    int size = 16 * 16;
    Mat trainData = Mat(Size(size, num), CV_32FC1);
    Mat responces = Mat(Size(1, num), CV_32FC1);

    int counter = 0;
    for(int i=0;i<=9;i++){		
    	DIR *dir;
    	struct dirent *ent;
    	char pathToImages[]="./digits3";
    	char path[255];
    	sprintf(path, "%s/%d", pathToImages, i);
    	if ((dir = opendir(path)) != NULL){		
	        while ((ent = readdir (dir)) != NULL){ 
	            if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 ){
	                char text[255];
	                sprintf(text,"/%s",ent->d_name);
	                string digit(text);
	                digit=path+digit; 

					Mat mat=imread(digit,1);
				
					cvtColor(mat,mat,COLOR_BGR2GRAY);

					threshold(mat , mat , 200, 255 ,THRESH_OTSU);					

					mat.convertTo(mat,CV_32FC1,1.0/255.0);

					resize(mat, mat, Size(16,16 ),0,0,INTER_NEAREST);
					
					mat.reshape(1,1);


	                for (int k=0; k<size; k++){
	                    trainData.at<float>(counter*size+k) = mat.at<float>(k);
	                }
	                responces.at<float>(counter) = i;
	                counter++;
	        	}
	         
	        }
        	
        }
        closedir(dir);
    }
    

    Ptr<KNearest> knn;
    knn = KNearest::create();

    knn->train(trainData,ROW_SAMPLE,responces);
 
    
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
            //imshow("current cell", currentCell);
            Mat cellClone = currentCell.clone();
            if(area > cellClone.rows*cellClone.cols/5){
                cellClone = preprocessImage(cellClone, 16, 16);
                cellClone.convertTo(cellClone, CV_32FC1, 1.0/255.0);             
                waitKey(0); 
		        Mat response, dist;
                resize(cellClone, cellClone, Size(16,16 ),0,0,INTER_NEAREST);		
			    float p=knn->findNearest(cellClone.reshape(1,1),1, noArray(), response, dist);
                cout << "P: " << p << endl;
                //cout << p <<" "; 
            }
            else{
                //cout <<" ";
            }
        }
        //cout << " " << endl; 
    }
    cout << endl;

    //vector <Mat> small; vector <Mat> smallt;
	
 
    /*int m = 0, n = 0; Mat smallimage; Mat smallimage2;
	for (; m < 450; m = m + 50){
		for (n = 0; n < 450; n = n + 50){ 
			smallimage = Mat(undistortedThreshed, cv::Rect(n, m, 50, 50));
					
			smallt.push_back(smallimage);
		}
	}
    
    int z[9][9];
	for(size_t i=0;i<smallt.size();i++){
		Mat img123 =Mat(Size(size, 1), CV_32FC1);
		if(countNonZero(smallt[i])>200){
		
			Mat thresholded3; Mat regionOfInterest; Mat img12;
		
			thresholded3=smallt[i].clone();

			vector < vector <Point> >contours2;
			
			findContours(thresholded3, contours2, RETR_LIST, CHAIN_APPROX_SIMPLE);

			Rect prevb; double areaprev = 0; double area2; int q;

			for (int j = 0; j < contours2.size(); j++){
				Rect bnd = boundingRect(contours2[j]);
					    
				area2 = bnd.height*bnd.width;
				
				if (area2 > areaprev){
					prevb = bnd;
					areaprev = area2;
				    q = j;
				}
			}
			
            Rect rec = prevb;
	
			regionOfInterest = smallt[i](rec);

			resize(regionOfInterest, img12, Size(16,16),0,0,INTER_NEAREST);

			img12.convertTo(img12,CV_32FC1,1.0/255.0);                 
			img12.reshape(1,1);   

			Mat output;
			if(countNonZero(img12)>50){
				imshow("display",img12);
				waitKey(0);			
			    for(int k=0;k<size;k++){
				    img123.at<float>(k) = img12.at<float>(k);
			    }
	            		
		        Mat response, dist;		
			    float p=knn->findNearest(img123.reshape(1,1),1, noArray(), response, dist);
			
			    z[i/9][i%9]=p;
			}
		    else
                z[i/9][i%9]=0;
		}
		else z[i/9][i%9]=0;
    }
    
	for(int i=0;i<9;i++){
		for(int j=0;j<9;j++){
			//cout << z[i][j]<<" ";
		}
		cout<<endl;
	}

		
	int grid[N][N];

	for(int i=0;i<9;i++){
		for(int j=0;j<9;j++){
			//grid[i][j]=z[i][j];
		}
	}

	if (SolveSudoku(grid) == true)
		printGrid(grid);
	else	
		cout << "please correct" << endl;
		
    */
	waitKey(0);
	return 0;
}
		

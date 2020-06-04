#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

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





int main( int argc, char* argv[] ){
	// Read original image 
	Mat src = imread("sudoku.jpg",IMREAD_UNCHANGED );
	resize(src,src,Size(540,540),0,0,INTER_NEAREST);

	//if fail to read the image
	if (!src.data){
		cout << "Error loading the image" << endl;
		return -1;
	}
	
	Mat srcb; // Copy of original image but in grey scale
	cvtColor(src, srcb, COLOR_BGR2GRAY);

	imshow("original image",src);
	
	Mat smooth;
	Mat thresholded;
	
	GaussianBlur(srcb, smooth, Size(11, 11), 0, 0); //removing noises
	
	adaptiveThreshold(smooth, thresholded, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 15, 5);// thresholding the image
	
	Mat thresholded2 = thresholded.clone();//creating a copy
	
	imshow("smooth image",thresholded);
	
	vector< vector < Point > >contours; 
	vector <Vec4i> hierarchy;
	
	findContours(thresholded2, contours, hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE);//FINDING CONTOUR
	
	//finding the sudoku with max area which will be our main grid
	double area; double maxarea = 0; int p;
	for (int i = 0; i < contours.size(); i++){
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

	drawContours(src, contours, p, Scalar(255, 0, 0), 1, 8);
	


	imshow("countour image",src);

	
	

	Point2f entry[4];
	Point2f out[4];

	
	double sum = 0; double prevsum = 0; int a; int b; double diff1; double diff2;  double diffprev2 = 0; double diffprev=0;double prevsum2=contours[p][0].x + contours[p][0].y;
	
	int c; int d;
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


	Mat wrap; Mat mat; 

	mat = Mat::zeros(src.size(), src.type());
	
	wrap = getPerspectiveTransform(entry, out);
	
	warpPerspective(src, mat, wrap, Size(450, 450));

	imshow("sudoku part",mat);

	Mat ch; Mat thresholded31;

	cvtColor(mat,ch,COLOR_BGR2GRAY);

	GaussianBlur(ch, ch, Size(11, 11), 0, 0);

	adaptiveThreshold(ch, thresholded31, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);
	bitwise_not(thresholded31, thresholded31);

	Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(thresholded31, thresholded31, kernel,Point(-1,-1),1);

	erode(thresholded31,thresholded31,2);

    int p2=0;int p3=0;
	
    while(p3<450){
        for(int i=p3;i<p3+10;i++){
	        for(int j=0;j<450;j++){
		        thresholded31.at<uchar>(j,i)=0;
	        }
        }
        p3=p3+50;
    }

    while(p2<450){
        for( int i=0;i<450;i++){
	        for(int j=p2;j<p2+10;j++){
		        thresholded31.at<uchar>(j,i)=0;
	        }
        }
        p2=p2+50;
    }

    for(int i=440;i<450;i++){
	    for(int j=0;j<450;j++){
		    thresholded31.at<uchar>(j,i)=0;
	    }
    }

    for(int i=0;i<450;i++){
	for(int j=440;j<450;j++){
		thresholded31.at<uchar>(j,i)=0;
	    }
    }

    for(int i=0;i<450;i++){
	    for(int j=150;j<160;j++){
		    thresholded31.at<uchar>(j,i)=0;
	    }

    }

	imshow("thresholded new",thresholded31);
	
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
 	
	
    vector <Mat> small; vector <Mat> smallt;
	
 
    int m = 0, n = 0; Mat smallimage; Mat smallimage2;
	    for (; m < 450; m = m + 50){
		    for (n = 0; n < 450; n = n + 50){ 
			    smallimage = Mat(thresholded31, cv::Rect(n, m, 50, 50));
					
			    smallt.push_back(smallimage);
		    }
	    }

	
	
    int z[9][9];
	for(size_t i=0;i<smallt.size();i++){
		Mat img123 =Mat(Size(size, 1), CV_32FC1);
		if(countNonZero(smallt[i])>200){
		
			Mat thresholded32; Mat regionOfInterest; Mat img12;
		
			thresholded32=smallt[i].clone();

			vector < vector <Point> >contours2;
			
			findContours(thresholded32, contours2, RETR_LIST, CHAIN_APPROX_SIMPLE);

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
			cout << z[i][j]<<" ";
		}
		cout<<endl;
	}

		
	int grid[N][N];

	for(int i=0;i<9;i++){
		for(int j=0;j<9;j++){
			grid[i][j]=z[i][j];
		}
	}

	if (SolveSudoku(grid) == true)
		printGrid(grid);
	else	
		cout << "please correct" << endl;
		

	waitKey(0);
	return 0;
}
		

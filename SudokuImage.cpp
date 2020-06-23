#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "digitrecognizer.h"
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <dirent.h>
using namespace cv;
using namespace std;
using namespace ml;

#include <stdio.h>

// UNASSIGNED is used for empty cells in sudoku
#define UNASSIGNED 0

// N is used for size of Sudoku grid. Size will be NxN
#define N 9

typedef unsigned char BYTE;


/*HOGDescriptor hog(
        Size(20,20), //winSize
        Size(10,10), //blocksize
        Size(5,5), //blockStride,
        Size(10,10), //cellSize,
                 9, //nbins,
                  1, //derivAper,
                 -1, //winSigma,
                  0, //histogramNormType,
                0.2, //L2HysThresh,
                  1,//gammal correction,
                  64,//nlevels=64
                  1);//Use signed gradients
*/

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
    /*for(int i=0;i<cloneImg.rows;i++)
    {
        floodFill(cloneImg, Point(0, i), Scalar(0,0,0));

        floodFill(cloneImg, Point(cloneImg.cols-1, i), Scalar(0,0,0));

        floodFill(cloneImg, Point(i, 0), Scalar(0));
        floodFill(cloneImg, Point(i, cloneImg.rows-1), Scalar(0));
    }*/
    Mat realClone = cloneImg.clone();
    vector<vector<Point>> countours;
    findContours(cloneImg, countours, RETR_LIST, CHAIN_APPROX_SIMPLE);
    Rect prevb; double areaprev = 0; double area; int q;
    for(int i = 0; i<countours.size();i++){
        Rect bnd = boundingRect(countours[i]);
        area = bnd.height*bnd.width;
        if(area > areaprev){
            prevb = bnd;
            areaprev = area;
            q = i;
        }
    }
    Rect rec = prevb;
    Mat region = realClone(rec);
    if(!region.empty())
        resize(region, cloneImg, Size(numRows, numCols), 0, 0, INTER_NEAREST);   

    cloneImg.convertTo(cloneImg, CV_32FC1, 1.0/255.0 );
    imshow("clone img", cloneImg);
    waitKey(0);
    cloneImg = cloneImg.reshape(1,1);
    return cloneImg;
}

int reverseInt(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool loadMNIST(string trainPath, string labelsPath, int &numRows, int &numCols, int &numImages, Mat &trainingVectors, Mat &trainingClasses){
    int n = trainPath.length();
    char trainName[n+1];
    strcpy(trainName, trainPath.c_str());    
    n = labelsPath.length();
    char labelsName[n+1];
    strcpy(labelsName, labelsPath.c_str());

    ifstream file (trainName, ios::binary);
    if(file.is_open()){
        int magic_number=0;int r; int c;
        Size size;unsigned char temp=0;

        file.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        file.read((char*)&numImages,sizeof(numImages));
        numImages= reverseInt(numImages);
        
        file.read((char*)&numRows,sizeof(numRows));
        numRows= reverseInt(numRows); 
        file.read((char*)&numCols,sizeof(numCols));
        numCols= reverseInt(numCols);
        printf("%d %d %d\n", numImages, numRows, numCols);
        trainingVectors = Mat(numImages, numRows*numCols, CV_8UC1);
        unsigned char arr[28][28];
        for(int i=0;i<numImages;i++){
            for(r=0;r<numRows;r++){
                for(c=0;c<numCols;c++){                 
                    file.read((char*)&temp,sizeof(temp));
                    arr[r][c]= temp;
                    trainingVectors.at<uchar>(i ,r*numCols+c) = arr[r][c];
                    //printf("%1.1f ", (float)trainingVectors.at<uchar>(i,r*numCols+c)/255.0 );
                }          
                //printf("\n"); 
            }
            size.height=r;size.width=c;
            //printf("\n");
        }
    }

    ifstream fileLabels (labelsName, ios::binary);
    if(fileLabels.is_open()){
        int magic_number=0;int r; int c;
        Size size;unsigned char temp=0;

        fileLabels.read((char*)&magic_number,sizeof(magic_number)); 
        magic_number= reverseInt(magic_number);

        fileLabels.read((char*)&numImages,sizeof(numImages));
        numImages= reverseInt(numImages);
        
        //printf("%d\n", numImages);
        trainingClasses = Mat(numImages, 1, CV_8UC1);
        for(int i=0;i<numImages;i++){
            fileLabels.read((char*)&temp,sizeof(temp));
            trainingClasses.at<uchar>(1, i) = temp;
            //printf("%d\n", trainingClasses.at<uchar>(1,i));
        }
    }

    /*for(int i=0;i<numImages;i++){
        for(int c=0;c<784;c++){
            printf("%1.1f ", (float)trainingVectors.at<uchar>(i,c)/255.0 );
            if((c+1) % 28 == 0) putchar('\n');

        }
        printf("\n");
    }*/
    return true;
}


/*
Mat deskew(Mat& img)
{
    Moments m = moments(img);
    if(abs(m.mu02) < 1e-2)
    {
        // No deskewing needed. 
        return img.clone();
    }
    // Calculate skew based on central momemts. 
    double skew = m.mu11/m.mu02;
    // Calculate affine transform to correct skewness. 
    Mat warpMat = (Mat_<double>(2,3) << 1, skew, -0.5*SZ*skew, 0, 1 , 0);

    Mat imgOut = Mat::zeros(img.rows, img.cols, img.type());
    warpAffine(img, imgOut, warpMat, imgOut.size(),affineFlags);

    return imgOut;
}*/


bool loadDigitsDataset(Mat &trainData, Mat &responces, int &numRows, int &numCols, int &numImages){
    int num = 775;
    numImages = num;
    int size = 16 * 16;
    trainData = Mat(Size(size, num), CV_32FC1);
    responces = Mat(Size(1, num), CV_32FC1);
    int counter = 0;
    for(int i=0;i<=9;i++){
        // reading the images from the folder of tarining samples
        DIR *dir;
        struct dirent *ent;
        char pathToImages[]="./digits3"; // name of the folder containing images
        char path[255];
        sprintf(path, "%s/%d", pathToImages, i);
        if ((dir = opendir(path)) != NULL){
            while ((ent = readdir (dir)) != NULL){
                if (strcmp(ent->d_name, ".") != 0 && strcmp(ent->d_name, "..") != 0 ){
                    char text[255];
                    sprintf(text,"/%s",ent->d_name);
                    string digit(text);
                    digit=path+digit;
                    Mat mat=imread(digit,1); //loading the image
                    cvtColor(mat,mat, COLOR_RGB2GRAY);  //converting into grayscale
                    threshold(mat , mat , 200, 255 ,THRESH_OTSU); // preprocessing
                    mat.convertTo(mat,CV_32FC1,1.0/255.0); //necessary to convert images to CV_32FC1 for using K nearest neighbour algorithm
                    numRows = 16;
                    numCols = 16;
                    resize(mat, mat, Size(numRows, numCols ),0,0,INTER_NEAREST); // same size as our testing samples
                    //cout << "number " << i << endl;
                    //imshow("mat", mat);
                    //waitKey(0);
                    //cout << "M = " << endl << " " << mat << endl << endl;
                    mat.reshape(1,1);
                    for (int k=0; k<size;k++) {
                        trainData.at<float>(counter*size+k) = mat.at<float>(k); // storing the pixels of the image
                          
                        //trainData.at<float>(i ,counter*numCols+k) = mat.at<float>(k);
                    }

                    responces.at<float>(counter) = i; // stroing the responce corresponding to image
                    counter++;
                }
            }
        }
        closedir(dir);
    }
    return true;
}


int main( int argc, char* argv[] ){
	// Read original image 
	Mat src = imread("sudoku.jpg", IMREAD_GRAYSCALE );

	//if fail to read the image
	if (!src.data){
		cout << "Error loading the image" << endl;
		return -12;
	}
	
    Mat original = src.clone();
	imshow("original image",src);
    //waitKey();
    
    /**************************** Thresholding ****************************/ 

	Mat thresholded;

    Mat outerBox = Mat(src.size(), CV_8UC1);

	GaussianBlur(src, src, Size(11, 11), 0); //removing noises
	
	adaptiveThreshold(src, outerBox, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 5, 2);// thresholding the image
    
    bitwise_not(outerBox, outerBox);    

    Mat kernel = (Mat_<uchar>(3,3) << 0,1,0,1,1,1,0,1,0);
    dilate(outerBox, outerBox, kernel);
    //imshow("outer box", outerBox);
	
	
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
    //imshow("borders", outerBox);
    
    /************************** Hough Lines ************************/

    vector<Vec2f> lines;
    HoughLines(outerBox, lines, 1, CV_PI/180, 200);
 
    /**************************** Merge Lines *************************/

    mergeRelatedLines(&lines, src);
    for(unsigned int i=0;i<lines.size();i++){
        drawLine(lines[i], outerBox, CV_RGB(0,0,128));
    }
    
    //imshow("HoughLines merged", outerBox);        

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

    //imshow("undistorted", undistorted);


    /******************************* Number Classification *************************/

    Mat undistortedThreshed = undistorted.clone();
    adaptiveThreshold(undistorted, undistortedThreshed, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 101, 1);
    imshow("new threshold", undistortedThreshed);
    //waitKey(0);
    //cout << "M = " << endl << " " << undistortedThreshed << endl << endl;

    Ptr<KNearest> knn;
    knn = KNearest::create();

    string trainPath = "test_train/train-images.idx3-ubyte";
    string labelsPath = "test_train/train-labels.idx1-ubyte";

    int numRows, numCols, numImages;
    Mat trainingVectors, trainingClasses;

    //loadMNIST(trainPath, labelsPath, numRows, numCols, numImages, trainingVectors, trainingClasses  );
     
    loadDigitsDataset(trainingVectors, trainingClasses, numRows, numCols, numImages  );
    //trainingVectors.convertTo(trainingVectors, CV_32FC1, 1.0/255.0);
    //trainingClasses.convertTo(trainingClasses, CV_32SC1);
    for(int i = 0; i<numImages; i++){
        printf("Number = %1.1f\n",trainingClasses.at<float>(1, i));
        for(int j = 0; j<256; j++){
           printf("%1.1f ",trainingVectors.at<float>(i, j));
           if((j+1) % 16 == 0) putchar('\n');
        }
        putchar('\n');
    }

    knn->setDefaultK(7);

    knn->train(trainingVectors, ml::ROW_SAMPLE,trainingClasses);
    if(knn->isTrained())
        std::cout << "training succ" << std::endl;

    int dist = ceil((double)maxLength/9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);


    Mat horizontal = undistortedThreshed.clone();
    Mat vertical = undistortedThreshed.clone();

    vector<vector<Point>> cnts;
    
    Mat thresh = undistortedThreshed.clone();

    findContours(thresh, cnts, RETR_TREE, CHAIN_APPROX_SIMPLE); 
    
     
    for(int i = 0; i<cnts.size(); i++){ 
        double contour = contourArea(cnts[i]);
        if(contour < 1000){
            drawContours(thresh, cnts, i, Scalar(0,0,0), -1, LINE_8, noArray(), 0);
        }
    }
    
    undistortedThreshed = undistortedThreshed - thresh;

    imshow("d", thresh);
    waitKey(0);
    
    int vertical_size = vertical.cols/30; 
    
    Mat vertical_kernel = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    morphologyEx(thresh, thresh ,MORPH_CLOSE, vertical_kernel, Point(-1,-1), 2);
    
    //erode(vertical, vertical, vertical_kernel, Point(-1, -1));
    //dilate(vertical, vertical, vertical_kernel, Point(-1,-1));
    imshow("vertical", thresh);
    waitKey(0);

    int horizontal_size = horizontal.rows/3;
    Mat horizontal_kernel = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    //morphologyEx(undistortedThreshed, undistortedThreshed ,MORPH_CLOSE, horizontal_kernel, Point(-1,1), 4);
    erode(horizontal, horizontal, horizontal_kernel, Point(-1, -1));
    dilate(horizontal, horizontal, horizontal_kernel, Point(-1,-1));
  
    imshow("hori", horizontal);
    waitKey();

    undistortedThreshed = undistortedThreshed - horizontal;
    undistortedThreshed = undistortedThreshed - vertical;

    imshow("loll", undistortedThreshed);
    waitKey();


    for(int j=0;j<9;j++){
        for(int i=0;i<9;i++){
            for(int y=0;y<dist && j*dist+y<undistortedThreshed.cols;y++){

                uchar* ptr = currentCell.ptr(y);

                for(int x=0;x<dist && i*dist+x<undistortedThreshed.rows;x++){
                    ptr[x] = undistortedThreshed.at<uchar>(j*dist+y, i*dist+x);
                }
            }
             
            
            // Now fill along the borders
            Moments m = cv::moments(currentCell, true);
            int area = m.m00;
            imshow("currentCell" , currentCell);
            waitKey(0);

            if(area > 20){
                Mat cloneImg = preprocessImage(currentCell, numRows, numCols);
                Mat response;
                float number = knn->findNearest(cloneImg, knn->getDefaultK(), response, noArray(), noArray());
                printf("%d ", (int)number);
            }
            else{
                printf("0 ");
            }
        }
        printf("\n");
    }

    
	//waitKey();
	return 0;
}
		

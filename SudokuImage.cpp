#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>

#include "ImageProcessing.h"
using namespace ml;

int main( int argc, char* argv[] ){
	// Read original image 
	Mat src = imread("testPhotos/sudoku.jpg", IMREAD_GRAYSCALE );

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
    Vec2f topEdge = Vec2f(1000,1000);    
    Vec2f bottomEdge = Vec2f(-1000,-1000);        
    Vec2f leftEdge = Vec2f(1000,1000);    double leftXIntercept=100000;
    Vec2f rightEdge = Vec2f(-1000,-1000);        double rightXIntercept=0;

    for(unsigned int i=0;i<lines.size();i++){
        Vec2f current = lines[i];

        float p=current[0];

        float theta=current[1];

        if(p==0 && theta==-100)
            continue;
        double xIntercept;
        xIntercept = p/cos(theta);
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
    //imshow("new threshold", undistortedThreshed);
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
    /*for(int i = 0; i<numImages; i++){
        printf("Number = %1.1f\n",trainingClasses.at<float>(1, i));
        for(int j = 0; j<256; j++){
           printf("%1.1f ",trainingVectors.at<float>(i, j));
           if((j+1) % 16 == 0) putchar('\n');
        }
        putchar('\n');
    }
    */
    knn->setDefaultK(7);

    knn->train(trainingVectors, ml::ROW_SAMPLE,trainingClasses);
    if(knn->isTrained())
        std::cout << "training DONE" << std::endl;

    int dist = ceil((double)maxLength/9);
    Mat currentCell = Mat(dist, dist, CV_8UC1);


    Mat horizontal = undistortedThreshed.clone();
    Mat vertical = undistortedThreshed.clone();

    vector<vector<Point>> cnts;
    
    Mat thresh = undistortedThreshed.clone();

    findContours(thresh, cnts, RETR_TREE, CHAIN_APPROX_SIMPLE); 
    
     
    for(unsigned int i = 0; i<cnts.size(); i++){ 
        double contour = contourArea(cnts[i]);
        if(contour < 500){
            drawContours(thresh, cnts, i, Scalar(0,0,0), -1, LINE_8, noArray(), 0);
        }
    }
    
    undistortedThreshed = undistortedThreshed - thresh;

    //imshow("d", thresh);
    //waitKey(0);
    
    int vertical_size = vertical.cols/50;
    vertical_size = 5; 
    Mat vertical_kernel = getStructuringElement(MORPH_RECT, Size(1, vertical_size));
    morphologyEx(thresh, thresh ,MORPH_CLOSE, vertical_kernel, Point(-1,-1), 2);
    
    //erode(vertical, vertical, vertical_kernel, Point(-1, -1));
    //dilate(vertical, vertical, vertical_kernel, Point(-1,-1));
    //imshow("vertical", thresh);
    //waitKey(0);

    int horizontal_size = horizontal.rows/50;
    horizontal_size = vertical_size;
    Mat horizontal_kernel = getStructuringElement(MORPH_RECT, Size(horizontal_size, 1));
    morphologyEx(thresh, thresh ,MORPH_CLOSE, horizontal_kernel, Point(-1,-1), 2);


    //erode(horizontal, horizontal, horizontal_kernel, Point(-1, -1));
    //dilate(horizontal, horizontal, horizontal_kernel, Point(-1,-1));
  
    //imshow("hori",thresh);
    //waitKey();

    //undistortedThreshed = undistortedThreshed - horizontal;
    
    //undistortedThreshed = undistortedThreshed - vertical;
    undistortedThreshed = undistortedThreshed - thresh;
    imshow("no lines", undistortedThreshed);
    waitKey();


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
		

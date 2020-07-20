#include "ImageProcessing.h"

/*********************************** Image Processing *************************************/
 
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
    //newImg = img.clone();

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
    Rect prevb; double areaprev = 0; double area; 
    for(unsigned int i = 0; i<countours.size();i++){
        Rect bnd = boundingRect(countours[i]);
        area = bnd.height*bnd.width;
        if(area > areaprev){
            prevb = bnd;
            areaprev = area;
        }
    }
    Rect rec = prevb;
    Mat region = realClone(rec);
    if(!region.empty())
        resize(region, cloneImg, Size(numRows, numCols), 0, 0, INTER_NEAREST);   

    cloneImg.convertTo(cloneImg, CV_32FC1, 1.0/255.0 );
    //imshow("clone img", cloneImg);
    //waitKey(0);
    cloneImg = cloneImg.reshape(1,1);
    return cloneImg;
}

/******************************************** Load Datasets **********************************************/


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
                    char text[257];
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


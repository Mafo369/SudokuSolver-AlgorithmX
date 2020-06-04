#include "digitrecognizer.h"

typedef unsigned char BYTE;

DigitRecognizer::DigitRecognizer()
{
    knn = KNearest::create();

}
   
DigitRecognizer::~DigitRecognizer()
{
    delete knn;
}

int DigitRecognizer::readFlippedInteger(FILE *fp)
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

bool DigitRecognizer::train(string trainPath, string labelsPath)
{
    int n = trainPath.length();
    char trainName[n+1];
    strcpy(trainName, trainPath.c_str());    
    n = labelsPath.length();
    char labelsName[n+1];
    strcpy(labelsName, labelsPath.c_str());
    
    FILE *fp = fopen(trainName, "rb");
    FILE *fp2 = fopen(labelsName, "rb");

    if(!fp || !fp2)
        return false;
    // Read bytes in flipped order
    int magicNumber = readFlippedInteger(fp);
    numImages = readFlippedInteger(fp);
    numRows = readFlippedInteger(fp);

    numCols = readFlippedInteger(fp);

    fseek(fp2, 0x08, SEEK_SET);


    if(numImages > MAX_NUM_IMAGES) numImages = MAX_NUM_IMAGES;

    //////////////////////////////////////////////////////////////////
    // Go through each training data entry and save a

    // label for each digit

    int size = numRows*numCols;

    Mat trainingVectors = Mat(numImages, size, CV_32FC1);

    Mat trainingClasses = Mat(numImages, 1, CV_32FC1);


    memset(trainingClasses.data, 0, sizeof(float)*numImages);

    BYTE temp[size];
    BYTE tempClass=0;
    for(int i=0;i<numImages;i++)
    {

        fread((void*)temp, size, 1, fp);

        fread((void*)(&tempClass), sizeof(BYTE), 1, fp2);

        trainingClasses.data[i] = tempClass;

        for(int k=0;k<size;k++)
            trainingVectors.data[i*size+k] = temp[k]; ///sumofsquares;

    }

    fclose(fp);

    fclose(fp2);
    
    //Ptr<KNearest> kk1 = KNearest::create();
    //kk1->train(trainingVectors, ml::ROW_SAMPLE, trainingClasses);   
    //std::cout << "Hey" << std::endl;
    bool l = knn->train(trainingVectors, ml::ROW_SAMPLE,trainingClasses);
    if(!l)
        std::cout << "training failed" << std::endl;

    return true;
}

int DigitRecognizer::classify(cv::Mat img)
{
    Mat cloneImg = preprocessImage(img);
    imshow("number", cloneImg);
    waitKey(0);
    Mat response, dist;
    return knn->findNearest(Mat_<float>(cloneImg), 1, noArray(), response, dist);
}

Mat DigitRecognizer::preprocessImage(Mat img)
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

    cloneImg = cloneImg.reshape(1, 1);

    return cloneImg;
}



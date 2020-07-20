#pragma once

#include <opencv2/opencv.hpp>
#include <stdlib.h>
#include <stdio.h>
#include <iostream>
#include <dirent.h>
#include <fstream>

using namespace cv;
using namespace std;

void drawLine(Vec2f line, Mat &img, Scalar rgb);
   
void mergeRelatedLines(vector<Vec2f> *lines, Mat &img);

Mat preprocessImage(Mat img, int numRows, int numCols);
 
bool loadDigitsDataset(Mat &trainData, Mat &responces, int &numRows, int &numCols, int &numImages);

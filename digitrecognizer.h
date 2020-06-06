#include <opencv2/opencv.hpp>
#include <opencv2/ml.hpp>
#include <string>
#include <cerrno>

using namespace cv;
using namespace ml;
using namespace std;
#define MAX_NUM_IMAGES    60000
class DigitRecognizer
{
public:
    DigitRecognizer();

    ~DigitRecognizer();

    bool train(string trainPath, string labelsPath);

    float classify(Mat img);

private:
    Mat preprocessImage(Mat img);

    int readFlippedInteger(FILE *fp);

private:
    Ptr<KNearest> knn;
    int numRows, numCols, numImages;

};

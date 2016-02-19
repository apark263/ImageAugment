#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <functional>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <map>
#include "media.hpp"

using namespace cv;
using namespace std;
typedef function<void (Mat &, Mat &, void *)> transform_func;
typedef map<string, int> KVmap;

KVmap parseFile(char *filename) {
    ifstream ifs;
    ifs.open(filename);
    KVmap dx;
    while (!ifs.eof()) {
        string key, val;
        getline(ifs, key, ',');
        getline(ifs, val);
        if (!key.empty()) {
            dx.insert(pair<string, int> (key, atoi(val.c_str())));
        }
    }
    ifs.close();
    return dx;
}

int main( int argc, char** argv ) {
    ImageParams ip;
    Mat orig = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Size2f origSize(orig.cols, orig.rows);
    Size2f finalSize(224, 224);
    if (argc > 2) {
        ip.set_keys(parseFile(argv[2]));
    }

    Image itt(ip);

    for (int i = 0; i < 10; i++) {
        Mat outMat;
        char fname[256];
        sprintf(fname, "example_%02d.jpg", i);
        itt.transform(orig, outMat);
        imwrite(fname, outMat);
    }

    return 0;
}

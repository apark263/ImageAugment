#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <time.h>
#include <functional>
#include <fstream>
#include <iostream>


using namespace cv;
using namespace std;
typedef function<void (Mat &, Mat &, void *)> transform_func;
typedef Rect_<float> Rectf;

Rectf proposeROI(Size2f origSize, float areaScale, float aspectRatio);

void flipImage(Mat &inMat, Mat &outMat, void *varg) {
    int *doFlip = static_cast<int *>(varg);
    if (*doFlip != 0) {
        flip(inMat, outMat, 1);
    } else {
        outMat = inMat;
    }
}

void cropImage(Mat &inMat, Mat &outMat, void *varg) {
    Rectf *roi = static_cast<Rectf *>(varg);
    outMat = inMat(*roi);
}

void adjustImage(Mat &inMat, Mat &outMat, void *varg) {
    float *adjustVals = static_cast<float *>(varg);
    inMat.convertTo(inMat, -1, 1.0f + adjustVals[0], adjustVals[1]);
}

void rotateImage(Mat &inMat, Mat &outMat, void *varg) {
    float angle = *static_cast<float *> (varg);
    if (angle == 0.0f) {
        outMat = inMat;
        return;
    }
    warpAffine( inMat, outMat,
                getRotationMatrix2D( Point2f(inMat.size()) * 0.5f, angle, 1.0f ),
                inMat.size() );
}

void resizeImage(Mat &inMat, Mat &outMat, void *varg) {
    Size2f *finalSize = static_cast<Size2f *>(varg);
    int interp_method = inMat.size().area() < finalSize->area() ? CV_INTER_AREA : CV_INTER_CUBIC;
    resize(inMat, outMat, *finalSize, 0, 0, interp_method);
}

Rectf proposeROI(Size2f origSize, float areaScale, float cAR, RNG& rng) {
    // oAR and cAR are the original and crop aspect ratios, respectively
    float oAR = origSize.width / origSize.height;
    float maxAreaRatio = cAR > oAR ? oAR / cAR : cAR / oAR;

    Size2f propSize(origSize.height, origSize.height * cAR);
    propSize = propSize * sqrt(rng.uniform(areaScale, maxAreaRatio));
    float x = rng.uniform(0.0f, origSize.width - propSize.width);
    float y = rng.uniform(0.0f, origSize.height - propSize.height);
    return Rectf(Point2f(x, y), propSize);
}

int main( int argc, char** argv ) {
    Mat orig = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    Size2f origSize(orig.rows, orig.cols);
    Size2f finalSize(224, 224);
    RNG _rng(time(0));
    bool deterministic = false;
    if(! orig.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    int flipRange = 1;
    float minScale = 0.08;
    float minAspectRatio = 0.75;
    float contrastRange = 0.0;
    float brightnessRange = 0.0;
    float angleRange = 20;
    if (argc == 4)
    {
        contrastRange = atof(argv[2]);
        brightnessRange = 127 * atof(argv[3]);
    }
    int do_flip = 0;
    float aspectRatio = finalSize.width / finalSize.height;
    float scale = 1.0;
    float contrast = 1.0;
    float brightness = 0.0;
    float angle = 0.0;
    Rectf cropbox(Point2f(), origSize);
    if (!deterministic)
    {
        do_flip     = _rng.uniform(0, flipRange);
        aspectRatio = _rng.uniform(minAspectRatio, 1.0f/minAspectRatio);
        contrast    = _rng.uniform(-contrastRange, contrastRange);
        brightness  = _rng.uniform(-brightnessRange, brightnessRange);
        angle       = _rng.uniform(-angleRange, angleRange);
        cropbox     = proposeROI(origSize, minScale, aspectRatio, _rng);
    }
    cout << "Rect " << cropbox << endl;
    cout << "Contrast " << contrast << endl;
    cout << "Scale " << scale << endl;
    cout << "brightness " << brightness << endl;
    cout << "aspectRatio " << aspectRatio << endl;
    cout << "angle " << angle << endl;

    Mat rotatedImg, adjustedImg;
    Mat flippedImg, resizedImg;
    rotateImage(orig, rotatedImg, (void *) &angle);
    cropImage(rotatedImg, adjustedImg, (void *) &cropbox);
    flipImage(adjustedImg, flippedImg, (void *) &do_flip);
    float adjustParams[2] = {contrast, brightness};
    adjustImage(flippedImg, flippedImg, (void *) &adjustParams);
    resizeImage(flippedImg, resizedImg, (void *) &finalSize);


    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", orig );
    namedWindow( "Adjusted window", WINDOW_AUTOSIZE );
    imshow( "Adjusted window", resizedImg );

    namedWindow( "rotated window", WINDOW_AUTOSIZE );
    imshow( "rotated window", flippedImg );
    waitKey(0);
    return 0;
}

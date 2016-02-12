#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include<fstream>
#include<iostream>

using namespace cv;
using namespace std;

int main( int argc, char** argv ) {
    Mat orig = imread(argv[1], CV_LOAD_IMAGE_COLOR);
    RNG& _rng = theRNG();
    if(! orig.data )                              // Check for invalid input
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }
    // Size2i output(224, 224);
    float minScale = 0.08;
    bool center = false;
    float minAspectRatio = 0.7;
    float contrastRange = 1.0;
    float brightnessRange = 0.0;
    float saturationRange = 0.0;

    if (argc == 4)
    {
        contrastRange = atof(argv[2]);
        brightnessRange = atof(argv[3]);
    }
    int b0 = _rng.uniform(0, 2);
    printf("Dump values %f %d %f %f %f %f ** %d\n", minScale, center, minAspectRatio, contrastRange,
            brightnessRange, saturationRange, b0);
    Mat adjustedImg;
    orig.convertTo(adjustedImg, -1, contrastRange, brightnessRange);

    namedWindow( "Display window", WINDOW_AUTOSIZE );
    imshow( "Display window", orig );
    namedWindow( "Adjusted window", WINDOW_AUTOSIZE );
    imshow( "Adjusted window", adjustedImg );

    waitKey(0);
    return 0;
}

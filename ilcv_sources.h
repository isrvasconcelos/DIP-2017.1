#ifndef FILTERS_H
#define FILTERS_H

#include <iostream>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <math.h>

#include <stdlib.h>
#include <time.h>

using namespace std;
using namespace cv;

//*******************************//
//*********** Auxiliar ***********//

void windowOpen(string imgName, const Mat &img, int xPos, int yPos);
void drawSomething(Mat *img, int r1, int s1, int r2, int s2);


//*******************************//
//*** Img Matrix Manipulation ***//

Mat transGraphics(Mat img, int r1, int s1, int r2, int s2);
Mat scaleImage2_uchar(InputArray src);
Mat fftshift(InputArray src);
Mat applyLogTransform(InputArray src);
Mat createWhiteDisk (const int &rows, const int &cols, const int cx, const int cy, const int &radius);
Mat createCosineImg(const int &rows, const int &cols, const float &freq, const float &theta);
Mat cvtImg2Colormap(const Mat &src, int colormap);

//*******************************//
//********* Experiments *********//

// [Aug/01 and 03]
void laplacianSharpeningPt1();
void laplacianSharpeningPt2();
void spatioGradientRun();

// [Aug/08 and 10]
void dftRun(bool setNormalize);
void whiteDisk();

void lpfilter();
void hpfilter();

void sinusoidImg();
void sinusoidNoise();

// [Aug/22 and 24]
void processingRGB();
void colormapProcessingRGB();
void colorSpaceNTSC();
void colorSpaceHSV();
void diskBGR();

//*******************************//
//********* Home Lesson *********//


// [Jul/25]
void threeChannelHistogram();

// [Aug/01]
void piecewiseLinearTransform();

// [Aug/11]
void runBlurring();

// [Aug/15]
void sinusoidNoiseFiltering();

#endif // FILTERS_H
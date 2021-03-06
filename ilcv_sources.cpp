#include "ilcv_sources.h"

//*************************************************************************//
// AUXILIAR //

void windowOpen(string imgName, const Mat &img, int xPos=0, int yPos=0) {

	namedWindow( imgName, WINDOW_AUTOSIZE );
	imshow( imgName, img );  
	moveWindow( imgName, xPos, yPos ); 

}

void drawSomething(Mat *img, int r1, int s1, int r2, int s2) {

	for(int i=0 ; i < 256 ; i++)
		for(int j=0 ; j < 256; j++)
			img->at<uchar>(i, j) = (uchar) 255;

	line(*img, Point(0, 255), Point(r1, 255-s1), 0, 4);
	line(*img, Point(255, 0), Point(r2, 255-s2), 0, 4);
	line(*img, Point(r1, 255-s1), Point(r2, 255-s2), 0, 4);
	circle(*img, Point(r1, 255-s1), 8, 0, -1);
	circle(*img, Point(r2, 255-s2), 8, 0, -1);
}


//*************************************************************************//

Mat computeHistogram1C(const cv::Mat &src) {

	// Establish the number of bins
	int histSize = 256;

	// Set the ranges ( for B,G,R) )
	float range[] = { 0, 256 } ;
	const float* histRange = { range };
	bool uniform = true; bool accumulate = false;

	Mat b_hist/*, g_hist, r_hist*/;

	// Compute the histograms:
	calcHist(	&src, 1, 0, Mat(), b_hist, 1, 
			&histSize, 
			&histRange, 
			uniform, 
			accumulate );

	// Draw the histograms for B, G and R
	int hist_w = 512; int hist_h = 400;
	int bin_w = cvRound( (double) hist_w/histSize );

	Mat histImage( hist_h, hist_w, CV_8UC1, Scalar( 0 ) );

	// Normalize the result to [ 0, histImage.rows ]
	normalize(b_hist, b_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

	// Draw for each channel
	for( int i = 1; i < histSize; i++ ) {

		line(	histImage, Point( bin_w*(i-1), 
			hist_h - cvRound(b_hist.at<float>(i-1)) ),
			Point( bin_w*(i), hist_h - cvRound(b_hist.at<float>(i)) ),
			Scalar( 255 ), 
			2, 8, 0 );
    }

    return histImage;
}


Mat transGraphics(Mat img, int r1, int s1, int r2, int s2) {

	Mat img2 = img.clone();
	float b;
	float a;
	int x;

	for(int i=0 ; i < img2.cols ; i++) {
		for(int j=0 ; j < img2.rows; j++) {
			x = img2.at<uchar>(i, j);

			if(x <= (uchar) r1) {
				b = 0;

				if(r1 == 0) 
					a = 1;
				else
					a = ((float) s1)/r1;
            		}

			if(x > r1 && x < r2) {
				b = ((float)(s2*r1-s1*r2))/(r1-r2);
				a = ((float)(s1-b))/r1;
			}

			if(x >= r2) {
				if(r2 == 255)
					b = 0;
				else
					b = ((float)(255*s2-255*r2))/(255-r2);

				a = ((float)(255-b))/255;
			}

		img2.at<uchar>(i, j) = (uchar)(a * x + b);
		}
	}
    return img2;

}

Mat scaleImage2_uchar(Mat &src){

	Mat tmp = src.clone();
	if (src.type()!= CV_32F){
		tmp.convertTo(tmp, CV_32F);
	}
	normalize(tmp,tmp,1,0, NORM_MINMAX);
	tmp = 255*tmp;
	tmp.convertTo(tmp, CV_8U, 1,0);
	return tmp;
}

Mat fftshift(const Mat &src){

	Mat tmp = src.clone();
	Mat tmp2;

	tmp = tmp (Rect(0,0,tmp.cols & -2, tmp.rows & -2));

	int cx = tmp.cols/2;
	int cy = tmp.rows/2;

	Mat q0(tmp, Rect(0,0,cx,cy));
	Mat q1(tmp, Rect(cx,0,cx,cy));
	Mat q2(tmp, Rect(0,cy,cx,cy));
	Mat q3(tmp, Rect(cx,cy,cx,cy));

	q1.copyTo(tmp2);
	q2.copyTo(q1);
	tmp2.copyTo(q2);

	q0.copyTo(tmp2);
	q3.copyTo(q0);
	tmp2.copyTo(q3);

//	cout << "INPUT:" << src.rows << endl;
//	cout << "OUTPUT: " << tmp.rows << endl;

	return tmp;
}

Mat applyLogTransform(const Mat &img) {
	Mat mag = img.clone();
	mag +=1;
	log(mag,mag);
	return mag;
}

Mat createWhiteDisk(const int &rows, const int &cols, const int cx, const int cy, const int &radius) {
	Mat img = Mat::zeros(rows, cols,CV_32F);
	for(int x=0; x <img.cols; x++){
		for(int y=0; y<img.rows; y++){
			float d = sqrt((x-cx)*(x-cx)+(y-cy)*(y-cy));
			if (d<=radius){
				img.at<float>(y,x )= 1;
				//img.at<float>(y,x )= 1-d/radius;
			}
		}
	}

	return img;
}

Mat createCosineImg(const int &rows, const int &cols, const float &freq, const float &theta) {  //[Aug, 08]
	Mat img = Mat::zeros(rows, cols, CV_32F);
	float rho;

	for (int x=0; x<img.cols; x++) {
		for(int y=0; y<img.rows; y++) {
			rho = x*cos(theta) - y*sin(theta);
			img.at<float> (y,x) = cos(CV_2PI*freq*rho);
		}
	}

	return img;
}

Mat cvtImg2Colormap(const Mat &src, int colormap) { //[Aug, 22]

	Mat output = src.clone();
	output = scaleImage2_uchar(output);
	applyColorMap(output, output, colormap);
	return output;
}


//*************************************************************************//
// HOME LESSONS //
//******************* [Jul/25] ******************//

void threeChannelHistogram() {

	Mat imgIn = imread("lena.png", CV_LOAD_IMAGE_COLOR);
	Mat imgOut = imgIn.clone();
	vector<Mat> imgChannel;

	split(imgOut, imgChannel);

	for(int i=0; i<imgChannel.size(); i++)
		imgChannel[i] = computeHistogram1C(imgChannel[i]);

	merge(imgChannel.data(), imgChannel.size(), imgOut);

	windowOpen("Original", imgIn, 0, 0);
	windowOpen("BGR Histogram", imgOut, 525, 0);
	waitKey(0);
}


//******************* [Aug/01] ******************//

void piecewiseLinearTransform() {

	namedWindow("My Sample Picture", WINDOW_NORMAL);
	resizeWindow("My Sample Picture", 512, 512);
	moveWindow( "My Sample Picture", 0, 0 ); 

	namedWindow("Histogram", WINDOW_AUTOSIZE);
	moveWindow( "Histogram", 1000, 0 ); 

	namedWindow("Parameters", WINDOW_AUTOSIZE);
	moveWindow( "Parameters", 550, 700 ); 

	Mat img = imread("ilcv_sample.jpeg", CV_LOAD_IMAGE_GRAYSCALE);
	Mat hist;
	Mat img_new = img.clone();
	Mat graphics;

	int row = 256;
	int col = 256;

	graphics.create(row, col, CV_8UC1);

	int s1 = 65, s2 = 195, r1 = 65, r2 = 195;

	createTrackbar("r1", "Parameters", &r1, 255, 0, 0);
	createTrackbar("s1", "Parameters", &s1, 255, 0, 0);
	createTrackbar("r2", "Parameters", &r2, 255, 0, 0);
	createTrackbar("s2", "Parameters", &s2, 255, 0, 0);

	while(true) {
		hist = computeHistogram1C(img);
		drawSomething(&graphics, r1, s1, r2, s2);
		img_new = transGraphics(img, r1, s1, r2, s2);

		imshow("My Sample Picture", img_new);
		imshow("Histogram", computeHistogram1C(img_new));
		imshow("Parameters", graphics);

		if((char) waitKey(1)=='q') 
			break;
	}

}


//******************* [Aug/11] ******************//

void runBlurring() {

	Mat img = imread("lena.png", CV_LOAD_IMAGE_GRAYSCALE);   // Read the file
	Mat img2 = img.clone();
	Mat hist;

	int option;

	// Kernel Size for filters
	int ksize=5;

	// Gaussian Blur Parameters
	int sigmaX=10, sigmaY=10;

	// Bilateral Filtering Parameters
	int d=1, sigmaColor=10, sigmaSpace=10;

	if(! img.data ) {
        	cout <<  "Could not open or find the image" << std::endl ;
        	exit(1);
	}

	cout << "Tap:" << endl;
	cout << "1 - Mean" << endl;
	cout << "2 - Median" << endl;
	cout << "3 - Gaussian" << endl;
	cout << "4 - Bilateral" << endl;

  	cin >> option;

	namedWindow("Parameters", WINDOW_AUTOSIZE);
	moveWindow( "Parameters", 0, 0 ); 

	namedWindow("Lena", WINDOW_AUTOSIZE);
	moveWindow( "Lena", 600, 0 ); 

	if(option==1) {
		createTrackbar("Kernel Size", "Parameters", &ksize, 50, 0);
		while(true) {
			// Only positive values for kernel size are accepted
			if(ksize == 0)
				ksize++;

			blur(img, img2, Size(ksize,ksize)); // normalized box filter.
			hist = computeHistogram1C(img2);

			imshow("Parameters", hist);
			imshow("Lena", img2);

			if((char) waitKey(1)=='q') 
				break;
		}
	}

	if(option==2) {
		createTrackbar("Kernel Size", "Parameters", &ksize, 50, 0);
		while(true) {
			// Only positive and even values for kernel size are accepted
			if(ksize == 0 || ksize%2 == 0) {
				ksize++;
			}

			medianBlur(img, img2, ksize);
			hist = computeHistogram1C(img2);

			imshow("Parameters", hist);
			imshow("Lena", img2);

			if((char) waitKey(1)=='q') 
				break;
		}
	}

	if(option==3) {
		createTrackbar("Kernel Size", "Parameters", &ksize, 50, 0);
		createTrackbar("X - Std. Deviation", "Parameters", &sigmaX, 100, 0);
		createTrackbar("Y - Std. Deviation", "Parameters", &sigmaY, 100, 0);

		while(true) {
			// Only positive and even values for kernel size are accepted
			if(ksize == 0 || ksize%2 == 0) {
				ksize++;
			}

			GaussianBlur(img, img2, Size(ksize,ksize), sigmaX, sigmaY);
			hist = computeHistogram1C(img2);

			imshow("Parameters", hist);
			imshow("Lena", img2);

			if((char) waitKey(1)=='q') 
				break;
		}
	}
	
	if(option==4) {
		createTrackbar("Filter Diameter Size", "Parameters", &d, 20, 0);
		createTrackbar("Sigma Color", "Parameters", &sigmaColor, 200, 0);
		createTrackbar("Sigma Space", "Parameters", &sigmaSpace, 200, 0);

		while(true) {
			// Only positive and even values for kernel size are accepted
			if(ksize == 0 || ksize%2 == 0) {
				ksize++;
			}

			bilateralFilter(img, img2, d, sigmaColor, sigmaSpace);
			hist = computeHistogram1C(img2);

			imshow("Parameters", hist);
			imshow("Lena", img2);

			if((char) waitKey(1)=='q') 
				break;
		}
	}
}


//******************* [Aug/15] ******************//

void sinusoidNoiseFiltering() {

	Mat img = imread("lena_noisy.png", IMREAD_GRAYSCALE);
	Mat img2 = img.clone();
	Mat imgToFourier;
	Mat invdft2;

	int radius = 40; // Optimal settings
	int offset = 15; // Optimal settings

        namedWindow("Band-Stop Filter", WINDOW_KEEPRATIO);
	moveWindow( "Band-Stop Filter", 0, 0 ); 

        namedWindow("Lena Noisy", WINDOW_KEEPRATIO);
	moveWindow( "Lena Noisy", 200, 0 ); 

        namedWindow("Reconstructed", WINDOW_KEEPRATIO);
	moveWindow( "Reconstructed", 400, 0 ); 

	createTrackbar("radius", "Band-Stop Filter", &radius, img2.cols,0,0);
	createTrackbar("offset", "Band-Stop Filter", &offset, img2.cols,0,0);

	while(true) {

		if(radius == 0) 
			radius=1;

		Mat maskLP = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						radius );

		Mat maskHP = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						(radius+offset) );

		maskLP = fftshift(maskLP);
		maskHP = 1-fftshift(maskHP);

		Mat mask = maskLP+maskHP;
		normalize(mask,mask,1,0,NORM_MINMAX); // Normalize the mask for [0,1] range

		// Even img2.cols and rows size fix
		if(mask.cols < img2.cols) { 
			img2 = img2.colRange(0, mask.cols);
			img2.pop_back();
		}

		imgToFourier = img2;

		Mat planes2[] ={Mat_<float>(imgToFourier), 
				Mat::zeros(imgToFourier.size(), CV_32F)};

		merge(planes2, 2, imgToFourier);
		dft(imgToFourier, imgToFourier);
		split(imgToFourier, planes2);

		Mat mag2;
		magnitude(planes2[0], planes2[1], mag2);
		mag2 = applyLogTransform(mag2);

		multiply(planes2[0], mask, planes2[0]);
		multiply(planes2[1], mask, planes2[1]);

		merge(planes2, 2, imgToFourier);

		dft(imgToFourier, invdft2, DFT_INVERSE|DFT_REAL_OUTPUT);
		normalize(invdft2, invdft2, 0, 1, CV_MINMAX);

		imshow("Lena Noisy", scaleImage2_uchar(img));
		imshow("Band-Stop Filter", fftshift(mask));
		imshow("Reconstructed", scaleImage2_uchar(invdft2));

		if((char)waitKey(1)=='q') 
			break;
	}

}

//******************* [Sep/20] ******************//

void hitOrMiss() {

    Mat input_image = (Mat_<uchar>(16, 16) <<
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0,
        0, 0, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
                       );

    input_image = input_image * 255;

    Mat kernel = (Mat_<int>(3, 3) <<
        0, 1, 0,
        -1, 1, 1,
        0, -1, 0);

    Mat kernel2 = (Mat_<int>(3, 3) <<
        0, 1, 0,
        1, 1, -1,
        0, -1, 0);

    Mat kernel3 = (Mat_<int>(3, 3) <<
        0, -1, 0,
        1, 1, -1,
        0, 1, 0);

    Mat kernel4 = (Mat_<int>(3, 3) <<
        0, -1, 0,
        -1, 1, 1,
        0, 1, 0);

	//kernel = kernel2;

    Mat output_image;
    Mat output_image2;
    Mat output_image3;
    Mat output_image4;
    Mat output_imageFinal;

    morphologyEx(input_image, output_image, MORPH_HITMISS, kernel);
    morphologyEx(input_image, output_image2, MORPH_HITMISS, kernel2);
    morphologyEx(input_image, output_image3, MORPH_HITMISS, kernel3);
    morphologyEx(input_image, output_image4, MORPH_HITMISS, kernel4);

    bitwise_or(output_image, output_image2, output_imageFinal);
    bitwise_or(output_imageFinal, output_image3, output_imageFinal);
    bitwise_or(output_imageFinal, output_image4, output_imageFinal);

    const int rate = 1;

    kernel = (kernel + 1) * 127;
    kernel.convertTo(kernel, CV_8U);

    namedWindow("Original", WINDOW_KEEPRATIO);
    moveWindow ( "Original", 300, 0 ); 

    namedWindow("Hit or Miss", WINDOW_KEEPRATIO);
    moveWindow ( "Hit or Miss", 600, 0 ); 

    resize(input_image, input_image, Size(), rate, rate, INTER_NEAREST);
    imshow("Original", input_image);

    resize(output_imageFinal, output_image, Size(), rate, rate, INTER_NEAREST);
    imshow("Hit or Miss", output_image);

    waitKey(0);

}

//******************* [Sep/29] ******************//

void characterSegmentation() {
	Mat img  = imread("calculator.tif", IMREAD_GRAYSCALE);

	Mat output = img.clone();
	Mat output2 = img.clone();
	Mat output3 = img.clone();

	Mat kernel1 = (Mat_ <float>(3,3) <<
			0, 1, 0,
			1, 1, 1,
			0, 1, 0);

	Mat kernel2 = (Mat_ <float>(3,3) <<
			0, 0, 0,
			1, 1, 1,
			0, 0, 0);

	Mat kernel3 = (Mat_ <float>(3,3) <<
			0, 1, 0,
			0, 1, 0,
			0, 1, 0);

	Mat kernel4 = (Mat_ <float>(5,5) <<
			0, 1, 1, 1, 0,
			0, 1, 1, 1, 0,
			1, 1, 1, 1, 1,
			0, 1, 1, 1, 0,
			0, 1, 1, 1, 0);
	Mat kernel5 = (Mat_ <float>(3,3) <<
			1, 1, 1,
			1, 1, 1,
			1, 1, 1);

	
	kernel1.convertTo(kernel1, CV_8U);
	kernel2.convertTo(kernel2, CV_8U);
	kernel3.convertTo(kernel3, CV_8U);
	kernel4.convertTo(kernel4, CV_8U);
	kernel4.convertTo(kernel5, CV_8U);

	// OBS.: Threshold types below
	// 0: Binary, 1: BinInverted, 2: Threshold Truncated, 3: Threshold to Zero, 4: Threshold to Zero Inverted
	const double rate = 0.75;
	int threshold_value = 150;
	int threshold_type = 0;
	int const max_type = 4;
	int const max_BINARY_value = 255;


	// *********** Pt. 1 *********** //
	namedWindow( "Pt. 1: Original Image", WINDOW_KEEPRATIO);
    	resize(img, img, Size(), rate, rate, INTER_NEAREST);
	moveWindow( "Pt. 1: Original Image", 0, 0 ); 
	imshow("Pt. 1: Original Image", img);

	waitKey(0);
	destroyAllWindows();

	// *********** Pt. 2 *********** //
	// Thresholding gray scale to binary
	namedWindow( "Pt. 2: Thresholding gray scale to binary", WINDOW_KEEPRATIO);
	moveWindow( "Pt. 2: Thresholding gray scale to binary", 0, 0 ); 

	createTrackbar("threshold_value", "Pt. 2: Thresholding gray scale to binary", &threshold_value, max_BINARY_value,0,0);
	createTrackbar("threshold_type", "Pt. 2: Thresholding gray scale to binary", &threshold_type, max_type,0,0);

	while(true) {

		threshold( output, output2, threshold_value, max_BINARY_value, threshold_type );
    		resize(output2, output2, Size(), rate, rate, INTER_NEAREST);
    		imshow("Pt. 2: Thresholding gray scale to binary", output2);

		if((char)waitKey(1)=='q') 
			break;
	}

	destroyAllWindows();

	// *********** Pt. 3 *********** //
	// Filtering horizontal lines
	output = output2.clone();
	output3 = output2.clone();

	int iterOpen=23; // Optimal setting
	int iterClose=0;
	int max_value=30;

	namedWindow( "Pt. 3: Filtering horizontal lines", WINDOW_KEEPRATIO);
	moveWindow( "Pt. 3: Filtering horizontal lines", 0, 0 ); 
	createTrackbar("It. Erode", "Pt. 3: Filtering horizontal lines", &iterOpen, max_value,0,0);
	createTrackbar("It. Dilate", "Pt. 3: Filtering horizontal lines", &iterClose, max_value,0,0);


	while(true) {

		morphologyEx(output2, output3, MORPH_OPEN, kernel2, Point(-1,-1), iterOpen);
		morphologyEx(output3, output3, MORPH_CLOSE, kernel2, Point(-1,-1), iterClose);
    		imshow("Pt. 3: Filtering horizontal lines", output3);

		if((char)waitKey(1)=='q') 
			break;
	}

	destroyAllWindows();

	// *********** Pt. 4 *********** //
	// Increasing horizontal lines thickness

	int iterErode=0;
	int iterDilate=2;  // Optimal setting

	namedWindow( "Pt. 4: Increasing horizontal lines thickness", WINDOW_KEEPRATIO);
	moveWindow( "Pt. 4: Increasing horizontal lines thickness", 0, 0 ); 
	createTrackbar("It. Erode", "Pt. 4: Increasing horizontal lines thickness", &iterErode, max_value,0,0);
	createTrackbar("It. Dilate", "Pt. 4: Increasing horizontal lines thickness", &iterDilate, max_value,0,0);

	Mat output4 = output3.clone();

	while(true) {

		morphologyEx(output3, output4, MORPH_ERODE, kernel4, Point(-1,-1), iterErode);
		morphologyEx(output4, output4, MORPH_DILATE, kernel4, Point(-1,-1), iterDilate);
    		imshow("Pt. 4: Increasing horizontal lines thickness", output4);

		if((char)waitKey(1)=='q') 
			break;
	}

	// *********** Pt. 5 *********** //
	// Merging highlighted horizontal lines with original image to erase them

	Mat output5 = output4.clone();
	threshold( output, output, 0, 255, 1 ); // Inverting colors
	output5 = output+output4; // Merging HERE!

	imshow("Pt. 5: Merging highlighted horizontal lines with original image to erase them", output5);
	waitKey(0);
	destroyAllWindows();

	// *********** Pt. 6 *********** //
	// Final fixes

	iterClose=1; // Optimal setting

	namedWindow( "Pt. 6: Final fixes", WINDOW_KEEPRATIO);
	moveWindow( "Pt. 6: Final fixes", 0, 0 ); 
	createTrackbar("It. Dilate", "Pt. 5", &iterClose, max_value,0,0);

	Mat output6 = output5.clone();
	morphologyEx(output5, output6, MORPH_DILATE, kernel2, Point(-1,-1), iterClose);
	imshow("Pt. 6: Final fixes", output6);

	waitKey(0);
}

//*************************************************************************//
// EXPERIMENTS //
//*************** [Aug/01 and 03] ***************//

void laplacianSharpeningPt1() {

	Mat img  = imread("lena.png", IMREAD_GRAYSCALE);
	Mat img2, img3;
	img.convertTo(img,CV_32F);

	int factor = 5; // Sharpening Factor

	Mat kernel = (Mat_ <float>(3,3) <<
			1.0, 1.0,  1.0,
			1.0, -8.0, 1.0,
			1.0, 1.0,  1.0);

	filter2D(img, img2, CV_32F,kernel, Point(-1,-1),0,BORDER_DEFAULT);

	namedWindow( "img3", WINDOW_KEEPRATIO);
	moveWindow ( "img3", 450, 0 ); 
	createTrackbar("factor", "img3", &factor, 100, 0, 0);

	while(true){
		Mat hist = computeHistogram1C(img3);
		add(img, -(factor/100.0)*img2,img3,noArray(), CV_8U);

		imshow ("img3", scaleImage2_uchar(img3));
		imshow ("hist", hist);
		windowOpen("img2", scaleImage2_uchar(img2), 1000, 700); // Filtered
//		windowOpen("img", scaleImage2_uchar(img), 0, 0); 	// Original

		if((char)waitKey(5)=='q') 
			break;
	    }

}

void laplacianSharpeningPt2() {

	Mat img  = imread ("lena.png", IMREAD_GRAYSCALE);
	Mat lap, img2;

	Laplacian(img,lap,CV_32F,1,1,0);
	add(img, -lap,img2,noArray(), CV_8U);

	windowOpen("img", img, 0, 0);
	windowOpen("img2", img2, 450, 700);
	windowOpen("lap", lap, 1000, 0);

	waitKey(0);
}

void spatioGradientRun() {

Mat img  = imread ("lena.png", IMREAD_GRAYSCALE);
	Mat gx, gy,g;
	spatialGradient(img,gx,gy,3,BORDER_DEFAULT);
	g = abs(gx) + abs(gy);

	windowOpen("img", scaleImage2_uchar(img), 300, 700);
	windowOpen("g", scaleImage2_uchar(g), 600, 700);
	windowOpen("gx", scaleImage2_uchar(gx), 0, 0);
	windowOpen("gy", scaleImage2_uchar(gy), 1000, 0);

	waitKey(0);
}

//*************** [Aug/08 and 10] ***************//

void dftRun(bool setNormalize=false) {

	// Discrete Fourier Transform
	Mat img = imread("lena.png", IMREAD_GRAYSCALE);
	Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};
	Mat img2;

	if(! img.data ) {
        	cout <<  "Could not open or find the image" << std::endl ;
        	exit(1);
	}


	merge(planes, 2, img2);
	dft(img2, img2);
	split(img2, planes);

	Mat mag;
	magnitude(planes[0], planes[1], mag);
	mag = applyLogTransform(mag);

	if(setNormalize) {
		normalize(planes[0],planes[0],1,0,NORM_MINMAX);
		normalize(planes[1],planes[1],1,0,NORM_MINMAX);
	}

	windowOpen("planes_0", fftshift(planes[0]), 300, 700);
	windowOpen("planes_1", fftshift(planes[1]), 600, 700);
	windowOpen("img", scaleImage2_uchar(img), 0, 0);
	windowOpen("mag", fftshift(scaleImage2_uchar(mag)), 1000, 0);

	waitKey(0);
	
}

void whiteDisk() {

	Mat disk0= Mat::zeros(200,200, CV_32F);
        Mat disk = disk0.clone();
        namedWindow( "disk", WINDOW_AUTOSIZE);
	moveWindow( "disk", 0, 0 ); 

        int xc =100;
        int yc = 100;
        int radius = 20;

        createTrackbar("xc", "disk", &xc,disk.cols,0 );
        createTrackbar("yc", "disk", &yc,disk.rows,0 );
        createTrackbar("radius", "disk", &radius,disk.cols,0 );

        for (int x=0; x<disk.cols;x++)
            for(int y=0;y<disk.rows;y++)
                if ((x-xc)*(x-xc)+(y-yc)*(y-yc) <= radius*radius)
                    disk.at<float>(y,x)=1;
 
        while (true){
		disk = disk0.clone();
		disk = createWhiteDisk(200,200,xc,yc,radius);
		imshow("disk", disk);

		if((char)waitKey(1)=='q') 
			break;
        }

}

void lpfilter() {

	Mat img = imread("lena.png", IMREAD_GRAYSCALE);
	Mat img2 = img.clone();

	int radius = 50; // [0, 512] (512 = img.cols)
        namedWindow("mask", WINDOW_AUTOSIZE);
	createTrackbar("radius", "mask", &radius, img2.cols,0,0);
	moveWindow( "mask", 500, 700 ); 

	while(true) {

		Mat mask = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						radius );

		mask = fftshift(mask);
		Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

		merge(planes, 2 , img2);
		dft(img2, img2);
		split(img2, planes);

		multiply(planes[0], mask, planes[0]);
		multiply(planes[1], mask, planes[1]);
		merge(planes, 2, img2);
		idft(img2, img2, DFT_REAL_OUTPUT);
		img2 = fftshift(img2);

		//windowOpen("planes_0", fftshift(planes[0]), 300, 700);
		//windowOpen("planes_1", fftshift(planes[1]), 600, 700);

		imshow("mask", fftshift(mask));
		windowOpen("img", scaleImage2_uchar(img), 0, 0);
		windowOpen("img2", fftshift(scaleImage2_uchar(img2)), 1000, 0);

		if((char)waitKey(1)=='q') 
			break;
	}
}


void hpfilter() {

	Mat img = imread("lena.png", IMREAD_GRAYSCALE);
	Mat img2 = img.clone();

	int radius = 50; // [0, 512] (512 = img.cols)

        namedWindow("img", WINDOW_AUTOSIZE);
	moveWindow( "img", 0, 0 ); 

        namedWindow("img2", WINDOW_AUTOSIZE);
	moveWindow( "img2", 1000, 0 ); 

        namedWindow("mask", WINDOW_AUTOSIZE);
	moveWindow( "mask", 400, 400 ); 

	createTrackbar("radius", "mask", &radius, img2.cols,0,0);


	while(true) {

		Mat mask = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						radius );

		mask = fftshift(mask);
		mask = 1-mask; // <- Chage here for high/low pass
		Mat planes[] = {Mat_<float>(img), Mat::zeros(img.size(), CV_32F)};

		merge(planes, 2 , img2);
		dft(img2, img2);
		split(img2, planes);

		multiply(planes[0], mask, planes[0]);
		multiply(planes[1], mask, planes[1]);

		merge(planes, 2, img2);

		idft(img2, img2, DFT_REAL_OUTPUT);
		img2 = fftshift(img2);

		imshow("img", scaleImage2_uchar(img));
		imshow("img2", fftshift(scaleImage2_uchar(img2)));
		imshow("mask", fftshift(mask));

		//windowOpen("planes_0", fftshift(planes[0]), 300, 700);
		//windowOpen("planes_1", fftshift(planes[1]), 600, 700);

		if((char)waitKey(1)=='q') 
			break;
	}
}

void sinusoidImg() {

	namedWindow("img", WINDOW_KEEPRATIO);
        Mat img;
        int rows = 500;
        int cols = 500;
        int freq = 1;
        int theta = 2;

        createTrackbar("Freq", "img", &freq, 500,0,0);
        createTrackbar("Theta", "img", &theta, 100,0,0);

        while(true){
		img = createCosineImg(	rows,cols,
					(float)freq/1e3, 
					(float)(CV_2PI*theta/100.0) );

		imshow("img", scaleImage2_uchar(img));

		if((char)waitKey(1)=='q') 
			break;
        }
}

void sinusoidNoise() {

	namedWindow("img", WINDOW_AUTOSIZE);
        namedWindow("mag", WINDOW_AUTOSIZE);

	moveWindow( "img", 0  , 0 ); 
	moveWindow( "mag", 750, 0 ); 

        Mat img  = imread("lena.png", IMREAD_GRAYSCALE);
        Mat noise;

        img.convertTo(img, CV_32F);
        img = img/255.0;

        int rows = img.rows;
        int cols = img.cols;

        int freq = 1;
        int theta= 2;
        int gain = 1;

        createTrackbar("Freq" , "img", &freq , 500,0,0);
        createTrackbar("Theta", "img", &theta, 100,0,0);
        createTrackbar("Gain" , "img", &gain , 100,0,0);

        while(true){
            noise = createCosineImg(rows,cols,(float)freq/1e3, (float)(CV_2PI*theta/100.0));
            noise = img + (float)(gain/100.0)*noise;

            Mat img3 = noise.clone();
            Mat img2, mag;
            Mat planes[] = {Mat_<float>(img3),Mat::zeros(img3.size(), CV_32F)};

            merge(planes,2,img2);
            dft(img2,img2);
            split(img2,planes);

            magnitude(planes[0], planes[1], mag);
            mag = applyLogTransform(mag);

            imshow("img", scaleImage2_uchar(noise));
            imshow("mag", fftshift(scaleImage2_uchar(mag)));

            if((char)waitKey(1)=='q') 
		break;
        }
}

//*************** [Aug/22 and 24] ***************//


void processingRGB() {

	Mat img = imread("../img/baboon.png", IMREAD_COLOR);
	vector<Mat> bgr;
	split(img, bgr);

	namedWindow("Original", WINDOW_KEEPRATIO);
	moveWindow( "Original", 0  , 0 ); 

	namedWindow("Blue", WINDOW_KEEPRATIO);
	moveWindow( "Blue", 250  , 0 ); 

	namedWindow("Green", WINDOW_KEEPRATIO);
	moveWindow( "Green", 500  , 0 ); 

	namedWindow("Red", WINDOW_KEEPRATIO);
	moveWindow( "Red", 750  , 0 ); 

	imshow("Original", img);
	imshow("Blue", bgr[0]);
	imshow("Green", bgr[1]);
	imshow("Red", bgr[2]);

	waitKey(0);
}

void colormapProcessingRGB() {

	Mat img = imread("../img/rgbcube_kBKG.png", IMREAD_COLOR);
	vector<Mat> bgr;
	split(img, bgr);
	int colormap = COLORMAP_JET;

	namedWindow("Original", WINDOW_KEEPRATIO);
	moveWindow( "Original", 0  , 0 ); 

	namedWindow("Blue", WINDOW_KEEPRATIO);
	moveWindow( "Blue", 250  , 0 ); 

	namedWindow("Green", WINDOW_KEEPRATIO);
	moveWindow( "Green", 500  , 0 ); 

	namedWindow("Red", WINDOW_KEEPRATIO);
	moveWindow( "Red", 750  , 0 ); 

	imshow("Original", img);
	imshow("Blue",  cvtImg2Colormap(bgr[0], colormap)  );
	imshow("Green", cvtImg2Colormap(bgr[1], colormap)  );
	imshow("Red",   cvtImg2Colormap(bgr[2], colormap)  );

	waitKey(0);
}

void colorSpaceNTSC() { //  Y Cb Cr, Livro Cap 7

	Mat img = imread("../img/baboon.png", IMREAD_COLOR);
	Mat img2;
	vector<Mat> yrb;
	cvtColor(img, img2, CV_BGR2YCrCb);
	split(img2,yrb);
	int colormap=0;

	imshow("img", img);
	moveWindow( "img", 0  , 0 ); 

	// y -> Iluminance
	imshow("y", yrb[0]);
	moveWindow( "y", 250  , 0 ); 

	// r -> Cr
	imshow("r", yrb[1]);
	moveWindow( "r", 500  , 0 ); 

	// b -> Cb
	imshow("b", yrb[2]);
	moveWindow( "b", 750  , 0 ); 

	waitKey(0);
}

void colorSpaceHSV() { // Livro Cap 7

	Mat img = imread("../img/chips.png", IMREAD_COLOR);
	Mat img2;
	vector<Mat> hsv;
	cvtColor(img, img2, CV_BGR2HSV);
	split(img2,hsv);
	int colormap=0;

	imshow("Original", img);
	moveWindow( "Original", 0  , 0 ); 

	// h -> hue
	imshow("Hue", hsv[0]);
	moveWindow( "Hue", 250  , 0 ); 

	// s -> saturation
	imshow("Saturation", hsv[1]);
	moveWindow( "Saturation", 500  , 0 ); 

	// v -> Iluminance
	imshow("v", hsv[2]);
	moveWindow( "v", 750  , 0 ); 

	waitKey(0);
}

void diskBGR() {

	int rows = 0.5e3;
	int radius = (int)(rows/4);
	int bx = (int) (rows/2), by = (int) (rows/2) - (int) (radius/2);
	int gx = (int) (rows/2) - radius/2;
	int gy = (int) (rows/2) + radius/2;
	int rx = (int) (rows/2) + radius/2;
	int ry = (int) (rows/2) + radius/2;

	Mat img;
	vector<Mat> bgr;

	bgr.push_back( createWhiteDisk(rows, rows, bx, by, radius) );
	bgr.push_back( createWhiteDisk(rows, rows, gx, gy, radius) );
	bgr.push_back( createWhiteDisk(rows, rows, rx, ry, radius) );
	
	merge(bgr, img);
	img = scaleImage2_uchar(img);

	windowOpen("img", img);

	waitKey(0);

}

//*************** [Sep/12 and 14] ***************//

void blurringBGR() {

	Mat img =  imread("../img/baboon.png", IMREAD_COLOR);
	Mat img2;
	int wsize = 3;

	namedWindow("img", WINDOW_KEEPRATIO);
	moveWindow( "img", 0  , 0 ); 

	namedWindow("img2", WINDOW_KEEPRATIO);
	moveWindow( "img2", 0  , 0 ); 
	createTrackbar("wsize", "img2", &wsize, 50, 0, 0);

	while(true) {

		blur(img, img2, Size(wsize + 1, wsize + 1), Point(-1,-1), BORDER_DEFAULT);
		imshow("img", img);
		imshow("img2", img2);

          	if((char)waitKey(1)=='q') 
			break;
	}
}

void sharpeningBGR() {

	Mat img =  imread("../img/baboon.png", IMREAD_COLOR);
	Mat img2;
	int wsize = 3;

	namedWindow("img", WINDOW_KEEPRATIO);
	moveWindow( "img", 0  , 0 ); 

	namedWindow("img2", WINDOW_KEEPRATIO);
	moveWindow( "img2", 0  , 0 ); 
	createTrackbar("wsize", "img2", &wsize, 10, 0, 0);

	while(true) {

		Laplacian(img, img2, CV_16S, 2*wsize+1, 1, 0, BORDER_DEFAULT);
		imshow("img", img);
		imshow("img2", img2);

          	if((char)waitKey(1)=='q') 
			break;
	}
}

void segmentationBGR() { // Image segmentation in BGR colorspace pt. 1

	Mat img =  imread("../img/baboon.png", IMREAD_COLOR);
	Mat img2;
	int wsize = 3;

	int sp = 10; // Color 
	int sr = 100; // Distance
	int maxLevel = 1;

	namedWindow("img2", WINDOW_KEEPRATIO);
	moveWindow( "img2", 0  , 0 ); 
	createTrackbar("sp", "img2", &sp, 20, 0, 0);
	createTrackbar("sr", "img2", &sr, 200, 0, 0);

	while(true) {

		TermCriteria criteria = TermCriteria(TermCriteria::MAX_ITER + TermCriteria::EPS, 5, 1);
		pyrMeanShiftFiltering(img, img2, sp, sr, maxLevel, criteria);	

		imshow("img", img);
		imshow("img2", img2);

          	if((char)waitKey(1)=='q') 
			break;
	}
}

void houghLineTransform() {

	Mat img =  imread("../img/building.tif", IMREAD_GRAYSCALE);
	Mat edges, edges2, dst, cdst, src;

	int edgeThresh = 1;
	int kernel_size = 3;
	int ratio = 3;
	int lowThreshold=45;
	int const max_lowThreshold = 100;

	namedWindow("Edge detect", WINDOW_KEEPRATIO);
	createTrackbar( "Threshold:", "Edge detect", &lowThreshold, max_lowThreshold);

	while(true) {
		blur( img, edges, Size(3,3) ); // Reduce noise with a kernel 3x3
		Canny(edges, edges2, lowThreshold, lowThreshold*ratio, kernel_size );

		imshow( "Edge detect", edges2 );

          	if((char)waitKey(1)=='q') 
			break;
	}

	int p1 = 0, p2 = 0, p3 = 0, r=0;
	namedWindow("hou", WINDOW_KEEPRATIO);
	createTrackbar("r","hou", &r, 360, 0,0 );
	createTrackbar("p1","hou", &p1, 100, 0,0 );
	createTrackbar("p2","hou", &p2, 100,0,0);
	createTrackbar("p3", "hou",&p3, 15, 0,0);

	while(true) {

		cdst = edges2.clone();
		vector<Vec4i> lines;
		HoughLinesP(cdst, lines, 1, CV_PI/r, p1, p2, p3);
		cvtColor(cdst, cdst, CV_GRAY2BGR);

		for( size_t i = 0; i < lines.size(); i++ ) {
				Vec4i l = lines[i];
				line( cdst, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,0,0), 3, CV_AA);
		}



		imshow("source", edges2);
		imshow("hou", cdst);

          	if((char)waitKey(1)=='q') 
			break;
	}


	waitKey(0);
}

//*************************************************************************//
// MISC //

void colorSegmentation() {

	Mat img = imread("../img/chips.png", IMREAD_COLOR);

	int h1=0;
	int h2=10;
	int h3=160;
	int h4=175;

	Mat hsv_image, hsv_image2;
  	cvtColor(img, hsv_image, cv::COLOR_BGR2HSV);
  
  	// Threshold the HSV image, keep only the red pixels
  	Mat lower_red_hue_range;
  	Mat upper_red_hue_range;

	namedWindow("LRH", WINDOW_KEEPRATIO);
	moveWindow( "LRH", 0  , 0 ); 
	createTrackbar("Lower Red Hue Range","LRH", &h1, 255, 0,0);
	createTrackbar("Upper Red Hue Range","LRH", &h2, 255, 0,0);

	namedWindow("Original", WINDOW_KEEPRATIO);
	moveWindow( "Original", 800  , 0 ); 


	while(true) {
  		inRange(hsv_image, cv::Scalar(h1, 100, 100), cv::Scalar(h2, 255, 255), lower_red_hue_range);

		imshow("LRH", lower_red_hue_range);
		imshow("Original", img);
          	if((char)waitKey(1)=='q') 
			break;
	}

	namedWindow("Final", WINDOW_KEEPRATIO);
	moveWindow( "Final", 400  , 0 ); 

	hsv_image2=hsv_image.clone();
	vector<Mat> bgr2;

	split(hsv_image2, bgr2);
	bgr2[0]=lower_red_hue_range;
	merge(bgr2,hsv_image2);
	
	cvtColor(hsv_image2, hsv_image2, cv::COLOR_HSV2BGR);

	split(hsv_image2, bgr2);

	bitwise_and(bgr2[0], lower_red_hue_range, bgr2[0]);
	bitwise_and(bgr2[1], lower_red_hue_range, bgr2[1]);
	bitwise_and(bgr2[2], lower_red_hue_range, bgr2[2]);

	merge(bgr2,hsv_image2);

	imshow("Final", hsv_image2);

	waitKey(0);


}


//*************************************************************************//
// REAV1 //
    // Criar uma imagem 100x100 contendo um padrão listrado preto e branco,
    // assim como o ilustrado na imagem 'q01.png'. O código deve contemplar um
    // slider que varia de 0 a 90, rotacionando o padrão, resultando numa
    // imagem como aquela ilustrada no arquivo 'build/israel_q01.png'.


void Q01() {

	namedWindow("img", WINDOW_KEEPRATIO);
        Mat img;
        int rows = 100;
        int cols = 100;
        int freq = 80;
        int theta = 14;

        //createTrackbar("Freq", "img", &freq, 500,0,0);
        createTrackbar("Theta", "img", &theta, 90,0,0);

        while(true){
		img = createCosineImg(	rows,cols,
					(float)freq/1e3, 
					(float)(CV_2PI*theta/100.0) );

		imshow("img", scaleImage2_uchar(img));

		if((char)waitKey(1)=='q') 
			break;
        }

	imwrite("israel_q01.png", scaleImage2_uchar(img));
}

//*************************************************************************//

    // Q03
    // Considere a imagem "build/tungsten.tif".
    // Deseja-se melhorar áres que têm:
    // 1 - baixa intensidade média (escuras) em relação à
    // toda a imagem (média total da imagem).
    // 2 - Têm baixo contraste (i. e., baixa variância).
    // 3 - Não têm contraste constante.
    // Aplique a transformação especificada na figura "build/contrast.png",
    // onde:
    // - S_{xy} é uma região 3x3.
    // - E, k_0, k_1 e k_2 são escalares.
    // - m_G é a média global da imagem.
    // - m_{S_{xy}} é a média na região S_{xy}.
    // - \sigma_G é o desvio padrão global da imagem
    // - \sigma_{S_{xy}} é o desvio padrão na região S_{xy}
    // O programa deve ser interativo. Isto é, o usuário poderá alterar
    // os valores das constantes da equação para analizar o resultado,
    // assim como ilustrado na imagem "build/tungsten_solution.png".
void Q03() {

	namedWindow("My Sample Picture", WINDOW_AUTOSIZE);
	moveWindow( "My Sample Picture", 0, 0 ); 

	namedWindow("Parameters", WINDOW_AUTOSIZE);
	moveWindow( "Parameters", 550, 0 ); 

	Mat img = imread("tungsten.tif", CV_LOAD_IMAGE_GRAYSCALE);
	Mat hist;
	Mat img_new = img.clone();

	int e = 6, k0 = 50, k1 = 1, k2 = 100;

	createTrackbar("e", "Parameters", &e, 100, 0, 0);
	createTrackbar("k0/1000", "Parameters", &k0, 1000, 0, 0);
	createTrackbar("k1/1000", "Parameters", &k1, 1000, 0, 0);
	createTrackbar("k2/1000", "Parameters", &k2, 1000, 0, 0);

	while(true) {
		img_new = transGraphics2(img, e, k0, k1, k2);

		imshow("My Sample Picture", img);
		imshow("Parameters", img_new);

		if((char) waitKey(1)=='q') 
			break;
	}

	imwrite("israel_q03.png", img_new);
}

Mat transGraphics2(Mat img, int e, int k0, int k1, int k2) {

	Mat img2 = img.clone();
	int kernelSize=3;
	int x;

	float globalMean=0;
	float globalStdDev=0;
	float localMean=0;
	float localStdDev=0;
	float thisPixel = 0;
	int imgSize = img.cols*img.rows;

	float K0 = k0*0.001;
	float K1 = k1*0.001;
	float K2 = k2*0.001;

	// Evaluating Mean
	for(int i=0 ; i < img2.cols ; i++) {
		for(int j=0 ; j < img2.rows; j++) {
			thisPixel=img2.at<uchar>(i, j);
			globalMean = globalMean+thisPixel/imgSize;
		}
	}

	// Evaluating Std Dev
	for(int i=0 ; i < img2.cols ; i++) {
		for(int j=0 ; j < img2.rows; j++) {
			thisPixel=img2.at<uchar>(i, j);
			globalStdDev = globalStdDev + ((globalMean-thisPixel)*(globalMean-thisPixel) / (imgSize-1));
		}
	}

	for(int i=0 ; i < img2.cols ; i=i+kernelSize) {
		for(int j=0 ; j < img2.rows; j=j+kernelSize) {
			localMean=0;
			localStdDev=0;

			for(int k=0 ; k < kernelSize ; k++) {
				for(int l=0 ; l < kernelSize; l++) {
					thisPixel=img2.at<uchar>(i+k, j+l);
					localMean = (localMean+thisPixel)/9;
				}
			}

			for(int k=0 ; k < kernelSize ; k++) {
				for(int l=0 ; l < kernelSize; l++) {
					thisPixel=img2.at<uchar>(i+k, j+l);
					localStdDev = localStdDev + (pow((localMean-thisPixel),2) / (9-1));
				}
			}

			localStdDev = sqrt(localStdDev);

			if((localMean <= K0*globalMean) && (K1*globalStdDev < localStdDev) && (K2*globalStdDev > localStdDev)) {
				//cout << "Filtering" << endl;
				for(int k=0 ; k < kernelSize ; k++) {
					for(int l=0 ; l < kernelSize; l++) {
							img2.at<uchar>(i+k, j+l)=e*img2.at<uchar>(i+k, j+l);
					}
				}
			}
		}
	}
    return img2;

}

//*************************************************************************//
    // Q04
    /// Carregar a imagem "build/rose.tif" e filtrar os ruídos presentes na imagem.
    /// Um resultado razoável pode ser visto na imagem "build/israel_q04.png".

void Q04() {

	Mat img = imread("rose.tif", IMREAD_GRAYSCALE);
	Mat img2 = img.clone();
	Mat imgToFourier;
	Mat invdft2;

	Mat hist;

	int option;

	// Kernel Size for filters
	int ksize=4;

	namedWindow("Parameters", WINDOW_KEEPRATIO);
	moveWindow( "Parameters", 0, 0 ); 

	namedWindow("Rose", WINDOW_KEEPRATIO);
	moveWindow( "Rose", 600, 0 ); 

	createTrackbar("Kernel Size", "Parameters", &ksize, 50, 0);
	while(true) {
		// Only positive and even values for kernel size are accepted
		if(ksize == 0 || ksize%2 == 0) {
			ksize++;
		}

		medianBlur(img, img2, ksize);
		hist = computeHistogram1C(img2);
		imshow("Parameters", hist);
		imshow("Rose", img2);
		if((char) waitKey(1)=='q') 
			break;
	}

	img = img2;

	int radius = 30; // Optimal settings
	int offset = 15; // Optimal settings

        namedWindow("Band-Stop Filter", WINDOW_KEEPRATIO);
	moveWindow( "Band-Stop Filter", 0, 0 ); 

        namedWindow("Rose Noisy", WINDOW_KEEPRATIO);
	moveWindow( "Rose Noisy", 200, 0 ); 

        namedWindow("Reconstructed", WINDOW_KEEPRATIO);
	moveWindow( "Reconstructed", 400, 0 ); 

	createTrackbar("radius", "Band-Stop Filter", &radius, img2.cols,0,0);
	createTrackbar("offset", "Band-Stop Filter", &offset, img2.cols,0,0);

	while(true) {

		if(radius == 0) 
			radius=1;

		Mat maskLP = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						radius );

		Mat maskHP = createWhiteDisk (	img2.rows, 
						img2.cols, 
						(int)img2.cols/2, 
						(int)img2.rows/2,
						(radius+offset) );

		maskLP = fftshift(maskLP);
		maskHP = 1-fftshift(maskHP);

		Mat mask = maskLP+maskHP;
		normalize(mask,mask,1,0,NORM_MINMAX); // Normalize the mask for [0,1] range

		// Even img2.cols and rows size fix
		if(mask.cols < img2.cols) { 
			img2 = img2.colRange(0, mask.cols);
			img2.pop_back();
		}

		imgToFourier = img2;

		Mat planes2[] ={Mat_<float>(imgToFourier), 
				Mat::zeros(imgToFourier.size(), CV_32F)};

		merge(planes2, 2, imgToFourier);
		dft(imgToFourier, imgToFourier);
		split(imgToFourier, planes2);

		Mat mag2;
		magnitude(planes2[0], planes2[1], mag2);
		mag2 = applyLogTransform(mag2);

		multiply(planes2[0], mask, planes2[0]);
		multiply(planes2[1], mask, planes2[1]);

		merge(planes2, 2, imgToFourier);

		dft(imgToFourier, invdft2, DFT_INVERSE|DFT_REAL_OUTPUT);
		normalize(invdft2, invdft2, 0, 1, CV_MINMAX);

		imshow("Rose Noisy", scaleImage2_uchar(img));
		imshow("Band-Stop Filter", fftshift(mask));
		imshow("Reconstructed", scaleImage2_uchar(invdft2));

		if((char)waitKey(1)=='q') 
			break;
	}

	imwrite("israel_q04.png", scaleImage2_uchar(invdft2));
}


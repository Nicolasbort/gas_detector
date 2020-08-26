#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>
#include <cv_bridge/cv_bridge.h>
#include <std_msgs/Bool.h>
#include <regex>

using namespace cv;
using namespace std;

#define cvCOLOR_RED     Scalar(0, 0, 255)
#define cvCOLOR_GREEN   Scalar(0, 255, 0)
#define cvCOLOR_BLUE    Scalar(255, 0, 0)

// Parametros de filtros
#define GAUSSIANFILTER 3
#define KERNELSIZE 7


// Limiares da cor azul ( Imagem HSV )
#define MINBLUE         95
#define MAXBLUE         120

#define MINSATBLUE      135
#define MAXSATBLUE      180

#define MINVALBLUE      190
#define MAXVALBLUE      235

// Limiares da cor amarela ( Imagem HSV )
#define MINYELLOW       0
#define MAXYELLOW       70

#define MINSATYELLOW    215
#define MAXSATYELLOW    255

#define MINVALYELLOW    215
#define MAXVALYELLOW    255


int ARR_MAXBLUE[3]   = {MAXBLUE, MAXSATBLUE, MAXVALBLUE};
int ARR_MINBLUE[3]   = {MINBLUE, MINSATBLUE, MINVALBLUE};

int ARR_MAXYELLOW[3] = {MAXYELLOW, MAXSATYELLOW, MAXVALYELLOW};
int ARR_MINYELLOW[3] = {MINYELLOW, MINSATYELLOW, MINVALYELLOW};

int ARR_MAX_C2[3] = {25, 35, 45};
int ARR_MIN_C2[3] = {0, 0, 0};

int ARR_MAXMARCADOR_HSV[3] = {140, 40, 255};
int ARR_MINMARCADOR_HSV[3] = {0, 0, 130};


const int MIN_CONTOUR_AREA = 100;

const int RESIZED_IMAGE_WIDTH = 20;
const int RESIZED_IMAGE_HEIGHT = 30;

Ptr<cv::ml::KNearest>  kNearest(cv::ml::KNearest::create());


// FUNCOES
float razao(RotatedRect rect)
{
    Point2f pts[4];

    rect.points(pts);

    return float( hypot( pts[1].x - pts[0].x, pts[1].y - pts[0].y ) / hypot( pts[2].x - pts[1].x, pts[2].y - pts[1].y ));
}

float rotated_area(RotatedRect rect)
{
  Point2f pts[4];
  float area;

  rect.points( pts );

  area = hypot( pts[1].x - pts[0].x, pts[1].y - pts[0].y ) * hypot( pts[2].x - pts[1].x, pts[2].y - pts[1].y );

  return area;
}

Mat rotateToImage(Mat img, RotatedRect rect)
{

	Mat rotated, cropped;
	float angle = rect.angle;
    Size rect_size = rect.size;

	Mat M = getRotationMatrix2D(rect.center, angle, 1.0);

    // Rotaciona a imagem
	warpAffine(img, rotated, M, img.size());

	// Isola o rotatedrect para dentro do cropped
	getRectSubPix(rotated, rect_size, rect.center, cropped);

    return cropped;
}

bool train()
{
    Mat matClassificationInts;      // we will read the classification numbers into this variable as though it is a vector

    FileStorage fsClassifications("/home/nicolas/hydrone/marcador/classifications_gazebo.xml", FileStorage::READ);        // open the classifications file

    if (fsClassifications.isOpened() == false) {                                                    // if the file was not opened successfully
        cout << "ERRO, falha ao abrir o xml classificador\n";    // show error message
        return false;                                                                                  // and exit program
    }

    fsClassifications["classifications"] >> matClassificationInts;      // read classifications section into Mat classifications variable
    fsClassifications.release();                                        // close the classifications file

                                                                        // read in training images ////////////////////////////////////////////////////////////

    Mat matTrainingImagesAsFlattenedFloats;         // we will read multiple images into this single image variable as though it is a vector

    FileStorage fsTrainingImages("/home/nicolas/hydrone/marcador/images_gazebo.xml", FileStorage::READ);          // open the training images file

    if (fsTrainingImages.isOpened() == false) {                                                 // if the file was not opened successfully
        std::cout << "ERRO, falha ao abrir o xml imagens\n";         // show error message
        return false;                                                                              // and exit program
    }

    fsTrainingImages["images"] >> matTrainingImagesAsFlattenedFloats;           // read images section into Mat training images variable
    fsTrainingImages.release();                                                 // close the traning images file

                                                                                // finally we get to the call to train, note that both parameters have to be of type Mat (a single Mat)
                                                                                // even though in reality they are multiple images / numbers
    kNearest->train(matTrainingImagesAsFlattenedFloats, cv::ml::ROW_SAMPLE, matClassificationInts); 

   ROS_INFO("TREINO FINALIZADO");
}

string getPercent(Mat img)
{
    if (img.empty()) {
        ROS_INFO("ERRO: a imagem está vazia\n");
        return ""; 
    }

    Mat matThresh;   

    // // filter image from grayscale to black and white
    adaptiveThreshold(img, matThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);    

    resize(matThresh, matThresh, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));


    Mat matROIFloat;
    matThresh.convertTo(matROIFloat, CV_32FC1);  
    Mat matROIFlattenedFloat = matROIFloat.reshape(1, 1);
    Mat matCurrentChar(0, 0, CV_32F);

    kNearest->findNearest(matROIFlattenedFloat, 1, matCurrentChar);

    string strFinalString = "";
    float fltCurrentChar = (float)matCurrentChar.at<float>(0, 0);   
    strFinalString += char(int(fltCurrentChar));

    return strFinalString;
}

void invert_color(Mat img)
{
    bitwise_not(img, img);
}

string getKNNChar(Mat img, const char* name)
{
    Mat matTestingNumbers = img.clone();

    Rect rect_left( Point(10, 0), Point(img.cols * 0.52, img.rows) );
    Rect rect_right( Point(img.cols * 0.52, 0), Point(img.cols, img.rows) );

    if (matTestingNumbers.empty()) {
        cout << "ERRO: a imagem está vazia\n";
        return ""; 
    }

    Mat matThresh;   

    // filter image from grayscale to black and white
    adaptiveThreshold(matTestingNumbers, matThresh, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 11, 2);                                   // constant subtracted from the mean or weighted mean


    Mat LEFT_ROI = matThresh(rect_left);
    Mat RIGHT_ROI = matThresh(rect_right);

    resize(LEFT_ROI, LEFT_ROI, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));  
    resize(RIGHT_ROI, RIGHT_ROI, Size(RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT));

    Mat LEFT_FLOAT, RIGHT_FLOAT;

    LEFT_ROI.convertTo(LEFT_FLOAT, CV_32FC1); 
    RIGHT_ROI.convertTo(RIGHT_FLOAT, CV_32FC1);      

    Mat LEFT_FLATTEN = LEFT_FLOAT.reshape(1, 1);
    Mat RIGHT_FLATTEN = RIGHT_FLOAT.reshape(1, 1);

    Mat LEFT_CHAR(0, 0, CV_32F);
    Mat RIGHT_CHAR(0, 0, CV_32F);

    kNearest->findNearest(LEFT_FLATTEN, 1, LEFT_CHAR); 
    kNearest->findNearest(RIGHT_FLATTEN, 1, RIGHT_CHAR);

    float FLOAT_LEFT_CHAR = (float)LEFT_CHAR.at<float>(0, 0);
    float FLOAT_RIGHT_CHAR = (float)RIGHT_CHAR.at<float>(0, 0);

    string strFinalString;
    strFinalString = char(int(FLOAT_LEFT_CHAR));
    strFinalString += char(int(FLOAT_RIGHT_CHAR));
 
    return strFinalString;
}
// FIM FUNCOES


class TesseractExtract
{
public:

  tesseract::TessBaseAPI *ocr;


  TesseractExtract()
  {
    ocr = new tesseract::TessBaseAPI();

    ocr->Init(NULL, "eng", tesseract::OEM_LSTM_ONLY);

    ocr->SetVariable("tessedit_char_whitelist","0123456789-%");

    ocr->SetPageSegMode(tesseract::PSM_SINGLE_LINE);

  }

  ~TesseractExtract()
  {
    ocr->End();
  }
  
  string extract(Mat img, string name = "name", int channels=1)
  {
    string numbers;

    GaussianBlur(img, img, Size(5, 5), 0.0);
    medianBlur(img, img, 9);

    imshow(name, img);

    ocr->SetImage(img.data, img.cols, img.rows, channels, img.step);

    ocr->SetSourceResolution(100);

    numbers = std::string(ocr->GetUTF8Text());

    return numbers;
  }
};


class LandingMark
{
public:

    //// Variaveis ////
    Mat image, main_imgHSV_C3, img_blue_C1, img_yellow_C1, img_final_C1;
    Mat img_lab_can1_C1, img_hsv_can3_C1;
    Mat morph_kernel;

    int rows, cols;
    int centerX, centerY;

    bool success, fstTime;

	Rect mark;
    RotatedRect markRotatedRect;

    vector< vector<Point>> contours;


    LandingMark()
    {
        morph_kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

        success = false;
        fstTime = true;
    }


    void camParam(Mat img)
    {
        rows = img.rows;
        cols = img.cols;

        centerX = img.size().width/2;
        centerY = img.size().height/2;
    }


    void setImage(Mat img)
    {
        if (fstTime)
        {
            LandingMark::camParam(img);
            fstTime = false;
        }

        image = img;
    }


    Mat imfill(Mat img)
    {
        morphologyEx(img, img, MORPH_CLOSE, morph_kernel, Point(-1, -1), 3);

        findContours(img, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        vector<vector<Point>> hull( contours.size() );

        for( size_t i = 0; i < contours.size(); i++ )
        {
            convexHull( contours[i], hull[i] );
        }

        if (hull.size() == 1)
        {
            drawContours(img, hull, 0, 255, -1);
        }
        else if (hull.size() > 1)
        {
            float biggestArea = 0;
            vector<Point> biggestContour;

            for ( size_t i = 0; i < hull.size(); i++ )
            {
                float area = contourArea(hull[i]);

                if (area > biggestArea)
                {
                    biggestArea = area;
                    biggestContour = hull[i];
                }
            }
            vector<vector<Point>> bigContours;
            bigContours.push_back(biggestContour);
            drawContours(img, bigContours, 0, 255, -1);
        }

        return img;
    }


    Mat imlimiares(Mat hsv, int hsvMin[3], int hsvMax[3])
    {
        Mat hsvtresh;

        inRange(hsv, Scalar(hsvMin[0], hsvMin[1], hsvMin[2]), Scalar(hsvMax[0], hsvMax[1], hsvMax[2]), hsvtresh);

        hsvtresh = imfill(hsvtresh);

        return hsvtresh;
    }


    void processImage()
    {
        Mat img_aux;

        cvtColor(image, main_imgHSV_C3, COLOR_BGR2HSV);;

        Mat hsv, output;

        // Pega a area azul
        img_blue_C1 = imlimiares(main_imgHSV_C3, ARR_MINBLUE, ARR_MAXBLUE);
        bitwise_and(main_imgHSV_C3, main_imgHSV_C3, hsv, img_blue_C1);

        // Pega a area amarela
        img_yellow_C1 = imlimiares(main_imgHSV_C3, ARR_MINYELLOW, ARR_MAXYELLOW);
        bitwise_and(hsv, hsv, output, img_yellow_C1);

        // Pega apenas a area do mark
        bitwise_and(img_blue_C1, img_yellow_C1, img_final_C1);
    }


    bool foundMark()
    {
        this->processImage();

        findContours(this->img_final_C1, this->contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        bool succ = false;

        RotatedRect rotRect;



        for (int i=0; i<this->contours.size(); i++)
        {
            rotRect = minAreaRect( this->contours[i] );

            if (rotRect.size.width > 0 && rotRect.size.height > 0)
            {
                if (int(rotRect.angle) % 90 != 0)
                {
                    rotRect.angle += abs(rotRect.angle - 90);
                }

                this->markRotatedRect = rotRect;
                succ = true;
            }else{
                succ = false;
            }
        }

        return succ;
    }


    void drawRotated()
    {
        Point2f vertices2f[4];
        markRotatedRect.points(vertices2f);

        Point vertices[4];    
        for(int i = 0; i < 4; ++i){
            vertices[i] = vertices2f[i];
        }

        fillConvexPoly(image, vertices, 4, cvCOLOR_RED);
    }


    Mat rotatedToImage()
    {
        Mat rotated;
        float angle = markRotatedRect.angle;

        if (markRotatedRect.angle < -45.) {
            angle += 90.0;
        }

        Mat M = getRotationMatrix2D(markRotatedRect.center, angle, 1.0);

        warpAffine(image, rotated, M, image.size());

        return rotated;
    }


    void show()
    {
        imshow("main", image);
    }
};


class ROI
{

public: 

    Mat image;
    Mat editable_image, clean_img;
    vector<Rect> numbers;
    RotatedRect biggest_rect;

    ROI()
    {
        kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);
    }


    ROI(Mat img)
    {
        image = img.clone();

        editable_image = image.clone();

        kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);
    }


    ROI(Mat img, Rect rect)
    {
        kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

        image = img(rect);

        editable_image = image.clone();
    }


    ROI(Mat img, RotatedRect rectROI)
    {
        this->kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

        Mat M, rotated, cropped;
        float angle = rectROI.angle;
        Size size = rectROI.size;

        if (rectROI.angle < -45.) {
            angle += 90.0;
            swap(size.width, size.height);
        }

        M = getRotationMatrix2D(rectROI.center, angle, 1.0);

        warpAffine(img, rotated, M, img.size(), INTER_CUBIC);

        getRectSubPix(rotated, size, rectROI.center, cropped);

        this->image = cropped.clone();

        this->editable_image = image.clone();
    }


    void set(Mat img, RotatedRect rectROI)
    {
        this->kernel = Mat::ones(KERNELSIZE, KERNELSIZE, CV_8U);

        Mat M, rotated, cropped;
        float angle = rectROI.angle;
        Size size = rectROI.size;

        if (rectROI.angle < -45.) {
            angle += 90.0;
            swap(size.width, size.height);
        }

        M = getRotationMatrix2D(rectROI.center, angle, 1.0);

        warpAffine(img, rotated, M, img.size(), INTER_CUBIC);

        getRectSubPix(rotated, size, rectROI.center, cropped);

        this->image = cropped.clone();

        this->editable_image = image.clone();
    }


    bool found(Mat img)
    {
        RotatedRect current;
        float biggest_area = 0, current_area;
        bool found = false;

        vector<vector<Point>> contours;
        findContours(img, contours, RETR_LIST, CHAIN_APPROX_SIMPLE);

        if (contours.size() >= 1)
        {
            for (int i=0; i<contours.size(); i++)
            {
                current = minAreaRect( contours[i] );
            
                current_area = rotated_area(current);

                if ( current_area < 25 )
                    continue;

                if (current_area > biggest_area)
                {
                    this->biggest_rect = current;
                    biggest_area = current_area;
                } 
            }
        }

        if (biggest_area == 0)
        {
            return false;
        }
        else if (razao(biggest_rect) <= 1.5)
        {
            return true;
        }
        else
        {
            return false;
        }
    }


    void rotatedToImage(Mat img)
    {

        Mat rotated, cropped;
        float angle = this->biggest_rect.angle;
        Size rect_size = this->biggest_rect.size;

        Mat M = getRotationMatrix2D(biggest_rect.center, angle, 1.0);

        // Rotaciona a imagem
        warpAffine(img, rotated, M, img.size());

        // Isola o rotatedrect para dentro do cropped
        getRectSubPix(rotated, rect_size, biggest_rect.center, cropped);

        this->image = cropped.clone();

        this->editable_image = this->image.clone();
    }


    void getRectNumbersStatic(Mat img)
    {
        int TOP_LEFT_CORNER_X = img.cols * 0.06;
        int TOP_LEFT_CORNER_Y_1 = img.rows * 0.015;
        int TOP_LEFT_CORNER_Y_2 = img.rows * 0.53;

        int BOT_RIGHT_CORNER_X = img.cols * 0.63;
        int BOT_RIGHT_CORNER_Y_1 = img.rows * 0.45;
        int BOT_RIGHT_CORNER_Y_2 = img.rows * 0.95;

        // Rect da porcentagem
        int P_TOP_LEFT_CORNER_X = img.cols * 0.64;
        int P_TOP_LEFT_CORNER_Y = img.rows * 0.015;

        int P_BOT_RIGHT_CORNER_X = img.cols * 0.95;
        int P_BOT_RIGHT_CORNER_Y = img.rows * 0.45;

        Rect PERCENTE( Point(P_TOP_LEFT_CORNER_X, P_TOP_LEFT_CORNER_Y), Point(P_BOT_RIGHT_CORNER_X, P_BOT_RIGHT_CORNER_Y) );
        Rect TOP( Point(TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y_1), Point(BOT_RIGHT_CORNER_X, BOT_RIGHT_CORNER_Y_1) );
        Rect BOT( Point(TOP_LEFT_CORNER_X, TOP_LEFT_CORNER_Y_2), Point(BOT_RIGHT_CORNER_X, BOT_RIGHT_CORNER_Y_2) );

        numbers.push_back(TOP);
        numbers.push_back(BOT);
        numbers.push_back(PERCENTE);

        // rectangle(img, TOP, cvCOLOR_BLUE, 2);
        // rectangle(img, BOT, cvCOLOR_RED, 2); 
        // rectangle(img, PERCENTE, Scalar(0, 255, 0), 2);
    }


    void improve_image()
    {
        detailEnhance(this->image, this->image);
    }


    void drawRotated(Mat img)
    {
        Point2f pts[4];

        biggest_rect.points(pts);

        vector<Point> ptss(4);

        for (int i=0;i<4; i++)
        {
            ptss[i] = pts[i];
        }

        fillConvexPoly(img, ptss, Scalar(255, 0, 0));
    }


    void resize(int width, int height)
    {
        cv::resize(this->image, this->image, Size(width, height));

        this->editable_image = this->image.clone();
    }

    void show(const char* title)
    {
        imshow(title, this->image);
    }


private:

    vector<vector<Point>> contours;
    Mat kernel;
};


class ContourWithData 
{
public:
    vector<Point> ptContour;
    Rect boundingRect;
    float fltArea;


    bool checkIfContourIsValid() {
        if (fltArea < MIN_CONTOUR_AREA) return false;
        return true;
    }

    static bool sortByBoundingRectXPosition(const ContourWithData& cwdLeft, const ContourWithData& cwdRight) { 
        return(cwdLeft.boundingRect.x < cwdRight.boundingRect.x);
    }

};


TesseractExtract tess;
LandingMark main_image;
ROI base, marcador;

Mat frame;


int SUM_ANGLE = 0, contImwrite = 0;
long count_non_percent = 0, count_percent = 0, contador_frames = 0;
long knn_10_percent = 0, tess_68_percent = 0;
vector<int> numbers_index_bot(99, 0);
vector<int> numbers_index_top(99, 0);


ros::Publisher pub;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
	try
	{
		main_image.setImage(cv_bridge::toCvShare(msg, "bgr8")->image);

        Mat base_C1;
        Mat img_teste, img_inrange;
        Mat img_percent;
        Mat img_number_top, img_number_bot;

		
		// Verifica se achou a base
		if ( main_image.foundMark() )
		{
            Mat kernel = Mat::ones(Size(1, 1), CV_8U);

            base.set( main_image.image, main_image.markRotatedRect );

            imshow("base_mark", base.image);

            inRange(base.image, Scalar(ARR_MIN_C2[0], ARR_MIN_C2[1], ARR_MIN_C2[2]), Scalar(ARR_MAX_C2[0], ARR_MAX_C2[1], ARR_MAX_C2[2]), base_C1);
            morphologyEx(base_C1, base_C1, MORPH_CLOSE, kernel);
            detailEnhance(base.image, base.image);

            if ( marcador.found(base_C1) )
            {
                kernel = Mat::ones(Size(3, 3), CV_8U);

                marcador.biggest_rect.angle += SUM_ANGLE;
                marcador.image = rotateToImage(base.image, marcador.biggest_rect);
                marcador.resize(400, 400);
                marcador.improve_image();


                ////// INICIO ROTAÇÃO DE IMAGEM //////

                    cvtColor(marcador.image, img_teste, COLOR_BGR2HSV);
                    inRange(img_teste, Scalar(ARR_MINMARCADOR_HSV[0], ARR_MINMARCADOR_HSV[1], ARR_MINMARCADOR_HSV[2]), Scalar(ARR_MAXMARCADOR_HSV[0], ARR_MAXMARCADOR_HSV[1], ARR_MAXMARCADOR_HSV[2]), img_inrange );
                    invert_color(img_inrange);
                    

                    marcador.getRectNumbersStatic(img_inrange);

                    img_percent = img_inrange(marcador.numbers[2]);

                    string percent = getPercent(img_percent);

                    imshow("img", img_inrange);

                    if (percent.compare("%") != 0){
                        count_non_percent++;

                        if (count_percent > 0){
                            count_percent--;
                        }
                    }
                    else{
                        contador_frames++;

                        img_number_bot = img_inrange(marcador.numbers[1]);
                        img_number_top = img_inrange(marcador.numbers[0]);

                        int BORDER = 50;

                        copyMakeBorder(img_number_bot, img_number_bot, BORDER, BORDER, BORDER, BORDER, BORDER_ISOLATED, 255);
                        copyMakeBorder(img_number_top, img_number_top, BORDER, BORDER, BORDER, BORDER, BORDER_ISOLATED, 255);

                        string raw_number_bot = tess.extract(img_number_bot, "bot");
                        string raw_number_top = tess.extract(img_number_top, "top");

                        string filtered_number_bot = regex_replace(raw_number_bot, regex(R"([^\d\-])"), "");
                        string filtered_number_top = regex_replace(raw_number_top, regex(R"([^\d\-])"), "");

                        // int int_number_bot = stoi(number_bot);
                        // int int_number_top = stoi(number_top);

                        // numbers_index_bot[int_number_bot]++;
                        // numbers_index_top[int_number_top]++;

                        // std::vector<int>::iterator iterator_bot = max_element(begin(numbers_index_bot), end(numbers_index_bot));
                        // std::vector<int>::iterator iterator_bot = max_element(begin(numbers_index_bot), end(numbers_index_bot));

                        ROS_INFO("TOP: %s   BOT: %s", filtered_number_top.c_str(), filtered_number_bot.c_str());

                        count_percent++;
                        if (count_non_percent > 0){
                            count_non_percent--;
                        }
                    }


                    if (count_non_percent >= 3)
                    {
                        SUM_ANGLE += 90;
                        count_non_percent--;
                        if (SUM_ANGLE == 360){
                            SUM_ANGLE = 0;
                        }
                    }

                // // string msg = "68: " + to_string((knn_10_percent / (float)contador_frames)*100) + "   10: " + to_string((tess_68_percent / (float)contador_frames)*100);
                // // string msg = "68: " + to_string(tess_68_percent) + "  10: " + to_string(knn_10_percent) + "\nFrames: " + to_string(contador_frames);
                // string msg = "68: " + number_top + "  10: " + number_bot + "\nFrames: " + to_string(contador_frames);
                
                // putText(marcador.image, msg, Point(10, 50), 1, 2.5, cvCOLOR_BLUE, 2);
                // marcador.show("Marcador");
            }


			// pub.publish("1");
		}  

		// Mostra a imagem
		main_image.show();
			
		int key = waitKey(20);

		if (key == 32)
		{
			imwrite("/home/nicolas/SENSORS_PHOTOSHOP/marcador"+to_string(contImwrite)+".jpeg", img_inrange);
			imwrite("/home/nicolas/SENSORS_PHOTOSHOP/main_"+to_string(contImwrite)+".jpeg", main_image.image);
            contImwrite++;

			key = 255;
		}else if (key == 10)
        {
            SUM_ANGLE += 90;
        }
        
	}
	catch (cv_bridge::Exception& e)
	{
		ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
	}
}

int main(int argc, char **argv)
{
    train();
 	ros::init(argc, argv, "image_listener");

	ros::NodeHandle n;

	pub = n.advertise<std_msgs::Bool>("/hydrone/gas_detector/marcador", 100);

	image_transport::ImageTransport it(n);
	// Imagem de vídeo
	image_transport::Subscriber sub = it.subscribe("/hydrone/camera_camera/image_raw", 1, imageCallback);
	// Imagem de câmera
	// image_transport::Subscriber sub = it.subscribe("/usb_cam/image_raw", 1, imageCallback);
	ros::spin();
		
	destroyWindow("view");
}
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include "kalmanfilter.h"
using namespace std;
using namespace cv;

bool mouseDown = false;

void CallBackFunc(int event, int x, int y, int flags, void* userdata)
{
     if ( event == CV_EVENT_LBUTTONDOWN )  { mouseDown = true; }
     if ( event == CV_EVENT_LBUTTONUP ) { mouseDown = false; }

     if ( event == EVENT_MOUSEMOVE && mouseDown )
     {
        *((int*) userdata) = x;
        *( ( (int*) userdata ) + 1) = y;
     }

     if ( !mouseDown )
     {
         *((int*) userdata) = -1;
         *( ( (int*) userdata ) + 1) = -1;
     }
}

int main()
{
    // 4-dimensional state, 2-dimensional measurements
    typedef KalmanFilter<4, 2> KF;
    typedef KF::MeasurementSpaceVector Measurement;

    // transition matrix
    KF::StateMatrix F =
            (KF::StateMatrix() <<
                    1, 0, 1, 0,
                    0, 1, 0, 1,
                    0, 0, 1, 0,
                    0, 0, 0, 1).finished();

    // sensor model
    KF::MeasurementStateConversionMatrix H =
            ( KF::MeasurementStateConversionMatrix() <<
            1, 0, 0, 0,
            0, 1, 0, 0).finished();

    // process noise covariance
    KF::StateMatrix Q = 4. *
            (KF::StateMatrix() <<
            1, .5, 0, 0,
            .5, 1, 0, 0,
            0, 0, 1, .5,
            0, 0, .5, 1).finished();

    // measurement noise covariance
    KF::MeasurementMatrix R = 4. *
            (KF::MeasurementMatrix() <<
            1., .5,
            .5, 1.).finished();

    // initialize filter with matrices
    KF filter(F, Q, R, H);

    namedWindow( "Kalman Demo", CV_WINDOW_AUTOSIZE );

    int * m = new int[2]();
    m[0] = m[1] = -1;

    setMouseCallback("Kalman Demo",CallBackFunc, m);

    bool reset = true;

    while ( cv::waitKey(30) )
    {
        Mat img(480, 640, CV_8UC3, Scalar(255, 255, 255));

        // reset filter if nothings incoming from callback handler
        if ( m[0] == -1 && m[1] == -1 && reset != true )
        {
            reset = true;
            filter.reset();
            std::cerr << "resetting filter ..." << std::endl;
        }

        if ( m[0] != -1 && m[1] != -1 && reset == true)
        {
            reset = false;
        }

        // if something's incoming from callback handler
        if ( reset == false )
        {
            reset = false;

            // update filter with mouse coordinates
            const Measurement x(m[0], m[1]);
            filter.update(x);

            // draw mouse position
            circle(img, Point(x(0), x(1)), 10, Scalar(0, 0, 0));

            const KF::StateSpaceVector s = filter.state();
            const KF::StateSpaceVector p = filter.prediction();


            // draw filter position
            circle(img, Point(s(0), s(1)), 5, Scalar(0, 0, 255));

            // draw filter prediction
            circle(img, Point(p(0), p(1)), 5, Scalar(255, 0, 0));

            // draw filter velocity (scaled)
            line(img, Point(s(0), s(1)), Point(s(0) + 5 * s(2), s(1) + 5 * s(3)), Scalar(0, 255, 0));
        }

        imshow( "Kalman Demo", img );

    }
    return 0;
}


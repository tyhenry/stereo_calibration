#pragma once

#include "ofMain.h"
#include "ofxVimba.h"
#include "ofxOpenCv.h"
#include "ofxCv.h"

class ofApp : public ofBaseApp{

	public:
		void setup();
		void update();
		void draw();
		void exit();

		bool calibrateIntrinsics();
		bool stereoCalibrate();

		bool fullCalibration();
		
		bool loadCalibration(string dir, bool absolute = true);

		bool saveIntrinsics(string dir, bool absolute = true);
		bool saveStereoCalibration(string dir, bool absolute = true);
	

		void keyPressed(int key);
		void keyReleased(int key);
		void mouseMoved(int x, int y );
		void mouseDragged(int x, int y, int button);
		void mousePressed(int x, int y, int button);
		void mouseReleased(int x, int y, int button);
		void mouseEntered(int x, int y);
		void mouseExited(int x, int y);
		void windowResized(int w, int h);
		void dragEvent(ofDragInfo dragInfo);
		void gotMessage(ofMessage msg);

		string camIds[2];
		
		ofxVimba::ofxVimbaCam cams[2];
		ofImage imgs[2];
		ofImage undImgs[2];

		ofxCv::Calibration calibrations[2];	// instrinsic calibrations

		float foundTime, waitTime;
		bool bSearching, bFound, bUndistort, bRectify;
		bool bHasIntrinsics, bHasExtrinsics;
		bool bCalibrating = false;

		vector<pair<ofImage,ofImage>> foundImgs;

		// intrinsics calibration matrices

		cv::Mat K0, K1, D0, D1;
		cv::Size sz;

		// stereo calibration matrices

		cv::Vec3d T;		// translation
		cv::Mat R, F, E;	// rotation, fundamental, essential

		cv::Mat R0, R1, P0, P1, Q;	// rectification matrices: rotation0, rotation1, projection0, projection1, disparity-to-depth

		cv::Mat mapX[2];
		cv::Mat mapY[2];	// rectification image maps


		// face finder for rough depth calc
		ofxCv::ObjectFinder finders[2];
		bool bFaceDepth, bHasFace;
		ofRectangle faces[2];
		float faceDepth;
};

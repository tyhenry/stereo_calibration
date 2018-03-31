#include "ofApp.h"

using namespace ofxCv;
using namespace cv;

//using namespace ofxVimba;

const string CONFIG_FILE = "config.yml"; // either absolute path or relative to bin/data

#define PAUSE_EXIT_FAILURE cout << "press any key to quit" << endl; std::cin.get(); ofExit(EXIT_FAILURE);

//--------------------------------------------------------------
void ofApp::setup() {

	// ----------------------- //
	// setup calib config file //
	// ----------------------- //

	FileStorage settings(ofToDataPath(CONFIG_FILE), FileStorage::READ);
	if (settings.isOpened()) {

		// save camera names
		camIds[0] = settings["cam0"];
		camIds[1] = settings["cam1"];

		for (auto& calibration : calibrations) {
			calibration.setPatternSize(settings["xCount"], settings["yCount"]);
			calibration.setSquareSize(settings["squareSize"]);
			CalibrationPattern patternType;
			switch (settings["patternType"]) {
				case 0: patternType = CHESSBOARD; break;
				case 1: patternType = CIRCLES_GRID; break;
				case 2: patternType = ASYMMETRIC_CIRCLES_GRID; break;
			}
			calibration.setPatternType(patternType);
		}
	}
	else {
		cout << "No config file found at: " << ofToDataPath(CONFIG_FILE, true);
		PAUSE_EXIT_FAILURE;
	}





	// ---------------- //
	// setup vimba cams //
	// ---------------- //

	//auto camIds = ofxVimba::listDevices();

	bool bHasCams =
		cams[0].open(camIds[0])
		&& cams[1].open(camIds[1]);


	if (!bHasCams) {
		PAUSE_EXIT_FAILURE;
	}
	else {
		for (int i = 0; i < 2; i++) {
			imgs[i].allocate(cams[i].getCamWidth(), cams[i].getCamHeight(), OF_IMAGE_COLOR); // RGB24, ofxVimba default
			undImgs[i] = imgs[i];
		}
	}

	//ofSetLogLevel("ofxVimbaCam", OF_LOG_ERROR);





	foundTime = 0.f;

	bSearching = false;
	bFound = false;
	bUndistort = false;
	bRectify = false;

	waitTime = 2.f; // sec between pattern searches



	// ------------------ //
	// setup face tracker //
	// ------------------ //

	for (auto& finder : finders) {
		finder.setup("haarcascade_frontalface_default.xml");
		finder.setPreset(ObjectFinder::Fast);
		finder.setFindBiggestObject(true); // ??
	}
	bFaceDepth = false;
	bHasFace = false;
}

//--------------------------------------------------------------
void ofApp::update() {

	float t = ofGetElapsedTimef();

	// update vimba cams

	for (int i = 0; i < 2; i++) {
		cams[i].update();

		if (cams[i].isFrameNew()) {

			// copy, rotate
			imgs[i] = cams[i].getFrame();
			imgs[i].rotate90(-1);

			if (bUndistort && !bRectify) {  // undistort only
				imitate(undImgs[i], imgs[i]);
				calibrations[i].undistort(toCv(imgs[i]), toCv(undImgs[i]));
				undImgs[i].update();
			}
			else if (bRectify) {
				imitate(undImgs[i], imgs[i]);
				cv::remap(toCv(imgs[i]), toCv(undImgs[i]), mapX[i], mapY[i], cv::INTER_LINEAR);
				undImgs[i].update();

				// face detection
				if (bFaceDepth) {	// only runs if rectification is on

					finders[i].update(undImgs[i]);
				}

			}

		}
	}

	if (bFaceDepth) {
	// face detection check
		if (finders[0].size() == 1 && finders[1].size() == 1) {	// both views see a single face, we assume the same one...
			bHasFace = true;
			faces[0] = finders[0].getObject(0);
			faces[1] = finders[1].getObject(0);

			// depth from stereo:
			// https://stackoverflow.com/questions/6241607/distance-to-the-object-using-stereo-camera
			// Construct a cv::Point3f(x,y,disp), transform using Q matrix
			//	the output will be points in an XYZ-coordinate system with the left camera at the center.

			cv::Vec3f pt = toCv(faces[0].getCenter());
			pt[2] = pt[0] - faces[1].getCenter().x;	// replace z with disparity
			vector<cv::Vec3f> src, dst;
			src.push_back(pt);
			//pt = pt * Q;	// convert to x,y,z in left cam space
			//faceDepth = pt[2];

			cv::perspectiveTransform(src, dst, Q);
			if (dst.size() > 0) {
				faceDepth = dst[0][2];	// z = depth
			}

		}
	}
	else {
		bHasFace = false;
	}


	// calibration pattern search

	if (bSearching && t - foundTime > waitTime) {

		Mat mats[2];
		mats[0] = toCv(imgs[0]);
		mats[1] = toCv(imgs[1]);

		std::vector<cv::Point2f> pointBufs[2];

		// find corners
		bFound = calibrations[0].findBoard(mats[0], pointBufs[0]) && calibrations[1].findBoard(mats[1], pointBufs[1]);

		if (bFound) {

			// save images

			//string dir = ofToDataPath("cal_imgs", false);
			string fn0 = ofFilePath::join("cal_imgs", "/L/L_" + ofToString(foundImgs.size()) + ".jpg");
			string fn1 = ofFilePath::join("cal_imgs", "/R/R_" + ofToString(foundImgs.size()) + ".jpg");

			ofLogNotice() << "found checkerboard, saving to " << fn0 << " and " << fn1;

			ofSaveImage(imgs[0].getPixelsRef(), fn0);
			ofSaveImage(imgs[1].getPixelsRef(), fn1);

			foundImgs.emplace_back(imgs[0], imgs[1]);

			foundTime = ofGetElapsedTimef();

		}

	}

	else if (!bSearching) {
		bFound = false;
	}

}

//--------------------------------------------------------------
void ofApp::draw() {

	float x = 0;
	float y = 0;
	float w = ofGetWidth() / float(2);
	float h = 0;

	if (!bUndistort && !bRectify) {
		for (auto& img : imgs) {
			h = w / img.getWidth() * img.getHeight();
			img.draw(x, y, w, h);
			x += w;
		}
	}
	else {
		for (auto& img : undImgs) {
			h = w / img.getWidth() * img.getHeight();
			img.draw(x, y, w, h);
			x += w;
		}
		if (bHasFace) {

			// draw face boxes
			ofPushMatrix();
			ofPushStyle();
			ofNoFill();
			ofScale(w / undImgs[0].getWidth());

			ofDrawRectangle(faces[0]);
			// print face depth
			ofDrawBitmapStringHighlight(ofToString(faceDepth), faces[0].getCenter());

			ofTranslate(undImgs[0].getWidth(), 0.f);
			ofDrawRectangle(faces[1]);
			// print face depth
			ofDrawBitmapStringHighlight(ofToString(faceDepth), faces[1].getCenter());

			ofPopStyle();
			ofPopMatrix();


		}
	}
	

	stringstream ss;
	ss << "Press SPACE to turn " << (bSearching ? "OFF" : "ON") << " checkerboard search.";
	if (bSearching) {
		ss << "\nSearching... saved " << ofToString(foundImgs.size(), 3, ' ') << " frames so far...";
		if (bFound) {
			ss << "\tFOUND board - Searching again in " << ofToString(max(0.f, waitTime - (ofGetElapsedTimef() - foundTime)),2) << " sec";
		}
	}
	if (!bSearching){
		if (foundImgs.size()) {
			ss << "\nPress ENTER to run full calibration on " << foundImgs.size() << " saved frames" << (bHasIntrinsics && bHasExtrinsics ? " - DONE" : "");
			ss << "\nPress DELETE to clear saved frames from memory and reset calibration";
		}
		else {
			ss << "\nNo calibration images saved or loaded yet. Press SPACE to run checkerboard search (or see advanced options below)";
		}
	}

	ss << "\n'L' - load calibration files from disk";
	if (bHasExtrinsics) ss << "\n'F' - track face and calc depth based on calibration.";
	ofDrawBitmapStringHighlight(ss.str(), 10, h + 20);

	stringstream ssa; // advanced

	ssa << "Advanced:";
	ssa << "\n'C' - load checkerboard images from disk ( /data/cal_imgs/L/ + /data/cal_imgs/R/ )";
	ssa << "\n'I' - perform intrinsic calibration on " << foundImgs.size() << " frames" << (bHasIntrinsics ? " - DONE" : "");
	ssa << "\n'E' - perform stereo (extrinsic) calibration based on intrinsic calibration";
	ssa << "\n'U' - toggle undistortion - " << (bUndistort || bRectify ? "ON" : "OFF");
	ssa << "\n'R' - toggle rectification based on stereo calibration - " << (bRectify ? "ON" : "OFF");

	ofDrawBitmapStringHighlight(ssa.str(), 10, h + 100, ofColor::black, ofColor::gray);


}

//--------------------------------------------------------------
void ofApp::exit()
{
	ofxVimba::exit();
}


//--------------------------------------------------------------
bool ofApp::calibrateIntrinsics()
{
	auto& cal0 = calibrations[0];
	auto& cal1 = calibrations[1];

	cal0.reset();
	cal1.reset();

	for (int i = 0; i < foundImgs.size(); ++i) {
		auto& p = foundImgs[i];
		if (!cal0.add(toCv(p.first))) {
			ofLogError() << "Calibration::add() failed on left img " << i;
		}
		if (!cal1.add(toCv(p.second))) {
			ofLogError() << "Calibration::add() failed on right img " << i;
		}
	}

	bHasIntrinsics = cal0.calibrate() && cal1.calibrate();

	if (bHasIntrinsics) {
		// cam matrices
		K0 = cal0.getDistortedIntrinsics().getCameraMatrix();
		K1 = cal1.getDistortedIntrinsics().getCameraMatrix();

		// distortion coefs
		D0 = cal0.getDistCoeffs();
		D1 = cal1.getDistCoeffs();

		// image size
		sz = cal0.getDistortedIntrinsics().getImageSize();

		// save to disk
		string dir = ofToDataPath("cal_imgs", true);
		calibrations[0].save(dir + "/L_calib.yml", true);
		calibrations[1].save(dir + "/R_calib.yml", true);
	}

	return bHasIntrinsics;
}


//--------------------------------------------------------------
bool ofApp::stereoCalibrate()
{

	if (!bHasIntrinsics) {
		cout << "error performing stereo calibration - cameras have not been intrinsically calibrated yet" << endl;
		return bHasExtrinsics = false;
	}

	vector< vector< Point3f > > objectPoints;
	//vector< vector< Point2f > > imagePoints0, imagePoints0;
	//vector< Point2f > corners1, corners2;
	vector< vector< Point2f > > left_img_points, right_img_points;

	auto& cal0 = calibrations[0];
	auto& cal1 = calibrations[1];

	// must be the same size and have data
	if (cal0.size() != cal1.size() || cal0.size() == 0) {
		cout << "error performing stereo calibration from calibration data vectors of size - [0]: " << cal0.size() << ", [1]: " << cal1.size() << endl;
		return bHasExtrinsics = false;
	}

	// make static object points
	std::vector<cv::Point3f> pts = cal0.createObjectPoints(cal0.getPatternSize(), cal0.getSquareSize(), ofxCv::CalibrationPattern::CHESSBOARD);
	objectPoints.resize(cal0.size(), pts);

	// cam matrices
	// K0 = cal0.getDistortedIntrinsics().getCameraMatrix();
	// K1 = cal1.getDistortedIntrinsics().getCameraMatrix();
	
	// distortion coefs
	// D0 = cal0.getDistCoeffs();
	// D1 = cal1.getDistCoeffs();

	// cv::Size sz = cal0.getDistortedIntrinsics().getImageSize();

	// output matrices
	// Vec3d T;		// translation
	// Mat R, F, E;	// rotation, fundamental, essential

	cv::stereoCalibrate(objectPoints, cal0.imagePoints, cal1.imagePoints, K0, D0, K1, D1, sz, R, T, E, F);

	string file = ofToDataPath("cal_imgs", true) + "/stereo_calib.yml";

	cv::FileStorage fs(file, cv::FileStorage::WRITE);
	fs << "K0" << K0;
	fs << "D0" << D0;
	fs << "K1" << K1;
	fs << "D1" << D1;
	fs << "R" << R;		// rotation 0->1
	fs << "T" << T;		// translation 0->1
	fs << "E" << E;		// essential 0->1
	fs << "F" << F;		// fundamental 0->1

	//cv::Mat R0, R1, P0, P1, Q;
	//flag = CV_CALIB_ZERO_DISPARITY;
	int flag = 0; double alpha = -1;

	cv::stereoRectify(K0, D0, K1, D1, cal0.getDistortedIntrinsics().getImageSize(), R, T, R0, R1, P0, P1, Q, flag, alpha);

	fs << "R0" << R0;	// rotation cam 0 
	fs << "R1" << R1;	// rotation cam 1
	fs << "P0" << P0;	// projection cam 0
	fs << "P1" << P1;	// projection cam 1
	fs << "Q" << Q;		// disparity-to-depth mapping matrix

	// calc rectification maps

	cv::initUndistortRectifyMap(K0, D0, R0, P0, cal0.getDistortedIntrinsics().getImageSize(), CV_32F, mapX[0], mapY[0]);
	cv::initUndistortRectifyMap(K1, D1, R1, P1, cal1.getDistortedIntrinsics().getImageSize(), CV_32F, mapX[1], mapY[1]);

	cout << "Stereo calibration complete! --> " << file << endl;

	bHasExtrinsics = true;
	return bHasExtrinsics;
}

//--------------------------------------------------------------
bool ofApp::fullCalibration()
{
	bool ok = calibrateIntrinsics();
	if (ok) {
		ok = stereoCalibrate();
	}
	if (ok) {
		auto res = ofSystemLoadDialog("choose folder where to save calibration files (L_calib.yml, R_calib.yml, stereo_calib.yml)", true, ofToDataPath("", true));
		if (res.bSuccess) {
			ok = ok && saveIntrinsics(res.getPath(), true);
			ok = ok && saveStereoCalibration(res.getPath(), true);
		}
		else {
			ok = false;
		}
	}
	return ok;
}

//--------------------------------------------------------------
bool ofApp::loadCalibration(string dir, bool absolute)
{
	// load calibration files from disk

	if (!absolute) dir = ofToDataPath(dir, true);
	string calLPath = ofFilePath::join(dir, "L_calib.yml");
	string calRPath = ofFilePath::join(dir, "R_calib.yml");
	string calSPath = ofFilePath::join(dir, "stereo_calib.yml");
	
	if (!std::filesystem::exists(calLPath)){
		ofLogError() << "can't load missing " << calLPath;
		return false;
	}
	if (!std::filesystem::exists(calRPath)) {
		ofLogError() << "can't load missing " << calRPath;
		return false;
	}
	if (!std::filesystem::exists(calSPath)) {
		ofLogError() << "can't load missing " << calSPath;
		return false;
	}

	ofLogNotice() << "loading: " << calLPath << ", " << calRPath << ", " << calSPath;


	// instrinsic calibrations
	calibrations[0].load(calLPath, true);
	calibrations[1].load(calRPath, true);

	ofLogNotice() << "loaded instrinsics";

	bHasIntrinsics = true;

	// extrinsic calibrations
	cv::FileStorage fs(calSPath, cv::FileStorage::READ);

	// stereo calibration
	fs["R"] >> R;		// rotation 0->1
	fs["T"] >> T;		// translation 0->1
	fs["E"] >> E;		// essential 0->1
	fs["F"] >> F;		// fundamental 0->1

	// rectification
	fs["R0"] >> R0;	// rotation cam 0 
	fs["R1"] >> R1;	// rotation cam 1
	fs["P0"] >> P0;	// projection cam 0
	fs["P1"] >> P1;	// projection cam 1
	fs["Q"] >> Q;	// disparity-to-depth mapping matrix

	// create rectification image maps

	const auto& I0 = calibrations[0].getDistortedIntrinsics();
	const auto& I1 = calibrations[0].getDistortedIntrinsics();

	cv::initUndistortRectifyMap(I0.getCameraMatrix(), calibrations[0].getDistCoeffs(), R0, P0, I0.getImageSize(), CV_32F, mapX[0], mapY[0]);
	cv::initUndistortRectifyMap(I1.getCameraMatrix(), calibrations[1].getDistCoeffs(), R1, P1, I1.getImageSize(), CV_32F, mapX[1], mapY[1]);

	ofLogNotice() << "loaded extrinsics";

	bHasExtrinsics = true;

	return true;
}

//--------------------------------------------------------------
bool ofApp::saveIntrinsics(string dir, bool absolute)
{
	// save to disk
	if (ofDirectory::doesDirectoryExist(dir, !absolute)) {

		calibrations[0].save(ofFilePath::join(dir, "L_calib.yml"), absolute);
		calibrations[1].save(ofFilePath::join(dir, "R_calib.yml"), absolute);
		return true;
	}
	else {
		ofLogError() << "can't save intrinsic calibration files! directory doesn't exist: " << dir;
		return false;
	}
}


//--------------------------------------------------------------
bool ofApp::saveStereoCalibration(string dir, bool absolute)
{
	// save to disk
	if (ofDirectory::doesDirectoryExist(dir, !absolute)) {

		string file = ofFilePath::join(dir, "stereo_calib.yml");
		if (!absolute) file = ofToDataPath(file, true);

		cv::FileStorage fs(file, cv::FileStorage::WRITE);
		fs << "K0" << K0;
		fs << "D0" << D0;
		fs << "K1" << K1;
		fs << "D1" << D1;
		fs << "R" << R;		// rotation 0->1
		fs << "T" << T;		// translation 0->1
		fs << "E" << E;		// essential 0->1
		fs << "F" << F;		// fundamental 0->1
		fs << "R0" << R0;	// rotation cam 0 
		fs << "R1" << R1;	// rotation cam 1
		fs << "P0" << P0;	// projection cam 0
		fs << "P1" << P1;	// projection cam 1
		fs << "Q" << Q;		// disparity-to-depth mapping matrix
		return true;
	}
	else {
		ofLogError() << "can't save intrinsic calibration files! directory doesn't exist: " << dir;
		return false;
	}
}


//--------------------------------------------------------------
void ofApp::keyPressed(int key) {

}

//--------------------------------------------------------------
void ofApp::keyReleased(int key) {

	// SEARCHING
	if (key == ' ') {
		bSearching = !bSearching;
	}


	// CALIBRATE
	else if (key == OF_KEY_RETURN) {
		fullCalibration();
	}

	// CLEAR
	else if (key == OF_KEY_DEL) {
		foundImgs.clear();
		bHasExtrinsics = false;
		bHasIntrinsics = false;
		bUndistort = false;
		bFaceDepth = false;
		bRectify = false;
	}


	else if (key == 'i' || key == 'I') {

		calibrateIntrinsics();
	}
	else if (key == 'e' || key == 'E') {

		// stereo calibration - extrinsics
		stereoCalibrate();
	}
	else if (key == 'u' || key == 'U') {

		if (bUndistort) {
			bUndistort = false; // turn off
			bFaceDepth = false;
			bRectify = false;
		}
		else if (bHasIntrinsics){

			// undistort on
			bUndistort = true;
		}
	}
	else if (key == 'r' || key == 'R') {

		// toggle stereo rectification of images
		if (bRectify) {
			bRectify = false; // turn off
			bFaceDepth = false;
			// leave undistort on
		}
		else if (bHasExtrinsics) {
			bRectify = true;
			bUndistort = true;
		}
	}
	else if (key == 'l' || key == 'L') {

		// load calibration files from disk
		auto res = ofSystemLoadDialog("load calibration files from folder (L_calib.yml, R_calib.yml, stereo_calib.yml", true, ofToDataPath("cal_imgs", true));
		if (res.bSuccess) {
			loadCalibration(res.getPath(), true);
		}
		else {
			//loadCalibration("cal_imgs", false);	// load from default folder
		}

	}
	else if (key == 'f' || key == 'F') {
		if (bFaceDepth) {
			bFaceDepth = false;
			bUndistort = false;
			bRectify = false;
		}
		else if (bHasIntrinsics && bHasExtrinsics) {
			bFaceDepth = true;
			bRectify = true; // rectification needed for face depth calculation
			bUndistort = true;
		}
	}
	else if (key == 'c' || key == 'C') {
		// load checkerboard images from default folder paths : bin/data/cal_imgs/L/ + /R/

		if (ofDirectory::doesDirectoryExist("cal_imgs/L/") && ofDirectory::doesDirectoryExist("cal_imgs/R/")) {

			foundImgs.clear();
			bHasIntrinsics = false;
			bHasExtrinsics = false;
			bUndistort = false;
			bRectify = false;
			bFaceDepth = false;

			ofDirectory dirL;
			dirL.allowExt("jpg");
			dirL.allowExt("png");
			dirL.listDir("cal_imgs/L/");

			ofDirectory dirR;
			dirR.allowExt("jpg");
			dirR.allowExt("png");
			dirR.listDir("cal_imgs/R/");

			ofLogNotice() << "found " << dirL.size() << " left, " << dirR.size() << " right images in ./data/cal_imgs/L and /R";

			int maxPairs = min(dirL.size(), dirR.size());

			ofImage L, R;

			for (std::size_t i = 0; i < maxPairs; i++) {

				if (L.loadImage(dirL[i]) && R.loadImage(dirR[i])) {
					foundImgs.emplace_back(L, R);
				}
				else {
					ofLogError() << "error loading image pair: " << dirL[i].getAbsolutePath() << ", " << dirR[i].getAbsolutePath();
				}
			}
		}
		else {
			ofLogError() << "error loading imges from disk - ./data/cal_imgs/L/ or /R/ doesn't exist!";
		}

	}
}

//--------------------------------------------------------------
void ofApp::mouseMoved(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseDragged(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mousePressed(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseReleased(int x, int y, int button) {

}

//--------------------------------------------------------------
void ofApp::mouseEntered(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::mouseExited(int x, int y) {

}

//--------------------------------------------------------------
void ofApp::windowResized(int w, int h) {

}

//--------------------------------------------------------------
void ofApp::gotMessage(ofMessage msg) {

}

//--------------------------------------------------------------
void ofApp::dragEvent(ofDragInfo dragInfo) {

}

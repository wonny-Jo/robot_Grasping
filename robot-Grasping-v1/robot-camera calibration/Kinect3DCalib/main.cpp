#include "define.h"
#include "BodyCalib.h"
#include "BodyDataLoader.h"
#include "DataRecord.h"


#define SAMPLECOUNT  702
int SAMPLECOUNT_cal = 0;


#ifndef MATLAB
#define BIN
#endif

int main(){

	////////////////////////////////////////////////mocap/////////////////////////////////////////////////////////////
	BodyCalib Calib;
	int iCount = 0, jCount = 0;
	int tDataCount = 0;
	

#ifdef BIN
 	DataRecord data;
	// data.OpenDataFile("C:\\Users\\PycharmProjects\\rgbd_2_calibration\\data\\collectedDATA.bin", 'r');
	//data.OpenDataFile("20180903_220710.bin", 'r');
	data.OpenDataFile("C:/Users/user/Desktop/AI_Project/ball_calib_python_20200803/Mocap_Robot-calibration-Matlab-ver--master/joint_list_20200813_RL_00_alignedxyz.bin", 'r'); // 
	data.ReadAllData();
	tDataCount = data.GetDataCount();
#else
	MatlabLoader matlab;
	matlab.ReadFile("C:\\Users\\Vision Demo\\Desktop\\Kinect3DCalib-master\\Kinect3DCalib-master\\Knect3dCalib\\Kinect3DCalib\\MatlabDataFile.mat", "data");
	tDataCount = matlab.GetDataCount();
#endif

	for(int i = 0; i < tDataCount; i++){
		cv::Point3f robot, mocap;
#ifdef BIN
		data.GetBodyData(&robot, &mocap);
#else
		matlab.GetBodyData(&robot, &mocap);
#endif
		/*if ((mocap.x > 3)|| (mocap.x < -3 ) || (isnan(mocap.x)))
			continue;*/
		if (((mocap.x == 0) && (mocap.y == 0) && (mocap.z == 0)) || (cvIsNaN(mocap.x)) || (cvIsNaN(mocap.y)) || (cvIsNaN(mocap.z)))
			continue;
		else
		{
			Calib.DataStore(mocap, robot);
			//printf("data_num %4d : [%+.6f %+.6f %+.6f] - [%+.6f %+.6f %+.6f]\n", jCount, robot.x, robot.y, robot.z, mocap.x, mocap.y, mocap.z);
			cout<<"data_num "<<jCount<<" : ["<< robot.x<<' '<< robot.y<<' '<< robot.z<<']'<<'\n';
			jCount++;
		}
	}
	//printf("Inserted Data Count : %d\n", jCount);
	cout << "Inserted Data Count : " << jCount << '\n';
#ifdef BIN
	data.CloseDataFile();
#else
#endif
	//////////////////////////////////////////////////////////////////////////////////////////////////////////////////

	int LoopCount = Calib.CalcLoopNUM(0.999, 0.85, 50); // 0.8
	//printf("Loop Count : %d\n", LoopCount);
	cout << "Loop Count : " << LoopCount << '\n';
	Calib.InitParam(LoopCount, 0.005, jCount);	// 0.003
	Calib.CalcMatrix();

	return 0;
}
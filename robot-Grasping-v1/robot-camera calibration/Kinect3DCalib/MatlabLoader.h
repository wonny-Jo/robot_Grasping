#pragma once
#include <opencv2\opencv.hpp>
//#include "engine.h"
//#include "mat.h"

class MatlabLoader
{
public:
	MatlabLoader(void);
	~MatlabLoader(void);

	int GetDataCount();
	void ReadFile(char *fileName, char *matrixName);
	void GetBodyData(cv::Point3f *robot, cv::Point3f *mocap);

private:
	std::list<std::pair<cv::Point3f, cv::Point3f>> m_list;
};


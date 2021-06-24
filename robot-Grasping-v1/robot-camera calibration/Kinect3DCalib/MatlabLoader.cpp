#include "MatlabLoader.h"


MatlabLoader::MatlabLoader(void)
{
}


MatlabLoader::~MatlabLoader(void)
{
}

int MatlabLoader::GetDataCount(){
	return m_list.size();
}

void MatlabLoader::ReadFile(char *fileName, char *matrixName){
	MATFile *pmat;
	mxArray *Data;
	cv::Point3f robot, mocap;
	std::pair<cv::Point3f, cv::Point3f> t_pair;

	int	  i, ndir, ndim;
	const char **dir;

	pmat = matOpen(fileName, "r");
	if (pmat == NULL) {
		printf("Error opening file %s\n", fileName);
		return;
	}

	//    pa = matGetNextVariableInfo(pmat, &name);
	Data = matGetVariable(pmat, matrixName);
	if (Data == NULL) {
		printf("Error reading in file %s\n", fileName);
		return;
	}

	//	ndim = mxGetNumberOfDimensions(pa);

	mwSize nRow = mxGetM(Data);
	mwSize nCol = mxGetN(Data);

	if(nCol != 6){
		printf("Matrix %s invalid format\n", matrixName);
	}

	double *pVal = (double*)mxGetPr(Data);

	/*noisyVel = new double[nCol*nRow];

	for (i = 0; i < nRow*nCol; i++)
	noisyVel[i] = *pVal++;*/
	for(int i = 0; i < nRow; i++){

		robot.x = *pVal++;
		robot.y = *pVal++;
		robot.z = *pVal++;

		mocap.x = *pVal++;
		mocap.y = *pVal++;
		mocap.z = *pVal++;

		t_pair.first = robot;
		t_pair.second = mocap;

		m_list.push_back(t_pair);
	}

	mxDestroyArray(Data);

	if (matClose(pmat) != 0) {
		printf("Error closing file %s\n",fileName);
		return;
	}
}

void MatlabLoader::GetBodyData(cv::Point3f *robot, cv::Point3f *mocap){
	if(m_list.empty()){
		printf("error - Data Vector is empty.\n");
		return;
	}

	std::list<std::pair<cv::Point3f, cv::Point3f>>::iterator it;
	it = m_list.begin();
	std::pair<cv::Point3f, cv::Point3f> t_pair;
	t_pair  = *it;
	m_list.pop_front();

	*robot = t_pair.first;
	*mocap = t_pair.second;
}
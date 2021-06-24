#include "DataRecord.h"


DataRecord::DataRecord(void)
{
}


DataRecord::~DataRecord(void)
{
}

void DataRecord::DataInsert(Pose3D robot, cv::Point3f mocap){
	cv::Point3f temp;
	temp.x = robot.x;
	temp.y = robot.y;
	temp.z = robot.z;

	//·Îº¿ ÁÂÇ¥, macap ÁÂÇ¥¼øÀ¸·Î
	fwrite(&temp, sizeof(cv::Point3f), 1, Datafp);
	fwrite(&mocap, sizeof(cv::Point3f), 1, Datafp);
}

void DataRecord::OpenDataFile(char *filename, char mod){
	if(mod == 'w'){
		Datafp = fopen(filename, "wb");
	}
	else if(mod == 'r'){
		Datafp = fopen(filename, "rb");
	}else{
		printf("Invalid File Open mode!\n");
		printf("-w : write mode\n");
		printf("-r : Read mode\n");

		return;
	}

	if(Datafp == NULL){
		printf("Can not open file\n");
	}
}

void DataRecord::ReadAllData(){
	cv::Point3d robot, mocap;
	std::pair<cv::Point3d, cv::Point3d> t_pair;
	int i = 0;

	while(!feof(Datafp)){
		fread(&robot, sizeof(cv::Point3d), 1, Datafp);
		fread(&mocap, sizeof(cv::Point3d), 1, Datafp);

		t_pair.first = robot;
		t_pair.second = mocap;

		m_list.push_back(t_pair);

		/*if(feof(Datafp))
			break;*/
	}
	m_list.pop_back();
	printf("Data Load Complete!\n");
}

void DataRecord::CloseDataFile(){
	if(Datafp != NULL){
		fclose(Datafp);
		Datafp = NULL;
	}
}

int DataRecord::GetDataCount(){
	return m_list.size();
}

void DataRecord::GetBodyData(cv::Point3f *robot, cv::Point3f *mocap){
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
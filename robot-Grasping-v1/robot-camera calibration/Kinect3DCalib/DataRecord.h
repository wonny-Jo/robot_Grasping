#include <opencv2\opencv.hpp>
#include <list>

typedef struct
{
	double x, y, z;
	double Roll, Pitch, Yaw;
}Pose3D;

class DataRecord
{
public:
	DataRecord(void);
	~DataRecord(void);

	void DataInsert(Pose3D robot, cv::Point3f mocap);

	/*mode 'w' : write mode
	mode 'r' : read mode*/
	void OpenDataFile(char *filename, char mod);
	void CloseDataFile();

	void ReadAllData();
	int GetDataCount();
	void GetBodyData(cv::Point3f *robot, cv::Point3f *mocap);

private:
	FILE *Datafp;

	std::list<std::pair<cv::Point3f, cv::Point3f>> m_list;
};


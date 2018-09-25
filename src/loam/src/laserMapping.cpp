#include <math.h>

#include "common.h"
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

const float scanPeriod = 0.1;      //velodyne激光雷达周期

const int stackFrameNum = 1;
const int mapFrameNum = 5;


double timeLaserCloudCornerLast = 0;     //次边缘点时间戳
double timeLaserCloudSurfLast = 0;       //次平面点时间戳
double timeLaserCloudFullRes = 0;        //完整点云时间戳
double timeLaserOdometry = 0;            //轨迹时间戳


bool newLaserCloudCornerLast = false;    //次边缘点云更新标志
bool newLaserCloudSurfLast = false;      //次平面点云更新标志
bool newLaserCloudFullRes = false;       //完整点云更新标志
bool newLaserOdometry = false;           //轨迹更新标志

//地图网格大小
int laserCloudCenWidth = 10;     //x轴
int laserCloudCenHeight = 10;     //y轴
int laserCloudCenDepth = 5;     //z轴


const int laserCloudWidth = 21;     //x轴
const int laserCloudHeight = 21;    //y轴
const int laserCloudDepth = 11;     //z轴
const int laserCloudNum = laserCloudWidth * laserCloudHeight * laserCloudDepth;


int laserCloudValidInd[125];
int laserCloudSurroundInd[125];


//次边缘点点云，次平面点点云
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>);


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerStack(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfStack(new pcl::PointCloud<pcl::PointXYZI>);


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerStack2(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfStack2(new pcl::PointCloud<pcl::PointXYZI>);

//待优化问题
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr coeffSel(new pcl::PointCloud<pcl::PointXYZI>);


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurround(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurround2(new pcl::PointCloud<pcl::PointXYZI>);

pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerFromMap(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfFromMap(new pcl::PointCloud<pcl::PointXYZI>);


//完整点云
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>);


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerArray[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfArray[laserCloudNum];


pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerArray2[laserCloudNum];
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfArray2[laserCloudNum];


//次边缘点kd树，次平面点kd树
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>);
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfFromMap(new pcl::KdTreeFLANN<pcl::PointXYZI>);



float transformSum[6] = {0};		//粗定位结果
float transformIncre[6] = {0};		//增量
float transformTobeMapped[6] = {6};  //最终的位姿估计
float transformBefMapped[6] = {0};   //transformBefMapped = transformSum 
float transformAftMapped[6] = {0};   //transformAftMapped = transformTobeMapped


void transformAssociateToMap()
{
	//绕y轴旋转
	float x1 = cos(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
	         - sin(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);
	float y1 = transformBefMapped[4] - transformSum[4];
	float z1 = sin(transformSum[1]) * (transformBefMapped[3] - transformSum[3])
			 + cos(transformSum[1]) * (transformBefMapped[5] - transformSum[5]);

    //绕x轴旋转
	float x2 = x1;
	float y2 = cos(transformSum[0]) * y1 + sin(transformSum[0]) * z1;
	float z2 = -sin(transformSum[0]) * y1 + cos(transformSum[0]) * z1;


	//绕z轴旋转
	transformIncre[3] = cos(transformSum[2]) * x2 + sin(transformSum[2]) * y2;
	transformIncre[4] = -sin(transformSum[2]) * x2 + cos(transformSum[2]) * y2;
	transformIncre[5] = z2;



	float sbcx = sin(transformSum[0]);
	float cbcx = cos(transformSum[0]);
	float sbcy = sin(transformSum[1]);
	float cbcy = cos(transformSum[1]);
	float sbcz = sin(transformSum[2]);
	float cbcz = cos(transformSum[2]);


	float sblx = sin(transformBefMapped[0]);
	float cblx = cos(transformBefMapped[0]);
	float sbly = sin(transformBefMapped[1]);
	float cbly = cos(transformBefMapped[1]);
	float sblz = sin(transformBefMapped[2]);
	float cblz = cos(transformBefMapped[2]);


	float salx = sin(transformAftMapped[0]);
	float calx = cos(transformAftMapped[0]);
	float saly = sin(transformAftMapped[1]);
	float caly = cos(transformAftMapped[1]);
	float salz = sin(transformAftMapped[2]);
	float calz = cos(transformAftMapped[2]);



	float srx = -sbcx * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx *cblz)
			  - cbcx * sbcy * (calx * calz * (cbly * sblz - cblz * sblx * sbly)
			  - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly)
			  - cbcx * cbcy * (calx * salz * (cblz * sbly - cbly * sblx * sblz)
			  - calx * calz *(sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx);
    transformTobeMapped[0] = -asin(srx);


	float srycrx = sbcx * (cblx * cblz * (caly * salz - calz * salx * saly)
				 - cblx * sblz * (caly * calz + salx * saly * salz) + calx * saly * sblx)
				 - cbcx * cbcy * ((caly * calz + salx * saly * salz) * (cblz * sbly - cbly * sblx * sblz)
				 + (caly * salz - calz * salx * saly) * (sbly * sblz + cbly * cblz * sblx) - calx * cblx * cbly * saly)
				 + cbcx * sbcy * ((caly * calz + salx * saly * salz) * (cbly * cblz + sblx * sbly * sblz)
				 + (caly * salz - calz * salx * saly) * (cbly * sblz - cblz * sblx * sbly) + calx * cblx * saly * sbly);
	float crycrx = sbcx * (cblx * sblz * (calz * saly - caly * salx * salz)
				 - cblx * cblz * (saly * salz + caly * calz * salx) + calx * caly * sblx)
				 + cbcx * cbcy * ((saly * salz + caly * calz * salx) * (sbly * sblz + cbly * cblz * sblx)
				 + (calz * saly - caly * salx * salz) * (cblz * sbly - cbly * sblx * sblz) + calx * caly * cblx * cbly)
				 - cbcx * sbcy * ((saly * salz + caly * calz * salx) * (cbly * sblz - cblz * sblx * sbly)
				 + (calz * saly - caly * salx * salz) * (cbly * cblz + sblx * sbly * sblz) - calx * caly * cblx * sbly);
	transformTobeMapped[1] = atan2(srycrx / cos(transformTobeMapped[0]),
                                   crycrx / cos(transformTobeMapped[0]));



	float srzcrx = (cbcz * sbcy - cbcy * sbcx * sbcz) * (calx * salz * (cblz * sbly - cbly * sblx * sblz)
				 - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx)
				 - (cbcy * cbcz + sbcx * sbcy * sbcz) * (calx * calz * (cbly * sblz - cblz * sblx * sbly)
				 - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly)
				 + cbcx * sbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
	float crzcrx = (cbcy * sbcz - cbcz * sbcx * sbcy) * (calx * calz * (cbly * sblz - cblz * sblx * sbly)
				 - calx * salz * (cbly * cblz + sblx * sbly * sblz) + cblx * salx * sbly)
				 - (sbcy * sbcz + cbcy * cbcz * sbcx) * (calx * salz * (cblz * sbly - cbly * sblx * sblz)
				 - calx * calz * (sbly * sblz + cbly * cblz * sblx) + cblx * cbly * salx)
				 + cbcx * cbcz * (salx * sblx + calx * cblx * salz * sblz + calx * calz * cblx * cblz);
	transformTobeMapped[2] = atan2(srzcrx / cos(transformTobeMapped[0]),
								   crzcrx / cos(transformTobeMapped[0]));


	

	x1 = cos(transformTobeMapped[2]) * transformIncre[3] - sin(transformTobeMapped[2]) * transformIncre[4];
	y1 = sin(transformTobeMapped[2]) * transformIncre[3] + cos(transformTobeMapped[2]) * transformIncre[4];
	z1 = transformIncre[5];
	     
	x2 = x1;
	y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
	z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;
		    
	transformTobeMapped[3] = transformAftMapped[3]
						   - (cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2);
	transformTobeMapped[4] = transformAftMapped[4] - y2;
	transformTobeMapped[5] = transformAftMapped[5]
				           - (-sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2);

}


//更新转换矩阵
void transformUpdate()
{
	for(int i = 0; i < 6; i ++)
	{
		transformBefMapped[i] = transformSum[i];
		transformAftMapped[i] = transformTobeMapped[i];
	}
}


void pointAssociateToMap(pcl::PointXYZI const * const pi, pcl::PointXYZI * const po)
{

	//绕z轴旋转
	float x1 = cos(transformTobeMapped[2]) * pi->x - sin(transformTobeMapped[2]) * pi->y; 
	float y1 = sin(transformTobeMapped[2]) * pi->x + cos(transformTobeMapped[2]) * pi->y;
	float z1 = pi->z;

	//绕y轴旋转
	float x2 = cos(transformTobeMapped[1]) * x1 + sin(transformTobeMapped[1]) * z1;
	float y2 = y1;
	float z2 = -sin(transformTobeMapped[1]) * x1 + cos(transformTobeMapped[1]) * z1;

	//绕x轴旋转后再平移
	po->x = x2 + transformTobeMapped[3];
	po->y = cos(transformTobeMapped[0]) * y2 - sin(transformTobeMapped[0]) * z2 + transformTobeMapped[4];
	po->z = sin(transformTobeMapped[0]) * y2 + cos(transformTobeMapped[0]) * z2 + transformTobeMapped[5];
	po->intensity = pi->intensity;

	/*
	//绕z轴旋转
	float x1 = cos(transformTobeMapped[2]) * pi->x - sin(transformTobeMapped[2]) * pi->y;
	float y1 = sin(transformTobeMapped[2]) * pi->x + cos(transformTobeMapped[2]) * pi->y;
	float z1 = pi->z;


	//绕x轴旋转
	float x2 = x1;
	float y2 = cos(transformTobeMapped[0]) * y1 - sin(transformTobeMapped[0]) * z1;
	float z2 = sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

    //绕y轴旋转
	po->x = cos(transformTobeMapped[1]) * x2 + sin(transformTobeMapped[1]) * z2 + transformTobeMapped[3];
	po->y = y2 + transformTobeMapped[4];
	po->z = -sin(transformTobeMapped[1]) * x2 + cos(transformTobeMapped[1]) * z2 + transformTobeMapped[5];

	po->intensity = pi->intensity;
	*/
}




void pointAssociateTobeMapped(pcl::PointXYZI const * const pi, pcl::PointXYZI * const po)
{
	//先平移再绕x轴逆旋转
	float x1 = pi->x - transformTobeMapped[3];
	float y1 = cos(transformTobeMapped[0]) * (pi->y - transformTobeMapped[4]) + sin(transformTobeMapped[0]) * (pi->z - transformTobeMapped[5]);
	float z1 = -sin(transformTobeMapped[0]) * (pi->y - transformTobeMapped[4]) + cos(transformTobeMapped[0]) * (pi->z - transformTobeMapped[5]);

	//绕y轴旋转
	float x2 = cos(transformTobeMapped[1]) * x1 - sin(transformTobeMapped[1]) * z1;
	float y2 = y1;
	float z2 = sin(transformTobeMapped[1]) * x1 + cos(transformTobeMapped[1]) * z1;

	//绕z轴旋转
	po->x = cos(transformTobeMapped[2]) * x2 + sin(transformTobeMapped[2]) * y2;
	po->y = -sin(transformTobeMapped[2]) * x2 + cos(transformTobeMapped[2]) * y2;
	po->z = z2;
	po->intensity = pi->intensity;

	/*
	float x1 = cos(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3])
			 - sin(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);
	float y1 = pi->y - transformTobeMapped[4];
	float z1 = sin(transformTobeMapped[1]) * (pi->x - transformTobeMapped[3])
			 + cos(transformTobeMapped[1]) * (pi->z - transformTobeMapped[5]);

	
	float x2 = x1;
	float y2 = cos(transformTobeMapped[0]) * y1 + sin(transformTobeMapped[0]) * z1;
	float z2 = -sin(transformTobeMapped[0]) * y1 + cos(transformTobeMapped[0]) * z1;

	po->x = cos(transformTobeMapped[2]) * x2 + sin(transformTobeMapped[2]) * y2;
	po->y = -sin(transformTobeMapped[2]) * x2 + cos(transformTobeMapped[2]) * y2;
	po->z = z2;
	po->intensity = pi->intensity;
	*/
}



//次边缘点回调函数
void laserCloudCornerLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudCornerLast2)
{
	timeLaserCloudCornerLast = laserCloudCornerLast2->header.stamp.toSec();
	laserCloudCornerLast->clear();
	pcl::fromROSMsg(*laserCloudCornerLast2, *laserCloudCornerLast);
	newLaserCloudCornerLast = true;
}

//次平面点回调函数
void laserCloudSurfLastHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudSurfLast2)
{
	timeLaserCloudSurfLast = laserCloudSurfLast2->header.stamp.toSec();
	laserCloudSurfLast->clear();
	pcl::fromROSMsg(*laserCloudSurfLast2, *laserCloudSurfLast);
	newLaserCloudSurfLast = true;
}

//完整点云回调函数
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullRes2)
{
	timeLaserCloudFullRes = laserCloudFullRes2->header.stamp.toSec();
	laserCloudFullRes->clear();
	pcl::fromROSMsg(*laserCloudFullRes2, *laserCloudFullRes);
	newLaserCloudFullRes = true;
}


//轨迹回调函数
void laserOdometryHandler(const nav_msgs::Odometry::ConstPtr &laserOdometry)
{
	timeLaserOdometry = laserOdometry->header.stamp.toSec();

	double roll,pitch,yaw;
	geometry_msgs::Quaternion geoQuat = laserOdometry->pose.pose.orientation;
	tf::Matrix3x3(tf::Quaternion(geoQuat.x, geoQuat.y, geoQuat.z, geoQuat.w)).getRPY(roll,pitch, yaw);

	transformSum[0] = roll;
	transformSum[1] = pitch;
	transformSum[2] = yaw;

	transformSum[3] = laserOdometry->pose.pose.position.x;
	transformSum[4] = laserOdometry->pose.pose.position.y;
	transformSum[5] = laserOdometry->pose.pose.position.z;

	newLaserOdometry = true;

}



int main(int argc,char **argv)
{
	ros::init(argc,argv,"laserMapping");
	ros::NodeHandle nh;

	ros::Subscriber subLaserCloudCornerLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_corner_last", 2, laserCloudCornerLastHandler);      //次边缘点订阅函数
	ros::Subscriber subLaserCloudSurfLast = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_surf_last", 2, laserCloudSurfLastHandler);     //次平面点订阅函数
	ros::Subscriber subLaserOdometry = nh.subscribe<nav_msgs::Odometry>("/laser_odom_to_init", 5, laserOdometryHandler);     //轨迹订阅函数
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_3", 2, laserCloudFullResHandler);    //完整点云订阅函数

	ros::Publisher pubLaserCloudSurround = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surround", 1);
	ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_registratiin", 2);
	ros::Publisher pubOdomAftMapped = nh.advertise<nav_msgs::Odometry>("/aft_mapped_to_init", 5);



	nav_msgs::Odometry odoAftMapped;
	odoAftMapped.header.frame_id = "/camera_init";
	odoAftMapped.child_frame_id = "/aft_mapped";


	std::vector<int> pointSearchInd;
	std::vector<float> pointSearchSqDis;

	pcl::PointXYZI pointOri, pointSel, pointProj, coeff;


	cv::Mat matA0(5, 3, CV_32F, cv::Scalar::all(0));
	cv::Mat matB0(5, 1, CV_32F, cv::Scalar::all(-1));
	cv::Mat matX0(3, 1, CV_32F, cv::Scalar::all(0));


	cv::Mat matA1(3, 3, CV_32F, cv::Scalar::all(0));
	cv::Mat matD1(1, 3, CV_32F, cv::Scalar::all(0));
	cv::Mat matV1(3, 3, CV_32F, cv::Scalar::all(0));


	bool isDegenerate = false;
	cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));


	//次边缘点下采样
	pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterCorner;
	downSizeFilterCorner.setLeafSize(0.2, 0.2, 0.2);

	//次平面点下采样
	pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterSurf;
	downSizeFilterSurf.setLeafSize(0.4, 0.4, 0.4);

	//地图点下采样
	pcl::VoxelGrid<pcl::PointXYZI> downSizeFilterMap;
	downSizeFilterMap.setLeafSize(0.6, 0.6, 0.6);


	//laserCloudNum = 21 * 21 * 11 = 4851
	//初始化
	for(int i = 0; i < laserCloudNum; i ++)
	{
		laserCloudCornerArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
		laserCloudSurfArray[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
		laserCloudCornerArray2[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
		laserCloudSurfArray2[i].reset(new pcl::PointCloud<pcl::PointXYZI>);
	}

	int frameCount = stackFrameNum - 1;    //stackFrameNum = 1
	int mapFrameCount = mapFrameNum - 1;   //mapFrameNum = 5
	ros::Rate rate(100);
	bool status = ros::ok();
	while(status)
	{
		ros::spinOnce();
		
		if(newLaserCloudCornerLast && newLaserCloudSurfLast && newLaserCloudFullRes && newLaserOdometry && fabs(timeLaserCloudCornerLast - timeLaserOdometry) < 0.005 && fabs(timeLaserCloudSurfLast - timeLaserOdometry) < 0.005 && fabs(timeLaserCloudFullRes - timeLaserOdometry) < 0.005)
		{
			newLaserCloudCornerLast = false;
			newLaserCloudSurfLast = false;
			newLaserCloudFullRes = false;
			newLaserOdometry = false;

			frameCount ++;
			if(frameCount >= stackFrameNum)
			{
				transformAssociateToMap();    //更新transformTobeMapped
				std::cout << "transformTobeMapped = " << transformTobeMapped[0] << '\t' << transformTobeMapped[1] << '\t' << transformTobeMapped[2] << '\t' << transformTobeMapped[3] << '\t' << transformTobeMapped[4] << '\t' << transformTobeMapped[5] << std::endl;

				int laserCloudCornerLaserNum = laserCloudCornerLast->points.size();
				for(int i = 0; i < laserCloudCornerLaserNum; i ++)
				{
					pointAssociateToMap(&laserCloudCornerLast->points[i], &pointSel);
					laserCloudCornerStack2->push_back(pointSel);
				}

				int laserCloudSurfLastNum = laserCloudSurfLast->points.size();
				for(int i = 0; i < laserCloudSurfLastNum; i ++)
				{
					pointAssociateToMap(&laserCloudSurfLast->points[i], &pointSel);
					laserCloudSurfStack2->push_back(pointSel);
				}
			}

			
			if(frameCount >= stackFrameNum)
			{
				frameCount = 0;

				pcl::PointXYZI pointOnYAxis;
				pointOnYAxis.x = 0.0;
				pointOnYAxis.y = 0.0;
				pointOnYAxis.z = 10.0;
				pointAssociateToMap(&pointOnYAxis, &pointOnYAxis);


				//laserCloudCenWidth = 10 && laserCloudCenHeight = 10 && laserCloudCenDepth = 5
				//cube = 50 * 50 * 50
				
				/*
				(-525,-475)中心为-500	centerCubeI = 0
				(-125,-75)中心为-100	centerCubeI = 8
				(-75,-25)中心为-50	centerCubeI = 9
				(-25,25)中心为0	centerCubeI = 10 

				*/



				//计算网格下标
				int centerCubeI = int((transformTobeMapped[3] + 25.0) / 50.0) + laserCloudCenWidth;
				int centerCubeJ = int((transformTobeMapped[4] + 25.0) / 50.0) + laserCloudCenHeight;
				int centerCubeK = int((transformTobeMapped[5] + 25.0) / 50.0) + laserCloudCenDepth;
				if(transformTobeMapped[3] + 25.0 < 0)	centerCubeI --;
				if(transformTobeMapped[4] + 25.0 < 0)	centerCubeJ --;
				if(transformTobeMapped[5] + 25.0 < 0)	centerCubeK --;



				while(centerCubeI < 3)
				{
					for(int j = 0; j < laserCloudHeight; j ++)
					{
						for(int k = 0; k < laserCloudDepth; k ++)
						{
							int i = laserCloudWidth - 1;    // i = 20 (i的最大下标)
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							//后移一位
							for(; i >= 1; i --)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[(i - 1) * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[(i - 1) * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							}

							//第一个清空
							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;

							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}

					centerCubeI ++;
					laserCloudCenWidth ++;
				}


	
				while(centerCubeI >= laserCloudWidth - 3)
				{
					for(int j = 0; j < laserCloudHeight; j ++)
					{
						for(int k = 0; k < laserCloudDepth; k ++)
						{
							int i = 0;
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							for(; i < laserCloudWidth - 1; i ++)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[(i + 1) * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[(i + 1) * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							}

							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;

							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}

					centerCubeI --;
					laserCloudCenWidth --;
				}




				while(centerCubeJ < 3)
				{
					for(int i = 0; i < laserCloudWidth; i ++)
					{
						for(int k = 0; k < laserCloudDepth; k ++)
						{
							int j = laserCloudHeight - 1;
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							for(; j >= 1; j --)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + (j - 1) * laserCloudDepth + k];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + (j - 1) * laserCloudDepth + k];  
							}

							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;
							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}
					centerCubeJ ++;
					laserCloudCenHeight ++;
				}

				while(centerCubeJ >= laserCloudHeight - 3)
				{
					for(int i = 0; i < laserCloudWidth; i ++)
					{
						for(int k = 0; k < laserCloudDepth; k ++)
						{
							int j = 0;
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							for(; j < laserCloudHeight - 1; j ++)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + (j + 1) * laserCloudDepth + k];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + (j + 1) * laserCloudDepth + k];
							}

							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;
							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}

					centerCubeJ --;
					laserCloudCenHeight --;
				}


				while(centerCubeK < 3)
				{
					for(int i = 0; i < laserCloudWidth; i ++)
					{
						for(int j = 0; j < laserCloudHeight; j ++)
						{
							int k = laserCloudDepth - 1;
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							for(; k >= 1; k --)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + (k - 1)];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + (k - 1)];
							}
							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;
							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}

					centerCubeK ++;
					laserCloudCenDepth ++;
				}


				while(centerCubeK >= laserCloudDepth - 3)
				{
					for(int i = 0; i < laserCloudWidth; i ++)
					{
						for(int j = 0; j < laserCloudHeight; j ++)
						{
							int k = 0;
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeCornerPointer = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];
							pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCubeSurfPointer = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k];

							for(; k < laserCloudDepth - 1; k ++)
							{
								laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + (k + 1)];
								laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + (k + 1)];
							}

							laserCloudCornerArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeCornerPointer;
							laserCloudSurfArray[i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k] = laserCloudCubeSurfPointer;
							laserCloudCubeCornerPointer->clear();
							laserCloudCubeSurfPointer->clear();
						}
					}

					centerCubeK --;
					laserCloudCenDepth --;
				}




				int laserCloudValidNum = 0;
				int laserCloudSurroundNum = 0;
				//前后5个cube中找,其中centerCube >= 3 && centerCube < laserCloudDepth - 3
				for(int i = centerCubeI - 2; i <= centerCubeI + 2; i ++)
				{
					for(int j = centerCubeJ - 2; j < centerCubeJ + 2; j ++)
					{
						for(int k = centerCubeK - 2; k < centerCubeK + 2; k ++)
						{
							if(i >= 0 && i < laserCloudWidth && j >= 0 && j < laserCloudHeight && k >= 0 && k < laserCloudDepth)
							{
								//网格中心点坐标
								float centerX = 50.0 * (i - laserCloudCenWidth);   
								float centerY = 50.0 * (j - laserCloudCenHeight);
								float centerZ = 50.0 * (k - laserCloudCenDepth);


								bool isInLaserFOV = false;
								
								for(int ii = -1; ii <= 1; ii +=2)
								{
									for(int jj = -1; jj <= 1; jj += 2)
									{
										for(int kk = -1; kk <= 1; kk += 2)
										{
											//计算边界
											float cornerX = centerX + 25.0 * ii;
											float cornerY = centerY + 25.0 * jj;
											float cornerZ = centerZ + 25.0 * kk;

											//transformTobeMapped距边缘点的距离
											float squaredSide1 = (transformTobeMapped[3] - cornerX)
															   * (transformTobeMapped[3] - cornerX)
															   + (transformTobeMapped[4] - cornerY)
															   * (transformTobeMapped[4] - cornerY)
															   + (transformTobeMapped[5] - cornerZ)
															   * (transformTobeMapped[5] - cornerZ);

											//pointOnYAxis与边缘点之间的距离
											float squaredSide2 = (pointOnYAxis.x - cornerX) * (pointOnYAxis.x - cornerX)
															   + (pointOnYAxis.y - cornerY) * (pointOnYAxis.y - cornerY)
															   + (pointOnYAxis.z - cornerZ) * (pointOnYAxis.z - cornerZ);

											float check1 = 100.0 + squaredSide1 - squaredSide2
														 - 10.0 * sqrt(3.0) * sqrt(squaredSide1);
											float check2 = 100.0 + squaredSide1 - squaredSide2
														 + 10.0 * sqrt(3.0) * sqrt(squaredSide1);

											if(check1 < 0 && check2 > 0)
											{
												isInLaserFOV = true;
											}
										}
									}
								}

								if(isInLaserFOV)
								{
									laserCloudValidInd[laserCloudValidNum] = i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k;
									laserCloudValidNum ++;
								}

								laserCloudSurroundInd[laserCloudSurroundNum] = i * laserCloudHeight * laserCloudDepth + j * laserCloudDepth + k;
								laserCloudSurroundNum ++;

							}
						}
					}
				}


				laserCloudCornerFromMap->clear();
				laserCloudSurfFromMap->clear();
			    for(int i = 0; i < laserCloudValidNum; i ++)
				{
					*laserCloudCornerFromMap += *laserCloudCornerArray[laserCloudValidInd[i]];
					*laserCloudSurfFromMap += *laserCloudSurfArray[laserCloudValidInd[i]];
				}
				int laserCloudCornerFromMapNum = laserCloudCornerFromMap->points.size();
				int laserCloudSurfFromMapNum = laserCloudSurfFromMap->points.size();



				int laserCloudCornerStackNum2 = laserCloudCornerStack2->points.size();
				for(int i = 0; i < laserCloudCornerStackNum2; i ++)
				{
					pointAssociateTobeMapped(&laserCloudCornerStack2->points[i], &laserCloudCornerStack2->points[i]);
				}


				int laserCloudSurfStackNum2 = laserCloudSurfStack2->points.size();
				for(int i = 0; i < laserCloudSurfStackNum2; i ++)
				{
					pointAssociateTobeMapped(&laserCloudSurfStack2->points[i], &laserCloudSurfStack2->points[i]);
				}

				
				//下采样
				laserCloudCornerStack->clear();
				downSizeFilterCorner.setInputCloud(laserCloudCornerStack2);
				downSizeFilterCorner.filter(*laserCloudCornerStack);
				int laserCloudCornerStackNum = laserCloudCornerStack->points.size();

				
				//下采样
				laserCloudSurfStack->clear();
				downSizeFilterSurf.setInputCloud(laserCloudSurfStack2);
				downSizeFilterSurf.filter(*laserCloudSurfStack);
				int laserCloudSurfStackNum = laserCloudSurfStack->points.size();


				laserCloudCornerStack2->clear();
				laserCloudSurfStack2->clear();


				if(laserCloudCornerFromMapNum > 10 && laserCloudSurfFromMapNum > 100)
				{
					//构建次边缘点和次平面点构建kd树
					kdtreeCornerFromMap->setInputCloud(laserCloudCornerFromMap);
					kdtreeSurfFromMap->setInputCloud(laserCloudSurfFromMap);
					
					//迭代10次
					for(int iterCount = 0; iterCount < 10; iterCount ++)
					{
						laserCloudOri->clear();
						coeffSel->clear();

						//次边缘点处理
						for(int i = 0; i < laserCloudCornerStackNum; i ++)
						{
							pointOri = laserCloudCornerStack->points[i];
							pointAssociateToMap(&pointOri, &pointSel);
							kdtreeCornerFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
							//找5个最近点
							if(pointSearchSqDis[4] < 1.0)
							{
								float cx = 0, cy = 0, cz = 0;
								for(int j = 0; j < 5; j ++)
								{
									cx += laserCloudCornerFromMap->points[pointSearchInd[j]].x;
									cy += laserCloudCornerFromMap->points[pointSearchInd[j]].y;
									cz += laserCloudCornerFromMap->points[pointSearchInd[j]].z;
								}

								//平均值
								cx /= 5;
								cy /= 5;
								cz /= 5;

								
								//计算方差
								float a11 = 0, a12 = 0, a13 = 0, a22 = 0, a23 = 0, a33 = 0;
								for(int j = 0; j < 5; j ++)
								{
									float ax = laserCloudCornerFromMap->points[pointSearchInd[j]].x - cx;
									float ay = laserCloudCornerFromMap->points[pointSearchInd[j]].y - cy;
									float az = laserCloudCornerFromMap->points[pointSearchInd[j]].z - cz;

									a11 += ax * ax;
									a23 += ax * ay;
									a13 += ax * az;
									a22 += ay * ay;
									a23 += ay * az;
									a33 += az * az;
								}

								a11 /= 5;
								a12 /= 5;
								a13 /= 5;
								a22 /= 5;
								a23 /= 5;
								a33 /= 5;

								matA1.at<float>(0, 0) = a11;
								matA1.at<float>(0, 1) = a12;
								matA1.at<float>(0, 2) = a13;
								matA1.at<float>(1, 0) = a12;
								matA1.at<float>(1, 1) = a22;
								matA1.at<float>(1, 2) = a23;
								matA1.at<float>(2, 0) = a13;
								matA1.at<float>(2, 1) = a23;
								matA1.at<float>(2, 2) = a33;




								cv::eigen(matA1, matD1, matV1);     //特征值分解
								if(matD1.at<float>(0, 0) > 3 * matD1.at<float>(0, 1))
								{
									float x0 = pointSel.x;
									float y0 = pointSel.y;
									float z0 = pointSel.z;

									float x1 = cx + 0.1 * matV1.at<float>(0, 0);    //第一列为5个点的方向向量
									float y1 = cy + 0.1 * matV1.at<float>(0, 1);	
									float z1 = cz + 0.1 * matV1.at<float>(0, 2);

									float x2 = cx - 0.1 * matV1.at<float>(0, 0);    //在中心点附近沿着方向向量前后取2个点
									float y2 = cy - 0.1 * matV1.at<float>(0, 1);
									float z2 = cz - 0.1 * matV1.at<float>(0, 2);


									float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
											   * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
											   + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
											   * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
											   + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
											   * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
												 
									float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(     z1 - z2));
												   
									float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
											 + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012      / l12;
												
									float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
											 - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012      / l12;
														 
									float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
											 + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012      / l12;
														
									float ld2 = a012 / l12;   //点线距
								    
									pointProj = pointSel;
									pointProj.x -= la * ld2;
									pointProj.y -= lb * ld2;
									pointProj.z -= lc * ld2;

									float s = 1 - 0.9 * fabs(ld2);

									coeff.x = s * la;
									coeff.y = s * lb;
									coeff.z = s * lc;
									coeff.intensity = s * ld2;

									if(s > 0.1)
									{
										laserCloudOri->push_back(pointOri);
										coeffSel->push_back(coeff);
									}

								}
							}
						}

						//次平面点处理
						for(int i = 0; i < laserCloudSurfStackNum; i ++)
						{
							pointOri = laserCloudSurfStack->points[i];
							pointAssociateToMap(&pointOri, &pointSel);
							kdtreeSurfFromMap->nearestKSearch(pointSel, 5, pointSearchInd, pointSearchSqDis);
							//选取最近5个点
							if(pointSearchSqDis[4] < 1.0)
							{
								for(int j = 0; j < 5; j ++)
								{
									matA0.at<float>(j, 0) = laserCloudSurfFromMap->points[pointSearchInd[j]].x;
									matA0.at<float>(j, 1) = laserCloudSurfFromMap->points[pointSearchInd[j]].y;
									matA0.at<float>(j, 2) = laserCloudSurfFromMap->points[pointSearchInd[j]].z;
								}

								cv::solve(matA0, matB0, matX0, cv::DECOMP_QR);

								float pa = matX0.at<float>(0, 0);
								float pb = matX0.at<float>(1, 0);
								float pc = matX0.at<float>(2, 0);
								float pd = 1;

								float ps = sqrt(pa * pa + pb * pb + pc * pc);
								pa /= ps;
								pb /= ps;
								pc /= ps;
								pd /= ps;

								//判断该点在不在面上
								bool planeValid = true;
								for(int j = 0; j < 5; j ++)
								{
									if(fabs(pa * laserCloudSurfFromMap->points[pointSearchInd[j]].x + pb * laserCloudSurfFromMap->points[pointSearchInd[j]].y + pc * laserCloudSurfFromMap->points[pointSearchInd[j]].z + pd) > 0.2)
									{
										planeValid = false;
										break;
									}
								}


								if(planeValid)
								{
									float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

									pointProj = pointSel;
									pointProj.x -= pa * pd2;
									pointProj.y -= pb * pd2;
									pointProj.z -= pc * pd2;

									float s = 1 - 0.9 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y * pointSel.y + pointSel.z + pointSel.z));


									coeff.x = s * pa;
									coeff.y = s * pb;
									coeff.z - s * pc;
									coeff.intensity = s * pd2;

									if(s > 0.1)
									{
										laserCloudOri->push_back(pointOri);
										coeffSel->push_back(coeff);
									}
								}
							}
						}


						float srx = sin(transformTobeMapped[0]);
						float crx = cos(transformTobeMapped[0]);
						float sry = sin(transformTobeMapped[1]);
						float cry = cos(transformTobeMapped[1]);
						float srz = sin(transformTobeMapped[2]);
						float crz = cos(transformTobeMapped[2]);


						int laserCloudSelNum = laserCloudOri->points.size();
						if(laserCloudSelNum < 50)
						{
							continue;
						}



						cv::Mat matA(laserCloudSelNum, 0, CV_32F, cv::Scalar::all(0));
						cv::Mat matAt(6, laserCloudSelNum, CV_32F, cv::Scalar::all(0));
						cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
						cv::Mat matB(laserCloudSelNum, 1, CV_32F, cv::Scalar::all(0));
						cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
						cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));


						for(int i = 0; i < laserCloudSelNum; i ++)
						{
							pointOri = laserCloudOri->points[i];
							coeff = coeffSel->points[i];

							float arx = (crx*sry*srz*pointOri.x + crx*crz*sry*pointOri.y - srx*sry*pointOri.z) * coeff.x
									  + (-srx*srz*pointOri.x - crz*srx*pointOri.y - crx*pointOri.z) * coeff.y
									  + (crx*cry*srz*pointOri.x + crx*cry*crz*pointOri.y - cry*srx*pointOri.z) * coeff.z;
							   
							float ary = ((cry*srx*srz - crz*sry)*pointOri.x
									  + (sry*srz + cry*crz*srx)*pointOri.y + crx*cry*pointOri.z) * coeff.x
									  + ((-cry*crz - srx*sry*srz)*pointOri.x
									  + (cry*srz - crz*srx*sry)*pointOri.y - crx*sry*pointOri.z) * coeff.z;
								    
							float arz = ((crz*srx*sry - cry*srz)*pointOri.x + (-cry*crz-srx*sry*srz)*pointOri.y)*coeff.x
									  + (crx*crz*pointOri.x - crx*srz*pointOri.y) * coeff.y
									  + ((sry*srz + cry*crz*srx)*pointOri.x + (crz*sry-cry*srx*srz)*pointOri.y)*coeff.z;

						
							matA.at<float>(i, 0) = arx;
							matA.at<float>(i, 1) = ary;
							matA.at<float>(i, 2) = arz;
							matA.at<float>(i, 3) = coeff.x;
							matA.at<float>(i, 4) = coeff.y;
							matA.at<float>(i, 5) = coeff.z;
							matB.at<float>(i, 0) = -coeff.intensity; 
						}

						cv::transpose(matA, matAt);
						matAtA = matAt * matA;
						matAtB = matAt * matB;

						cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);

						if(iterCount == 0)
						{
							cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
							cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
							cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

							cv::eigen(matAtA, matE, matV);
							matV.copyTo(matV2);

							isDegenerate = false;
							float eigenThre[6] = {100, 100, 100, 100, 100, 100};
							for(int i = 5; i >= 0; i --)
							{
								if(matE.at<float>(0, i) < eigenThre[i])
								{
									for(int j = 0; j < 6; j ++)
									{
										matV2.at<float>(i, j) = 0;
									}
									isDegenerate = true;
								}
								else
								{
									break;
								}
							}
							matP = matV.inv() * matV2;
						}



						if(isDegenerate)
						{
							cv::Mat matX2(6, 1, CV_32F, cv::Scalar::all(0));
							matX.copyTo(matX2);
							matX = matP * matX2;
						}



						transformTobeMapped[0] += matX.at<float>(0, 0);
						transformTobeMapped[1] += matX.at<float>(1, 0);
						transformTobeMapped[2] += matX.at<float>(2, 0);
						transformTobeMapped[3] += matX.at<float>(3, 0);
						transformTobeMapped[4] += matX.at<float>(4, 0);
						transformTobeMapped[5] += matX.at<float>(5, 0);

						float deltaR = sqrt(pow(rad2deg(matX.at<float>(0, 0)), 2) +
											pow(rad2deg(matX.at<float>(1, 0)), 2) +
											pow(rad2deg(matX.at<float>(2, 0)), 2));

						float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2) +
											pow(matX.at<float>(4, 0) * 100, 2) +
											pow(matX.at<float>(5, 0) * 100, 2));


						if(deltaR < 0.05 && deltaT < 0.05)
						{
							break;
						}

					}
					
					transformUpdate();
				}


				for(int i = 0; i < laserCloudCornerStackNum; i ++)
				{
					pointAssociateToMap(&laserCloudCornerStack->points[i], &pointSel);

					//计算该点所处的网格下标
					int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
					int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
					int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudDepth;

					if(pointSel.x + 25.0 < 0)  cubeI --;
					if(pointSel.y + 25.0 < 0)  cubeJ --;
					if(pointSel.z + 25.0 < 0)  cubeK --;


					if(cubeI >= 0 && cubeI < laserCloudWidth &&
					   cubeJ >= 0 && cubeJ < laserCloudHeight &&
					   cubeK >= 0 && cubeK < laserCloudDepth)
					{
						int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
						laserCloudCornerArray[cubeInd]->push_back(pointSel);
					}
				}


				for(int i = 0; i < laserCloudSurfStackNum; i ++)
				{
					pointAssociateToMap(&laserCloudSurfStack->points[i], &pointSel);

					int cubeI = int((pointSel.x + 25.0) / 50.0) + laserCloudCenWidth;
					int cubeJ = int((pointSel.y + 25.0) / 50.0) + laserCloudCenHeight;
					int cubeK = int((pointSel.z + 25.0) / 50.0) + laserCloudCenDepth;

					if(pointSel.x + 25.0 < 0)   cubeI --;
					if(pointSel.y + 25.0 < 0)   cubeJ --;
					if(pointSel.z + 25.0 < 0)   cubeK --;

					if(cubeI >= 0 && cubeI < laserCloudWidth &&
					   cubeJ >= 0 && cubeJ < laserCloudHeight &&
					   cubeK >= 0 && cubeK < laserCloudDepth)
					{
						int cubeInd = cubeI + laserCloudWidth * cubeJ + laserCloudWidth * laserCloudHeight * cubeK;
						laserCloudSurfArray[cubeInd]->push_back(pointSel);
					}
				}


				for(int i = 0; i < laserCloudValidNum; i ++)
				{
					int ind = laserCloudValidInd[i];

					laserCloudCornerArray2[ind]->clear();
					downSizeFilterCorner.setInputCloud(laserCloudCornerArray[ind]);
					downSizeFilterCorner.filter(*laserCloudCornerArray2[ind]);

					laserCloudSurfArray2[ind]->clear();
					downSizeFilterSurf.setInputCloud(laserCloudSurfArray[ind]);
					downSizeFilterSurf.filter(*laserCloudSurfArray2[ind]);

					pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = laserCloudCornerArray[ind];
					laserCloudCornerArray[ind] = laserCloudCornerArray2[ind];
					laserCloudCornerArray2[ind] = laserCloudTemp;


					laserCloudTemp = laserCloudSurfArray[ind];
					laserCloudSurfArray[ind] = laserCloudSurfArray2[ind];
					laserCloudSurfArray2[ind] = laserCloudTemp;
				}

				mapFrameCount ++;
				if(mapFrameCount >= mapFrameNum)
				{
					mapFrameCount = 0;

					laserCloudSurround2->clear();
					for(int i = 0; i < laserCloudSurroundNum; i ++)
					{
						int ind = laserCloudSurroundInd[i];
						*laserCloudSurround2 += *laserCloudCornerArray[ind];
						*laserCloudSurround2 += *laserCloudSurfArray[ind];
					}

					laserCloudSurround->clear();
					downSizeFilterCorner.setInputCloud(laserCloudSurround2);
					downSizeFilterCorner.filter(*laserCloudSurround);


					sensor_msgs::PointCloud2 laserCloudSurround3;
					pcl::toROSMsg(*laserCloudSurround, laserCloudSurround3);
					laserCloudSurround3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
					laserCloudSurround3.header.frame_id = "/camera_init";
					pubLaserCloudSurround.publish(laserCloudSurround3);

				}

				int laserCloudFullResNum = laserCloudFullRes->points.size();
				for(int i = 0; i < laserCloudFullResNum; i ++)
				{
					pointAssociateToMap(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
				}


				sensor_msgs::PointCloud2 laserCloudFullRes3;
				pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
				laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				laserCloudFullRes3.header.frame_id = "/camera_init";
				pubLaserCloudFullRes.publish(laserCloudFullRes3);

				std::cout << transformTobeMapped[0] << '\t' << transformTobeMapped[1] << '\t' << transformTobeMapped[2] << '\t' << transformTobeMapped[3] << '\t' << transformTobeMapped[4] << '\t' << transformTobeMapped[5] << std::endl;
				//std::cout << transformAftMapped[0] << '\t' << transformAftMapped[1] << '\t' << transformAftMapped[2] << '\t' << transformAftMapped[3] << '\t' << transformAftMapped[4] << '\t' << transformAftMapped[5] << std::endl;
				/*
				geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYa(transformAftMapped[2], -transformAftMapped[0], -transformAftMapped[1]);
				odometryAftMapped.header.stamp = ros::Time().fromSec(timeLaserOdometry);
				odometryAftMapped.pose.pose.orientation.x = -geoQuat.y;
				odometryAftMapped.pose.pose.orientation.y = -geoQuat.z;
				odometryAftMapped.pose.pose.orientation.z = geoQuat.x;
				odometryAftMapped.pose.pose.orientation.w = geoQuat.w;
				odometryAftMapped.pose.pose.position.x = transformAftMapped[3];
				odometryAftMapped.pose.pose.position.y = transformAftMapped[4];
				odometryAftMapped.pose.pose.position.z = transformAftMapped[5];
				pubOdomAftMapped.publish(odomAftMapped);


				aftMappedTrans.stamp_ = ros::Time().fromSec(timeLaserOdometry);
				aftMappedTrans.setRotation(tf::Quaternion(-geoQuat.y, -geoQuat.z, geoQuat.xm geoQuat.w));
				aftMappedTrans.setOrigin(tf::Vector3(transformAftMapped[3], transformAftMapped[4], transformAftMapped[5]));
				tfBroadcaster.sendTransform(aftMappedTrans);
				*/
			}
			

		}
		

		status = ros::ok();
		rate.sleep();
	}

	return 0;
}

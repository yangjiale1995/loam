#include <cmath>
#include <nav_msgs/Odometry.h>
#include <opencv/cv.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl_conversions/pcl_conversions.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

#include "common.h"

const float scanPeriod = 0.1;	//velodyne激光点云周期

const int skipFrameNum = 1;    //跳帧频率
bool systemInited = false;     //laserOdometry初始化

double timeCornerPointsSharp = 0;		//边缘点时间戳
double timeCornerPointsLessSharp = 0;   //次边缘点时间戳
double timeSurfPointsFlat = 0;          //平面点时间戳
double timeSurfPointsLessFlat = 0;      //次平面点时间戳
double timeLaserCloudFullRes = 0;       //点云时间戳

bool newCornerPointsSharp = false;        //边缘点消息更新标志
bool newCornerPointsLessSharp = false;    //次边缘点消息更新标志
bool newSurfPointsFlat = false;           //平面点消息更新标志
bool newSurfPointsLessFlat = false;       //次平面点消息更新标志
bool newLaserCloudFullRes = false;        //点云消息更新标志

//边缘点和次边缘点
pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsSharp(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr cornerPointsLessSharp(new pcl::PointCloud<pcl::PointXYZI>);

//平面点和次平面点
pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsFlat(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr surfPointsLessFlat(new pcl::PointCloud<pcl::PointXYZI>);

//保存上一帧的次边缘点和次平面点
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudCornerLast(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudSurfLast(new pcl::PointCloud<pcl::PointXYZI>);

//保存匹配点和距离信息
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudOri(new pcl::PointCloud<pcl::PointXYZI>);
pcl::PointCloud<pcl::PointXYZI>::Ptr coeffSel(new pcl::PointCloud<pcl::PointXYZI>);

//点云
pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudFullRes(new pcl::PointCloud<pcl::PointXYZI>);

//边缘点kd树，次边缘点kd树
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeCornerLast(new pcl::KdTreeFLANN<pcl::PointXYZI>);
pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr kdtreeSurfLast(new pcl::KdTreeFLANN<pcl::PointXYZI>);

//保存上一帧次边缘点和次平面点个数
int laserCloudCornerLastNum;
int laserCloudSurfLastNum;


float transform[6] = {0};      //前三个表示旋转，后三个表示平移
float transformSum[6] = {0};   //???

//边缘点和对应的边缘线两个点
int pointSelCornerInd[40000];
float pointSearchCornerInd1[40000];
float pointSearchCornerInd2[40000];

//平面点和对应的三个平面三个点
int pointSelSurfInd[40000];
float pointSearchSurfInd1[40000];
float pointSearchSurfInd2[40000];
float pointSearchSurfInd3[40000];


//将所有点云转换到第一个点的坐标系下
void TransformToStart(pcl::PointXYZI const * const pi, pcl::PointXYZI * const po)
{
	float s = 10 * (pi->intensity - int(pi->intensity));
	float rx = s * transform[0];
	float ry = s * transform[1];
	float rz = s * transform[2];
	float tx = s * transform[3];
	float ty = s * transform[4];
	float tz = s * transform[5];

	//先平移，然后绕z轴旋转
	float x1 = cos(rz) * (pi->x - tx) - sin(rz) * (pi->y - ty);
	float y1 = sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
	float z1 = (pi->z - tz);

	//绕y轴旋转
	float x2 = cos(ry) * x1 - sin(ry) * z1;
	float y2 = y1;
	float z2 = sin(ry) * x1 + cos(ry) * z1;

	//绕x轴旋转
	po->x = x2;
	po->y = cos(rx) * y2 - sin(rx) * z2;
	po->z = sin(rx) * y2 + cos(rx) * z2;
	po->intensity = pi->intensity;

}


//将所有点云转换到最后一个点的坐标系下
void TransformToEnd(pcl::PointXYZI const * const pi, pcl::PointXYZI * const po)
{
	float s = 10 * (pi->intensity - int(pi->intensity));

	float rx = s * transform[0];
	float ry = s * transform[1];
	float rz = s * transform[2];
	float tx = s * transform[3];
	float ty = s * transform[4];
	float tz = s * transform[5];

	//先平移再绕z轴旋转
	float x1 = cos(rz) * (pi->x - tx) - sin(rz) * (pi->y - ty);
	float y1 = sin(rz) * (pi->x - tx) + cos(rz) * (pi->y - ty);
	float z1 = (pi->z - tz);

	//绕y轴旋转
	float x2 = cos(ry) * x1 - sin(ry) * z1;
	float y2 = y1;
	float z2 = sin(ry) * x1 + cos(ry) * z1;

	//绕x轴旋转
	float x3 = x2;
	float y3 = cos(rx) * y2 - sin(rx) * z2;
	float z3 = sin(rx) * y3 + cos(rx) * z2;

	rx = transform[0];
	ry = transform[1]; 
	rz = transform[2];
	tx = transform[3];
	ty = transform[4];
	tz = transform[5];


	//绕x轴旋转
	float x4 = x3;
	float y4 = cos(rx) * y3 - sin(rx) * z3;
	float z4 = sin(rx) * y3 + cos(rx) * z3;

	float x5 = cos(ry) * x4 - sin(ry) * z4;
	float y5 = y4;
	float z5 = sin(ry) * x4 + cos(ry) * z4;

	float x6 = cos(rz) * x5 - sin(rz) * y5;
	float y6 = sin(rz) * x5 + cos(rz) * y5;
	float z6 = z5;

	po->x = x6 + tx;
	po->y = y6 + ty;
	po->z = z6 + tz;
	po->intensity = int(pi->intensity);

}



//相对于第一个点云即原点，积累旋转量
void AccumulateRotation(float cx, float cy, float cz, float lx, float ly, float lz,
						float &ox, float &oy, float &oz)
{
	float srx = cos(lx) * cos(cx) * sin(ly) * sin(cz) - cos(cx) * cos(cz) * sin(lx) - cos(lx) * cos(ly) * sin(cx);
	ox = -asin(srx);
		  
	float srycrx = sin(lx) * (cos(cy) * sin(cz) - cos(cz) * sin(cx) * sin(cy)) + cos(lx) * sin(ly) * (cos(cy) * cos(cz) + sin(cx) * sin(cy) * sin(cz)) + cos(lx) * cos(ly) * cos(cx) * sin(cy);
	float crycrx = cos(lx) * cos(ly) * cos(cx) * cos(cy) - cos(lx) * sin(ly) * (cos(cz) * sin(cy) - cos(cy) * sin(cx) * sin(cz)) - sin(lx) * (sin(cy) * sin(cz) + cos(cy) * cos(cz) * sin(cx));
	oy = atan2(srycrx / cos(ox), crycrx / cos(ox));
			    
	float srzcrx = sin(cx) * (cos(lz) * sin(ly) - cos(ly) * sin(lx) * sin(lz)) + cos(cx) * sin(cz) * (cos(ly) * cos(lz) + sin(lx) * sin(ly) * sin(lz)) + cos(lx) * cos(cx) * cos(cz) * sin(lz);
	float crzcrx = cos(lx) * cos(lz) * cos(cx) * cos(cz) - cos(cx) * sin(cz) * (cos(ly) * sin(lz) - cos(lz) * sin(lx) * sin(ly)) - sin(cx) * (sin(ly) * sin(lz) + cos(ly) * cos(lz) * sin(lx));
	oz = atan2(srzcrx / cos(ox), crzcrx / cos(ox));
}



//边缘点
void laserCloudSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsSharpMsg)
{
	timeCornerPointsSharp = cornerPointsSharpMsg->header.stamp.toSec();
	cornerPointsSharp->clear();
	pcl::fromROSMsg(*cornerPointsSharpMsg, *cornerPointsSharp);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cornerPointsSharp, *cornerPointsSharp, indices);
	newCornerPointsSharp = true;
	
}


//次边缘点
void laserCloudLessSharpHandler(const sensor_msgs::PointCloud2ConstPtr &cornerPointsLessSharpMsg)
{
	timeCornerPointsLessSharp = cornerPointsLessSharpMsg->header.stamp.toSec();
	cornerPointsLessSharp->clear();
	pcl::fromROSMsg(*cornerPointsLessSharpMsg, *cornerPointsLessSharp);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*cornerPointsLessSharp, *cornerPointsLessSharp, indices);
	newCornerPointsLessSharp = true;

}


//平面点
void laserCloudFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsFlatMsg)
{
	timeSurfPointsFlat = surfPointsFlatMsg->header.stamp.toSec();
	surfPointsFlat->clear();
	pcl::fromROSMsg(*surfPointsFlatMsg, *surfPointsFlat);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*surfPointsFlat, *surfPointsFlat, indices);
	newSurfPointsFlat = true;

}


//次平面点
void laserCloudLessFlatHandler(const sensor_msgs::PointCloud2ConstPtr &surfPointsLessFlatMsg)
{
	timeSurfPointsLessFlat = surfPointsLessFlatMsg->header.stamp.toSec();
	surfPointsLessFlat->clear();
	pcl::fromROSMsg(*surfPointsLessFlatMsg, *surfPointsLessFlat);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*surfPointsLessFlat, *surfPointsLessFlat, indices);
	newSurfPointsLessFlat = true;

}



//完整点云
void laserCloudFullResHandler(const sensor_msgs::PointCloud2ConstPtr &laserCloudFullResMsg)
{
	timeLaserCloudFullRes = laserCloudFullResMsg->header.stamp.toSec();
	laserCloudFullRes->clear();
	pcl::fromROSMsg(*laserCloudFullResMsg, *laserCloudFullRes);
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(*laserCloudFullRes, *laserCloudFullRes, indices);
	newLaserCloudFullRes = true;

}




int main(int argc,char **argv)
{
	ros::init(argc,argv,"laserOdometry");
	ros::NodeHandle nh;

	ros::Subscriber subCornerPointsSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_sharp",2,laserCloudSharpHandler);    //订阅边缘点消息
	ros::Subscriber subCornerPointsLessSharp = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp",2,laserCloudLessSharpHandler);     //订阅次边缘点消息
	ros::Subscriber subSurfPointsFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_flat",2,laserCloudFlatHandler);           //订阅平面点消息
	ros::Subscriber subSurfPointsLessFlat = nh.subscribe<sensor_msgs::PointCloud2>("/laser_cloud_less_flat",2,laserCloudLessFlatHandler);     //订阅次平面点消息
	ros::Subscriber subLaserCloudFullRes = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_cloud_2",2,laserCloudFullResHandler);       //订阅点云消息
	ros::Publisher pubLaserCloudCornerLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_corner_last",2);            //发布次边缘点消息
	ros::Publisher pubLaserCloudSurfLast = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_surf_last",2);                //发布次平面点消息
	ros::Publisher pubLaserCloudFullRes = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_3",2);                      //发布点云消息
	ros::Publisher pubLaserOdometry = nh.advertise<nav_msgs::Odometry>("/laser_odom_to_init",5);                              //发布odometry消息

	
	nav_msgs::Odometry laserOdometry;
	laserOdometry.header.frame_id = "/camera_init";
	laserOdometry.child_frame_id = "/laser_odom";

	tf::TransformBroadcaster tfBroadcaster;
	tf::StampedTransform laserOdometryTrans;
	laserOdometryTrans.frame_id_ = "/camera_init";
	laserOdometryTrans.child_frame_id_ = "/laser_odom";



	std::vector<int> pointSearchInd;
	std::vector<float> pointSearchSqDis;

	pcl::PointXYZI pointOri, pointSel, tripod1, tripod2, tripod3, pointProj, coeff;

	bool isDegenerate = false;

	cv::Mat matP(6, 6, CV_32F, cv::Scalar::all(0));

	int frameCount = skipFrameNum;   //初始化frameCount = skipFrameNum = 1
	ros::Rate rate(100);
	bool status = ros::ok();
	while(status)
	{
		ros::spinOnce();
		
		if(newCornerPointsSharp && newCornerPointsLessSharp && newSurfPointsFlat &&
			 newSurfPointsLessFlat && newLaserCloudFullRes &&
		     fabs(timeCornerPointsSharp - timeSurfPointsLessFlat) < 0.005 &&
		     fabs(timeCornerPointsLessSharp - timeSurfPointsLessFlat) < 0.005 &&
		     fabs(timeSurfPointsFlat - timeSurfPointsLessFlat) < 0.005 &&
			 fabs(timeLaserCloudFullRes - timeSurfPointsLessFlat) < 0.005) 
		{  //同步作用，确保同时收到同一个点云的特征点以及IMU信息才进入
			 newCornerPointsSharp = false;
			 newCornerPointsLessSharp = false;
			 newSurfPointsFlat = false;
			 newSurfPointsLessFlat = false;
			 newLaserCloudFullRes = false;
			

			 if(!systemInited)     //第一帧消息用来做初始化
			 {

				 //laserCloudCornerLast = cornerPointsLessSharp赋值操作
				 pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = cornerPointsLessSharp;
				 cornerPointsLessSharp = laserCloudCornerLast;
				 laserCloudCornerLast = laserCloudTemp;

				 //laserCloudSurfLast = surfPointsLessFlat赋值操作
				 laserCloudTemp = surfPointsLessFlat;
				 surfPointsLessFlat = laserCloudSurfLast;
				 laserCloudSurfLast = laserCloudTemp;

				 //构造次边缘点和次平面点的kd树
				 kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
				 kdtreeSurfLast->setInputCloud(laserCloudSurfLast);

				 laserCloudCornerLastNum = laserCloudCornerLast->points.size();
				 laserCloudSurfLastNum = laserCloudSurfLast->points.size();

				 //将次边缘点消息发布出去
				 sensor_msgs::PointCloud2 laserCloudCornerLast2;
				 pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
				 laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
				 laserCloudCornerLast2.header.frame_id = "/camera";
				 pubLaserCloudCornerLast.publish(laserCloudCornerLast2);


				 //将次平面点消息发布出去
				 sensor_msgs::PointCloud2 laserCloudSurfLast2;
				 pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
				 laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
				 laserCloudSurfLast2.header.frame_id = "/camera";
				 pubLaserCloudSurfLast.publish(laserCloudSurfLast2);


				 systemInited = true;
				 continue;
			 }

			 //上一帧中次边缘点和次平面点的个数足够多
			 if(laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100)
			 {

				int cornerPointsSharpNum = cornerPointsSharp->points.size();   //边缘点个数
				int surfPointsFlatNum = surfPointsFlat->points.size();		   //平面点个数

				//迭代次数
				for(int iterCount = 0; iterCount < 25; iterCount ++)
				{
					laserCloudOri->clear();     //保存匹配好的点云
					coeffSel->clear();          //保存点线距或者点面距

					//边缘点循环
					for(int i = 0; i < cornerPointsSharpNum; i ++)	//边缘点个数循环
					{ 
						TransformToStart(&cornerPointsSharp->points[i], &pointSel);  //将每个边缘点转换到第一个点的坐标系下


						if(iterCount % 5 == 0)
						{
						kdtreeCornerLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);		//kd树查找最近点
						int closestPointInd = -1, minPointInd2 = -1;    //保存构成边缘线的两个点下标

						if(pointSearchSqDis[0] < 25)  //pointSearchSqDis保存距离信息,从近到远排序
						{
							closestPointInd = pointSearchInd[0];   //最近的点下标

							int closestPointScan = int(laserCloudCornerLast->points[closestPointInd].intensity);		//最近的点对应的线号

							float pointSqDis, minPointSqDis2 = 25;

							//寻找距离目标点最近距离的点
							for(int j = closestPointInd + 1; j < laserCloudCornerLastNum; j ++)
							{
								if(int(laserCloudCornerLast->points[j].intensity)> closestPointScan + 2.5)    //非相邻线
								{
									break;
								}

								pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
											 (laserCloudCornerLast->points[j].x - pointSel.x) +
											 (laserCloudCornerLast->points[j].y - pointSel.y) *
											 (laserCloudCornerLast->points[j].y - pointSel.y) +
											 (laserCloudCornerLast->points[j].z - pointSel.z) * 
											 (laserCloudCornerLast->points[j].z - pointSel.z);    //距离

								if(int(laserCloudCornerLast->points[j].intensity) > closestPointScan)    //相邻线
								{
									if(pointSqDis < minPointSqDis2)
									{
										minPointSqDis2 = pointSqDis;  //更新最小距离
										minPointInd2 = j;             //更新下标
									}
								}
							}
							for(int j = closestPointInd - 1; j >= 0; j --)
							{
								if(int(laserCloudCornerLast->points[j].intensity) < closestPointScan - 2.5)    //非相邻线
								{
									break;
								}

								pointSqDis = (laserCloudCornerLast->points[j].x - pointSel.x) * 
											 (laserCloudCornerLast->points[j].x - pointSel.x) +
											 (laserCloudCornerLast->points[j].y - pointSel.y) *
											 (laserCloudCornerLast->points[j].y - pointSel.y) +
											 (laserCloudCornerLast->points[j].z - pointSel.z) * 
											 (laserCloudCornerLast->points[j].z - pointSel.z);         //距离
								
								if(int(laserCloudCornerLast->points[j].intensity) < closestPointScan)    //相邻线
								{
									if(pointSqDis < minPointSqDis2)
									{
										minPointSqDis2 = pointSqDis;   //更新最小距离
										minPointInd2 = j;              //更新下标
									}
								}
							}
						}
						pointSearchCornerInd1[i] = closestPointInd;
						pointSearchCornerInd2[i] = minPointInd2;
						}


						if(pointSearchCornerInd2[i] >= 0)
						{
							//tripod1 = laserCloudCornerLast->points[closestPointInd];
							//tripod2 = laserCloudCornerLast->points[minPointInd2];

							tripod1 = laserCloudCornerLast->points[pointSearchCornerInd1[i]];
							tripod2 = laserCloudCornerLast->points[pointSearchCornerInd2[i]];

							float x0 = pointSel.x;
							float y0 = pointSel.y;
							float z0 = pointSel.z;
							float x1 = tripod1.x;
							float y1 = tripod1.y;
							float z1 = tripod1.z;
							float x2 = tripod2.x;
							float y2 = tripod2.y;
							float z2 = tripod2.z;

							/*
							点线距公式：d = |(x0 - x1)×(x0 - x2)|
											-------------------
												|x1 - x2|
							a012 = |(x0 - x1)×(x0 - x2)|    分子
							l12 = |x1 - x2|    分母
							ld2 = a012 / l12  点线距
							la = 
							lb =
							lc = 
							*/
							float a012 = sqrt(((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
									   * ((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
									   + ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
									   * ((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
									   + ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))
									   * ((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1)));
  
							//kd-tree和真实最近距离点两点之间的距离
							float l12 = sqrt((x1 - x2)*(x1 - x2) + (y1 - y2)*(y1 - y2) + (z1 - z2)*(z1 - z2));
	 
							
							float la = ((y1 - y2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
									 + (z1 - z2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))) / a012 / l12;     
		 
							float lb = -((x1 - x2)*((x0 - x1)*(y0 - y2) - (x0 - x2)*(y0 - y1))
									 - (z1 - z2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
		    
							float lc = -((x1 - x2)*((x0 - x1)*(z0 - z2) - (x0 - x2)*(z0 - z1))
			                         + (y1 - y2)*((y0 - y1)*(z0 - z2) - (y0 - y2)*(z0 - z1))) / a012 / l12;
			                
							//点到线的距离
							float ld2 = a012 / l12;
		
							float s = 1;
							if(iterCount >= 5)
							{
								s = 1 - 1.8 * fabs(ld2);
							}

							coeff.x = s * la;
							coeff.y = s * lb;
							coeff.z = s * lc;
							coeff.intensity = s * ld2;

							if(s > 0.1 && ld2 != 0)
							{
								laserCloudOri->push_back(cornerPointsSharp->points[i]);
								coeffSel->push_back(coeff);
							}
						}
					}

				

					//平面点同样处理
					for(int i = 0; i < surfPointsFlatNum; i ++)
					{
						TransformToStart(&surfPointsFlat->points[i], &pointSel);

						if(iterCount % 5 == 0)
						{

						kdtreeSurfLast->nearestKSearch(pointSel, 1, pointSearchInd, pointSearchSqDis);    //kd树查找最近点
						int closestPointInd = -1, minPointInd2 = -1, minPointInd3 = -1;
						if(pointSearchSqDis[0] < 25)
						{
							closestPointInd = pointSearchInd[0];
							int closestPointScan = int(laserCloudSurfLast->points[closestPointInd].intensity);   //最近点线号
							
							float pointSqDis, minPointSqDis2 = 25, minPointSqDis3 = 25;
							for(int j = closestPointInd + 1; j < laserCloudSurfLastNum; j ++)
							{
								if(int(laserCloudSurfLast->points[j].intensity) > closestPointScan + 2.5)    //非相邻线
								{
									break;
								}

								pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
											 (laserCloudSurfLast->points[j].x - pointSel.x) +
										     (laserCloudSurfLast->points[j].y - pointSel.y) *
											 (laserCloudSurfLast->points[j].y - pointSel.y) +
											 (laserCloudSurfLast->points[j].z - pointSel.z) *
											 (laserCloudSurfLast->points[j].z - pointSel.z);
								
	
								if(int(laserCloudSurfLast->points[j].intensity) <= closestPointScan)
								{
									if(pointSqDis < minPointSqDis2)
									{
										minPointSqDis2 = pointSqDis;
										minPointInd2 = j;
									}
								}
								else
								{
									if(pointSqDis < minPointSqDis3)
									{
										minPointSqDis3 = pointSqDis;
										minPointInd3 = j;
									}
								}

							}
	
							for(int j = closestPointInd - 1; j >= 0; j --)
							{
								if(int(laserCloudSurfLast->points[j].intensity) < closestPointScan - 2.5)
								{
									break;
								}

								pointSqDis = (laserCloudSurfLast->points[j].x - pointSel.x) *
											 (laserCloudSurfLast->points[j].x - pointSel.x) +
											 (laserCloudSurfLast->points[j].y - pointSel.y) *
											 (laserCloudSurfLast->points[j].y - pointSel.y) +
											 (laserCloudSurfLast->points[j].z - pointSel.z) *
											 (laserCloudSurfLast->points[j].z - pointSel.z);


								if(int(laserCloudSurfLast->points[j].intensity) >= closestPointScan)
								{
									if(pointSqDis < minPointSqDis2)
									{
										minPointSqDis2 = pointSqDis;
										minPointInd2 = j;
									}
								}
								else
								{
									if(pointSqDis < minPointSqDis3)
									{
										minPointSqDis3 = pointSqDis;
										minPointInd3 = j;
									}
								}
							}

						}

						pointSearchSurfInd1[i] = closestPointInd;
						pointSearchSurfInd2[i] = minPointInd2;
						pointSearchSurfInd3[i] = minPointInd3;
						}


						if(pointSearchSurfInd2[i] >= 0 && pointSearchSurfInd3[i] >= 0)
						{
							//tripod1 = laserCloudSurfLast->points[closestPointInd];
							//tripod2 = laserCloudSurfLast->points[minPointInd2];
							//tripod3 = laserCloudSurfLast->points[minPointInd3];

							tripod1 = laserCloudSurfLast->points[pointSearchSurfInd1[i]];
							tripod2 = laserCloudSurfLast->points[pointSearchSurfInd2[i]];
							tripod3 = laserCloudSurfLast->points[pointSearchSurfInd3[i]];

							float pa = (tripod2.y - tripod1.y) * (tripod3.z - tripod1.z)
									 - (tripod3.y - tripod1.y) * (tripod2.z - tripod1.z);
							float pb = (tripod2.z - tripod1.z) * (tripod3.x - tripod1.x)
									 - (tripod3.z - tripod1.z) * (tripod2.x - tripod1.x);
							float pc = (tripod2.x - tripod1.x) * (tripod3.y - tripod1.y)
									 - (tripod3.x - tripod1.x) * (tripod2.y - tripod1.y);
							float pd = -(pa * tripod1.x + pb * tripod1.y + pc * tripod1.z);

							float ps = sqrt(pa * pa + pb * pb + pc * pc);

							pa /= ps;
							pb /= ps;
							pc /= ps;
							pd /= ps;

							float pd2 = pa * pointSel.x + pb * pointSel.y + pc * pointSel.z + pd;

							float s = 1;
							if(iterCount >= 5)
							{
								s = 1 - 1.8 * fabs(pd2) / sqrt(sqrt(pointSel.x * pointSel.x + pointSel.y + pointSel.y + pointSel.z * pointSel.z));
							}

							coeff.x = s * pa;
							coeff.y = s * pb;
							coeff.z = s * pc;
							coeff.intensity = s * pd2;

							if(s > 0.1 && pd2 != 0)
							{
								laserCloudOri->push_back(surfPointsFlat->points[i]);
								coeffSel->push_back(coeff);
							}
						}
					}

					int pointSelNum = laserCloudOri->points.size();   //laserCloudOri保存特征点	
					if(pointSelNum < 10)    //匹配点太少，不能进行计算
					{
						continue;
					}


					cv::Mat matA(pointSelNum, 6, CV_32F, cv::Scalar::all(0));
					cv::Mat matAt(6, pointSelNum, CV_32F, cv::Scalar::all(0));
					cv::Mat matAtA(6, 6, CV_32F, cv::Scalar::all(0));
					cv::Mat matB(pointSelNum, 1, CV_32F, cv::Scalar::all(0));
					cv::Mat matAtB(6, 1, CV_32F, cv::Scalar::all(0));
					cv::Mat matX(6, 1, CV_32F, cv::Scalar::all(0));

					//构建优化方程
					for(int i = 0; i < pointSelNum; i ++)
					{
						pointOri = laserCloudOri->points[i];
						coeff = coeffSel->points[i];


						float srx = sin(transform[0]);
						float crx = cos(transform[0]);
						float sry = sin(transform[1]);
						float cry = cos(transform[1]);
						float srz = sin(transform[2]);
						float crz = cos(transform[2]);
						float tx = transform[3];
						float ty = transform[4];
						float tz = transform[5];

						float arx = (crx * sry * srz *pointOri.x + crx * crz * sry * pointOri.y + srx * sry * pointOri.z + tx * crx * sry * srz - ty * crx * crz * sry - tz * srx * sry) * coeff.x + (srx * srz * pointOri.x - crz * srx * pointOri.y + crx *pointOri.z + ty * crz * srx - tz * crx - tx * srx * srz) * coeff.y + (crx * cry * srz * pointOri.x - crx * cry * crz * pointOri.y - cry * srx * pointOri.z + tz * cry * srx + ty * crx * cry * crz - tx * crx * cry * srz) * coeff.z;
							   
						float ary = ((-crz * sry - cry * srx * srz) * pointOri.x + (cry * crz * srx - sry * srz) * pointOri.y - crx * cry * pointOri.z + tx * (crz * sry + cry * srx * srz) + ty * (sry * srz - cry * crz * srx) + tz * crx * cry) * coeff.x + ((cry * crz - srx * sry * srz) * pointOri.x + (cry * srz + crz * srx * sry) * pointOri.y - crx * sry * pointOri.z + tz * crx * sry - ty * (cry * srz + crz * srx * sry) - tx * (cry * crz - srx * sry * srz)) * coeff.z;
									     
						float arz = ((-cry * srz - crz * srx * sry) * pointOri.x + (cry * crz - srx * sry * srz) * pointOri.y + tx * (cry * srz + crz * srx * sry) - ty * (cry * crz - srx * sry * srz)) * coeff.x + (-crx * crz * pointOri.x - crx * srz * pointOri.y + ty * crx * srz + tx * crx * crz) * coeff.y + ((cry * crz * srx - sry * srz) * pointOri.x + (crz * sry + cry * srx * srz) * pointOri.y + tx*(sry * srz - cry * crz * srx) - ty * (crz * sry + cry * srx * srz)) * coeff.z;
											   
						float atx = -(cry * crz - srx * sry * srz) * coeff.x + crx * srz * coeff.y - (crz * sry + cry * srx * srz) * coeff.z;
							
						float aty = -(cry * srz + crz * srx * sry) * coeff.x - crx * crz * coeff.y - (sry * srz - cry * crz * srx) * coeff.z;
													
						float atz = crx * sry * coeff.x - srx * coeff.y - crx * cry * coeff.z;
													
						float d2 = coeff.intensity;

						//matA保存雅克比矩阵
						matA.at<float>(i,0) = arx;
						matA.at<float>(i,1) = ary;
						matA.at<float>(i,2) = arz;
						matA.at<float>(i,3) = atx;
						matA.at<float>(i,4) = aty;
						matA.at<float>(i,5) = atz;
						matB.at<float>(i, 0) = -0.05 * d2;
					}

					
					cv::transpose(matA, matAt);
					matAtA = matAt * matA;
					matAtB = matAt * matB;

					cv::solve(matAtA, matAtB, matX, cv::DECOMP_QR);   //(JTJ)X=-JF

					if(iterCount == 0)
					{
						cv::Mat matE(1, 6, CV_32F, cv::Scalar::all(0));
						cv::Mat matV(6, 6, CV_32F, cv::Scalar::all(0));
						cv::Mat matV2(6, 6, CV_32F, cv::Scalar::all(0));

						cv::eigen(matAtA, matE, matV);    //matATA进行特征值分解

						//退化场景检查
						isDegenerate = false;
						float eignThre[6] = {10, 10, 10, 10, 10, 10};
						for(int i = 5; i >= 0; i --)
						{
							if(matE.at<float>(0,i) < eignThre[i])
							{
								for(int j = 0; j < 6; j ++)
								{
									matV2.at<float>(i,j) = 0;
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

					//matX保存的是更新量
					transform[0] += matX.at<float>(0,0);
					transform[1] += matX.at<float>(1,0);
					transform[2] += matX.at<float>(2,0);
					transform[3] += matX.at<float>(3,0);
					transform[4] += matX.at<float>(4,0);
					transform[5] += matX.at<float>(5,0);


					for(int i = 0; i < 6; i ++)
					{
						if(isnan(transform[i]))
						{
							transform[i] = 0;
						}
					}

					//更新量足够小就跳出循环
					float deltaR = sqrt(pow(rad2deg(matX.at<float>(0,0)), 2)
									  + pow(rad2deg(matX.at<float>(1,0)), 2)
									  + pow(rad2deg(matX.at<float>(2,0)), 2));

					float deltaT = sqrt(pow(matX.at<float>(3, 0) * 100, 2)
							
							+ pow(matX.at<float>(4, 0) * 100, 2)
									  + pow(matX.at<float>(5, 0) * 100, 2));

					if(deltaR < 0.1 && deltaT < 0.1)
					{
						break;
					}
				}

			}

			float rx, ry, rz, tx, ty, tz;
			AccumulateRotation(transformSum[0], transformSum[1], transformSum[2], -transform[0], -transform[1], -transform[2], rx, ry, rz);

			//绕z轴旋转
			float x1 = cos(rz) * transform[3] - sin(rz) * transform[4];
			float y1 = sin(rz) * transform[3] + cos(rz) * transform[4];
			float z1 = transform[5];

			//绕y轴旋转
			float x2 = cos(ry) * x1 - sin(ry) * z1;
			float y2 = y1;
			float z2 = sin(ry) * x1 + cos(ry) * z1;

			//绕x轴旋转
			tx = transformSum[3] - x2; 
			ty = transformSum[4] - (cos(rx) * y2 - sin(rx) * z2);
			tz = transformSum[5] - (sin(rx) * y2 + cos(rx) * z2);


			transformSum[0] = rx; 
			transformSum[1] = ry;
			transformSum[2] = rz;
			transformSum[3] = tx;
			transformSum[4] = ty;
			transformSum[5] = tz;


			geometry_msgs::Quaternion geoQuat = tf::createQuaternionMsgFromRollPitchYaw(rx, ry, rz);
	
			laserOdometry.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
			laserOdometry.pose.pose.orientation.x = geoQuat.x;
			laserOdometry.pose.pose.orientation.y = geoQuat.y;
			laserOdometry.pose.pose.orientation.z = geoQuat.z;
			laserOdometry.pose.pose.orientation.w = geoQuat.w;
			laserOdometry.pose.pose.position.x = tx;
			laserOdometry.pose.pose.position.y = ty;
			laserOdometry.pose.pose.position.z = tz;
			pubLaserOdometry.publish(laserOdometry);


			std::cout << std::setprecision(15) << timeSurfPointsLessFlat << '\t' << tx << '\t' << ty << '\t' << tz << std::endl;


			//将次边缘点转换到地图坐标系下
			int cornerPointsLessSharpNum = cornerPointsLessSharp->points.size();
			for(int i = 0; i < cornerPointsLessSharpNum; i ++)
			{
				TransformToEnd(&cornerPointsLessSharp->points[i], &cornerPointsLessSharp->points[i]);
			}

			//将次平面点转换到地图坐标系下
			int surfPointsLessFlatNum = surfPointsLessFlat->points.size();
			for(int i = 0; i < surfPointsLessFlatNum; i ++)
			{
				TransformToEnd(&surfPointsLessFlat->points[i], &surfPointsLessFlat->points[i]);
			}

			frameCount ++;

			if(frameCount >= skipFrameNum + 1)  //每隔两帧发一次完整点云信息
			{
				int laserCloudFullResNum = laserCloudFullRes->points.size();
				for(int i = 0; i < laserCloudFullResNum; i ++)
				{
					TransformToEnd(&laserCloudFullRes->points[i], &laserCloudFullRes->points[i]);
				}
			}

			//laserCloudCornerLast = cornerPointsLessSharp赋值操作
			pcl::PointCloud<pcl::PointXYZI>::Ptr laserCloudTemp = cornerPointsLessSharp;
			cornerPointsLessSharp = laserCloudCornerLast;
			laserCloudCornerLast = laserCloudTemp;

			//laserCloudSurfLast = surfPointsLessFlat;
			laserCloudTemp = surfPointsLessFlat;
			surfPointsLessFlat = laserCloudSurfLast;
			laserCloudSurfLast = laserCloudTemp;

			//次边缘点和次平面点个数
			laserCloudCornerLastNum = laserCloudCornerLast->points.size();
			laserCloudSurfLastNum = laserCloudSurfLast->points.size();

			//如果足够多可以构造kd树
			if(laserCloudCornerLastNum > 10 && laserCloudSurfLastNum > 100)
			{
				kdtreeCornerLast->setInputCloud(laserCloudCornerLast);
				kdtreeSurfLast->setInputCloud(laserCloudSurfLast);
			}


			if(frameCount >= skipFrameNum + 1)
			{
				frameCount = 0;

				//发布次边缘点消息
				sensor_msgs::PointCloud2 laserCloudCornerLast2;
				pcl::toROSMsg(*laserCloudCornerLast, laserCloudCornerLast2);
				laserCloudCornerLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat     );
				laserCloudCornerLast2.header.frame_id = "/camera";
				pubLaserCloudCornerLast.publish(laserCloudCornerLast2);
		
				//发布次平面点消息
				sensor_msgs::PointCloud2 laserCloudSurfLast2;
				pcl::toROSMsg(*laserCloudSurfLast, laserCloudSurfLast2);
				laserCloudSurfLast2.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
				laserCloudSurfLast2.header.frame_id = "/camera";
				pubLaserCloudSurfLast.publish(laserCloudSurfLast2);
	
				//发布完整点云消息
				sensor_msgs::PointCloud2 laserCloudFullRes3;
				pcl::toROSMsg(*laserCloudFullRes, laserCloudFullRes3);
				laserCloudFullRes3.header.stamp = ros::Time().fromSec(timeSurfPointsLessFlat);
				laserCloudFullRes3.header.frame_id = "/camera";
				pubLaserCloudFullRes.publish(laserCloudFullRes3);

			}

		}

		status = ros::ok();
		rate.sleep();
	}

	return 0;
}

#include <cmath>
#include <vector>

#include "common.h"
#include <opencv/cv.h>
#include <nav_msgs/Odometry.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>

const double scanPeriod = 0.1;      //点云周期0.1s

const int N_SCANS = 16;		//16线

ros::Publisher pubLaserCloud;               //点云
ros::Publisher pubCornerPointsSharp;        //边缘点
ros::Publisher pubCornerPointsLessSharp;    //次边缘点
ros::Publisher pubSurfPointsFlat;           //平面点
ros::Publisher pubSurfPointsLessFlat;       //次平面点


//点云回调函数
void laserCloudHandler(const sensor_msgs::PointCloud2ConstPtr& laserCloudMsg)
{

	std::vector<int> scanStartInd(N_SCANS,0);     //每一线的开始下标
	std::vector<int> scanEndInd(N_SCANS,0);       //每一线的结束下标

	double timeScanCur = laserCloudMsg->header.stamp.toSec();  //时间戳
	pcl::PointCloud<PointType> laserCloudIn;
	pcl::fromROSMsg(*laserCloudMsg, laserCloudIn);		//sensor_msgs::PoinrCloud2->pcl::PointCloud<PointType>
	std::vector<int> indices;
	pcl::removeNaNFromPointCloud(laserCloudIn, laserCloudIn, indices);   //剔除NaN点
	int cloudSize = laserCloudIn.points.size();		//点云数量

	float startOri = -atan2(laserCloudIn.points[0].y, laserCloudIn.points[0].x);   //开始角度
	float endOri = -atan2(laserCloudIn.points[cloudSize - 1].y, laserCloudIn.points[cloudSize - 1].x) + 2 * M_PI;   //结束角度
	if(endOri - startOri > 3 * M_PI)
	{
		endOri -= 2 * M_PI;
	}
	else if(endOri - startOri < M_PI)
	{
		endOri += 2 * M_PI;
	}

	int count = cloudSize;
	PointType point;
	std::vector<pcl::PointCloud<PointType> > laserCloudScans(N_SCANS);
	bool halfPassed = false;

	for(int i = 0; i < cloudSize; i ++)
	{
		point = laserCloudIn.points[i];

		//计算线号
		float angle = rad2deg(atan(point.z / sqrt(point.x * point.x + point.y * point.y)));    
		int scanID;
		int roundedAngle = round(angle);   //角度进行四舍五入转换成整数

		if(roundedAngle > 0)
		{
			scanID = roundedAngle;
		}
		else
		{
			scanID = roundedAngle + (N_SCANS - 1);
		}
		if(scanID > (N_SCANS - 1) || scanID < 0)
		{
			count --;
			continue;
		}

		float ori = -atan2(point.y, point.x);
		if(!halfPassed)
		{
			if(ori < startOri - M_PI / 2)
			{
				ori += 2 * M_PI;
			}
			else if(ori > startOri + M_PI * 3 / 2)
			{
				ori -= 2 * M_PI;
			}

			if(ori - startOri > M_PI)
			{
				halfPassed = true;
			}
		}
		else
		{
			ori += 2 * M_PI;
			if(ori < endOri - M_PI * 3 / 2)
			{
				ori += 2 * M_PI;
			}
			else if(ori > endOri + M_PI / 2)
			{
				ori - 2 * M_PI;
			}
		}


		float relTime = (ori - startOri) / (endOri - startOri);
		point.intensity = scanID + scanPeriod * relTime;   //整数部分表示线号，小数部分表示角度比例
		laserCloudScans[scanID].push_back(point);   //按照线号分类

	}
	cloudSize = count;

	//按照线号从小到大排序
	pcl::PointCloud<PointType>::Ptr laserCloud(new pcl::PointCloud<PointType>());
	for(int i = 0; i < N_SCANS; i ++)
	{
		*laserCloud += laserCloudScans[i];
	}

	std::vector<float> cloudCurvature(cloudSize,0);    //曲率
	std::vector<int> cloudSortInd(cloudSize,0);        //下标
	std::vector<int> cloudNeighborPicked(cloudSize,0); //是否参与特征点选取
	std::vector<int> cloudLabel(cloudSize,0);          //哪一类特征点
	
	int scanCount = -1;     //初始化线号为-1
	for (int i = 5; i < cloudSize - 5; i++)    //从第5个点开始到倒数第5个点结束，使用前后5个点用来算曲率 
	{
		float diffX = laserCloud->points[i - 5].x + laserCloud->points[i - 4].x
					+ laserCloud->points[i - 3].x + laserCloud->points[i - 2].x
					+ laserCloud->points[i - 1].x - 10 * laserCloud->points[i].x
					+ laserCloud->points[i + 1].x + laserCloud->points[i + 2].x
					+ laserCloud->points[i + 3].x + laserCloud->points[i + 4].x
					+ laserCloud->points[i + 5].x;
		float diffY = laserCloud->points[i - 5].y + laserCloud->points[i - 4].y
					+ laserCloud->points[i - 3].y + laserCloud->points[i - 2].y
					+ laserCloud->points[i - 1].y - 10 * laserCloud->points[i].y
					+ laserCloud->points[i + 1].y + laserCloud->points[i + 2].y
					+ laserCloud->points[i + 3].y + laserCloud->points[i + 4].y
					+ laserCloud->points[i + 5].y;
		float diffZ = laserCloud->points[i - 5].z + laserCloud->points[i - 4].z
					+ laserCloud->points[i - 3].z + laserCloud->points[i - 2].z
					+ laserCloud->points[i - 1].z - 10 * laserCloud->points[i].z
					+ laserCloud->points[i + 1].z + laserCloud->points[i + 2].z
					+ laserCloud->points[i + 3].z + laserCloud->points[i + 4].z
					+ laserCloud->points[i + 5].z;

		cloudCurvature[i] = diffX * diffX + diffY * diffY + diffZ * diffZ;  //曲率
		cloudSortInd[i] = i;                //初始化下标
		cloudNeighborPicked[i]= 0;          //初始化
		cloudLabel[i] = 0;                  //初始化

		//计算每一线的开始和结束下标
		if(int(laserCloud->points[i].intensity) != scanCount)
		{
			scanCount = int(laserCloud->points[i].intensity);
			if(scanCount > 0 && scanCount < N_SCANS)
			{
				scanStartInd[scanCount] = i + 5;
				scanEndInd[scanCount - 1] = i - 5;
			}
		}
	}
	scanStartInd[0] = 5;
	scanEndInd.back() = cloudSize - 5;


	//两种情况不能作为特征点选取，第一种为平行于激光束，第二种为遮挡
	for(int i = 5; i < cloudSize - 6; i ++)
	{
		float diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x;
		float diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y;
		float diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z;
		float diff = diffX * diffX + diffY * diffY + diffZ * diffZ;


		if(diff > 0.1)
		{
			float depth1 = sqrt(laserCloud->points[i].x * laserCloud->points[i].x + 
						   laserCloud->points[i].y * laserCloud->points[i].y +
						   laserCloud->points[i].z * laserCloud->points[i].z);
			
			float depth2 = sqrt(laserCloud->points[i + 1].x * laserCloud->points[i + 1].x +
						   laserCloud->points[i + 1].y * laserCloud->points[i + 1].y + 
						   laserCloud->points[i + 1].z * laserCloud->points[i + 1].z);

			if(depth1 > depth2)
			{
				diffX = laserCloud->points[i + 1].x - laserCloud->points[i].x * depth2 / depth1;
				diffY = laserCloud->points[i + 1].y - laserCloud->points[i].y * depth2 / depth1;
				diffZ = laserCloud->points[i + 1].z - laserCloud->points[i].z * depth2 / depth1;

				if(sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth2 < 0.1)
				{
					cloudNeighborPicked[i - 5] = 1;
					cloudNeighborPicked[i - 4] = 1;
					cloudNeighborPicked[i - 3] = 1;
					cloudNeighborPicked[i - 2] = 1;
					cloudNeighborPicked[i - 1] = 1;
					cloudNeighborPicked[i] = 1;
				}
			}
			else
			{
				diffX = laserCloud->points[i + 1].x * depth1 / depth2 - laserCloud->points[i].x;
				diffY = laserCloud->points[i + 1].y * depth1 / depth2 - laserCloud->points[i].y;
				diffZ = laserCloud->points[i + 1].z * depth1 / depth2 - laserCloud->points[i].z;

				if(sqrt(diffX * diffX + diffY * diffY + diffZ * diffZ) / depth1 < 0.1)
				{
					cloudNeighborPicked[i + 1] = 1;
					cloudNeighborPicked[i + 2] = 1;
					cloudNeighborPicked[i + 3] = 1;
					cloudNeighborPicked[i + 4] = 1;
					cloudNeighborPicked[i + 5] = 1;
					cloudNeighborPicked[i + 6] = 1;
				}
			}
		}

		float diffX2 = laserCloud->points[i].x - laserCloud->points[i - 1].x;
		float diffY2 = laserCloud->points[i].y - laserCloud->points[i - 1].y;
		float diffZ2 = laserCloud->points[i].z - laserCloud->points[i - 1].z;
		float diff2 = diffX2 * diffX2 + diffY2 * diffY2 + diffZ2 * diffZ2;

		float dis = laserCloud->points[i].x * laserCloud->points[i].x
				  + laserCloud->points[i].y * laserCloud->points[i].y
				  + laserCloud->points[i].z * laserCloud->points[i].z;

		if(diff > 0.0002 * dis && diff2 > 0.0002 * dis)
		{
			cloudNeighborPicked[i] = 1;
		}
	}


	pcl::PointCloud<PointType> cornerPointsSharp;           //边缘点
	pcl::PointCloud<PointType> cornerPointsLessSharp;       //次边缘点
	pcl::PointCloud<PointType> surfPointsFlat;              //平面点
	pcl::PointCloud<PointType> surfPointsLessFlat;          //次平面点

	//每一线选取若干个特征点
	for(int i = 0; i < N_SCANS; i ++)
	{
		pcl::PointCloud<PointType>::Ptr surfPointsLessFlatScan(new pcl::PointCloud<PointType>);

		//将每一线平均分成6份
		for(int j = 0; j < 6; j ++)
		{
			int sp = (scanStartInd[i] * (6 - j) + scanEndInd[i] * j) / 6;    //开始下标
			int ep = (scanStartInd[i] * (5 - j) + scanEndInd[i] * (j + 1)) / 6 - 1;   //结束下标
		
			//按照曲率进行从小到大排序
			for(int k = sp + 1; k <= ep; k ++)
			{
				for(int l = k; l >= sp + 1; l --)
				{
					if(cloudCurvature[cloudSortInd[l]] < cloudCurvature[cloudSortInd[l - 1]])
					{
						int temp = cloudSortInd[l - 1];
						cloudSortInd[l - 1] = cloudSortInd[l];
						cloudSortInd[l] = temp;
					}
				}
			}

			//选取边缘点
			int largestPickedNum = 0;
			for(int k = ep; k >= sp; k --)
			{
				int ind = cloudSortInd[k];
				if(cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] > 0.1)
				{
					largestPickedNum ++;
					if(largestPickedNum <= 2)     //选取2个点为边缘点
					{
						cloudLabel[ind] = 2;
						cornerPointsSharp.push_back(laserCloud->points[ind]);
						cornerPointsLessSharp.push_back(laserCloud->points[ind]);
					}
					else if(largestPickedNum <= 20)     //选取20个点为次边缘点
					{
						cloudLabel[ind] = 1;
						cornerPointsLessSharp.push_back(laserCloud->points[ind]);
					}
					else
					{
						break;
					}

					cloudNeighborPicked[ind] = 1;     //已经被选过

					//前后5个距离近的点不作为特征点被选取
					for(int l = 1; l <= 5; l ++)
					{
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l - 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l - 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l - 1].z;
						if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
						{
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
					for(int  l = -1; l >= -5; l --)
					{
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l + 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l + 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l + 1].z;
						if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
						{
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}

			//选取平面点
			int smallestPickedNum = 0;
			for(int k = sp; k <= ep; k ++)
			{
				int ind = cloudSortInd[k];
				if(cloudNeighborPicked[ind] == 0 && cloudCurvature[ind] < 0.1)
				{
					cloudLabel[ind] = -1;
					surfPointsFlat.push_back(laserCloud->points[ind]);

					smallestPickedNum ++;
					if(smallestPickedNum >= 4)     //选取4个平面点作为平面点
					{
						break;
					}

					cloudNeighborPicked[ind] = 1;

					//前后距离近的点不作为特征点
					for(int l = 1; l <= 5; l ++)
					{
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l - 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l - 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l - 1].z;
						if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
						{
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
					for(int l = -1; l >= -5; l --)
					{
						float diffX = laserCloud->points[ind + l].x
									- laserCloud->points[ind + l + 1].x;
						float diffY = laserCloud->points[ind + l].y
									- laserCloud->points[ind + l + 1].y;
						float diffZ = laserCloud->points[ind + l].z
									- laserCloud->points[ind + l + 1].z;
						if(diffX * diffX + diffY * diffY + diffZ * diffZ > 0.05)
						{
							break;
						}
						cloudNeighborPicked[ind + l] = 1;
					}
				}
			}
		
			//剩下的未被选中的点作为平面点
			for(int k = sp; k <= ep; k ++)
			{
				if(cloudLabel[k] <= 0)
				{
					surfPointsLessFlatScan->push_back(laserCloud->points[k]);
				}
			}

		}

		//平面点太多，进行下采样
		pcl::PointCloud<PointType> surfPointsLessFlatScanDS;
		pcl::VoxelGrid<PointType> downSizeFilter;
		downSizeFilter.setInputCloud(surfPointsLessFlatScan);
		downSizeFilter.setLeafSize(0.2,0.2,0.2);
		downSizeFilter.filter(surfPointsLessFlatScanDS);
		surfPointsLessFlat += surfPointsLessFlatScanDS;
	}


	//发布点云
	sensor_msgs::PointCloud2 laserCloudOutMsg;
	pcl::toROSMsg(*laserCloud, laserCloudOutMsg);
	laserCloudOutMsg.header.stamp = laserCloudMsg->header.stamp;
	laserCloudOutMsg.header.frame_id = "/camera";
	pubLaserCloud.publish(laserCloudOutMsg);


	//边缘点
	sensor_msgs::PointCloud2 cornerPointsSharpMsg;
	pcl::toROSMsg(cornerPointsSharp, cornerPointsSharpMsg);
	cornerPointsSharpMsg.header.stamp = laserCloudMsg->header.stamp;
	cornerPointsSharpMsg.header.frame_id = "/camera";
	pubCornerPointsSharp.publish(cornerPointsSharpMsg);


	//次边缘点
	sensor_msgs::PointCloud2 cornerPointsLessSharpMsg;
	pcl::toROSMsg(cornerPointsLessSharp, cornerPointsLessSharpMsg);
	cornerPointsLessSharpMsg.header.stamp = laserCloudMsg->header.stamp;
	cornerPointsLessSharpMsg.header.frame_id = "/camera";
	pubCornerPointsLessSharp.publish(cornerPointsLessSharpMsg);


	//平面点
	sensor_msgs::PointCloud2 surfPointsFlatMsg;
	pcl::toROSMsg(surfPointsFlat, surfPointsFlatMsg);
	surfPointsFlatMsg.header.stamp = laserCloudMsg->header.stamp;
	surfPointsFlatMsg.header.frame_id = "/camera";
	pubSurfPointsFlat.publish(surfPointsFlatMsg);

	//次平面点
	sensor_msgs::PointCloud2 surfPointsLessFlatMsg;
	pcl::toROSMsg(surfPointsLessFlat, surfPointsLessFlatMsg);
	surfPointsLessFlatMsg.header.stamp = laserCloudMsg->header.stamp;
	surfPointsLessFlatMsg.header.frame_id = "/camera";
	pubSurfPointsLessFlat.publish(surfPointsLessFlatMsg);

}


int main(int argc,char **argv)
{
	ros::init(argc,argv,"scanRegistration");
	ros::NodeHandle nh;

	ros::Subscriber subLaserCloud = nh.subscribe<sensor_msgs::PointCloud2>("/velodyne_points",2,laserCloudHandler);    

	pubLaserCloud = nh.advertise<sensor_msgs::PointCloud2>("/velodyne_cloud_2",2);

	pubCornerPointsSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_sharp",2);

	pubCornerPointsLessSharp = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_sharp",2);

	pubSurfPointsFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_flat",2);

	pubSurfPointsLessFlat = nh.advertise<sensor_msgs::PointCloud2>("/laser_cloud_less_flat",2);

	ros::spin();

	return 0;
}

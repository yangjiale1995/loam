#include <cmath>
#include <pcl/point_types.h>

#define POINTSNUM 60000

#define HEIGHT 10


typedef pcl::PointXYZI PointTyoe;


inline double rad2deg(double radians)
{
	return radians * 180.0 / M_PI;
}

inline double deg2rad(double degrees)
{
	return degrees * M_PI / 180.0;
}

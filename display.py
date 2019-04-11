import numpy
from open3d import *
import argparse

# parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--position", type = str, required=True, help="path of the position vector")
ap.add_argument("-t", "--texture", type = str, required=True, help="path of the texture vector")
args = vars(ap.parse_args())

# function to convert numpy arrays to ply files
def convert_to_pointCloud(X,Y):
	intensity_f = numpy.reshape(Y,(len(Y),1))
	my_img_position = X
	my_img_color = np.concatenate((intensity_f,intensity_f,intensity_f),axis=1)
	pcd = PointCloud()
	pcd.points = Vector3dVector(my_img_position)
	pcd.colors = Vector3dVector(my_img_color)
	write_point_cloud("./X_Y.ply", pcd)
	pcd_load = read_point_cloud("./X_Y.ply")
	draw_geometries([pcd_load]) 
	return 1

position = numpy.load(args["position"])
position = position['position']

texture = numpy.load(args["texture"])
texture = texture['texture']

tmp = convert_to_pointCloud(position,texture)

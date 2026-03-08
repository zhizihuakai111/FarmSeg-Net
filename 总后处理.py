import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KDTree
from sklearn.cluster import DBSCAN
from skimage.morphology import skeletonize, remove_small_objects, binary_closing, binary_dilation, disk
from skimage import filters
from skimage.measure import label, regionprops
from scipy.ndimage import binary_fill_holes
from scipy.spatial import ConvexHull, cKDTree, Delaunay
import cv2
import networkx as nx
import open3d as o3d
import os
import warnings
from collections import deque
from lxml import etree
from pyproj import Transformer
import traceback  # 用于打印详细的错误堆栈信息

# 设置Matplotlib支持中文
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 抑制来自numpy.polyfit的RankWarning
warnings.simplefilter('ignore', np.RankWarning)


# ===========================================
# 第一部分：自适应均值滤波
# ===========================================
def compute_point_cloud_curvature(pointCloud, radius):
    """计算点云曲率"""
    kdtree = o3d.geometry.KDTreeFlann(pointCloud)
    curvatures = np.zeros((len(pointCloud.points),))

    for i in range(len(pointCloud.points)):
        point = pointCloud.points[i]
        [k, idx, _] = kdtree.search_radius_vector_3d(point, radius)

        if k < 3:
            continue

        neighbors = np.asarray(pointCloud.points)[idx[1:], :]
        mean = np.mean(neighbors, axis=0)

        covariance_matrix = np.cov((neighbors - mean).T)
        eigenvalues = np.linalg.eigvalsh(covariance_matrix)
        curvatures[i] = eigenvalues[0] / np.sum(eigenvalues)

    return curvatures


def pointCloud_edge_and_smooth_filter(pointCloud, radius, curvature_threshold=0.02, high_curvature_smooth_factor=1.5):
    """边缘保护的自适应均值滤波"""
    kdtree = o3d.geometry.KDTreeFlann(pointCloud)
    np_points = np.array(pointCloud.points)
    np_colors = np.array(pointCloud.colors) if pointCloud.has_colors() else None

    curvatures = compute_point_cloud_curvature(pointCloud, radius)

    for i in range(len(pointCloud.points)):
        point = pointCloud.points[i]
        [k, idx, dists] = kdtree.search_radius_vector_3d(point, radius)

        if k < 1:
            continue

        neighbors = np.asarray(pointCloud.points)[idx]
        weights = np.exp(-np.asarray(dists) / radius)

        if curvatures[i] < curvature_threshold:
            # 平坦区域，标准均值平滑
            np_points[i] = np.average(neighbors, axis=0, weights=weights)
            if np_colors is not None:
                neighbors_colors = np.asarray(pointCloud.colors)[idx]
                np_colors[i] = np.average(neighbors_colors, axis=0, weights=weights)
        elif curvatures[i] >= curvature_threshold and curvatures[
            i] < high_curvature_smooth_factor * curvature_threshold:
            # 高曲率区域，执行增强的平滑
            enhanced_weights = weights * high_curvature_smooth_factor
            np_points[i] = np.average(neighbors, axis=0, weights=enhanced_weights)
            if np_colors is not None:
                neighbors_colors = np.asarray(pointCloud.colors)[idx]
                np_colors[i] = np.average(neighbors_colors, axis=0, weights=enhanced_weights)
        else:
            # 边缘区域：保留原始点
            np_points[i] = point
            if np_colors is not None:
                np_colors[i] = pointCloud.colors[i]

    pointCloud_filtered = o3d.geometry.PointCloud()
    pointCloud_filtered.points = o3d.utility.Vector3dVector(np_points)
    if np_colors is not None:
        pointCloud_filtered.colors = o3d.utility.Vector3dVector(np_colors)

    return pointCloud_filtered


# ===========================================
# 第二部分：区域生长分割
# ===========================================
class RegionGrowing:
    """区域生长分割算法"""

    def __init__(self, cloud,
                 min_pts_per_cluster=1,
                 max_pts_per_cluster=np.inf,
                 theta_threshold=30,
                 curvature_threshold=0.05,
                 neighbour_number=30):

        self.cure = None
        self.pcd = cloud
        self.min_pts_per_cluster = min_pts_per_cluster
        self.max_pts_per_cluster = max_pts_per_cluster
        self.theta_threshold = np.deg2rad(theta_threshold)
        self.curvature_threshold = curvature_threshold
        self.neighbour_number = neighbour_number
        self.point_neighbours = []
        self.point_labels = []
        self.num_pts_in_segment = []
        self.clusters = []
        self.number_of_segments = 0

    def prepare_for_segment(self):
        points = np.asarray(self.pcd.points)
        normals = np.asarray(self.pcd.normals) if self.pcd.has_normals() else np.array([])

        if not points.shape[0]:
            return False
        if self.neighbour_number == 0:
            return False
        if points.shape[0] != normals.shape[0]:
            self.pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))

        return True

    def find_neighbour_points(self):
        number = len(self.pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        self.point_neighbours = np.zeros((number, self.neighbour_number))
        for ik in range(number):
            [_, idx, _] = kdtree.search_knn_vector_3d(self.pcd.points[ik], self.neighbour_number)
            self.point_neighbours[ik, :] = idx

    def validate_points(self, point, nebor):
        try:
            is_seed = True
            cosine_threshold = np.cos(self.theta_threshold)

            # 先检查索引是否有效
            if point < 0 or point >= len(self.pcd.normals) or nebor < 0 or nebor >= len(self.pcd.normals):
                return False, is_seed

            curr_seed_normal = self.pcd.normals[point]
            seed_nebor_normal = self.pcd.normals[nebor]

            # 检查法向量是否为零向量
            if np.all(np.abs(curr_seed_normal) < 1e-10) or np.all(np.abs(seed_nebor_normal) < 1e-10):
                return False, is_seed

            # 计算点积前对法向量进行归一化，避免因为长度为0导致的问题
            curr_seed_normal_norm = np.linalg.norm(curr_seed_normal)
            seed_nebor_normal_norm = np.linalg.norm(seed_nebor_normal)

            if curr_seed_normal_norm < 1e-10 or seed_nebor_normal_norm < 1e-10:
                return False, is_seed

            curr_seed_normal = curr_seed_normal / curr_seed_normal_norm
            seed_nebor_normal = seed_nebor_normal / seed_nebor_normal_norm

            dot_normal = np.fabs(np.dot(seed_nebor_normal, curr_seed_normal))

            if dot_normal < cosine_threshold:
                return False, is_seed
            if nebor < len(self.cure) and self.cure[nebor] > self.curvature_threshold:
                is_seed = False

            return True, is_seed
        except Exception as e:
            print(f"验证点时出错: {e}")
            return False, False

    def label_for_points(self, initial_seed, segment_number):
        seeds = deque([initial_seed])
        self.point_labels[initial_seed] = segment_number
        num_pts_in_segment = 1

        while len(seeds):
            curr_seed = seeds[0]
            seeds.popleft()
            i_nebor = 0
            # 这里确保curr_seed不会超出范围
            if curr_seed < 0 or curr_seed >= len(self.point_neighbours):
                continue

            # 确保self.point_neighbours[curr_seed]不为空且是有效的数组
            if len(self.point_neighbours) <= curr_seed or len(self.point_neighbours[curr_seed]) == 0:
                continue

            while i_nebor < self.neighbour_number and i_nebor < len(self.point_neighbours[curr_seed]):
                try:
                    index = int(self.point_neighbours[curr_seed, i_nebor])
                    if index < 0 or index >= len(self.point_labels) or self.point_labels[index] != -1:
                        i_nebor += 1
                        continue

                    belongs_to_segment, is_seed = self.validate_points(curr_seed, index)
                    if not belongs_to_segment:
                        i_nebor += 1
                        continue

                    self.point_labels[index] = segment_number
                    num_pts_in_segment += 1

                    if is_seed:
                        seeds.append(index)

                    i_nebor += 1
                except Exception as e:
                    print(f"处理邻居点时出错: {e}")
                    i_nebor += 1
                    continue

        return num_pts_in_segment

    def region_growing_process(self):
        num_of_pts = len(self.pcd.points)
        self.point_labels = -np.ones(num_of_pts)

        try:
            self.pcd.estimate_covariances(o3d.geometry.KDTreeSearchParamKNN(self.neighbour_number))
            cov_mat = self.pcd.covariances
            self.cure = np.zeros(num_of_pts)

            for i_n in range(num_of_pts):
                try:
                    if i_n >= len(cov_mat):
                        continue

                    # 确保协方差矩阵是有效的
                    if cov_mat[i_n] is None or np.any(np.isnan(cov_mat[i_n])) or np.any(np.isinf(cov_mat[i_n])):
                        self.cure[i_n] = 0
                        continue

                    eignvalue, _ = np.linalg.eig(cov_mat[i_n])

                    # 检查特征值是否有效
                    if np.any(np.isnan(eignvalue)) or np.any(np.isinf(eignvalue)):
                        self.cure[i_n] = 0
                        continue

                    idx = eignvalue.argsort()[::-1]
                    eignvalue = eignvalue[idx]

                    # 避免除以零
                    denominator = eignvalue[0] + eignvalue[1] + eignvalue[2]
                    if denominator < 1e-10:
                        self.cure[i_n] = 0
                    else:
                        self.cure[i_n] = eignvalue[2] / denominator
                except Exception as e:
                    print(f"计算曲率时出错 (点 {i_n}): {e}")
                    self.cure[i_n] = 0

            # 创建点的曲率索引
            point_curvature_index = np.zeros((num_of_pts, 2))
            for i_cu in range(num_of_pts):
                point_curvature_index[i_cu, 0] = self.cure[i_cu]
                point_curvature_index[i_cu, 1] = i_cu

            # 按曲率排序
            temp_cure = np.argsort(point_curvature_index[:, 0])
            point_curvature_index = point_curvature_index[temp_cure, :]

            seed_counter = 0
            if num_of_pts > 0:
                seed = int(point_curvature_index[seed_counter, 1])
            else:
                print("警告: 点云为空，无法进行区域生长分割")
                return

            segmented_pts_num = 0
            number_of_segments = 0

            max_iterations = min(num_of_pts * 2, 10000)  # 设置最大迭代次数，避免无限循环
            iteration = 0

            while segmented_pts_num < num_of_pts and iteration < max_iterations:
                try:
                    pts_in_segment = self.label_for_points(seed, number_of_segments)
                    segmented_pts_num += pts_in_segment
                    self.num_pts_in_segment.append(pts_in_segment)
                    number_of_segments += 1

                    found_next_seed = False
                    for i_seed in range(seed_counter + 1, num_of_pts):
                        index = int(point_curvature_index[i_seed, 1])
                        if index >= 0 and index < len(self.point_labels) and self.point_labels[index] == -1:
                            seed = index
                            seed_counter = i_seed
                            found_next_seed = True
                            break

                    if not found_next_seed:
                        print("无法找到更多的种子点，区域生长结束")
                        break
                except Exception as e:
                    print(f"区域生长过程出错: {e}")
                    iteration += 1
                    continue

                iteration += 1

            if iteration >= max_iterations:
                print(f"警告: 区域生长达到最大迭代次数 {max_iterations}，强制结束")
        except Exception as e:
            print(f"区域生长处理出错: {e}")

    def region_growing_clusters(self):
        number_of_segments = len(self.num_pts_in_segment)
        number_of_points = np.asarray(self.pcd.points).shape[0]

        for i in range(number_of_segments):
            tmp_init = list(np.zeros(self.num_pts_in_segment[i]))
            self.clusters.append(tmp_init)

        counter = list(np.zeros(number_of_segments))
        for i_point in range(number_of_points):
            segment_index = int(self.point_labels[i_point])
            if segment_index != -1:
                point_index = int(counter[segment_index])
                self.clusters[segment_index][point_index] = i_point
                counter[segment_index] = point_index + 1

        self.number_of_segments = number_of_segments

    def extract(self):
        try:
            if not self.prepare_for_segment():
                print("区域生长算法预处理失败！")
                return []

            self.find_neighbour_points()
            self.region_growing_process()
            self.region_growing_clusters()

            all_cluster = []
            for i in range(len(self.clusters)):
                if self.min_pts_per_cluster <= len(self.clusters[i]) <= self.max_pts_per_cluster:
                    all_cluster.append(self.clusters[i])
                else:
                    if len(self.clusters[i]) > 0:  # 确保簇不为空
                        self.point_labels[self.clusters[i]] = -1

            self.clusters = all_cluster
            return all_cluster
        except Exception as e:
            print(f"区域生长分割失败: {e}")
            return []


# ===========================================
# 第三部分：边界提取
# ===========================================
def boundary_extraction(point_cloud, resolution):
    """经纬边界提取"""
    x_min = np.min(point_cloud[:, 0])
    x_max = np.max(point_cloud[:, 0])
    delta_x = (x_max - x_min) / resolution
    minmax_x = np.full((resolution + 1, 2), [np.inf, -np.inf])
    indexs_x = np.zeros(2 * resolution + 2, dtype=int)

    for i in range(point_cloud.shape[0]):
        x = point_cloud[i, 0]
        y = point_cloud[i, 1]
        id_x = int((x - x_min) / delta_x)
        if id_x >= resolution + 1:  # 防止越界
            id_x = resolution
        if y < minmax_x[id_x, 0]:
            minmax_x[id_x, 0] = y
            indexs_x[id_x] = i
        elif y > minmax_x[id_x, 1]:
            minmax_x[id_x, 1] = y
            indexs_x[id_x + resolution + 1] = i

    y_min = np.min(point_cloud[:, 1])
    y_max = np.max(point_cloud[:, 1])
    delta_y = (y_max - y_min) / resolution
    minmax_y = np.full((resolution + 1, 2), [np.inf, -np.inf])
    indexs_y = np.zeros(2 * resolution + 2, dtype=int)

    for i in range(point_cloud.shape[0]):
        x = point_cloud[i, 0]
        y = point_cloud[i, 1]
        id_y = int((y - y_min) / delta_y)
        if id_y >= resolution + 1:  # 防止越界
            id_y = resolution
        if x < minmax_y[id_y, 0]:
            minmax_y[id_y, 0] = x
            indexs_y[id_y] = i
        elif x > minmax_y[id_y, 1]:
            minmax_y[id_y, 1] = x
            indexs_y[id_y + resolution + 1] = i

    # 过滤掉无效的索引
    valid_indices_x = indexs_x[indexs_x != 0]
    valid_indices_y = indexs_y[indexs_y != 0]

    boundary_x = point_cloud[valid_indices_x]
    boundary_y = point_cloud[valid_indices_y]

    if len(boundary_x) > 0 and len(boundary_y) > 0:
        boundary_points = np.vstack((boundary_x, boundary_y))
    elif len(boundary_x) > 0:
        boundary_points = boundary_x
    elif len(boundary_y) > 0:
        boundary_points = boundary_y
    else:
        boundary_points = np.array([])

    return boundary_points


# ===========================================
# 第四部分：Alpha边界点滤波
# ===========================================
def compute_alpha_shape(points, alpha):
    """计算Alpha形状"""
    xyz_points = points[:, :3]

    # 使用 Delaunay 三角剖分
    tri = Delaunay(xyz_points)
    edges = set()

    # 遍历每个三角形
    for simplex in tri.simplices:
        # 遍历每个三角形的边
        for i in range(3):
            for j in range(i + 1, 3):
                # 计算边长
                p1 = xyz_points[simplex[i]]
                p2 = xyz_points[simplex[j]]
                edge_length = np.linalg.norm(p1 - p2)

                # 如果边长小于 alpha，保留该边
                if edge_length < alpha:
                    edge = tuple(sorted([simplex[i], simplex[j]]))
                    edges.add(edge)

    # 提取属于边界的点，并保持其原始信息
    boundary_indices = set()
    for edge in edges:
        boundary_indices.add(edge[0])
        boundary_indices.add(edge[1])

    boundary_points = points[list(boundary_indices)]
    return boundary_points


# ===========================================
# 第五部分：点云排序
# ===========================================
def greedy_nearest_neighbor_sort_kdtree(points):
    """使用KD树加速邻近点查询进行贪心排序"""
    if len(points) == 0:
        return np.array([])

    sorted_points = []
    remaining_points = points.copy()

    # 选择初始点
    current_point = remaining_points[0]
    sorted_points.append(current_point)

    # 从未排序的点集中删除该点
    remaining_points = np.delete(remaining_points, 0, axis=0)

    while len(remaining_points) > 0:
        # 使用KDTree加速邻近查询
        tree = KDTree(remaining_points[:, :3])  # 仅使用XYZ部分构建KD树

        # 查询最近邻点
        dist, nearest_index = tree.query([current_point[:3]], k=1)
        nearest_index = nearest_index[0][0]  # 获取最近邻点的索引

        # 更新当前点为最近的点，并将该点加入排序结果
        current_point = remaining_points[nearest_index]
        sorted_points.append(current_point)

        # 删除已处理的点
        remaining_points = np.delete(remaining_points, nearest_index, axis=0)

    return np.array(sorted_points)


# ===========================================
# 第六部分：枢轴点提取
# ===========================================
def detect_corners(points, eps=0.15, min_samples=10):
    """使用DBSCAN聚类来检测点云中的拐角点和转折点"""
    if len(points) == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(points[:, :3])
    labels = clustering.labels_
    return labels


def optimize_boundary(points):
    """使用凸包拟合田块边界，以优化点云中的枢轴点"""
    if len(points) < 3:
        return points

    try:
        hull = ConvexHull(points[:, :2])  # 只使用XY平面上的点
        hull_points = points[hull.vertices]
        return hull_points
    except Exception as e:
        print(f"凸包计算错误: {e}")
        return points


# ===========================================
# 第七部分：坐标转换
# ===========================================
def transform_to_wgs84_utm51n(points, x_offset, y_offset, z_offset):
    """
    对点云进行X、Y、Z位移，转换为WGS84 UTM 51N投影坐标系

    参数:
    points - 输入点云数组
    x_offset, y_offset, z_offset - X, Y, Z方向的位移量

    返回:
    转换后的点云数组
    """
    # 对点云进行位移
    transformed_points = np.copy(points)
    transformed_points[:, 0] += x_offset  # X坐标位移
    transformed_points[:, 1] += y_offset  # Y坐标位移
    transformed_points[:, 2] += z_offset  # Z坐标位移

    return transformed_points


def convert_utm_to_wgs84(points):
    """
    将UTM坐标系(WGS84 UTM 51N)转换为WGS84地理坐标系(经纬度)

    参数:
    points - UTM坐标系下的点云数组

    返回:
    WGS84地理坐标系下的点云数组(经度,纬度,高程)
    """
    # 定义坐标系
    utm_zone = 51
    utm_proj = f"+proj=utm +zone={utm_zone} +datum=WGS84 +units=m +no_defs"
    wgs84_proj = "+proj=longlat +datum=WGS84 +no_defs"

    # 创建转换器
    transformer = Transformer.from_proj(utm_proj, wgs84_proj, always_xy=True)

    # 进行坐标转换
    wgs84_points = np.copy(points)
    for i in range(len(points)):
        lon, lat = transformer.transform(points[i, 0], points[i, 1])
        wgs84_points[i, 0] = lon  # 经度
        wgs84_points[i, 1] = lat  # 纬度
        # Z坐标保持不变

    return wgs84_points


# ===========================================
# 第八部分：生成KML文件
# ===========================================
def create_kml_file(points, output_file, field_name="田块"):
    """
    将点数据生成为KML文件，采用指定格式

    参数:
    points - WGS84地理坐标系下的点云数组(经度,纬度,高程)
    output_file - 输出KML文件路径
    field_name - 田块名称
    """
    # 创建KML结构
    kml = etree.Element("kml", xmlns="http://www.opengis.net/kml/2.2")
    doc = etree.SubElement(kml, "Document", id="root_doc")

    # 创建Schema
    schema = etree.SubElement(doc, "Schema", name=field_name, id=field_name)
    etree.SubElement(schema, "SimpleField", name="id", type="float")

    # 创建Folder
    folder = etree.SubElement(doc, "Folder")
    etree.SubElement(folder, "name").text = field_name

    # 创建Placemark
    placemark = etree.SubElement(folder, "Placemark")

    # 添加样式
    style = etree.SubElement(placemark, "Style")
    line_style = etree.SubElement(style, "LineStyle")
    etree.SubElement(line_style, "color").text = "ff0000ff"  # 红色

    poly_style = etree.SubElement(style, "PolyStyle")
    etree.SubElement(poly_style, "fill").text = "0"  # 无填充

    # 添加ExtendedData
    extended_data = etree.SubElement(placemark, "ExtendedData")
    schema_data = etree.SubElement(extended_data, "SchemaData", schemaUrl=f"#{field_name}")
    etree.SubElement(schema_data, "SimpleData", name="id").text = "6"

    # 添加几何
    multi_geom = etree.SubElement(placemark, "MultiGeometry")
    polygon = etree.SubElement(multi_geom, "Polygon")
    outer_boundary = etree.SubElement(polygon, "outerBoundaryIs")
    linear_ring = etree.SubElement(outer_boundary, "LinearRing")

    # 格式化坐标
    coords_str = " ".join([f"{point[0]},{point[1]}" for point in points])
    # 确保多边形闭合
    if len(points) > 0 and (points[0][0] != points[-1][0] or points[0][1] != points[-1][1]):
        coords_str += f" {points[0][0]},{points[0][1]}"

    etree.SubElement(linear_ring, "coordinates").text = coords_str

    # 写入KML文件
    with open(output_file, 'wb') as f:
        f.write(b'<?xml version="1.0" encoding="utf-8" ?>\n')
        f.write(etree.tostring(kml, pretty_print=True, encoding='utf-8'))


# ===========================================
# 第九部分：辅助函数
# ===========================================
def read_point_cloud(file_path):
    """读取点云数据（支持PCD和TXT格式）"""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pcd':
        # 读取PCD格式
        return o3d.io.read_point_cloud(file_path)
    elif file_extension == '.txt':
        # 从TXT文件读取数据
        try:
            data = np.loadtxt(file_path)

            # 判断数据列数以确定是否包含颜色信息
            if data.shape[1] >= 6:  # 假设至少包含XYZRGB
                points = data[:, :3]
                colors = data[:, 3:6] / 255.0  # 假设RGB范围是0-255

                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                return pcd
            else:  # 只有XYZ坐标
                points = data[:, :3]
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                return pcd
        except Exception as e:
            print(f"读取TXT点云时出错: {e}")
            return None
    else:
        print(f"不支持的文件格式: {file_extension}")
        return None


def save_point_cloud(point_cloud, file_path, with_label=False, labels=None):
    """保存点云数据（支持多种格式）"""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension == '.pcd':
        o3d.io.write_point_cloud(file_path, point_cloud)
    elif file_extension == '.txt':
        points = np.asarray(point_cloud.points)

        if point_cloud.has_colors():
            colors = np.asarray(point_cloud.colors) * 255.0
            data = np.hstack((points, colors))
            if with_label and labels is not None:
                data = np.hstack((data, labels.reshape(-1, 1)))
            np.savetxt(file_path, data, fmt='%.6f')
        else:
            if with_label and labels is not None:
                data = np.hstack((points, labels.reshape(-1, 1)))
                np.savetxt(file_path, data, fmt='%.6f')
            else:
                np.savetxt(file_path, points, fmt='%.6f')
    else:
        print(f"不支持的文件格式: {file_extension}")


def visualize_point_clouds(point_clouds, window_name="点云可视化", width=1024, height=768):
    """可视化点云"""
    o3d.visualization.draw_geometries(point_clouds, window_name=window_name, width=width, height=height)

    # ===========================================
    # 第十部分：农田点云处理
    # ===========================================


def save_boundary_points_with_labels(points, output_file, cluster_id=0, class_label=1):
    """
    保存边界点云数据，格式为 x y z L1 L2
    其中L1为农田类别标签(默认为1)，L2为该点所在农田的实例ID

    参数:
    points - 输入的边界点云数组
    output_file - 输出文件路径
    cluster_id - 实例ID (默认为0)
    class_label - 类别标签 (默认为1，表示农田)
    """
    # 添加类别标签(L1)和实例ID(L2)列
    L1 = np.full((points.shape[0], 1), class_label)  # 类别标签，1表示农田
    L2 = np.full((points.shape[0], 1), cluster_id)  # 实例ID

    # 合并数据
    labeled_points = np.hstack((points[:, :3], L1, L2))

    # 创建目录（如果不存在）
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    np.savetxt(output_file, labeled_points, fmt='%.6f %.6f %.6f %d %d')
    print(f"带标签的边界点已保存至: {output_file}")


def complete_processing_pipeline_for_kml(input_file, output_dir='.', x_offset=0, y_offset=0, z_offset=0):
    """完整的点云后处理流程，生成KML文件"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    kml_output_dir = os.path.join(output_dir, 'kml')
    os.makedirs(kml_output_dir, exist_ok=True)
    # 创建边界点输出目录
    boundary_output_dir = os.path.join(output_dir, 'boundary')
    os.makedirs(boundary_output_dir, exist_ok=True)

    print(f"处理点云文件: {input_file}")

    # 1. 读取点云
    print("1. 读取点云...")
    pcd = read_point_cloud(input_file)
    if pcd is None:
        print("无法读取点云文件，请检查文件路径和格式。")
        return

    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # 2. 自适应均值滤波
    print("2. 应用自适应均值滤波...")
    try:
        radius = 0.5  # 滤波窗口半径
        curvature_threshold = 0.01  # 边缘保护的曲率阈值
        high_curvature_smooth_factor = 20  # 高曲率区域平滑增强系数
        pcd_filtered = pointCloud_edge_and_smooth_filter(pcd, radius, curvature_threshold, high_curvature_smooth_factor)
    except Exception as e:
        print(f"滤波处理出错: {e}")
        print("跳过滤波步骤，使用原始点云继续处理")
        pcd_filtered = pcd

    # 3. 区域生长分割
    print("3. 应用区域生长分割...")
    try:
        # 确保点云有法向量
        if not pcd_filtered.has_normals():
            print("   计算点云法向量...")
            pcd_filtered.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(30))

        rg = RegionGrowing(pcd_filtered,
                           min_pts_per_cluster=20000,  # 每个聚类的最小点数
                           max_pts_per_cluster=100000000,  # 每个聚类的最大点数
                           neighbour_number=70,  # 邻域搜索点数
                           theta_threshold=5,  # 平滑阈值（角度制）
                           curvature_threshold=0.015)  # 曲率阈值

        clusters = rg.extract()
        if not clusters or len(clusters) == 0:
            print("区域生长分割未能找到任何聚类，将使用整个点云作为单个聚类")
            # 如果分割失败，将所有点视为一个聚类
            clusters = [list(range(len(pcd_filtered.points)))]

        print(f"   区域生长分割得到 {len(clusters)} 个聚类")
    except Exception as e:
        print(f"区域生长分割失败: {e}")
        print("使用简单分割算法...")
        try:
            # 使用DBSCAN进行简单聚类
            points = np.asarray(pcd_filtered.points)
            clustering = DBSCAN(eps=0.3, min_samples=100).fit(points)
            labels = clustering.labels_

            # 统计聚类结果
            unique_labels = np.unique(labels)
            n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
            print(f"找到了 {n_clusters} 个聚类")

            # 将聚类结果转换为索引列表
            clusters = []
            for label in unique_labels:
                if label != -1:  # 忽略噪声点
                    cluster_indices = np.where(labels == label)[0]
                    if len(cluster_indices) >= 100:
                        clusters.append(cluster_indices.tolist())

            if len(clusters) == 0:
                print("所有分割方法都失败，将整个点云作为一个聚类")
                clusters = [list(range(len(pcd_filtered.points)))]
        except Exception as e:
            print(f"简单分割也失败: {e}")
            clusters = [list(range(len(pcd_filtered.points)))]

    # 为合并同类型的边界点准备容器
    all_boundary_points = []
    all_alpha_boundary_points = []
    all_hull_points = []

    # 对每个聚类进行后续处理
    for i, cluster_indices in enumerate(clusters):
        try:
            print(f"处理聚类 {i + 1}/{len(clusters)}...")
            cluster_points = np.asarray(pcd_filtered.points)[cluster_indices]

            # 4. 边界提取
            print(f"4. 应用边界提取...")
            boundary_points = boundary_extraction(cluster_points, resolution=1000)

            if len(boundary_points) == 0:
                print(f"   未提取到边界点，跳过该聚类")
                continue

            # 保存每个聚类的边界点并添加到合并列表
            boundary_file = os.path.join(boundary_output_dir, f"{base_name}_聚类_{i}_边界点.txt")
            save_boundary_points_with_labels(boundary_points, boundary_file, cluster_id=i, class_label=1)

            # 添加到合并列表 - 包含标签信息
            labeled_boundary_points = np.hstack((
                boundary_points,
                np.full((boundary_points.shape[0], 1), 1),  # L1 = 1
                np.full((boundary_points.shape[0], 1), i)  # L2 = 聚类ID
            ))
            all_boundary_points.append(labeled_boundary_points)

            # 添加标签列（聚类ID）
            labels = np.full((boundary_points.shape[0], 1), i)
            boundary_with_label = np.hstack((boundary_points, labels))

            # 5. Alpha边界点滤波
            print(f"5. 应用Alpha边界点滤波...")
            try:
                # 输入格式为XYZL，其中L是标签列
                alpha = 0.05  # Alpha值
                alpha_boundary_points = compute_alpha_shape(boundary_with_label, alpha)

                # 保存Alpha边界点
                alpha_boundary_file = os.path.join(boundary_output_dir, f"{base_name}_聚类_{i}_Alpha边界点.txt")
                save_boundary_points_with_labels(alpha_boundary_points[:, :3], alpha_boundary_file, cluster_id=i,
                                                 class_label=1)

                # 添加到合并列表 - 包含标签信息
                labeled_alpha_points = np.hstack((
                    alpha_boundary_points[:, :3],
                    np.full((alpha_boundary_points.shape[0], 1), 1),  # L1 = 1
                    np.full((alpha_boundary_points.shape[0], 1), i)  # L2 = 聚类ID
                ))
                all_alpha_boundary_points.append(labeled_alpha_points)

                # 6. 点云排序
                print(f"6. 应用点云排序...")
                sorted_points = greedy_nearest_neighbor_sort_kdtree(alpha_boundary_points)

                # 添加序号列O
                num_points = sorted_points.shape[0]
                indices = np.arange(num_points).reshape(-1, 1)
                sorted_points_with_index = np.hstack((sorted_points, indices))

                # 7. 凸包优化提取枢轴点
                print(f"7. 提取枢轴点...")
                hull_points = optimize_boundary(sorted_points_with_index)

                # 保存枢轴点
                hull_file = os.path.join(boundary_output_dir, f"{base_name}_聚类_{i}_枢轴点.txt")
                save_boundary_points_with_labels(hull_points[:, :3], hull_file, cluster_id=i, class_label=1)

                # 添加到合并列表 - 包含标签信息
                labeled_hull_points = np.hstack((
                    hull_points[:, :3],
                    np.full((hull_points.shape[0], 1), 1),  # L1 = 1
                    np.full((hull_points.shape[0], 1), i)  # L2 = 聚类ID
                ))
                all_hull_points.append(labeled_hull_points)

                # 8. 坐标转换和KML文件生成
                print(f"8. 生成KML文件...")

                # 提取坐标部分 (XYZ)
                pivot_coords = hull_points[:, :3]

                # 转换为UTM51N坐标系
                pivot_utm = transform_to_wgs84_utm51n(pivot_coords, x_offset, y_offset, z_offset)

                # 转换为WGS84地理坐标系(经纬度)
                pivot_wgs84 = convert_utm_to_wgs84(pivot_utm)

                # 生成田块KML文件
                field_kml_file = os.path.join(kml_output_dir, f"{base_name}_聚类_{i}.kml")
                try:
                    # 确保KML输出目录存在
                    os.makedirs(os.path.dirname(field_kml_file), exist_ok=True)
                    create_kml_file(pivot_wgs84, field_kml_file, field_name=f"{base_name}_聚类_{i}")
                    print(f"   田块KML文件已保存至: {field_kml_file}")
                except Exception as e:
                    print(f"   创建KML文件失败: {e}")
                    traceback.print_exc()

            except Exception as e:
                print(f"   处理聚类 {i + 1} 时出错: {e}")
                traceback.print_exc()
                continue
        except Exception as e:
            print(f"处理聚类 {i + 1} 时出错: {e}")
            traceback.print_exc()
            continue

    # 保存合并后的边界点文件
    try:
        if all_boundary_points:
            merged_boundary_file = os.path.join(boundary_output_dir, f"{base_name}_所有边界点.txt")
            merged_boundary_points = np.vstack(all_boundary_points)
            np.savetxt(merged_boundary_file, merged_boundary_points, fmt='%.6f %.6f %.6f %d %d')
            print(f"合并的边界点已保存至: {merged_boundary_file}")

        if all_alpha_boundary_points:
            merged_alpha_file = os.path.join(boundary_output_dir, f"{base_name}_所有Alpha边界点.txt")
            merged_alpha_points = np.vstack(all_alpha_boundary_points)
            np.savetxt(merged_alpha_file, merged_alpha_points, fmt='%.6f %.6f %.6f %d %d')
            print(f"合并的Alpha边界点已保存至: {merged_alpha_file}")

        if all_hull_points:
            merged_hull_file = os.path.join(boundary_output_dir, f"{base_name}_所有枢轴点.txt")
            merged_hull_points = np.vstack(all_hull_points)
            np.savetxt(merged_hull_file, merged_hull_points, fmt='%.6f %.6f %.6f %d %d')
            print(f"合并的枢轴点已保存至: {merged_hull_file}")
    except Exception as e:
        print(f"保存合并点云文件时出错: {e}")
        traceback.print_exc()

    print("点云后处理完成！")

    # ===========================================
    # 第十一部分：道路处理
    # ===========================================


def read_road_point_cloud(file_path):
    """
    读取点云数据，格式为xyzl
    """
    data = np.loadtxt(file_path)
    x = data[:, 0]
    y = data[:, 1]
    z = data[:, 2]
    l = data[:, 3] if data.shape[1] > 3 else None
    # 可视化原始点云
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=1, c='b')
    plt.title('Original Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    return x, y, z, l


def project_to_2d(x, y, z):
    """
    将三维点云投影到二维平面，这里选择XY平面
    """
    # 可视化投影后的点云
    plt.figure(figsize=(10, 10))
    plt.scatter(x, y, s=1, c='g')
    plt.title('2D Projected Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()
    return x, y


def generate_density_map(x, y, grid_size=0.5):
    """
    将点云数据栅格化，生成密度图
    """
    x_min, x_max = x.min(), x.max()
    y_min, y_max = y.min(), y.max()
    x_bins = np.arange(x_min, x_max + grid_size, grid_size)
    y_bins = np.arange(y_min, y_max + grid_size, grid_size)
    density, _, _ = np.histogram2d(x, y, bins=[x_bins, y_bins])

    # 可视化密度图
    plt.figure(figsize=(10, 10))
    plt.imshow(density.T, origin='lower', cmap='hot', extent=[x_min, x_max, y_min, y_max])
    plt.title('Density Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.colorbar(label='Point Density')
    plt.axis('equal')
    plt.show()

    return density.T, x_bins, y_bins


def fill_gaps_morphology(binary_image, radius=5):
    """
    使用形态学操作填补缺口
    """
    # 创建结构元素
    selem = disk(radius)
    # 先进行膨胀
    dilated = binary_dilation(binary_image, selem)
    # 再进行闭运算
    closed = binary_closing(dilated, selem)
    return closed


def process_density_map(density, threshold='auto'):
    """
    对密度图进行图像处理，提取骨架
    """
    # 二值化
    if threshold == 'auto':
        thresh = filters.threshold_otsu(density)
    else:
        thresh = threshold
    binary = density > thresh

    # 可视化二值化结果
    plt.figure(figsize=(10, 10))
    plt.imshow(binary, origin='lower', cmap='gray')
    plt.title('Binarized Density Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

    # 使用形态学操作填补小缺口
    binary = fill_gaps_morphology(binary, radius=3)

    # 可视化填补缺口后的二值图
    plt.figure(figsize=(10, 10))
    plt.imshow(binary, origin='lower', cmap='gray')
    plt.title('Binary Map after Gap Filling')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

    # 去除小区域
    binary = remove_small_objects(binary, min_size=10)

    # 骨架提取
    skeleton = skeletonize(binary)

    # 可视化骨架
    plt.figure(figsize=(10, 10))
    plt.imshow(skeleton, origin='lower', cmap='gray')
    plt.title('Skeletonized Map')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.show()

    return skeleton


def skeleton_to_graph(skeleton):
    """
    将骨架图转换为图结构
    """
    G = nx.Graph()
    # 获取骨架图中为 True 的像素坐标
    coords = np.column_stack(np.nonzero(skeleton))
    for y, x in coords:
        G.add_node((y, x))
        # 检查邻居（8邻域）
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                ny, nx_ = y + dy, x + dx
                if 0 <= ny < skeleton.shape[0] and 0 <= nx_ < skeleton.shape[1]:
                    if skeleton[ny, nx_]:
                        G.add_edge((y, x), (ny, nx_))
    return G


def find_endpoints_and_junctions(G):
    """
    识别端点和交叉口
    """
    endpoints = [node for node, degree in G.degree() if degree == 1]  # 末端点 #1
    junctions = [node for node, degree in G.degree() if degree > 2]  # 交叉口 # >2
    return endpoints, junctions


def is_curve(path, angle_threshold=30):
    """
    判断路径是否包含转弯
    """
    if len(path) < 3:
        return False
    # 计算路径上每个点的方向
    directions = []
    for i in range(1, len(path)):
        y0, x0 = path[i - 1]
        y1, x1 = path[i]
        dy, dx = y1 - y0, x1 - x0
        angle = np.arctan2(dy, dx) * 180 / np.pi
        directions.append(angle)
    # 计算方向变化
    direction_changes = np.diff(directions)
    if np.any(np.abs(direction_changes) > angle_threshold):
        return True
    else:
        return False


def extract_paths(G, endpoints, junctions):
    """
    提取道路段，返回路径列表
    """
    from collections import deque

    # 所有关键点
    keypoints = set(endpoints + junctions)
    # 标记已访问的边
    visited_edges = set()
    # 存储道路段
    paths = []

    for node in keypoints:
        neighbors = list(G.neighbors(node))
        for neighbor in neighbors:
            edge = frozenset([node, neighbor])
            if edge in visited_edges:
                continue
            path = [node, neighbor]
            visited_edges.add(edge)
            prev = node
            current = neighbor
            while True:
                neighbors = list(G.neighbors(current))
                neighbors.remove(prev)
                if len(neighbors) == 0:
                    # 到达端点
                    break
                elif current in keypoints and current != node:
                    # 到达另一个关键点
                    break
                else:
                    next_node = neighbors[0]
                    edge = frozenset([current, next_node])
                    if edge in visited_edges:
                        break
                    path.append(next_node)
                    visited_edges.add(edge)
                    prev = current
                    current = next_node
            paths.append(path)
    return paths


def merge_paths(paths, endpoints, junctions):
    """
    根据规则合并路径，返回合并后的道路段列表，并可视化
    """
    road_segments = []
    for idx, path in enumerate(paths):
        start, end = path[0], path[-1]
        # 判断起点和终点的类型
        start_type = 'junction' if start in junctions else 'endpoint'
        end_type = 'junction' if end in junctions else 'endpoint'
        # 判断是否包含转弯
        curve = is_curve(path)
        # 根据规则判断是否合并
        # 可以根据您的具体需求调整规则
        road_segments.append(path)

    # 可视化合并后的道路段
    plt.figure(figsize=(10, 10))
    # 创建颜色映射
    cmap = plt.cm.get_cmap('tab20', len(road_segments))
    for idx, segment in enumerate(road_segments):
        y_coords = [coord[0] for coord in segment]
        x_coords = [coord[1] for coord in segment]
        plt.plot(x_coords, y_coords, color=cmap(idx), linewidth=2)
    plt.title('Merged Road Segments')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.gca().invert_yaxis()  # 如果图像上下颠倒，可以取消这一行
    plt.show()

    return road_segments


def map_back_to_point_cloud(x, y, z, road_segments, x_bins, y_bins, grid_size=0.5, output_dir='output',
                            output_file='segmented_point_cloud.txt'):
    """
    将道路段标签映射回原始点云，基于新的道路段定义，并保存结果
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_file)

    # 构建KD树
    points = np.vstack((x, y)).T
    tree = KDTree(points)
    labels = np.zeros(len(x), dtype=int)

    # 存储所有分段的点云数据
    all_segment_data = []

    for idx, segment in enumerate(road_segments):
        # 获取道路段中的坐标
        y_indices = [coord[0] for coord in segment]
        x_indices = [coord[1] for coord in segment]
        # 检查索引范围
        valid_indices = (np.array(x_indices) >= 0) & (np.array(x_indices) < len(x_bins) - 1) & \
                        (np.array(y_indices) >= 0) & (np.array(y_indices) < len(y_bins) - 1)
        x_indices = np.array(x_indices)[valid_indices]
        y_indices = np.array(y_indices)[valid_indices]
        if len(x_indices) == 0 or len(y_indices) == 0:
            continue  # 跳过空的索引列表
        # 计算实际坐标
        x_coords = x_bins[x_indices] + grid_size / 2
        y_coords = y_bins[y_indices] + grid_size / 2
        segment_points = np.vstack((x_coords, y_coords)).T
        # 在原始点云中查找邻近点
        indices = tree.query_radius(segment_points, r=grid_size)
        indices = [item for sublist in indices for item in sublist]
        labels[indices] = idx + 1  # 道路段ID从1开始

    # 可视化映射后的结果
    plt.figure(figsize=(10, 10))
    scatter = plt.scatter(x, y, c=labels, cmap='tab20', s=1)
    plt.title('Mapped Road Segments to Original Point Cloud')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.axis('equal')
    plt.colorbar(label='Segment Labels')
    plt.show()

    # 将结果保存为txt文件，格式为 x y z label
    data = np.column_stack((x, y, z, labels))
    np.savetxt(output_path, data, fmt='%.6f %.6f %.6f %d')

    # 为每个道路段收集点，并按照 x y z L1 L2 格式保存，其中L1为6（道路类别），L2=道路段标签
    all_segments_data = []
    unique_labels = np.unique(labels)
    for lbl in unique_labels:
        if lbl == 0:
            continue  # 跳过未标记的点
        idx = labels == lbl
        # 创建新的数据格式：x y z L1 L2，其中L1=6（道路类别），L2=道路段标签
        segment_data = np.column_stack((x[idx], y[idx], z[idx], np.full(np.sum(idx), 6), labels[idx]))
        all_segments_data.append(segment_data)

    # 合并所有分段点云数据并保存为一个文件
    if all_segments_data:
        combined_segments = np.vstack(all_segments_data)
        combined_file = os.path.join(output_dir, 'all_road_segments.txt')
        np.savetxt(combined_file, combined_segments, fmt='%.6f %.6f %.6f %d %d')
        print(f"所有道路段点云已合并保存到: {combined_file}")

    print(f"分割后的点云已保存到 {output_path}")

    return labels


def load_point_cloud_from_txt(file_path):
    """
    从txt文件加载点云数据
    """
    data = np.loadtxt(file_path)
    # 检查数据维度，确保即使只有4列也能正常工作
    if data.shape[1] == 4:  # x, y, z, label
        # 扩展为5列，最后一列全为0
        extended_data = np.zeros((data.shape[0], 5))
        extended_data[:, :4] = data
        return extended_data
    elif data.shape[1] >= 5:  # x, y, z, label1, label2, ...
        return data[:, :5]  # 仅取前5列
    else:
        raise ValueError(f"无效的数据格式，需要至少4列 (x,y,z,label)，当前有 {data.shape[1]} 列")


def load_point_cloud(file_path):
    """
    加载点云数据。
    假设数据以空格分隔，格式为：x y z a b
    """
    try:
        data = np.loadtxt(file_path)
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        a = data[:, 3] if data.shape[1] > 3 else np.zeros(len(x))
        b = data[:, 4] if data.shape[1] > 4 else np.zeros(len(x))
        points = data[:, :3]
        return points, a, b
    except Exception as e:
        print(f"加载点云数据失败: {e}")
        return None, None, None


def fit_curve(points):
    """
    使用多项式曲线拟合优化枢轴点位置
    """
    x = points[:, 0]
    y = points[:, 1]

    # 使用2次多项式进行拟合
    coefficients = np.polyfit(x, y, 2)
    polynomial = np.poly1d(coefficients)

    # 生成拟合曲线
    y_fit = polynomial(x)

    # 返回优化后的点，保留原始的信息
    if points.shape[1] >= 5:
        fitted_points = np.column_stack((x, y_fit, points[:, 2], points[:, 3], points[:, 4]))
    else:
        # 如果点云不包含足够的列，使用零填充
        fitted_points = np.column_stack((x, y_fit, points[:, 2], np.zeros(len(x)), np.zeros(len(x))))

    return fitted_points


def save_points_to_txt(points, file_path):
    """
    保存点云数据到txt文件
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    np.savetxt(file_path, points, fmt='%.6f', delimiter=' ', comments='')
    print(f"已保存点云数据到 {file_path}")


def separate_points(points, a, b):
    """
    根据标签 'a' 分离道路和农田点。
    a=6 代表道路，a=1 代表农田
    """
    road_mask = a == 6
    farmland_mask = a == 1

    road_points = points[road_mask]
    farmland_points = points[farmland_mask]
    farmland_labels = b[farmland_mask]

    print(f"Number of road points: {road_points.shape[0]}")
    print(f"Number of farmland points: {farmland_points.shape[0]}")

    return road_points, farmland_points, farmland_labels


def extract_road_skeleton(road_points, grid_size=0.5):
    """
    使用骨架化方法提取道路的骨架（中心线）。
    """
    # 转换为二维（忽略z轴）
    xy = road_points[:, :2]

    # 创建二维栅格
    min_xy = xy.min(axis=0) - grid_size  # 扩展边界以避免边缘问题
    max_xy = xy.max(axis=0) + grid_size
    grid_shape = ((max_xy - min_xy) / grid_size).astype(int) + 1
    grid = np.zeros(grid_shape, dtype=np.uint8)

    # 填充栅格
    indices = ((xy - min_xy) / grid_size).astype(int)
    # 确保索引在有效范围内
    valid_mask = (indices[:, 0] >= 0) & (indices[:, 0] < grid_shape[0]) & (indices[:, 1] >= 0) & (
            indices[:, 1] < grid_shape[1])
    indices = indices[valid_mask]
    grid[indices[:, 0], indices[:, 1]] = 1

    # 进行骨架化
    skeleton = skeletonize(grid).astype(np.uint8)

    # 提取骨架点坐标
    skeleton_coords = np.argwhere(skeleton > 0)
    skeleton_points = skeleton_coords * grid_size + min_xy

    print(f"Number of skeleton points extracted: {skeleton_points.shape[0]}")
    return skeleton_points


def build_road_graph(skeleton_points, distance_threshold=0.9):
    """
    根据骨架点构建道路网络图。
    节点代表骨架点；边连接距离在阈值内的点。
    """
    G = nx.Graph()

    # 添加节点
    for idx, point in enumerate(skeleton_points):
        G.add_node(idx, pos=point)

    # 使用KD树寻找邻近点，添加边
    tree = cKDTree(skeleton_points)
    pairs = tree.query_pairs(r=distance_threshold)

    for i, j in pairs:
        G.add_edge(i, j)

    print(f"Road network graph nodes: {G.number_of_nodes()}")
    print(f"Road network graph edges: {G.number_of_edges()}")
    return G


def find_potential_entrances(G, skeleton_points, farmland_points, association_radius=3.0):
    """
    通过筛选度数大于2或等于1且附近有农田的节点，识别潜在出入口。
    """
    entrances = []
    tree_farmland = cKDTree(farmland_points[:, :2])

    for node, degree in G.degree():
        if degree > 2 or degree == 1:
            pos = G.nodes[node]['pos']
            # 检查出入口附近是否有农田点
            indices = tree_farmland.query_ball_point(pos[:2], r=association_radius)
            if indices:
                entrances.append(pos)

    print(f"Number of potential entrances (degree > 2 or degree == 1 and near farmland): {len(entrances)}")
    return np.array(entrances)


def cluster_entrances(entrances, eps=2.5, min_samples=1):
    """
    使用DBSCAN聚类算法合并重复识别的出入口点。
    """
    if len(entrances) == 0:
        return np.array([])

    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(entrances)
    labels = clustering.labels_

    unique_labels = set(labels)
    clustered_entrances = []

    for label in unique_labels:
        if label == -1:
            # 噪声点
            clustered_entrances.append(entrances[labels == label][0])
        else:
            cluster = entrances[labels == label]
            centroid = cluster.mean(axis=0)
            clustered_entrances.append(centroid)

    clustered_entrances = np.array(clustered_entrances)
    print(f"Number of entrances after clustering: {clustered_entrances.shape[0]}")
    return clustered_entrances


def compute_angle(skeleton_points, entrance, neighborhood_size=10):
    """
    计算出入口点与道路骨架的夹角（单位：度）。
    使用向量点积的方法来计算夹角，确保准确性。
    """
    # 找到最近的邻域点
    distances = np.linalg.norm(skeleton_points - entrance, axis=1)
    idx = np.argsort(distances)[:neighborhood_size]
    neighborhood = skeleton_points[idx]

    if neighborhood.shape[0] < 2:
        return None  # 无法计算角度

    # 计算邻域点的方向向量（从第一个点到最后一个点）
    direction_vector = neighborhood[-1] - neighborhood[0]
    if np.linalg.norm(direction_vector) == 0:
        return None  # 无法计算方向

    road_direction = direction_vector / np.linalg.norm(direction_vector)

    # 计算出入口与道路的连接向量
    connection_vector = entrance[:2] - neighborhood[-1][:2]
    if np.linalg.norm(connection_vector) == 0:
        return None  # 无法计算方向

    entrance_direction = connection_vector / np.linalg.norm(connection_vector)

    # 计算两个方向向量之间的夹角
    dot_product = np.dot(road_direction, entrance_direction)
    # 防止数值误差导致的超出范围
    dot_product = np.clip(dot_product, -1.0, 1.0)
    angle_diff_rad = np.arccos(dot_product)
    angle_diff = np.degrees(angle_diff_rad)

    return angle_diff


def compute_geometric_features(G, skeleton_points, entrances, farmland_points, farmland_labels, neighborhood_size=10):
    """
    为每个潜在出入口点计算几何特征（曲率、道路方向变化、角度差）及网络特征（介数中心性、接近中心性）。
    并为每个出入口分配一个序号。
    """
    features = []
    # 计算全图的介数中心性和接近中心性
    print("Calculating graph centrality measures...")
    betweenness = nx.betweenness_centrality(G, normalized=True)
    closeness = nx.closeness_centrality(G)

    # 创建一个字典，将节点索引映射到中心性
    node_centrality = {node: {'betweenness': betweenness.get(node, 0), 'closeness': closeness.get(node, 0)} for node in
                       G.nodes()}

    # 创建KD树用于计算局部道路密度
    tree_skeleton = cKDTree(skeleton_points[:, :2])

    # 创建KD树用于计算出入口到农田的距离
    tree_farmland = cKDTree(farmland_points[:, :2])

    for idx, entrance in enumerate(entrances, 1):
        # 找到最近的邻域点
        distances = np.linalg.norm(skeleton_points - entrance, axis=1)
        idx_neighborhood = np.argsort(distances)[:neighborhood_size]
        neighborhood = skeleton_points[idx_neighborhood]

        if neighborhood.shape[0] < 3:
            curvature = 0
            direction_change = 0
            angle_diff = 180  # 默认最大角度
        else:
            x = neighborhood[:, 0]
            y = neighborhood[:, 1]
            try:
                coeffs = np.polyfit(x, y, 2)
                curvature = 2 * coeffs[0]  # 二次项系数的两倍
                direction_change = abs(coeffs[0])
            except np.RankWarning:
                curvature = 0
                direction_change = 0

            # 计算角度差
            angle_diff = compute_angle(skeleton_points, entrance, neighborhood_size=neighborhood_size)
            angle_diff = angle_diff if angle_diff is not None else 180

        # 获取节点在图中的索引
        # 由于clustered_entrances是道路骨架的点的聚类结果，可能不直接对应到图中的节点
        # 需要找到最近的图节点
        distances_to_nodes = np.linalg.norm(skeleton_points - entrance, axis=1)
        nearest_node_idx = np.argmin(distances_to_nodes)
        betweenness_centrality = node_centrality.get(nearest_node_idx, {}).get('betweenness', 0)
        closeness_centrality = node_centrality.get(nearest_node_idx, {}).get('closeness', 0)

        # 计算局部道路密度（在一定半径内的道路点数量）
        local_density_radius = 1.0  # 可根据需要调整
        local_density = len(tree_skeleton.query_ball_point(entrance[:2], r=local_density_radius))

        # 计算出入口到最近农田的距离
        distance_to_farmland, _ = tree_farmland.query(entrance[:2], k=1)

        features.append([
            curvature,
            direction_change,
            angle_diff,
            betweenness_centrality,
            closeness_centrality,
            local_density,
            distance_to_farmland
        ])

        print(f"Entrance {idx}: Coordinates {entrance[:2]}: Curvature={curvature}, "
              f"Direction Change={direction_change}, Angle Diff={angle_diff}, "
              f"Betweenness Centrality={betweenness_centrality:.4f}, "
              f"Closeness Centrality={closeness_centrality:.4f}, "
              f"Local Density={local_density}, Distance to Nearest Farmland={distance_to_farmland:.2f}")

    features = np.array(features)
    print(f"Computed geometric and network features shape: {features.shape}")
    return features


def filter_entrances_by_features(entrances, features,
                                 curvature_threshold=0.001,
                                 direction_change_threshold=0.002,
                                 angle_min=3, angle_max=150,
                                 betweenness_threshold=0.01,
                                 closeness_threshold_min=0.0008,
                                 closeness_threshold_max=0.0015,
                                 local_density_min=1,
                                 distance_to_farmland_max=4.7):
    """
    根据几何特征、网络特征和距离阈值筛选出更可能的出入口。
    """
    selected_indices = (
            (features[:, 0] > curvature_threshold) &
            (features[:, 1] > direction_change_threshold) &
            (features[:, 2] > angle_min) &
            (features[:, 2] < angle_max) &
            (features[:, 3] < betweenness_threshold) &  # 低介数中心性
            (features[:, 4] > closeness_threshold_min) &
            (features[:, 4] < closeness_threshold_max) &  # 高接近中心性
            (features[:, 5] > local_density_min) &  # 高局部道路密度
            (features[:, 6] < distance_to_farmland_max)  # 近距离到农田
    )
    final_entrances = entrances[selected_indices]
    print(f"Number of entrances after filtering based on all features: {final_entrances.shape[0]}")
    return final_entrances


def associate_entrances_with_farmland(entrances, farmland_points, farmland_labels, association_radius=5.0,
                                      max_associations=2):
    """
    将出入口与附近的农田进行关联。
    允许一个出入口关联最多2块农田，并基于距离选择最近的农田。
    """
    farmland_tree = cKDTree(farmland_points[:, :2])
    associations = []

    for entrance in entrances:
        # 查询出入口点附近的农田边界点
        indices = farmland_tree.query_ball_point(entrance[:2], r=association_radius)
        if indices:
            associated_labels = farmland_labels[indices]
            associated_points = farmland_points[indices]
            # 计算距离
            distances = np.linalg.norm(associated_points[:, :2] - entrance[:2], axis=1)
            # 获取排序索引
            sorted_indices = np.argsort(distances)
            sorted_labels = associated_labels[sorted_indices]
            sorted_distances = distances[sorted_indices]
            # 去除重复标签，确保每个农田只关联一次
            unique_labels = []
            for label in sorted_labels:
                if label not in unique_labels:
                    unique_labels.append(label)
                if len(unique_labels) == max_associations:
                    break
            associations.append((entrance, unique_labels))
        else:
            associations.append((entrance, None))

    associated_count = sum(1 for assoc in associations if assoc[1] is not None)
    print(f"Number of entrances associated with farmland: {associated_count}")
    return associations


def associate_farmlands_with_entrances(entrances, farmland_points, farmland_labels, association_radius=5.0):
    """
    将农田与附近的出入口进行关联。
    每块农田关联到最近的出入口，前提是该出入口在association_radius范围内。
    """
    entrances_tree = cKDTree(entrances[:, :2])
    distances, indices = entrances_tree.query(farmland_points[:, :2], distance_upper_bound=association_radius)

    # 初始化每个入口的关联列表
    entrance_associations = [[] for _ in range(len(entrances))]

    for farmland_idx, (distance, entrance_idx) in enumerate(zip(distances, indices)):
        if entrance_idx != len(entrances):
            entrance_associations[entrance_idx].append(farmland_labels[farmland_idx])

    return entrance_associations


def visualize_final_results(farmland_points, road_points, skeleton_points, entrances, associations,
                            farmland_labels):
    """
    可视化最终识别出的出入口及其与农田的关联。
    为每个出入口添加序号标注，并用不同颜色表示是否关联农田。
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(farmland_points[:, 0], farmland_points[:, 1], c='green', s=10, label='Farmland')
    plt.scatter(road_points[:, 0], road_points[:, 1], c='gray', s=10, label='Road Skeleton')

    for idx, (entrance, farmland_ids) in enumerate(associations, 1):
        if farmland_ids is not None and len(farmland_ids) > 0:
            plt.plot(entrance[0], entrance[1], 'ro')  # 红点表示出入口
            # 显示序号
            plt.text(entrance[0], entrance[1], f'{idx}', color='white', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))
            # 标注关联的农田ID
            for farmland_id in farmland_ids:
                farmland_idx = np.where(farmland_labels == farmland_id)[0]
                if len(farmland_idx) > 0:
                    farmland_point = farmland_points[farmland_idx[0]]
                    plt.plot([entrance[0], farmland_point[0]], [entrance[1], farmland_point[1]], 'r--',
                             linewidth=0.5)
        else:
            plt.plot(entrance[0], entrance[1], 'kx')  # 黑叉表示未关联
            plt.text(entrance[0], entrance[1], f'{idx}', color='white', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='black', alpha=0.5, edgecolor='none'))

    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title('Identified Entrances and Farmland Associations')
    plt.grid(True)
    plt.show()


def visualize_points(title, farmland_points, road_points, skeleton_points=None, entrances=None, candidates=None):
    """
    通用的可视化函数，用于不同处理阶段。
    为每个潜在出入口添加序号标注。
    """
    plt.figure(figsize=(12, 10))
    plt.scatter(farmland_points[:, 0], farmland_points[:, 1], c='green', s=10, label='Farmland')
    plt.scatter(road_points[:, 0], road_points[:, 1], c='gray', s=10, label='Road')

    if skeleton_points is not None:
        plt.scatter(skeleton_points[:, 0], skeleton_points[:, 1], c='blue', s=10, label='Road Skeleton')

    if candidates is not None:
        plt.scatter(candidates[:, 0], candidates[:, 1], c='yellow', s=30, label='Entrance Candidates')

    if entrances is not None:
        plt.scatter(entrances[:, 0], entrances[:, 1], c='red', s=50, label='Potential Entrances')
        for idx, entrance in enumerate(entrances, 1):
            plt.text(entrance[0], entrance[1], f'{idx}', color='white', fontsize=12,
                     ha='center', va='center', bbox=dict(facecolor='red', alpha=0.5, edgecolor='none'))

    plt.legend()
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(title)
    plt.grid(True)
    plt.show()


def visualize_features(features):
    """
    可视化几何特征和网络特征的分布。
    """
    feature_names = ['Curvature', 'Direction Change', 'Angle Difference',
                     'Betweenness Centrality', 'Closeness Centrality',
                     'Local Road Density', 'Distance to Nearest Farmland']
    plt.figure(figsize=(21, 12))

    for i in range(features.shape[1]):
        plt.subplot(3, 3, i + 1)
        plt.hist(features[:, i], bins=30, color='blue', alpha=0.7)
        plt.xlabel(feature_names[i])
        plt.ylabel('Frequency')
        plt.title(f'Distribution of {feature_names[i]}')

    plt.tight_layout()
    plt.show()


# ===========================================
# 第十二部分：合并农田和道路点云数据
# ===========================================
def merge_road_farm_points(farm_file, road_file, output_file):
    """
    合并农田和道路点云数据。
    农田数据格式：x y z 1 L2
    道路数据格式：x y z 6 L2

    参数:
    - farm_file: 农田点云文件路径（所有Alpha边界点.txt）
    - road_file: 道路点云文件路径（all_road_segments.txt）
    - output_file: 输出文件路径
    """
    try:
        # 加载农田点云数据
        farm_data = np.loadtxt(farm_file)
        print(f"加载农田点云，共{farm_data.shape[0]}个点，数据维度：{farm_data.shape[1]}")

        # 加载道路点云数据
        road_data = np.loadtxt(road_file)
        print(f"加载道路点云，共{road_data.shape[0]}个点，数据维度：{road_data.shape[1]}")

        # 合并两个点云数据
        merged_data = np.vstack((farm_data, road_data))
        print(f"合并后点云共{merged_data.shape[0]}个点")

        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 保存合并后的点云数据
        np.savetxt(output_file, merged_data, fmt='%.6f %.6f %.6f %d %d')
        print(f"合并点云已保存至: {output_file}")

        return output_file
    except Exception as e:
        print(f"合并点云数据失败: {e}")
        traceback.print_exc()
        return None


# ===========================================
# 第十三部分：集成处理流程
# ===========================================
def road_segmentation(input_file, output_dir='output', grid_size=0.45, pivot_detection=True):
    """
    道路分割与枢纽点检测的完整流程
    参数:
    - input_file: 输入点云文件路径
    - output_dir: 输出文件目录
    - grid_size: 栅格大小
    - pivot_detection: 是否执行枢纽点检测
    """
    print("===== 开始道路分割与枢纽点检测流程 =====")
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 1. 道路实例化
    print("\n== 第一阶段: 道路实例化 ==")

    # 读取点云数据
    print("正在读取点云数据...")
    x, y, z, l = read_road_point_cloud(input_file)

    # 投影到二维平面
    print("正在投影到二维平面...")
    x_2d, y_2d = project_to_2d(x, y, z)

    # 生成密度图
    print("正在生成密度图...")
    density, x_bins, y_bins = generate_density_map(x_2d, y_2d, grid_size=grid_size)

    # 图像处理与骨架提取
    print("正在提取道路骨架...")
    skeleton = process_density_map(density)

    # 构建道路网络图
    print("正在构建道路网络图...")
    G = skeleton_to_graph(skeleton)

    # 识别端点和交叉口
    print("正在识别端点和交叉口...")
    endpoints, junctions = find_endpoints_and_junctions(G)

    # 提取道路段
    print("正在提取道路段...")
    paths = extract_paths(G, endpoints, junctions)

    # 合并道路段并可视化
    print("正在合并道路段...")
    road_segments = merge_paths(paths, endpoints, junctions)

    # 映射回原始点云并保存结果
    print("正在映射回原始点云并保存结果...")
    labels = map_back_to_point_cloud(
        x_2d, y_2d, z, road_segments, x_bins, y_bins,
        grid_size=grid_size / 5,  # 使用更小的邻域半径
        output_dir=output_dir,
        output_file='segmented_point_cloud.txt'
    )

    # 2. 枢纽点检测 (对每个道路段进行处理)
    if pivot_detection:
        print("\n== 第二阶段: 枢纽点检测 ==")

        # 读取合并后的点云文件
        all_segments_file = os.path.join(output_dir, 'all_road_segments.txt')

        # 创建一个列表来存储所有枢纽点
        all_pivot_points = []

        # 为每个道路段ID分别处理
        unique_labels = np.unique(labels)
        for lbl in unique_labels:
            if lbl == 0:
                continue  # 跳过未标记的点

            print(f"正在处理道路段 {lbl}...")

            # 提取该道路段的点
            idx = labels == lbl
            segment_data = np.column_stack(
                (x_2d[idx], y_2d[idx], z[idx], np.full(np.sum(idx), 6), np.full(np.sum(idx), lbl)))

            try:
                # 检测拐角与转折点
                print(f"道路段 {lbl}: 正在检测拐角和转折点...")

                # 优化边界：使用凸包提取边界点
                print(f"道路段 {lbl}: 正在优化边界...")
                hull_points = optimize_boundary(segment_data)

                # 将凸包点作为枢轴点并添加道路段ID
                hull_points_with_id = np.column_stack((hull_points, np.full(hull_points.shape[0], lbl)))
                all_pivot_points.append(hull_points_with_id)

            except Exception as e:
                print(f"处理道路段 {lbl} 时出错: {str(e)}")

        # 合并所有道路段的枢纽点并保存
        if all_pivot_points:
            combined_pivot_points = np.vstack(all_pivot_points)
            combined_pivot_file = os.path.join(output_dir, 'all_pivot_points.txt')
            np.savetxt(combined_pivot_file, combined_pivot_points, fmt='%.6f', delimiter=' ', comments='')
            print(f"\n所有枢纽点已合并保存到: {combined_pivot_file}")

        print(f"\n枢纽点检测完成。共处理了 {len(unique_labels) - (1 if 0 in unique_labels else 0)} 个道路段。")

    print("\n===== 道路分割与枢纽点检测流程完成 =====")
    print(f"输出文件保存在: {output_dir}")
    print(f"- 完整点云（带标签）: segmented_point_cloud.txt")
    print(f"- 所有道路段合并: all_road_segments.txt")
    if pivot_detection:
        print(f"- 所有枢纽点合并: all_pivot_points.txt")

    return labels


def analyze_road_farm_topology(point_cloud_file, output_dir='topology_results', grid_size=0.5):
    """
    分析道路与农田的拓扑关系
    参数:
    - point_cloud_file: 道路和农田点云文件路径
    - output_dir: 输出目录
    - grid_size: 栅格大小
    """
    print("===== 开始分析道路与农田的拓扑关系 =====")

    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)

    # 步骤 1: 加载点云数据
    print("正在加载点云数据...")
    points, a, b = load_point_cloud(point_cloud_file)
    if points is None:
        print("加载点云数据失败，退出分析。")
        return None

    # 步骤 2: 分离道路和农田点
    print("正在分离道路和农田点...")
    road_points, farmland_points, farmland_labels = separate_points(points, a, b)

    # 可视化初始数据分布
    visualize_points("道路和农田分布", farmland_points, road_points)

    # 步骤 3: 提取道路骨架
    print("正在提取道路骨架...")
    skeleton_points = extract_road_skeleton(road_points, grid_size=grid_size)

    # 可视化道路骨架
    visualize_points("道路骨架和农田分布", farmland_points, road_points, skeleton_points=skeleton_points)

    # 步骤 4: 构建道路网络图
    print("正在构建道路网络图...")
    G = build_road_graph(skeleton_points, distance_threshold=0.9)

    # 步骤 5: 识别潜在出入口
    print("正在识别潜在出入口...")
    potential_entrances = find_potential_entrances(G, skeleton_points, farmland_points, association_radius=3.0)

    # 步骤 6: 聚类合并重复识别的出入口
    print("正在聚类合并重复出入口...")
    clustered_entrances = cluster_entrances(potential_entrances, eps=2.5, min_samples=1)

    # 可视化聚类后的出入口
    visualize_points("聚类后的潜在出入口分布", farmland_points, road_points, skeleton_points=skeleton_points,
                     entrances=clustered_entrances)

    # 步骤 7: 计算潜在出入口的几何特征和网络特征
    print("正在计算几何和网络特征...")
    features = compute_geometric_features(G, skeleton_points, clustered_entrances, farmland_points, farmland_labels,
                                          neighborhood_size=10)

    # 可视化特征分布
    visualize_features(features)

    # 步骤 8: 根据特征筛选最终出入口
    print("正在根据特征筛选出入口...")
    final_entrances = filter_entrances_by_features(
        clustered_entrances,
        features,
        curvature_threshold=0.39,
        direction_change_threshold=0.01,
        angle_min=3,
        angle_max=200,
        betweenness_threshold=0.0012,
        closeness_threshold_min=0.0008,
        closeness_threshold_max=0.0015,
        local_density_min=1,
        distance_to_farmland_max=4.7
    )

    # 可视化筛选后的出入口
    visualize_points("筛选后的出入口分布", farmland_points, road_points, skeleton_points=skeleton_points,
                     entrances=final_entrances)

    # 步骤 9: 关联出入口与农田
    print("正在关联出入口与农田...")
    associations_entrance_to_farmland = associate_entrances_with_farmland(
        final_entrances, farmland_points, farmland_labels, association_radius=5.0, max_associations=2
    )
    associations_farmland_to_entrance = associate_farmlands_with_entrances(
        final_entrances, farmland_points, farmland_labels, association_radius=5.0
    )

    # 合并两个关联结果
    for i in range(len(associations_entrance_to_farmland)):
        entrance, farmland_ids = associations_entrance_to_farmland[i]
        additional_farmland_ids = associations_farmland_to_entrance[i]
        if additional_farmland_ids:
            # 合并农田ID
            combined_farmland_ids = farmland_ids + additional_farmland_ids if farmland_ids else additional_farmland_ids
            # 去除重复的农田ID
            combined_farmland_ids = list(set(combined_farmland_ids))
            # 限制最大关联数
            if len(combined_farmland_ids) > 2:
                combined_farmland_ids = combined_farmland_ids[:2]
            # 更新关联
            associations_entrance_to_farmland[i] = (entrance, combined_farmland_ids)

    associations = associations_entrance_to_farmland

    # 步骤 10: 输出出入口与农田的关联信息
    print("\n出入口与农田的关联结果：")
    entrances_output_file = os.path.join(output_dir, 'entrance_farmland_associations.txt')
    with open(entrances_output_file, 'w', encoding='utf-8') as f:
        for idx, (entrance, farmland_ids) in enumerate(associations, 1):
            if farmland_ids is not None and len(farmland_ids) > 0:
                ids_str = ', '.join(map(str, farmland_ids))
                info = f"出入口 {idx}: 坐标 ({entrance[0]:.2f}, {entrance[1]:.2f}) 关联农田ID {ids_str}"
                print(info)
                f.write(info + '\n')
            else:
                info = f"出入口 {idx}: 坐标 ({entrance[0]:.2f}, {entrance[1]:.2f}) 未关联农田"
                print(info)
                f.write(info + '\n')

    # 统计关联的农田数量及ID
    all_associated_farmlands = []
    for _, farmland_ids in associations:
        if farmland_ids:
            all_associated_farmlands.extend(farmland_ids)

    unique_associated_farmlands = set(all_associated_farmlands)
    all_farmland_ids = set(farmland_labels)
    unassociated_farmlands = all_farmland_ids - unique_associated_farmlands

    # 打印统计信息
    print(f"\n农田统计信息:")
    print(f"- 农田总数: {len(all_farmland_ids)}")
    print(f"- 已关联出入口的农田数: {len(unique_associated_farmlands)}")
    print(f"- 未关联出入口的农田数: {len(unassociated_farmlands)}")
    if unassociated_farmlands:
        print(f"- 未关联农田ID: {sorted(unassociated_farmlands)}")

    # 尝试关联未关联的农田
    if unassociated_farmlands:
        print("\n正在尝试关联未关联的农田...")
        unassociated_farmlands = sorted(unassociated_farmlands)

        # 创建出入口当前关联数映射
        entrance_current_associations = [len(assoc[1]) if assoc[1] else 0 for assoc in associations]

        # 为每个未关联的农田寻找最近的出入口
        for farmland_id in unassociated_farmlands:
            farmland_idx = np.where(farmland_labels == farmland_id)[0]
            if len(farmland_idx) == 0:
                print(f"找不到农田ID {farmland_id} 的点。")
                continue

            farmland_point = farmland_points[farmland_idx[0]][:2]
            entrance_distances = np.linalg.norm(final_entrances[:, :2] - farmland_point, axis=1)
            nearest_entrance_idx = np.argmin(entrance_distances)
            nearest_distance = entrance_distances[nearest_entrance_idx]

            # 检查出入口是否可以接受更多关联
            if entrance_current_associations[nearest_entrance_idx] < 2:
                # 为该出入口添加农田
                associations[nearest_entrance_idx][1].append(farmland_id)
                entrance_current_associations[nearest_entrance_idx] += 1
                print(
                    f"农田ID {farmland_id} 关联到出入口 {nearest_entrance_idx + 1} (距离: {nearest_distance:.2f}m)")
            else:
                # 如果最近的出入口已满，寻找下一个最近的
                sorted_indices = np.argsort(entrance_distances)
                assigned = False
                for idx in sorted_indices[1:]:
                    if entrance_current_associations[idx] < 2:
                        associations[idx][1].append(farmland_id)
                        entrance_current_associations[idx] += 1
                        print(f"农田ID {farmland_id} 关联到出入口 {idx + 1} (距离: {entrance_distances[idx]:.2f}m)")
                        assigned = True
                        break
                if not assigned:
                    print(f"农田ID {farmland_id} 无法关联到任何出入口（所有出入口已满）。")

    # 更新统计信息
    all_associated_farmlands = []
    for _, farmland_ids in associations:
        if farmland_ids:
            all_associated_farmlands.extend(farmland_ids)

    unique_associated_farmlands = set(all_associated_farmlands)
    unassociated_farmlands = all_farmland_ids - unique_associated_farmlands

    print(f"\n关联处理后的农田统计信息:")
    print(f"- 已关联出入口的农田数: {len(unique_associated_farmlands)}")
    print(f"- 未关联出入口的农田数: {len(unassociated_farmlands)}")
    if unassociated_farmlands:
        print(f"- 未关联农田ID: {sorted(unassociated_farmlands)}")
    else:
        print("- 所有农田已成功关联到出入口。")

    # 可视化最终结果
    visualize_final_results(farmland_points, road_points, skeleton_points, final_entrances, associations,
                            farmland_labels)

    # 保存最终结果
    final_output_file = os.path.join(output_dir, 'final_associations.txt')
    # 检查final_entrances的形状并相应调整fmt
    if final_entrances.shape[1] > 3:
        # 只保存XYZ坐标（前三列）
        np.savetxt(final_output_file, final_entrances[:, :3], header="X Y Z", fmt="%.6f %.6f %.6f")
    else:
        np.savetxt(final_output_file, final_entrances, header="X Y Z", fmt="%.6f " * final_entrances.shape[1])
    print(f"\n最终关联结果已保存到: {entrances_output_file}")
    print(f"最终出入口坐标已保存到: {final_output_file}")

    print("\n===== 道路与农田的拓扑关系分析完成 =====")
    return associations


# ===========================================
# 第十四部分：PLY文件合并处理
# ===========================================
def read_ply_file(filename):
    """
    读取PLY文件，返回点云数据
    支持读取包含xyz、rgb、class、pred等字段的PLY文件
    """
    # PLY数据类型定义
    ply_dtypes = dict([
        (b'int8', 'i1'), (b'char', 'i1'),
        (b'uint8', 'u1'), (b'uchar', 'u1'),
        (b'int16', 'i2'), (b'short', 'i2'),
        (b'uint16', 'u2'), (b'ushort', 'u2'),
        (b'int32', 'i4'), (b'int', 'i4'),
        (b'uint32', 'u4'), (b'uint', 'u4'),
        (b'float32', 'f4'), (b'float', 'f4'),
        (b'float64', 'f8'), (b'double', 'f8')
    ])
    
    valid_formats = {'ascii': '', 'binary_big_endian': '>', 'binary_little_endian': '<'}
    
    with open(filename, 'rb') as plyfile:
        # 检查文件是否以ply开头
        if b'ply' not in plyfile.readline():
            raise ValueError('文件不是有效的PLY格式')
        
        # 获取编码格式
        fmt = plyfile.readline().split()[1].decode()
        if fmt == "ascii":
            raise ValueError('仅支持二进制PLY文件')
        
        ext = valid_formats[fmt]
        
        # 解析头部
        line = []
        properties = []
        num_points = None
        
        while b'end_header' not in line and line != b'':
            line = plyfile.readline()
            
            if b'element vertex' in line:
                line = line.split()
                num_points = int(line[2])
            
            elif b'property' in line:
                line = line.split()
                properties.append((line[2].decode(), ext + ply_dtypes[line[1]]))
        
        # 读取数据
        data = np.fromfile(plyfile, dtype=properties, count=num_points)
    
    return data


def merge_original_and_prediction(original_ply_path, prediction_ply_path, output_txt_path):
    """
    合并原始点云和预测结果，生成用于总后处理的点云文件
    
    参数:
    - original_ply_path: 原始点云PLY文件路径 (包含xyz, rgb, class)
    - prediction_ply_path: 预测结果PLY文件路径 (包含pred, label等)
    - output_txt_path: 输出TXT文件路径
    
    输出格式: x y z L1 L2
    - L1: 预测的类别标签 (1=农田, 6=道路等)
    - L2: 实例ID (初始设为0，后续处理中会更新)
    """
    print(f"正在读取原始点云: {original_ply_path}")
    original_data = read_ply_file(original_ply_path)
    
    print(f"正在读取预测结果: {prediction_ply_path}")
    prediction_data = read_ply_file(prediction_ply_path)
    
    # 检查点数是否一致
    if len(original_data) != len(prediction_data):
        raise ValueError(f"点云数量不匹配: 原始={len(original_data)}, 预测={len(prediction_data)}")
    
    print(f"点云数量: {len(original_data)}")
    
    # 提取坐标 (xyz)
    x = original_data['x']
    y = original_data['y']
    z = original_data['z']
    
    # 提取预测标签
    # 预测文件中可能有'pred'或'class'字段
    if 'pred' in prediction_data.dtype.names:
        pred_labels = prediction_data['pred']
    elif 'class' in prediction_data.dtype.names:
        pred_labels = prediction_data['class']
    else:
        raise ValueError("预测文件中未找到'pred'或'class'字段")
    
    # L1 = 预测类别, L2 = 实例ID (初始为0)
    L1 = pred_labels.astype(int)
    L2 = np.zeros(len(x), dtype=int)
    
    # 合并数据
    merged_data = np.column_stack((x, y, z, L1, L2))
    
    # 保存为TXT文件
    os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
    np.savetxt(output_txt_path, merged_data, fmt='%.6f %.6f %.6f %d %d')
    
    print(f"合并点云已保存至: {output_txt_path}")
    print(f"数据格式: x y z L1(类别) L2(实例ID)")
    
    # 统计各类别点数
    unique_labels, counts = np.unique(L1, return_counts=True)
    print("\n类别统计:")
    label_names = {0: 'building', 1: 'farmland', 2: 'hardground', 3: 'if', 
                   4: 'otherfarmland', 5: 'others', 6: 'road', 7: 'tree'}
    for label, count in zip(unique_labels, counts):
        label_name = label_names.get(label, f'unknown_{label}')
        print(f"  类别 {label} ({label_name}): {count} 点")
    
    return output_txt_path


def extract_road_and_farmland(merged_txt_path, output_dir):
    """
    从合并的点云中分别提取道路和农田点云
    
    参数:
    - merged_txt_path: 合并后的点云文件路径 (x y z L1 L2格式)
    - output_dir: 输出目录
    
    返回:
    - road_file: 道路点云文件路径
    - farmland_file: 农田点云文件路径
    """
    print(f"\n正在从合并点云中提取道路和农田...")
    
    # 读取合并点云
    data = np.loadtxt(merged_txt_path)
    x, y, z, L1, L2 = data[:, 0], data[:, 1], data[:, 2], data[:, 3], data[:, 4]
    
    # 提取道路点 (L1 = 6)
    road_mask = L1 == 6
    road_data = data[road_mask]
    
    # 提取农田点 (L1 = 1)
    farmland_mask = L1 == 1
    farmland_data = data[farmland_mask]
    
    # 保存道路点云
    os.makedirs(output_dir, exist_ok=True)
    road_file = os.path.join(output_dir, 'road_points.txt')
    np.savetxt(road_file, road_data, fmt='%.6f %.6f %.6f %d %d')
    print(f"道路点云已保存: {road_file} ({road_data.shape[0]} 点)")
    
    # 保存农田点云
    farmland_file = os.path.join(output_dir, 'farmland_points.txt')
    np.savetxt(farmland_file, farmland_data, fmt='%.6f %.6f %.6f %d %d')
    print(f"农田点云已保存: {farmland_file} ({farmland_data.shape[0]} 点)")
    
    return road_file, farmland_file


# ===========================================
# 第十五部分：主函数
# ===========================================
def main():
    """
    主函数：集成道路实例化与拓扑分析的完整流程
    """
    print("======================================")
    print("   道路与农田点云数据处理集成系统")
    print("======================================")
    
    # 询问是否需要合并PLY文件
    merge_ply = input("\n是否需要合并原始PLY和预测PLY文件? (y/n, 默认n): ").strip().lower()
    
    if merge_ply == 'y':
        print("\n[PLY文件合并阶段]")
        original_ply = input("请输入原始点云PLY文件路径: ").strip()
        prediction_ply = input("请输入预测结果PLY文件路径: ").strip()
        output_dir = input("请输入输出目录 (默认: ./merged_output): ").strip() or './merged_output'
        
        # 合并PLY文件
        merged_txt = os.path.join(output_dir, 'merged_points.txt')
        try:
            merge_original_and_prediction(original_ply, prediction_ply, merged_txt)
            
            # 提取道路和农田
            road_file, farmland_file = extract_road_and_farmland(merged_txt, output_dir)
            
            print(f"\n合并完成！生成的文件:")
            print(f"  - 合并点云: {merged_txt}")
            print(f"  - 道路点云: {road_file}")
            print(f"  - 农田点云: {farmland_file}")
            
            # 询问是否继续后续处理
            continue_process = input("\n是否继续进行后续处理? (y/n, 默认y): ").strip().lower()
            if continue_process == 'n':
                print("处理完成，程序退出。")
                return
            
            # 使用合并后的文件作为输入
            use_merged = input("是否使用刚才合并的文件进行后续处理? (y/n, 默认y): ").strip().lower()
            if use_merged != 'n':
                merged_file_for_topo = merged_txt
            else:
                merged_file_for_topo = None
        
        except Exception as e:
            print(f"合并PLY文件失败: {e}")
            traceback.print_exc()
            return
    else:
        merged_file_for_topo = None

    # 设置输入输出路径
    print("\n[设置输出目录]")
    farm_dir = input("请输入农田处理输出目录 (默认: ./farm_output): ").strip() or './farm_output'
    road_dir = input("请输入道路处理输出目录 (默认: ./road_output): ").strip() or './road_output'
    topo_dir = input("请输入拓扑关系分析输出目录 (默认: ./topo_output): ").strip() or './topo_output'

    # 1. 农田点云处理
    print("\n[第一阶段: 农田点云处理]")
    farm_input_file = input("请输入农田点云文件路径 (留空跳过): ").strip()
    if farm_input_file:
        print("\n[第一阶段] 执行农田点云处理...")

        # 询问坐标偏移参数
        print("\n请输入坐标偏移参数（用于转换为WGS84 UTM 51N投影坐标系）:")
        x_offset = float(input("X坐标偏移量: ").strip() or "0")
        y_offset = float(input("Y坐标偏移量: ").strip() or "0")
        z_offset = float(input("Z坐标偏移量: ").strip() or "0")

        complete_processing_pipeline_for_kml(farm_input_file, farm_dir, x_offset, y_offset, z_offset)

        # 获取生成的Alpha边界点文件路径
        base_name = os.path.splitext(os.path.basename(farm_input_file))[0]
        farm_alpha_file = os.path.join(farm_dir, 'boundary', f"{base_name}_所有Alpha边界点.txt")
    else:
        farm_alpha_file = input("请输入农田Alpha边界点文件路径: ").strip()

    # 2. 道路点云处理
    print("\n[第二阶段: 道路点云处理]")
    road_input_file = input("请输入道路点云文件路径 (留空跳过): ").strip()
    if road_input_file:
        print("执行道路分割与枢纽点检测...")
        road_segmentation(
            input_file=road_input_file,
            output_dir=road_dir,
            grid_size=0.45,
            pivot_detection=True
        )

        # 获取生成的all_road_segments.txt文件路径
        road_segments_file = os.path.join(road_dir, 'all_road_segments.txt')
    else:
        road_segments_file = input("请输入道路分段文件路径(all_road_segments.txt, 留空跳过): ").strip()

    # 3. 合并农田和道路点云数据（如果需要）
    if farm_alpha_file and road_segments_file:
        print("\n[第三阶段: 合并农田和道路点云数据]")
        merged_file = os.path.join(topo_dir, "merged_road_farm.txt")
        topo_input_file = merge_road_farm_points(farm_alpha_file, road_segments_file, merged_file)
    elif merged_file_for_topo:
        # 使用之前合并的PLY文件
        print("\n[第三阶段: 使用已合并的点云文件]")
        topo_input_file = merged_file_for_topo
    else:
        topo_input_file = input("\n请输入用于拓扑分析的合并点云文件路径 (留空跳过): ").strip()

    # 4. 道路与农田拓扑关系分析
    if topo_input_file:
        print("\n[第四阶段: 执行道路与农田拓扑关系分析]")
        try:
            associations = analyze_road_farm_topology(
                point_cloud_file=topo_input_file,
                output_dir=topo_dir,
                grid_size=0.5
            )
        except Exception as e:
            print(f"拓扑关系分析失败: {e}")
            traceback.print_exc()

    print("\n======================================")
    print("   点云数据处理完成")
    print("======================================")
    if farm_dir and os.path.exists(farm_dir):
        print(f"农田点云处理结果保存在: {farm_dir}")
    if road_dir and os.path.exists(road_dir):
        print(f"道路分割与枢纽点检测结果保存在: {road_dir}")
    if topo_dir and os.path.exists(topo_dir):
        print(f"道路与农田拓扑关系分析结果保存在: {topo_dir}")

    print("\n感谢使用道路与农田点云数据处理集成系统！")


if __name__ == "__main__":
    main()
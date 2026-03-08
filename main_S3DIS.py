from os.path import join

from matplotlib import pyplot as plt
import matplotlib.colors as colors

from FarmSegNet import Network
from tester_S3DIS import ModelTester
from helper_ply import read_ply
from helper_tool import ConfigS3DIS as cfg
from helper_tool import DataProcessing as DP
from helper_tool import Plot
# import tensorflow as tf
import numpy as np
import time, pickle, argparse, glob, os
from umap import UMAP
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()


class S3DIS:
    def __init__(self, test_area_idx):
        self.name = 'S3DIS'
        self.path = './data/S3DIS'
        self.label_to_names = {0: 'building',
                               1: 'farmland',
                               2: 'hardground',
                               3: 'if',
                               4: 'otherfarmland',
                               5: 'others',
                               6: 'road',
                               7: 'tree'}
        self.num_classes = len(self.label_to_names)
        self.label_values = np.sort([k for k, v in self.label_to_names.items()])
        self.label_to_idx = {l: i for i, l in enumerate(self.label_values)}
        self.ignored_labels = np.array([])

        self.val_split = 'Area_' + str(test_area_idx)
        self.all_files = glob.glob(join(self.path, 'original_ply', '*.ply'))

        # Initiate containers
        self.val_proj = []
        self.val_labels = []
        self.possibility = {}
        self.min_possibility = {}
        self.input_trees = {'training': [], 'validation': []}
        self.input_colors = {'training': [], 'validation': []}
        self.input_labels = {'training': [], 'validation': []}
        self.input_names = {'training': [], 'validation': []}
        self.load_sub_sampled_clouds(cfg.sub_grid_size)

    def load_sub_sampled_clouds(self, sub_grid_size):
        tree_path = join(self.path, 'input_{:.3f}'.format(sub_grid_size))
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]
            if self.val_split in cloud_name:
                cloud_split = 'validation'
            else:
                cloud_split = 'training'

            # Name of the input files
            kd_tree_file = join(tree_path, '{:s}_KDTree.pkl'.format(cloud_name))
            sub_ply_file = join(tree_path, '{:s}.ply'.format(cloud_name))

            data = read_ply(sub_ply_file)
            sub_colors = np.vstack((data['red'], data['green'], data['blue'])).T
            sub_labels = data['class']

            # Read pkl with search tree
            with open(kd_tree_file, 'rb') as f:
                search_tree = pickle.load(f)

            self.input_trees[cloud_split] += [search_tree]
            self.input_colors[cloud_split] += [sub_colors]
            self.input_labels[cloud_split] += [sub_labels]
            self.input_names[cloud_split] += [cloud_name]

            size = sub_colors.shape[0] * 4 * 7
            print('{:s} {:.1f} MB loaded in {:.1f}s'.format(kd_tree_file.split('/')[-1], size * 1e-6, time.time() - t0))

        print('\nPreparing reprojected indices for testing')

        # Get validation and test reprojected indices
        for i, file_path in enumerate(self.all_files):
            t0 = time.time()
            cloud_name = file_path.split('/')[-1][:-4]

            # Validation projection and labels
            if self.val_split in cloud_name:
                proj_file = join(tree_path, '{:s}_proj.pkl'.format(cloud_name))
                with open(proj_file, 'rb') as f:
                    proj_idx, labels = pickle.load(f)
                self.val_proj += [proj_idx]
                self.val_labels += [labels]
                print('{:s} done in {:.1f}s'.format(cloud_name, time.time() - t0))

    # Generate the input data flow
    def get_batch_gen(self, split):
        if split == 'training':
            num_per_epoch = cfg.train_steps * cfg.batch_size
        elif split == 'validation':
            num_per_epoch = cfg.val_steps * cfg.val_batch_size

        self.possibility[split] = []
        self.min_possibility[split] = []
        # Random initialize
        for i, tree in enumerate(self.input_colors[split]):
            self.possibility[split] += [np.random.rand(tree.data.shape[0]) * 1e-3]
            self.min_possibility[split] += [float(np.min(self.possibility[split][-1]))]

        def spatially_regular_gen():
            # Generator loop
            for i in range(num_per_epoch):

                # Choose the cloud with the lowest probability
                cloud_idx = int(np.argmin(self.min_possibility[split]))

                # choose the point with the minimum of possibility in the cloud as query point
                point_ind = np.argmin(self.possibility[split][cloud_idx])

                # Get all points within the cloud from tree structure
                points = np.array(self.input_trees[split][cloud_idx].data, copy=False)

                # Center point of input region
                center_point = points[point_ind, :].reshape(1, -1)

                # Add noise to the center point
                noise = np.random.normal(scale=cfg.noise_init / 10, size=center_point.shape)
                pick_point = center_point + noise.astype(center_point.dtype)

                # Check if the number of points in the selected cloud is less than the predefined num_points
                if len(points) < cfg.num_points:
                    # Query all points within the cloud
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=len(points))[1][0]
                else:
                    # Query the predefined number of points
                    queried_idx = self.input_trees[split][cloud_idx].query(pick_point, k=cfg.num_points)[1][0]

                # Shuffle index
                queried_idx = DP.shuffle_idx(queried_idx)
                # Get corresponding points and colors based on the index
                queried_pc_xyz = points[queried_idx]
                queried_pc_xyz = queried_pc_xyz - pick_point
                queried_pc_colors = self.input_colors[split][cloud_idx][queried_idx]
                queried_pc_labels = self.input_labels[split][cloud_idx][queried_idx]

                # Update the possibility of the selected points
                dists = np.sum(np.square((points[queried_idx] - pick_point).astype(np.float32)), axis=1)
                delta = np.square(1 - dists / np.max(dists))
                self.possibility[split][cloud_idx][queried_idx] += delta
                self.min_possibility[split][cloud_idx] = float(np.min(self.possibility[split][cloud_idx]))

                # up_sampled with replacement
                if len(points) < cfg.num_points:
                    queried_pc_xyz, queried_pc_colors, queried_idx, queried_pc_labels = \
                        DP.data_aug(queried_pc_xyz, queried_pc_colors, queried_pc_labels, queried_idx, cfg.num_points)

                if True:
                    yield (queried_pc_xyz.astype(np.float32),
                           queried_pc_colors.astype(np.float32),
                           queried_pc_labels,
                           queried_idx.astype(np.int32),
                           np.array([cloud_idx], dtype=np.int32))

        gen_func = spatially_regular_gen
        gen_types = (tf.float32, tf.float32, tf.int32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, 3], [None], [None], [None])
        return gen_func, gen_types, gen_shapes

    @staticmethod
    def get_tf_mapping2():
        # Collect flat inputs
        def tf_map(batch_xyz, batch_features, batch_labels, batch_pc_idx, batch_cloud_idx):
            batch_features = tf.concat([batch_xyz, batch_features], axis=-1)
            input_points = []
            input_neighbors = []
            input_pools = []
            input_up_samples = []

            for i in range(cfg.num_layers):
                neighbour_idx = tf.py_func(DP.knn_search, [batch_xyz, batch_xyz, cfg.k_n], tf.int32)
                sub_points = batch_xyz[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                pool_i = neighbour_idx[:, :tf.shape(batch_xyz)[1] // cfg.sub_sampling_ratio[i], :]
                up_i = tf.py_func(DP.knn_search, [sub_points, batch_xyz, 1], tf.int32)
                input_points.append(batch_xyz)
                input_neighbors.append(neighbour_idx)
                input_pools.append(pool_i)
                input_up_samples.append(up_i)
                batch_xyz = sub_points

            input_list = input_points + input_neighbors + input_pools + input_up_samples
            input_list += [batch_features, batch_labels, batch_pc_idx, batch_cloud_idx]

            return input_list

        return tf_map

    def init_input_pipeline(self):
        print('Initiating input pipelines')
        cfg.ignored_label_inds = [self.label_to_idx[ign_label] for ign_label in self.ignored_labels]
        gen_function, gen_types, gen_shapes = self.get_batch_gen('training')
        gen_function_val, _, _ = self.get_batch_gen('validation')
        self.train_data = tf.data.Dataset.from_generator(gen_function, gen_types, gen_shapes)
        self.val_data = tf.data.Dataset.from_generator(gen_function_val, gen_types, gen_shapes)

        self.batch_train_data = self.train_data.batch(cfg.batch_size)
        self.batch_val_data = self.val_data.batch(cfg.val_batch_size)
        map_func = self.get_tf_mapping2()

        self.batch_train_data = self.batch_train_data.map(map_func=map_func)
        self.batch_val_data = self.batch_val_data.map(map_func=map_func)

        self.batch_train_data = self.batch_train_data.prefetch(cfg.batch_size)
        self.batch_val_data = self.batch_val_data.prefetch(cfg.val_batch_size)

        iter = tf.data.Iterator.from_structure(self.batch_train_data.output_types, self.batch_train_data.output_shapes)
        self.flat_inputs = iter.get_next()
        self.train_init_op = iter.make_initializer(self.batch_train_data)
        self.val_init_op = iter.make_initializer(self.batch_val_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=int, default=0, help='the number of GPUs to use [default: 0]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')
    parser.add_argument('--mode', type=str, default='train', help='options: train, test, vis')
    parser.add_argument('--model_path', type=str, default='None', help='pretrained model path')
    parser.add_argument('--vis_method', type=str, default='umap', help='visualization method: umap, pca, tsne')
    parser.add_argument('--vis_dim', type=int, default=2, help='visualization dimensions: 2 or 3')
    parser.add_argument('--max_points', type=int, default=400000, help='maximum number of points for visualization')
    parser.add_argument('--custom_colors', action='store_true', help='use custom colors for classes')
    FLAGS = parser.parse_args()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    Mode = FLAGS.mode

    test_area = FLAGS.test_area
    dataset = S3DIS(test_area)
    dataset.init_input_pipeline()

    if Mode == 'train':
        model = Network(dataset, cfg)
        model.train(dataset)
    elif Mode == 'test':
        cfg.saving = False
        model = Network(dataset, cfg)
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            chosen_folder = logs[-1]
            snap_path = join(chosen_folder, 'snapshots')
            snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
            chosen_step = np.sort(snap_steps)[-1]
            chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
        tester = ModelTester(model, dataset, restore_snap=chosen_snap)
        tester.test(model, dataset)

    else:
        ##################
        # Visualize data #
        ##################

        # 加载或创建模型
        if FLAGS.model_path is not 'None':
            chosen_snap = FLAGS.model_path
        else:
            chosen_snapshot = -1
            logs = np.sort([os.path.join('results', f) for f in os.listdir('results') if f.startswith('Log')])
            if len(logs) > 0:
                chosen_folder = logs[-1]
                snap_path = join(chosen_folder, 'snapshots')
                snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']
                chosen_step = np.sort(snap_steps)[-1]
                chosen_snap = os.path.join(snap_path, 'snap-{:d}'.format(chosen_step))
            else:
                chosen_snap = None

        # 创建FarmSeg-Net模型
        model = Network(dataset, cfg)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())

            # 恢复模型权重
            if chosen_snap is not None:
                model.saver.restore(sess, chosen_snap)
                print(f"模型已从 {chosen_snap} 恢复")

            # 初始化数据迭代器
            sess.run(dataset.val_init_op)

            # 收集特征和标签
            all_features = []
            all_labels = []

            try:
                # 收集一批样本数据
                batch_count = 0
                max_batches = 5  # 限制收集的批次数

                print("开始收集特征数据...")
                # 使用tqdm创建进度条
                progress_bar = tqdm(total=max_batches, desc="收集数据批次", ncols=100)

                while batch_count < max_batches:
                    # 获取特征和标签
                    ops = (model.feature_fused, model.inputs['labels'])
                    features, labels = sess.run(ops, {model.is_training: False})

                    # 处理特征和标签
                    features = features.squeeze(2)  # [B, N, 1, C] -> [B, N, C]

                    # 展平批次维度
                    batch_size = features.shape[0]
                    for b in range(batch_size):
                        batch_features = features[b]  # [N, C]
                        batch_labels = labels[b]  # [N]

                        # 筛选有效标签的特征和标签
                        valid_indices = np.where(~np.isin(batch_labels, cfg.ignored_label_inds))[0]
                        valid_features = batch_features[valid_indices]
                        valid_labels = batch_labels[valid_indices]

                        all_features.append(valid_features)
                        all_labels.append(valid_labels)

                    batch_count += 1
                    # 更新进度条
                    progress_bar.update(1)
                    progress_bar.set_postfix({"特征点数": sum(f.shape[0] for f in all_features)})

                # 关闭进度条
                progress_bar.close()

                # 合并所有样本
                all_features = np.vstack(all_features)
                all_labels = np.concatenate(all_labels)

                # 如果点太多，随机采样以加快可视化速度
                if all_features.shape[0] > FLAGS.max_points:
                    print(f"采样点数从 {all_features.shape[0]} 减少到 {FLAGS.max_points}...")
                    indices = np.random.choice(all_features.shape[0], FLAGS.max_points, replace=False)
                    all_features = all_features[indices]
                    all_labels = all_labels[indices]

                print(f"收集特征形状: {all_features.shape}")
                print(f"收集标签形状: {all_labels.shape}")


                # 定义类别颜色映射（可自定义颜色）
                def generate_class_colors(dataset, use_custom=False):
                    unique_labels = np.unique(all_labels)
                    n_classes = len(unique_labels)

                    if use_custom:
                        # 直接使用十六进制颜色代码
                        color_map = {
                            0: '#0010EF',  # 建筑
                            1: '#006A94',  # 农田
                            2: '#00D925',  # 硬地
                            3: '#46FF00',  # if
                            4: '#B4FF00',  # 其他农田
                            5: '#FFDD00',  # 其他
                            6: '#FF6E00',  # 道路
                            7: '#FF1211',  # 树木
                        }

                        # 为每个标签分配颜色，直接使用十六进制代码
                        colors = {}
                        for i, label in enumerate(unique_labels):
                            if label in color_map:
                                colors[label] = color_map[label]
                            else:
                                # 如果没有预定义颜色，使用默认颜色
                                colors[label] = '#CCCCCC'  # 默认灰色
                    else:
                        # 使用matplotlib默认颜色循环
                        cmap = plt.cm.get_cmap('tab10', n_classes)
                        colors = {label: cmap(i % 10)[:3] for i, label in enumerate(unique_labels)}

                    return colors


                # 获取颜色映射
                class_colors = generate_class_colors(dataset, use_custom=FLAGS.custom_colors)

                # 保存数据用于重复实验（可选）
                data_save_path = os.path.join("visualization", "feature_data")
                os.makedirs(data_save_path, exist_ok=True)
                np.save(os.path.join(data_save_path, "features.npy"), all_features)
                np.save(os.path.join(data_save_path, "labels.npy"), all_labels)
                print(f"特征和标签数据已保存至: {data_save_path}")

                # 使用选定的降维方法(PCA或UMAP)
                print(f"使用{FLAGS.vis_method.upper()}降维到{FLAGS.vis_dim}维...")

                # 标准化特征
                print("正在进行特征标准化...")
                scaler = StandardScaler()
                normalized_features = scaler.fit_transform(all_features)

                # 根据选择的方法进行降维
                if FLAGS.vis_method.lower() == 'pca':
                    # 使用PCA进行降维
                    from sklearn.decomposition import PCA

                    progress_bar = tqdm(desc="PCA降维进行中", ncols=100)
                    progress_bar.set_postfix({"状态": "正在计算..."})
                    reducer = PCA(n_components=FLAGS.vis_dim)
                    embedding = reducer.fit_transform(normalized_features)
                    explained_variance = reducer.explained_variance_ratio_.sum()
                    progress_bar.close()
                    print(f"PCA降维完成，解释方差: {explained_variance:.2%}")

                elif FLAGS.vis_method.lower() == 'tsne':
                    # 使用t-SNE进行降维
                    from sklearn.manifold import TSNE

                    progress_bar = tqdm(desc="t-SNE降维进行中", ncols=100)
                    progress_bar.set_postfix({"状态": "正在计算..."})
                    reducer = TSNE(n_components=FLAGS.vis_dim, random_state=42)
                    embedding = reducer.fit_transform(normalized_features)
                    progress_bar.close()
                    print("t-SNE降维完成")

                else:  # 默认使用UMAP
                    # 使用标准UMAP降维
                    print("开始进行UMAP降维...")
                    progress_bar = tqdm(desc="UMAP降维进行中", ncols=100)
                    progress_bar.set_postfix({"状态": "正在计算..."})

                    # 简化UMAP参数
                    reducer = UMAP(n_neighbors=15,
                                   min_dist=0.1,
                                   n_components=FLAGS.vis_dim,
                                   random_state=42,
                                   n_epochs=200,  # 减少训练轮数
                                   low_memory=True,  # 低内存模式
                                   metric='euclidean')

                    try:
                        # 直接计算UMAP结果
                        embedding = reducer.fit_transform(normalized_features)
                        print("UMAP降维完成")
                    except Exception as e:
                        # 如果UMAP失败，使用PCA作为备选方案
                        print(f"UMAP计算出错: {e}")
                        print("尝试使用PCA替代...")
                        from sklearn.decomposition import PCA

                        pca = PCA(n_components=FLAGS.vis_dim)
                        embedding = pca.fit_transform(normalized_features)
                        print("已使用PCA完成降维")

                    # 关闭进度条
                    progress_bar.close()

                # 关闭进度条
                progress_bar.close()

                print("降维完成，开始绘制可视化图...")

                # 创建颜色映射（在任何可视化分支前定义）
                unique_labels = np.unique(all_labels)

                # 设置使用通用字体
                plt.rcParams['font.family'] = 'sans-serif'
                plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Helvetica', 'Arial', 'Verdana', 'sans-serif']

                # 定义不同类别的标记形状
                marker_shapes = {
                    0: 'o',  # 圆形
                    1: '^',  # 三角形
                    2: 's',  # 正方形
                    3: 'p',  # 五角星
                    4: '*',  # 星形
                    5: 'h',  # 六边形
                    6: 'x',  # x形
                    7: 'd',  # 钻石形
                    8: '+',  # 加号形
                    9: 'v',  # 倒三角形
                }

                # 可视化
                if FLAGS.vis_dim == 3:
                    # 3D可视化
                    fig = plt.figure(figsize=(12, 10))
                    ax = fig.add_subplot(111, projection='3d')

                    # 绘制散点图
                    for i, label in enumerate(unique_labels):
                        if label in cfg.ignored_label_inds:
                            continue
                        mask = all_labels == label
                        # 为当前类别选择标记形状
                        marker = marker_shapes.get(label, 'o')  # 默认使用圆形
                        ax.scatter(
                            embedding[mask, 0],
                            embedding[mask, 1],
                            embedding[mask, 2],
                            c=[class_colors[label]],
                            label=dataset.label_to_names[label],
                            alpha=1.0,
                            s=5,  # 略微增大标记大小以便更好地显示形状
                            marker=marker,
                            edgecolors='none'
                        )

                    # ax.set_title(f"{FLAGS.vis_method.upper()} 3D Visualization - feature_fused Features", fontsize=16)
                    # ax.set_xlabel("Dimension 1", fontsize=12)
                    # ax.set_ylabel("Dimension 2", fontsize=12)
                    # ax.set_zlabel("Dimension 3", fontsize=12)
                    # 设置刻度字体大小
                    ax.tick_params(axis='x', labelsize=10)
                    ax.tick_params(axis='y', labelsize=10)
                    ax.tick_params(axis='z', labelsize=10)
                    # ax.legend(title="Classes") # 注释掉图例

                    # 添加旋转角度，仅保存一张固定角度的图片
                    vis_path = "visualization"
                    os.makedirs(vis_path, exist_ok=True)
                    plt.savefig(os.path.join(vis_path, f"{FLAGS.vis_method}_3d_feature_fused.png"),
                                dpi=600, bbox_inches='tight')

                else:
                    # 2D可视化
                    plt.figure(figsize=(12, 10))

                    # 绘制散点图
                    for i, label in enumerate(unique_labels):
                        if label in cfg.ignored_label_inds:
                            continue
                        mask = all_labels == label
                        # 为当前类别选择标记形状
                        marker = marker_shapes.get(label, 'o')  # 默认使用圆形
                        plt.scatter(
                            embedding[mask, 0],
                            embedding[mask, 1],
                            c=[class_colors[label]],
                            label=dataset.label_to_names[label],
                            alpha=1.0,
                            s=5,  # 略微增大标记大小以便更好地显示形状
                            marker=marker,
                            edgecolors='none'
                        )

                # plt.title(f"{FLAGS.vis_method.upper()} {FLAGS.vis_dim}D Visualization - feature_fused Features", fontsize=16)
                # plt.xlabel("Dimension 1", fontsize=12)
                # plt.ylabel("Dimension 2", fontsize=12)
                # 设置刻度字体大小
                plt.tick_params(axis='x', labelsize=32)
                plt.tick_params(axis='y', labelsize=32)
                # plt.legend(title="Classes") # 注释掉图例

                # 保存图像
                vis_path = "visualization"
                os.makedirs(vis_path, exist_ok=True)
                plt.savefig(os.path.join(vis_path, f"{FLAGS.vis_method}_{FLAGS.vis_dim}d_feature_fused.png"),
                            dpi=600, bbox_inches='tight')

                print(f"可视化图像已保存至: {os.path.join(vis_path)}")
                plt.show()

            except tf.errors.OutOfRangeError:
                print("数据已处理完毕")
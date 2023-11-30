import numpy as np
from collections import OrderedDict
from concurrent import futures as futures
from os import path as osp
from pathlib import Path
from skimage import io


def get_image_index_str(img_idx, use_prefix_id=False):
    if use_prefix_id:
        return '{:07d}'.format(img_idx)
    else:
        return '{:06d}'.format(img_idx)


def get_hid_info_path(idx,
                       prefix,
                       info_type='image_2',
                       file_tail='.png',
                       training=True,
                       relative_path=True,
                       exist_check=True,
                       use_prefix_id=False):
    img_idx_str = get_image_index_str(idx, use_prefix_id)
    img_idx_str += file_tail
    prefix = Path(prefix)
    if training:
        file_path = Path('training') / info_type / img_idx_str
    else:
        file_path = Path('testing') / info_type / img_idx_str
    if exist_check and not (prefix / file_path).exists():
        raise ValueError('file not exist: {}'.format(file_path))
    if relative_path:
        return str(file_path)
    else:
        return str(prefix / file_path)


def get_label_path(idx,
                   prefix,
                   training=True,
                   relative_path=True,
                   exist_check=True,
                   info_type='labels',
                   use_prefix_id=False):
    return get_hid_info_path(idx, prefix, info_type, '.txt', training,
                              relative_path, exist_check, use_prefix_id)


def get_velodyne_path(idx,
                      prefix,
                      training=True,
                      relative_path=True,
                      exist_check=True,
                      use_prefix_id=False):
    return get_hid_info_path(idx, prefix, 'velodyne', '.bin', training,
                              relative_path, exist_check, use_prefix_id)


def get_label_anno(label_path):
    annotations = {}
    annotations.update({
        'name': [],
        'truncated': [],
        'occluded': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    })
    with open(label_path, 'r') as f:
        lines = f.readlines()
    # if len(lines) == 0 or len(lines[0]) < 15:
    #     content = []
    # else:
    content = [line.strip().split(' ') for line in lines]
    num_objects = len([x[0] for x in content if x[0] != 'DontCare'])

    annotations['name'] = np.array([x[0] for x in content])
    num_gt = len(annotations['name'])
    annotations['truncated'] = np.array([float(x[1]) for x in content])
    annotations['occluded'] = np.array([int(x[2]) for x in content])
    # dimensions will convert hwl format to standard lhw(camera) format.

    annotations['dimensions'] = np.array([[float(info) for info in x[3:6]]
                                          for x in content
                                          ]).reshape(-1, 3)
    annotations['location'] = np.array([[float(info) for info in x[6:9]]
                                        for x in content]).reshape(-1, 3)

    annotations['rotation_y'] = np.array([float(x[9])
                                          for x in content]).reshape(-1)
    index = list(range(num_objects)) + [-1] * (num_gt - num_objects)
    annotations['index'] = np.array(index, dtype=np.int32)
    return annotations


def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat


def get_hid_info(path,
                  training=True,
                  label_info=True,
                  velodyne=False,
                  image_ids=7481,
                  extend_matrix=True,
                  num_worker=8,
                  relative_path=True,
                  with_imageshape=True):
    """
    Base annotation format:
    {
        point_cloud: {
            num_features: 4
            velodyne_path: ...
        }
        annos: {
            location: [num_gt, 3] array
            dimensions: [num_gt, 3] array
            rotation_y: [num_gt] angle array
            name: [num_gt] ground truth name array
        }
    }
    """
    root_path = Path(path)
    if not isinstance(image_ids, list):
        image_ids = list(range(image_ids))

    def map_func(idx):
        info = {}
        pc_info = {'num_features': 4}
        calib_info = {}
        image_info = {'image_idx': idx}

        annotations = None
        if velodyne:
            pc_info['velodyne_path'] = get_velodyne_path(
                idx, path, training, relative_path)
            # points = np.fromfile(
            #     Path(path) / pc_info['velodyne_path'], dtype=np.float32)
            # points = np.copy(points).reshape(-1, pc_info['num_features'])
            # info['timestamp'] = np.int64(points[0, -1])
        if label_info:
            label_path = get_label_path(idx, path, training, relative_path)
            if relative_path:
                label_path = str(root_path / label_path)
            annotations = get_label_anno(label_path)

        info['image'] = image_info
        info['point_cloud'] = pc_info
        if annotations is not None:
            info['annos'] = annotations
        return info

    with futures.ThreadPoolExecutor(num_worker) as executor:
        image_infos = executor.map(map_func, image_ids)

    return list(image_infos)
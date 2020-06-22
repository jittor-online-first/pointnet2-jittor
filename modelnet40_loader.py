import os
import subprocess
from pathlib import Path

import lmdb
import msgpack_numpy
import numpy as np
from tqdm import tqdm

from jittor.dataset.dataset import Dataset


BASE_DIR = Path(__file__).parent

class ModelNet40(Dataset):
    def __init__(self, n_points: int, train: bool, batch_size=1, shuffle=False):
        super().__init__()
        self.n_points = n_points
        self.batch_size = batch_size
        self.train = train
        self.shuffle = shuffle

        self.path = BASE_DIR / 'data' / 'modelnet40_normal_resampled'
        self.cache = BASE_DIR / 'data' / 'modelnet40_normal_resampled_cache'
        self.cache.mkdir(exist_ok=True)

        if not self.path.exists():
            self.path.mkdir(parents=True)
            self.url = (
                "https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip"
            )
            zipfile = self.path / '..' / 'modelnet40_normal_resampled.zip'

            if not zipfile.exists():
                subprocess.check_call([
                    'curl', self.url, '-o', str(zipfile)
                ])

            subprocess.check_call([
                'unzip', str(zipfile), '-d', str(self.path / '..')
            ])

        cats_file = self.path / 'modelnet40_shape_names.txt'
        with cats_file.open() as f:
            cats = [line.rstrip() for line in f.readlines()]
            self.classes = dict(zip(cats, range(len(cats))))

        train = 'train' if self.train else 'test'
        shapes_file = self.path / f'modelnet40_{train}.txt'
        with shapes_file.open() as f:
            self.shapes = []
            for line in f.readlines():
                shape_id = line.rstrip()
                shape_name = '_'.join(shape_id.split('_')[0:-1])
                self.shapes.append((
                    shape_name,
                    shape_id + '.txt',
                ))

        self.lmdb_file = self.cache / train
        self.lmdb_env = None
        if not self.lmdb_file.exists():
            # create lmdb file
            with lmdb.open(str(self.lmdb_file), map_size=1 << 36) as lmdb_env:
                with lmdb_env.begin(write=True) as txn:
                    for i, (shape_name, shape_file) in enumerate(tqdm(self.shapes)):
                        shape_path = self.path / shape_name / shape_file
                        pts = np.loadtxt(shape_path, delimiter=',', dtype=np.float32)
                        cls = self.classes[shape_name]

                        txn.put(
                            str(i).encode(),
                            msgpack_numpy.packb(pts, use_bin_type=True),
                        )

        self.set_attrs(
            batch_size=self.batch_size,
            total_len=len(self.shapes),
            shuffle=self.shuffle
        )

    def __getitem__(self, idx):
        if self.lmdb_env is None:
            self.lmdb_env = lmdb.open(str(self.lmdb_file), map_size=1<<36, 
                                      readonly=True, lock=False)
        
        shape_name, _ = self.shapes[idx]
        with self.lmdb_env.begin(buffers=True) as txn:
            pts = msgpack_numpy.unpackb(txn.get(str(idx).encode()), raw=False)

        pt_idxs = np.arange(0, self.n_points)
        np.random.shuffle(pt_idxs)

        pts = pts[pt_idxs, :]
        pts, normals = pts[:, :3], pts[:, 3:]
        pts = self.normalize_pointclouds(pts)

        return pts, normals, self.classes[shape_name]

    def collect_batch(self, batch):
        pts = np.stack([b[0] for b in batch], axis=0)
        normals = np.stack([b[1] for b in batch], axis=0)
        cls = np.stack([b[2] for b in batch])
        return pts, cls

    def normalize_pointclouds(self, pts):
        pts = pts - pts.mean(axis=0)
        scale = np.sqrt((pts ** 2).sum(axis=1).max())
        pts = pts / scale
        return pts


if __name__ == '__main__':
    modelnet40 = ModelNet40(n_points=2048, train=True, batch_size=32, shuffle=True)
    for pts, normals, cls in modelnet40:
        break
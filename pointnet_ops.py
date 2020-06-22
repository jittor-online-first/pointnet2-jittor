import math

import jittor as jt
from jittor import nn

jt.flags.use_cuda = 1


def optimal_block(batch_size):
    return 2 ** int(math.log(batch_size))


class FurthestPointSampler(nn.Module):
    cuda_src='''
        __device__ void __update(float *__restrict__ dists, int *__restrict__ dists_i,
                                int idx1, int idx2) {
            const float v1 = dists[idx1], v2 = dists[idx2];
            const int i1 = dists_i[idx1], i2 = dists_i[idx2];
            dists[idx1] = max(v1, v2);
            dists_i[idx1] = v2 > v1 ? i2 : i1;
        }

        __global__ void furthest_point_sampling_kernel (
            int b, int n, int m, int block_size,
            const float *__restrict__ dataset,
            float *__restrict__ temp, 
            int *__restrict__ idxs) {

            if (m <= 0) return;

            extern __shared__ int dists_i[];
            float *dists =  (float *) &dists_i[block_size];

            int batch_index = blockIdx.x;
            dataset += batch_index * n * 3;
            temp += batch_index * n;
            idxs += batch_index * m;

            int tid = threadIdx.x;
            const int stride = block_size;

            int old = 0;
            if (threadIdx.x == 0) idxs[0] = old;

            // initialize temp with INF
            for (int k = tid; k < n; k += stride)
                temp[k] = 1e10;

            __syncthreads();
            for (int j = 1; j < m; j++) {
                int besti = 0;
                float best = -1;
                float x1 = dataset[old * 3 + 0];
                float y1 = dataset[old * 3 + 1];
                float z1 = dataset[old * 3 + 2];
                for (int k = tid; k < n; k += stride) {
                    float x2, y2, z2;
                    x2 = dataset[k * 3 + 0];
                    y2 = dataset[k * 3 + 1];
                    z2 = dataset[k * 3 + 2];
                    float mag = (x2 * x2) + (y2 * y2) + (z2 * z2);
                    if (mag <= 1e-3) continue;

                    float d = (x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1) + (z2 - z1) * (z2 - z1);

                    float d2 = min(d, temp[k]);
                    temp[k] = d2;
                    besti = d2 > best ? k : besti;
                    best = d2 > best ? d2 : best;
                }
                dists[tid] = best;
                dists_i[tid] = besti;
                __syncthreads();

                if (block_size >= 512) {
                    if (tid < 256) {
                        __update(dists, dists_i, tid, tid + 256);
                    }
                    __syncthreads();
                }
                if (block_size >= 256) {
                    if (tid < 128) {
                        __update(dists, dists_i, tid, tid + 128);
                    }
                    __syncthreads();
                }
                if (block_size >= 128) {
                    if (tid < 64) {
                        __update(dists, dists_i, tid, tid + 64);
                    }
                    __syncthreads();
                }
                if (block_size >= 64) {
                    if (tid < 32) {
                        __update(dists, dists_i, tid, tid + 32);
                    }
                    __syncthreads();
                }
                if (block_size >= 32) {
                    if (tid < 16) {
                        __update(dists, dists_i, tid, tid + 16);
                    }
                    __syncthreads();
                }
                if (block_size >= 16) {
                    if (tid < 8) {
                        __update(dists, dists_i, tid, tid + 8);
                    }
                    __syncthreads();
                }
                if (block_size >= 8) {
                    if (tid < 4) {
                        __update(dists, dists_i, tid, tid + 4);
                    }
                    __syncthreads();
                }
                if (block_size >= 4) {
                    if (tid < 2) {
                        __update(dists, dists_i, tid, tid + 2);
                    }
                    __syncthreads();
                }
                if (block_size >= 2) {
                    if (tid < 1) {
                        __update(dists, dists_i, tid, tid + 1);
                    }
                    __syncthreads();
                }

                old = dists_i[0];
                if (tid == 0) idxs[j] = old;
            }
        }

        int block_size = #block_size;

        float *temp;
        cudaMallocManaged(&temp, in0_shape0 * in0_shape1 * sizeof(float));

        furthest_point_sampling_kernel<<<in0_shape0, block_size, 2*block_size*sizeof(int)>>>(
            in0_shape0,
            in0_shape1,
            out_shape1,
            block_size,
            in0_p,
            temp,
            out_p
        );
        cudaDeviceSynchronize();
        cudaFree(temp);
    '''
    def __init__(self, n_samples):
        super().__init__()
        self.n_samples = n_samples

    def execute(self, x):
        '''
        Parameters
        ----------
        x: jt.Var, (B, N, 3)

        Returns
        -------
        y: jt.Var, (B, n_samples, 3)
        '''
        batch_size, n_points, n_coords = x.shape

        assert self.n_samples <= n_points
        assert n_coords == 3
        assert x.dtype == 'float32'

        block_size = optimal_block(batch_size)

        cuda_src = self.cuda_src.replace('#block_size', str(block_size))

        idxs_shape = [batch_size, self.n_samples]
        idxs = jt.code(idxs_shape, 'int32', [x,], cuda_src=cuda_src)
        
        y = x.reindex([batch_size, self.n_samples, 3], [
            'i0',               # Batchid
            '@e0(i0, i1)',      # Nid
            'i2'
        ], extras=[idxs])

        return y


class BallQueryGrouper(nn.Module):
    cuda_src = '''
        __global__ void query_ball_point_kernel(int b, int n, int m, float radius,
                                                int nsample,
                                                const float *__restrict__ new_xyz,
                                                const float *__restrict__ xyz,
                                                int *__restrict__ idx,
                                                int *__restrict__ cnt) {
            int batch_index = blockIdx.x;
            xyz += batch_index * n * 3;
            new_xyz += batch_index * m * 3;
            idx += m * nsample * batch_index;
            cnt += batch_index * m;

            int index = threadIdx.x;
            int stride = blockDim.x;

            float radius2 = radius * radius;
            for (int j = index; j < m; j += stride) {
                float new_x = new_xyz[j * 3 + 0];
                float new_y = new_xyz[j * 3 + 1];
                float new_z = new_xyz[j * 3 + 2];
                cnt[j] = 0;

                for (int k = 0; k < n && cnt[j] < nsample; ++k) {
                    float x = xyz[k * 3 + 0];
                    float y = xyz[k * 3 + 1];
                    float z = xyz[k * 3 + 2];
                    float d2 = (new_x - x) * (new_x - x) + (new_y - y) * (new_y - y) +
                                (new_z - z) * (new_z - z);

                    if (d2 < radius2) {
                        if (cnt[j] == 0) {
                            for (int l = 0; l < nsample; ++l)
                                idx[j * nsample + l] = k;
                        }
                        idx[j * nsample + cnt[j]] = k;
                        ++cnt[j];
                    }
                }
            }
        }

        int block_size = #block_size;

        query_ball_point_kernel<<<in0_shape0, block_size>>>(
            in0_shape0, in1_shape1, in0_shape1, #radius, #nsample,
            in0_p, in1_p, out0_p, out1_p
        );
    '''
    def __init__(self, radius, n_samples, use_xyz):
        super().__init__()
        self.radius = radius
        self.n_samples = n_samples
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        '''
        Parameters
        ----------
        xyz: jt.Var, (B, N, 3)
        features: jt.Var, (B, N, C)

        Returns
        -------
        new_feature: jt.Var, (B, N, n_samples, C)
        '''
        batch_size_x, n_input, n_coords = new_xyz.shape
        assert n_coords == 3

        batch_size_p, n_points, n_coords = pointset.shape
        assert n_coords == 3
        assert batch_size_x == batch_size_p

        if feature is not None:
            batch_size_f, n_points_f, n_feature = feature.shape
            assert batch_size_x == batch_size_f
            assert n_points == n_points_f

        block_size = optimal_block(batch_size_x)

        cuda_src = self.cuda_src.replace('#block_size', str(block_size)) \
                                .replace('#radius', str(self.radius)) \
                                .replace('#nsample', str(self.n_samples))

        idxs_shape = [batch_size_x, n_input, self.n_samples]
        cnts_shape = [batch_size_x, n_input]
        idxs, cnts = jt.code(
            [idxs_shape, cnts_shape],
            ['int32', 'int32'],
            [new_xyz, pointset],
            cuda_src=cuda_src
        )

        pc_shape = [batch_size_x, n_input, self.n_samples, 3]
        new_pointset = pointset.reindex(pc_shape, [
            'i0',
            '@e0(i0, i1, i2)',
            'i3',
        ], extras=[idxs])

        if feature is not None:
            feature_shape = [batch_size_x, n_input, self.n_samples, n_feature]
            new_feature = feature.reindex(feature_shape, [
                'i0',               # Batchid
                '@e0(i0, i1, i2)',  # Nid
                'i3',               # Featureid
            ], extras=[idxs])
        else:
            new_feature = None

        if self.use_xyz:
            local_xyz = new_pointset - new_xyz.unsqueeze(dim=2)
            if new_feature is not None:
                new_feature = jt.contrib.concat([local_xyz, new_feature], dim=-1)
            else:
                new_feature = local_xyz

        return new_feature


class GroupAll(nn.Module):
    def __init__(self, use_xyz):
        super().__init__()
        self.use_xyz = use_xyz

    def execute(self, new_xyz, pointset, feature):
        if self.use_xyz:
            new_feature = jt.contrib.concat([pointset, feature], dim=-1)
        new_feature = new_feature.unsqueeze(dim=1) # [B, 1, N, C]
        return new_feature

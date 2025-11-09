from typing import Optional, Tuple, Any, List

from triton_kernel import attention_hash_varlen

import numpy as np
import torch
import torch.nn.functional as F
from labml import tracker
from torch import distributed as dist, nn

consts: Any = None

class SendRecv:
    def __init__(self, msg: str = ""):
        self._pending_operations: List[dist.P2POp] = []
        self._active_requests = None
        self.rank = consts.rank
        self.world_size = consts.world_size
        self.send_rank = (self.rank + 1) % self.world_size
        self.recv_rank = (self.rank + self.world_size - 1) % self.world_size

    def send_recv(self, tensor_to_send, recv_tensor=None):
        if recv_tensor is None:
            result_tensor = torch.zeros_like(tensor_to_send)
        else:
            result_tensor = recv_tensor

        send_operation = dist.P2POp(dist.isend, tensor_to_send, self.send_rank, group=consts.process_group)
        recv_operation = dist.P2POp(dist.irecv, result_tensor, self.recv_rank, group=consts.process_group)

        if self.send_rank > self.rank:
            self._pending_operations.extend([send_operation, recv_operation])
        else:
            self._pending_operations.extend([recv_operation, send_operation])

        return result_tensor

    def commit(self):
        if self._active_requests is not None:
            raise RuntimeError("Commit called twice")
        self._active_requests = dist.batch_isend_irecv(self._pending_operations)
        self._pending_operations = []

    def wait(self):
        if self._active_requests is None:
            raise RuntimeError("Wait called before commit")

        for request in self._active_requests:
            request.wait()
        self._active_requests = None
        assert not self._pending_operations


@torch.compile
def update_out_and_lse(
        out: Optional[torch.Tensor],
        lse: Optional[torch.Tensor],
        block_out: torch.Tensor,
        block_lse: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.to(torch.float32)
    block_lse = block_lse.unsqueeze(dim=-1)

    if out is None:
        return block_out, block_lse

    lse_d = block_lse - lse
    out -= F.sigmoid(lse_d) * (out - block_out)
    lse -= F.logsigmoid(-lse_d)

    return out, lse


@torch.compile
def ring_attention_masked_forward_ref(k_hashes, q_hashes, h_idx, q, k, v, sm_scale, rank):
    batch_size, n_heads, seq_len, d = q.shape
    _, k_heads, context_len, _ = k.shape
    n_groups = n_heads // k_heads

    mask_shape = (batch_size, k_heads, n_groups, seq_len, context_len + 1)
    mask = k_hashes.new_zeros(mask_shape, dtype=torch.bool)
    k_idx = k_hashes[q_hashes, h_idx[:, None, None]]  # H G I J' -> J
    m = torch.logical_and(rank * context_len <= k_idx, k_idx < (rank + 1) * context_len)
    k_idx = (k_idx - rank * context_len) * m + (context_len * ~m)
    mask.scatter_(-1, k_idx.unsqueeze(0), m.unsqueeze(0))  # Add batch dimension
    mask = mask[:, :, :, :, :-1]

    scores = torch.einsum('bhgid,bhjd->bhgij',
                          q.view(q.shape[0], k_heads, n_groups, *q.shape[2:]),
                          k) * sm_scale

    scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)

    # Online softmax
    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - scores_max)
    exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
    log_sum_exp = torch.log(exp_sum) + scores_max
    P = exp_scores / exp_sum
    P.masked_fill_(~mask, 0.)
    out = torch.einsum('bhgij,bhjd->bhgid', P, v)
    # O = torch.matmul(P, v)
    return out.view(*q.shape), log_sum_exp.view(*q.shape[:-1])


@torch.compile
def ring_attention_masked_forward(k_hashes, q_hashes, q, k, v, sm_scale, rank):
    """
    :param k_hashes: [n_buckets, k_heads, bucket_size]
    :param q_hashes: [k_heads, n_groups, seq_len]
    :param q: [batch_size, n_heads, seq_len, d]
    :param k: [batch_size, k_heads, context_len, d]
    :param v: [batch_size, k_heads, context_len, d]
    :param sm_scale:
    :param rank:
    :return:
    """
    # torch.cuda.synchronize()
    batch_size, n_heads, seq_len, d_head = q.shape
    _, k_heads, context_len, _ = k.shape
    n_groups = n_heads // k_heads
    assert batch_size == 1
    assert k_hashes.shape[1] == k_heads
    assert q_hashes.shape[0] == k_heads
    assert q_hashes.shape[1] == n_groups
    assert q_hashes.shape[2] == seq_len

    k_hashes = k_hashes.permute(1, 0, 2)  # B Z J'
    q_hashes = q_hashes.reshape(k_heads, -1)

    k_hashes_mask = torch.logical_and(rank * context_len <= k_hashes, k_hashes < (rank + 1) * context_len)
    k_hash_counts = k_hashes_mask.to(torch.long).sum(-1)

    q = q.reshape(k_heads, -1, d_head)
    k = k.reshape(k_heads, -1, d_head)
    v = v.reshape(k_heads, -1, d_head)

    assert k_hashes.shape == k_hashes_mask.shape
    assert k_hashes_mask.dtype == torch.bool, k_hashes.dtype
    assert k_hashes.numel() > 0, k_hashes.shape

    k_hashes = k_hashes[k_hashes_mask] - rank * context_len

    # assert k_hashes.max() < context_len, k_hashes.max()
    # assert k_hashes.min() >= 0, k_hashes.min()

    # torch.cuda.synchronize()
    out, lse = attention_hash_varlen(q, k, v, k_hashes, k_hash_counts, q_hashes, sm_scale)
    # torch.cuda.synchronize()

    out = out.reshape(batch_size, n_heads, seq_len, d_head)
    lse = lse.reshape(batch_size, n_heads, seq_len)

    # lse = torch.nan_to_num(lse, nan=0.0, posinf=0.0, neginf=0.0) * torch.log(lse.new_tensor(2.))
    # assert not torch.isnan(lse).any()
    # assert not torch.isnan(out).any()
    # assert not torch.isinf(lse).any()
    # assert not torch.isinf(out).any()

    assert out.dtype == torch.bfloat16
    assert lse.dtype == torch.bfloat16

    return out, lse


@torch.compile
def ring_attention_masked_backward(k_hashes, q_hashes, h_idx, d_out, q, k, v, out, softmax_lse, sm_scale, rank):
    batch_size, n_heads, seq_len, d = q.shape
    _, k_heads, context_len, _ = k.shape
    n_groups = n_heads // k_heads

    mask_shape = (batch_size, k_heads, n_groups, seq_len, context_len + 1)

    mask = k_hashes.new_zeros(mask_shape, dtype=torch.bool)
    k_idx = k_hashes[q_hashes, h_idx[:, None, None]]  # H G I J' -> J
    m = torch.logical_and(rank * context_len <= k_idx, k_idx < (rank + 1) * context_len)
    k_idx = (k_idx - rank * context_len) * m + (context_len * ~m)
    mask.scatter_(-1, k_idx.unsqueeze(0), m.unsqueeze(0))  # Add batch dimension
    mask = mask[:, :, :, :, :-1].contiguous()

    shape = (q.shape[0], k_heads, n_groups, *q.shape[2:])
    q = q.view(*shape)
    d_out = d_out.view(*shape)
    out = out.view(*shape)

    # Recreate S and P from log_sum_exp
    scores = torch.einsum('bhgid,bhjd->bhgij', q, k) * sm_scale

    scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)

    P = torch.exp(scores - softmax_lse.view(*shape[:-1], 1))  # BHGIJ
    # Step 1: Compute dV
    dV = torch.einsum('bhgij,bhgid->bhjd', P, d_out)
    # Step 2: Compute dP
    dP = torch.einsum('bhgid,bhjd->bhgij', d_out, v)
    # Step 3: Compute D
    D = torch.sum(d_out * out, dim=-1, keepdim=True)  # BHGI1
    # Step 4: Compute dS
    dS = P * (dP - D)  # BHGIJ
    # Apply causal mask to dS if is_causal is True
    dS.masked_fill_(~mask, 0)
    # Step 5: Compute dQ
    dQ = torch.einsum('bhgij,bhjd->bhgid', dS, k) * sm_scale
    # Step 6: Compute dK
    dK = torch.einsum('bhgij,bhgid->bhjd', dS, q) * sm_scale
    return dQ.view(batch_size, n_heads, seq_len, d), dK, dV


@torch.no_grad()
def compute_centroids(kq: torch.Tensor, centroids, n_hashes, n_groups) -> torch.Tensor:
    n_heads, context_len, d_qkv = kq.shape
    k_heads = n_heads // n_groups

    kq_hash = kq.contiguous()
    kq_norm = torch.norm(kq_hash, p=2, dim=-1, keepdim=True)
    kq_norm = kq_norm.clamp(min=1e-6)
    kq_hash = kq_hash / kq_norm

    buckets = torch.einsum('zd,hid->zhi', centroids, kq_hash)  # Z HG I
    centroid_top_k = consts.centroid_topk
    buckets = torch.topk(buckets, k=centroid_top_k, dim=-1, largest=True)[1]  # Z HG I` -> I

    h_idx = torch.arange(n_heads, device=kq.device)
    centroids = kq_hash[h_idx[None, :, None], buckets]  # Z HG I` D

    del kq_hash, kq, buckets

    centroids = centroids.reshape(n_hashes, k_heads, n_groups * centroid_top_k, d_qkv)  # Z H GI` D
    centroids = centroids.mean(dim=2)  # Z H D

    return centroids


class _RingApproxCrossAttentionFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, kq, k_self, v_self, k, v, rnd_centroids, sm_scale, n_hashes, n_results, sampling_method):
        # q: query tensor [bs, n_heads, seq_len, d_qkv]
        # kq: key query for context [bs, n_heads, context_len, d_qkv]
        # k_self, v_self: self attention key/value [bs, k_heads, seq_len, d_qkv]
        # k, v: cross attention key/value [bs, k_heads, context_len, d_qkv]
        # rnd_centroids: random centroids for hashing [n_hashes, d_qkv]

        batch_size, n_heads, seq_len, d_qkv = q.shape
        assert batch_size == 1
        _, k_heads, context_len, _ = k.shape
        n_groups = n_heads // k_heads
        assert n_heads % k_heads == 0, f'{q.shape} == {k.shape}'
        assert n_groups > 0, f'{q.shape} == {k.shape}'
        assert kq.shape[0] == k_self.shape[0] == k.shape[0] == batch_size
        assert rnd_centroids.shape[0] == n_hashes

        centroids = compute_centroids(kq[0], rnd_centroids, n_hashes, n_groups)

        # centroids shape: [n_hashes, k_heads, d_qkv]
        # Reduce centroids across all devices
        dist.all_reduce(centroids, op=dist.ReduceOp.AVG)
        # Normalize centroids to unit vectors
        centroids = centroids / torch.norm(centroids, p=2, dim=-1, keepdim=True)

        # Find top-k matches for each centroid in the context keys
        k_hashes = torch.einsum('zhd,hjd->zhj', centroids, k[0])  # Z H J
        assert k_hashes.shape[0] % consts.world_size == 0
        hashes_per_rank = k_hashes.shape[0] // consts.world_size
        k_hashes = k_hashes.reshape(consts.world_size, hashes_per_rank, *k_hashes.shape[1:]).contiguous()
        k_hashes_all = k_hashes.new_zeros(k_hashes.shape)
        dist.all_to_all_single(k_hashes_all, k_hashes)

        k_hashes = k_hashes_all.permute(1, 2, 0, 3)

        del k_hashes_all
        k_hashes = k_hashes.reshape(*k_hashes.shape[:2], -1)
        assert k_hashes.shape[-1] == context_len * consts.world_size
        k_hashes = torch.topk(k_hashes, k=min(n_results, context_len * consts.world_size), dim=-1)[1]  # Z H J' -> J
        k_hashes_all = k_hashes.new_zeros((consts.world_size, *k_hashes.shape))
        dist.all_gather_into_tensor(k_hashes_all, k_hashes)
        k_hashes = k_hashes_all.reshape(-1, *k_hashes.shape[1:])
        del k_hashes_all
        assert k_hashes.shape[0] == hashes_per_rank * consts.world_size

        # Assign queries to centroids based on similarity
        q_hashes = torch.einsum('zhd,hgid->hgiz', centroids, q.reshape(k_heads, n_groups, seq_len, d_qkv))

        q_hashes = torch.argmax(q_hashes, dim=-1)  # H G I -> Z

        bin_counts = q_hashes.new_zeros((k_heads, n_hashes))
        bin_counts.scatter_add_(1, q_hashes.view(k_heads, -1), q_hashes.new_ones((k_heads, n_groups * seq_len)))
        bin_counts = bin_counts.to(torch.bfloat16)
        tracker.add('bin_counts', bin_counts.mean().cpu().item())
        tracker.add('bin_counts.std', bin_counts.std().cpu().item())

        # Ring attention communication setup
        comm = SendRecv("comm")

        # Store original tensors for backward pass
        k_og = k.clone()
        v_og = v.clone()

        out, lse = None, None
        next_k, next_v = None, None

        # Process self-attention component first
        block_out, block_lse = attention_forward(
            q, k_self, v_self, sm_scale, True
        )
        out, lse = update_out_and_lse(out, lse, block_out, block_lse)

        rank = comm.rank
        world_size = comm.world_size

        # Ring attention loop across all devices
        for step in range(world_size):
            current_rank = (rank - step + world_size) % world_size
            # Exchange keys, values, and hash indices with next device
            if step + 1 != world_size:
                next_k = comm.send_recv(k)
                next_v = comm.send_recv(v)
                comm.commit()

            block_out, block_lse = ring_attention_masked_forward(
                k_hashes, q_hashes, q, k, v, sm_scale, current_rank
            )

            # block_out, block_lse = ring_attention_masked_forward_ref(
            #     k_hashes, q_hashes, h_idx, q, k, v, sm_scale, current_rank
            # )
            #
            # ring_attention_masked_forward_ref_compare(
            #     k_hashes, q_hashes, h_idx, q, k, v, sm_scale, current_rank,
            #     block_out, block_lse
            # )

            out, lse = update_out_and_lse(out, lse, block_out, block_lse)

            # Update tensors for next iteration
            if step + 1 != world_size:
                comm.wait()
                k = next_k
                v = next_v

        out = out.to(q.dtype)
        # Save tensors needed for backward pass
        ctx.save_for_backward(q, k_self, v_self, k_og, v_og, q_hashes, k_hashes, out, lse.squeeze(-1))
        ctx.sm_scale = sm_scale
        ctx.sampling_method = sampling_method
        return out

    @staticmethod
    def backward(ctx, dout, *args):
        # Backward pass for ring approximate cross attention
        # Uses saved centroids, k_hashes, and q_hashes from forward pass

        q, k_self, v_self, k, v, q_hashes, k_hashes, out, softmax_lse = ctx.saved_tensors
        batch_size, n_heads, seq_len, d_qkv = q.shape
        _, k_heads, context_len, _ = k.shape
        n_groups = n_heads // k_heads

        sm_scale = ctx.sm_scale

        # Communication contexts for backward pass
        kv_comm = SendRecv("kv_comm")
        d_kv_comm = SendRecv("d_kv_comm")
        dq, dk, dv = None, None, None

        next_dk, next_dv = None, None
        next_k, next_v = None, None

        # Compute gradients for self-attention component
        dq, dk_self, dv_self = attention_backward(
            dout, q, k_self, v_self, out, softmax_lse, sm_scale, True
        )

        rank = kv_comm.rank
        world_size = kv_comm.world_size
        mask_shape = (batch_size, k_heads, n_groups, seq_len, context_len)
        h_idx = torch.arange(k_heads, device=k.device)

        # Backward ring attention loop
        for step in range(kv_comm.world_size):
            current_rank = (rank - step + world_size) % world_size
            # Exchange keys, values, and hash indices with next device
            if step + 1 != kv_comm.world_size:
                next_k = kv_comm.send_recv(k)
                next_v = kv_comm.send_recv(v)
                kv_comm.commit()

            block_dq_buffer, block_dk_buffer, block_dv_buffer = ring_attention_masked_backward(
                k_hashes, q_hashes, h_idx,
                dout, q, k, v, out, softmax_lse, sm_scale,
                current_rank,
            )

            # Accumulate query gradients
            dq += block_dq_buffer

            # Update key/value gradients
            if step == 0:
                assert dk is None and dv is None
                dk = block_dk_buffer
                dv = block_dv_buffer
            else:
                d_kv_comm.wait()
                dk = block_dk_buffer + next_dk
                dv = block_dv_buffer + next_dv

            # Update tensors for next iteration
            if step + 1 != kv_comm.world_size:
                kv_comm.wait()
                k = next_k
                v = next_v

            # Exchange gradients with next device
            next_dk = d_kv_comm.send_recv(dk)
            next_dv = d_kv_comm.send_recv(dv)
            d_kv_comm.commit()

        d_kv_comm.wait()

        return dq, None, dk_self, dv_self, next_dk, next_dv, None, None, None, None, None


def ring_cross_attention_approx(q, kq, k_self, v_self, k, v, rnd_centroids, sm_scale, n_hashes, n_results, *,
                                sampling_method: str):
    # Wrapper function for ring approximate cross attention
    return _RingApproxCrossAttentionFunc.apply(q, kq, k_self, v_self, k, v, rnd_centroids, sm_scale, n_hashes,
                                               n_results, sampling_method)


@torch.compile
def calc_ref_centroids(q_self, k):
    batch_size, n_heads, seq_len, d_qkv = q_self.shape
    batch_size, k_heads, context_len, _ = k.shape
    n_groups = n_heads // k_heads

    q_self_all = q_self.reshape(batch_size, k_heads, n_groups, seq_len, d_qkv)

    scores = torch.einsum('bhgid,bhjd->bhgij',
                          q_self_all,
                          k)
    # topk=128
    _topk = min(128, scores.shape[-1])
    best_q = torch.topk(scores, k=_topk, dim=4)[1]  # BHGIJ
    del scores
    best_q = best_q.unsqueeze(-1).expand(-1, -1, -1, -1, -1, d_qkv)  # BGHIJD
    q_gather = q_self_all.unsqueeze(-2)
    del q_self_all
    q_gather = q_gather.expand(-1, -1, -1, -1, _topk, -1)
    q_gather = q_gather.reshape(batch_size, k_heads, n_groups, -1, d_qkv)  # BGH(IJ)D
    best_q = best_q.view(batch_size, k_heads, n_groups, -1, d_qkv)

    centroids = q_gather.new_zeros((*best_q.shape[:3], context_len, d_qkv))  # BGHJD
    centroids.scatter_add_(3, best_q, q_gather)
    centroids = centroids / _topk  # BHGJD

    centroid_norm = torch.norm(centroids, p=2, dim=-1, keepdim=True)
    centroids = centroids / centroid_norm.clamp(min=1e-6)

    return centroids


def learn_emb_proj(kq, q_self, k):
    batch_size, n_heads, seq_len, d_qkv = q_self.shape
    batch_size, k_heads, context_len, _ = k.shape
    n_groups = n_heads // k_heads

    kq = kq.view(*kq.shape[:2], n_heads, d_qkv)

    # normalize
    kq = kq.permute(0, 2, 1, 3).contiguous()  # B H I D
    kq = kq.reshape(batch_size, k_heads, n_groups, -1, d_qkv)
    kq_norm = torch.norm(kq, p=2, dim=-1, keepdim=True)
    with torch.no_grad():
        tracker.add('kq_norm', kq_norm.mean().cpu().item())

    kq = kq / kq_norm.clamp(min=1e-6)

    with torch.no_grad():
        centroids = calc_ref_centroids(q_self, k)

    assert list(kq.shape) == list(centroids.shape), f'{kq.shape} != {centroids.shape}'
    with torch.no_grad():
        tracker.add('centroids.mean', centroids.mean().cpu().item())
        tracker.add('centroids.norm', torch.norm(centroids, p=2, dim=-1, keepdim=True).mean().cpu().item())

    return kq.reshape(batch_size, n_heads, -1, d_qkv), nn.MSELoss(reduction='mean')(kq, centroids)


def compute_attention(self: 'SelfAttention', q_self, k_self, v_self, q, k, v, x_e, pos_e, *, aux):
    # Compute both self-attention and context-query cross-attention
    if self.centroids is None:
        # Initialize random centroids for hashing on first use
        rng = np.random.RandomState(42)
        centroids = rng.randn(consts.n_buckets, q_self.shape[-1])
        self.centroids = torch.tensor(centroids, device=k.device, dtype=k.dtype, requires_grad=False)

    with torch.no_grad():
        x_e = x_e / torch.norm(x_e, p=2, dim=-1, keepdim=True)

    kq, f_loss = learn_emb_proj(self.kq_proj(x_e), q_self, k)
    aux[f'kq_mse_{self.layer_idx}'] = f_loss

    # Compute approximate cross attention from query to context
    out_d = ring_cross_attention_approx(
        q_self,
        kq,
        k_self,
        v_self,
        k,
        v,
        self.centroids,
        128 ** -0.5,
        consts.n_buckets,
        consts.n_bucket_size,
        sampling_method='',
    )

    return out_d

import math
from typing import Tuple

from rich import print

import torch
import triton
import triton.language as tl
from labml import monit, logger
from torch.library import triton_op, wrap_triton

_BLOCK_K = 128
_BLOCK_Q = 32


@triton_op("notbad::attention_hash_varlen", mutates_args={})
def _attention_hash_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                           k_hashes: torch.Tensor, k_hash_counts: torch.Tensor, k_hash_offset: torch.Tensor,
                           q_hashes_z: torch.Tensor, q_hashes_count: torch.Tensor, q_hashes_offset: torch.Tensor,
                           sm_scale: float, q_seq_len: int, kv_seq_len: int,
                           d_head: int, n_buckets: int,
                           ) -> Tuple[torch.Tensor, torch.Tensor]:
    o = torch.empty_like(q)  # B S D
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)

    grid = lambda meta: (len(q_hashes_count), 1, 1)
    wrap_triton(_attn_fwd)[grid](
        q, k, v,
        k_hashes, k_hash_counts, k_hash_offset,
        q_hashes_z, q_hashes_count, q_hashes_offset,
        lse, o,
        sm_scale=sm_scale,
        q_seq_len=q_seq_len,
        kv_seq_len=kv_seq_len,
        d_head=d_head,
        BLOCK_K=_BLOCK_K,
        BLOCK_Q=_BLOCK_Q,
        n_buckets=n_buckets,
    )

    return o, lse


def _attention_hash_varlen_backward(ctx, grad_out1, grad_out2):
    return None, None, None, None, None, None, None, None, None, None, None, None, None, None


_attention_hash_varlen.register_autograd(_attention_hash_varlen_backward)


def attention_hash_varlen(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                          k_hashes: torch.Tensor, k_hash_counts: torch.Tensor,
                          q_hashes: torch.Tensor,
                          sm_scale: float) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Group query attention forward pass. Returns the output in shape `[batch_size, n_heads, q_seq_len, d_head]`.

    :param q: has shape `[batch_size, q_seq_len, d_head]`
    :param k: has shape `[batch_size, kv_seq_len, d_head]`
    :param v: has shape `[batch_size, kv_seq_len, d_head]`
    :param k_hashes: has shape `[batch_size * C * J]`
    :param k_hash_counts: has shape `[batch_size, C]`
    :param q_hashes: has shape `[batch_size,  q_seq_len] -> C
    :param sm_scale: softmax scale factor
    """
    batch_size, q_seq_len, d_head = q.shape
    _, kv_seq_len, _ = k.shape
    _, n_buckets = k_hash_counts.shape

    assert k.shape[0] == batch_size
    assert k_hash_counts.shape[0] == batch_size
    assert q_hashes.shape[0] == batch_size
    assert q_hashes.shape[1] == q_seq_len, f'{q_hashes.shape} != {q_seq_len}'

    q_hashes_z, q_hashes_idx = torch.sort(q_hashes, dim=1, descending=False, stable=True)
    b_idx = torch.arange(batch_size, device=k.device)
    q = q[b_idx[:, None], q_hashes_idx]
    assert q.shape == (batch_size, q_seq_len, d_head)

    q_hashes_z = q_hashes_z + torch.arange(batch_size, device=k.device)[:, None] * n_buckets
    q_hashes_z, q_hashes_count = torch.unique_consecutive(q_hashes_z, return_counts=True)
    q_hashes_z = q_hashes_z % n_buckets

    q_hashes_offset = torch.cat([q_hashes_count.new_full([1], 0), torch.cumsum(q_hashes_count, dim=0)], dim=0)[:-1]

    k_hash_offset = torch.cumsum(k_hash_counts.view(-1), dim=0)
    k_hash_offset = torch.cat([k_hash_counts.new_full([1], 0), k_hash_offset[:-1]])
    k_hash_offset = k_hash_offset.reshape(k_hash_counts.shape)

    # Shape constraints
    assert d_head == k.shape[-1] == v.shape[-1]

    # Make sure the tensors are contiguous and the strides are same
    assert q.is_contiguous()
    assert k.is_contiguous()
    assert v.is_contiguous()
    assert k.stride() == v.stride()

    o, lse = _attention_hash_varlen(q, k, v,
                                    k_hashes, k_hash_counts, k_hash_offset,
                                    q_hashes_z, q_hashes_count, q_hashes_offset,
                                    sm_scale=sm_scale,
                                    q_seq_len=q_seq_len,
                                    kv_seq_len=kv_seq_len,
                                    d_head=d_head,
                                    n_buckets=n_buckets, )

    o_out = torch.empty_like(o)
    o_out[b_idx[:, None], q_hashes_idx] = o

    lse = lse.to(q.dtype) * math.log(2)
    lse_out = torch.empty_like(lse)
    lse_out[b_idx[:, None], q_hashes_idx] = lse
    lse_out = torch.nan_to_num(lse_out,
                               nan=0.0,
                               posinf=torch.finfo(lse_out.dtype).max,
                               neginf=torch.finfo(lse_out.dtype).min)

    return o_out, lse_out


def _get_autotune_configs() -> list:
    """
    #### Configs for auto-tuning
    """

    configs = []
    for s in [1, 3]:
        for w in [1, 2, 4]:
            configs.append(triton.Config({}, num_stages=s, num_warps=w))

    return configs


@triton.autotune(_get_autotune_configs(), key=[])
@triton.jit
def _attn_fwd(t_q, t_k, t_v,
              t_k_hashes, t_k_hashes_count, t_k_hashes_off,
              t_h, t_h_count, t_h_off,
              t_lse, t_o,
              sm_scale: tl.constexpr,
              q_seq_len: tl.int64,
              kv_seq_len: tl.int64,
              d_head: tl.constexpr,
              BLOCK_K: tl.constexpr,
              BLOCK_Q: tl.constexpr,
              n_buckets: tl.constexpr,
              ):
    """
    :param t_q: query [batch_size, q_seq_len, d_head] ordered by hash along dim=1
    :param t_k: keys [batch_size, kv_seq_len, d_head]
    :param t_v: values [batch_size, kv_seq_len, d_head]
    :param t_k_hashes: key indexes of buckets [batch_size * C * Z]
    :param t_k_hashes_count: key bucket sizes [batch_size, C]
    :param t_k_hashes_off: key bucket offsets [batch_size, C]

    :param t_h: query hashes [batch_size * C`]
    :param t_h_count: number of queries with the same hash [batch_size * C`]
    :param t_h_off: query offset along batch_size * q_seq_len [batch_size, C`]

    :param t_lse: $\log_2 \sum_j e^{S_{ij}}$ (out)
    :param t_o: output (out)

    :param sm_scale: softmax scale
    :param q_seq_len: query sequence length
    :param kv_seq_len: key/value sequence length
    :param d_head: size of a head
    :param BLOCK_K: key block size
    :param BLOCK_Q: query block size
    :param n_buckets: number of hashes C

    Strides `z`, `h`, `m` and  `d` denote the stride of the corresponding dimensions
     (`batch_size`, `n_heads`, `seq_len`, `d_head`) in the query.
    Stride `n` denote the stride on `seq_len` of key.

    """
    pid = tl.program_id(0)
    n_queries = tl.load(t_h_count + pid)
    bucket_id = tl.load(t_h + pid)
    offset = tl.load(t_h_off + pid)
    z = offset // q_seq_len
    i = offset % q_seq_len

    n_blocks = tl.cdiv(n_queries, BLOCK_Q)

    b_k_hashes_count = tl.load(t_k_hashes_count + z * n_buckets + bucket_id)
    b_k_hashes_off = tl.load(t_k_hashes_off + z * n_buckets + bucket_id)
    p_k_hashes = t_k_hashes + b_k_hashes_off

    offs_d = tl.arange(0, d_head)
    offs_q = tl.arange(0, BLOCK_Q)
    offs_k = tl.arange(0, BLOCK_K)
    offs_qd = offs_q[:, None] * d_head + offs_d[None, :]

    sm_scale = sm_scale * 1.44269504

    for s in tl.range(n_blocks):
        p_q = t_q + z * q_seq_len * d_head + i * d_head
        p_o = t_o + z * q_seq_len * d_head + i * d_head
        p_lse = t_lse + z * q_seq_len + i

        q_mask = offs_q < n_queries

        b_m = tl.zeros([BLOCK_Q], dtype=tl.float32) - float("inf")
        b_l = tl.zeros([BLOCK_Q], dtype=tl.float32) + 1.0
        b_acc = tl.zeros([BLOCK_Q, d_head], dtype=tl.float32)

        b_q = tl.load(p_q + offs_qd, mask=q_mask[:, None])

        b_acc, b_l, b_m = _attn_fwd_inner(b_acc, b_l, b_m, b_q,
                                          p_k_hashes,
                                          b_k_hashes_count,
                                          t_k + z * kv_seq_len * d_head,
                                          t_v + z * kv_seq_len * d_head,
                                          offs_d,
                                          offs_k,
                                          sm_scale,
                                          d_head,
                                          BLOCK_K=BLOCK_K,
                                          )

        tl.store(p_lse + offs_q, b_m + tl.math.log2(b_l), mask=q_mask)
        tl.store(p_o + offs_qd, (b_acc / b_l[:, None]).to(t_o.type.element_ty), mask=q_mask[:, None])

        i += BLOCK_Q
        n_queries -= BLOCK_Q


@triton.jit
def _attn_fwd_inner(b_acc, b_l, b_m, b_q,
                    p_k_hashes,
                    b_k_hashes_count,
                    t_k, t_v,
                    offs_d,
                    offs_k,
                    scale: tl.constexpr,
                    d_head: tl.constexpr,
                    BLOCK_K: tl.constexpr,
                    ):
    steps = tl.cdiv(b_k_hashes_count, BLOCK_K)

    for j in tl.range(steps):
        k_mask = offs_k < b_k_hashes_count
        b_idx = tl.load(p_k_hashes + offs_k, mask=k_mask)  # [i]
        p_kT = t_k + b_idx[None, :] * d_head + offs_d[:, None]
        b_kT = tl.load(p_kT, mask=k_mask[None, :])

        b_s = tl.dot(b_q, b_kT, out_dtype=tl.float32)
        b_s = b_s * scale
        b_s = tl.where(k_mask[None, :], b_s, -1.0e6)

        b_m_new = tl.maximum(b_m, tl.max(b_s, -1))
        b_p = tl.math.exp2(b_s - b_m_new[:, None])
        b_l_new = tl.sum(b_p, -1)

        b_m_m_new = tl.math.exp2(b_m - b_m_new)
        b_l = b_l * b_m_m_new + b_l_new

        p_v = t_v + b_idx[:, None] * d_head + offs_d[None, :]
        b_v = tl.load(p_v, mask=k_mask[:, None])
        b_acc = b_acc * b_m_m_new[:, None]
        b_p = b_p.to(b_q.dtype)
        b_acc += tl.dot(b_p, b_v, out_dtype=tl.float32)

        b_m = b_m_new

        p_k_hashes += BLOCK_K
        b_k_hashes_count -= BLOCK_K

    tl.static_assert(b_acc.dtype == tl.float32, "attn_fwd_inner requires accumulator to be in tl.float32 precision")

    return b_acc, b_l, b_m


@torch.compile()
def attention_hash(q, k, v, centroids, bucket_size, sm_scale):
    k_hashes = torch.einsum('bzd,bjd->bzj', centroids, k)  # Z H J
    k_hashes = torch.topk(k_hashes, k=bucket_size, dim=-1)[1]  # Z H J' -> J

    q_hashes = torch.einsum('bzd,bid->biz', centroids, q)
    q_hashes = torch.argmax(q_hashes, dim=-1)

    k_hashes_mask = torch.logical_and(1 * 8 * 1024 <= k_hashes, k_hashes < 8 * 8 * 1024)
    k_hash_counts = k_hashes_mask.sum(-1)

    return attention_hash_varlen(q, k, v, k_hashes[k_hashes_mask], k_hash_counts, q_hashes, sm_scale)


@torch.no_grad()
def _calc_abs_rel_error(a: torch.Tensor, b: torch.Tensor, atol=1e-2):
    d = (a - b).abs()
    max_abs = d.max()
    d = (d - atol).clamp(min=0)
    d = d / b.abs().clamp(min=1e-6)
    max_rel = d.max()

    return max_abs.cpu().item(), max_rel.cpu().item()


def _test_op(batch_size, n_groups, q_seq_len, kv_seq_len, n_buckets, bucket_size, d_head, dtype, device):
    with monit.section(f'Init {q_seq_len} {kv_seq_len} {d_head}'):
        torch.manual_seed(20)
        q = (torch.empty((batch_size, n_groups * q_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        k = (torch.empty((batch_size, kv_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        v = (torch.empty((batch_size, kv_seq_len, d_head),
                         dtype=dtype, device=device).normal_(mean=0.0, std=0.5).requires_grad_())
        sm_scale = d_head ** -0.5

        q_hashes = torch.randint(n_buckets, (batch_size, n_groups * q_seq_len), device=device)
        k_hashes = torch.rand(batch_size, n_buckets, kv_seq_len, device=device)
        k_hashes = torch.topk(k_hashes, dim=-1, k=bucket_size, sorted=True)[1]
        if kv_seq_len > 32 * 1024:
            print(f'[yellow]Large KV[/yellow]{k_hashes.shape}')
            k_hashes_mask = torch.logical_and(24 * 1024 <= k_hashes, k_hashes < 24 * 1024 + 100)
            # k_hashes_mask = torch.logical_and(1024 <= k_hashes, k_hashes < 1030)
        else:
            mask_seq_len = int(kv_seq_len * 0.9)
            k_hashes_mask = k_hashes < mask_seq_len

        k_hash_counts = k_hashes_mask.to(torch.long).sum(-1)

        # k_hashes = torch.arange(bucket_size, device=device)[None, None, :]
        torch.cuda.synchronize()

    with monit.section('Pytorch reference'):
        scores = torch.einsum('bid,bjd->bij', q, k) * sm_scale

        mask = k_hashes.new_zeros((batch_size, n_groups * q_seq_len, kv_seq_len), dtype=torch.bool)
        assert mask.shape == scores.shape

        b_idx = torch.arange(batch_size, device=k.device)
        k_idx = k_hashes[b_idx[:, None], q_hashes]  # B I J' -> J
        mask.scatter_(-1, k_idx, True)  # Add batch dimension
        if kv_seq_len > 32 * 1024:
            mm = torch.zeros_like(mask)
            mm[:, :, 24 * 1024:24 * 1024 + 100] = True
            mask = torch.logical_and(mm, mask)
        else:
            mask[:, :, mask_seq_len:] = False

        scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)
        # Online softmax
        scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
        exp_scores = torch.exp(scores - scores_max)
        exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
        ref_lse = torch.log(exp_sum) + scores_max
        ref_lse = ref_lse.squeeze(-1)
        P = exp_scores / exp_sum
        ref_out = torch.einsum('bij,bjd->bid', P, v)

        torch.cuda.synchronize()

    with monit.section('Triton Top-K'):
        assert q.dtype == dtype
        tri_out, tri_lse = attention_hash_varlen(q, k, v, k_hashes[k_hashes_mask], k_hash_counts, q_hashes, sm_scale)
        torch.cuda.synchronize()

    with monit.section('Test') as s:
        # compare
        # assert not torch.isnan(tri_out).any()
        # assert not torch.isinf(tri_out).any()
        # assert not torch.isnan(ref_out).any()
        # assert not torch.isinf(ref_out).any()
        print(f'Triton NaN=[cyan]{bool(torch.isnan(tri_out).any())}[/cyan]')
        print(f'Ref NaN=[cyan]{bool(torch.isnan(ref_out).any())}[/cyan]')
        print(f'Triton Inf=[cyan]{bool(torch.isinf(tri_out).any())}[/cyan]')
        print(f'Ref Inf=[cyan]{bool(torch.isinf(ref_out).any())}[/cyan]')

        # assert not torch.isnan(tri_lse).any()
        print(f'Triton LSE NaN=[cyan]{bool(torch.isnan(tri_lse).any())}[/cyan]')
        print(f'Ref LSE NaN=[cyan]{bool(torch.isnan(ref_lse).any())}[/cyan]')
        # assert not torch.isnan(ref_lse).any()
        # assert not torch.isinf(tri_lse).any()
        print(f'Triton LSE Inf=[cyan]{bool(torch.isinf(tri_lse).any())}[/cyan]')
        print(f'Ref LSE Inf=[cyan]{bool(torch.isinf(ref_lse).any())}[/cyan]')
        # assert not torch.isinf(ref_lse).any()
        passed = True
        if not torch.allclose(tri_out, ref_out, atol=1e-2, rtol=0.):
            abs_err, rel_err = _calc_abs_rel_error(ref_out, tri_out)
            logger.log(('[FAILED]', logger.Text.danger), f' Out mismatch {abs_err} {rel_err}')
            passed = False
            # for i in range(q_seq_len):
            #     if not torch.allclose(tri_out[0, i], ref_out[0, i], atol=1e-2, rtol=0.):
            #         print(i)
            #         print(torch.cat([ref_out.detach()[0, i].cpu()[:, None],
            #                          tri_out.detach()[0, i].cpu()[:, None]], dim=1).numpy().tolist())
            #         # break
        print(tri_lse.shape, ref_lse.shape)
        if not torch.allclose(tri_lse, ref_lse, atol=1e-2, rtol=0.):
            abs_err, rel_err = _calc_abs_rel_error(ref_lse, tri_lse)
            logger.log(('[FAILED]', logger.Text.danger), f' LSE mismatch {abs_err} {rel_err}')
            passed = False

        if passed:
            logger.log('[PASSED]', logger.Text.success)
            s.success = True
        else:
            s.success = False
        torch.cuda.synchronize()


def _perf_triton_fn(*, device, dtype, batch_size, n_groups, seq_len, context_len, n_buckets, bucket_size, d_head):
    sm_scale = d_head ** -0.5

    q = torch.randn((batch_size, n_groups * seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, context_len, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, context_len, d_head), dtype=dtype, device=device, requires_grad=True)
    centroids = torch.randn((batch_size, n_buckets, d_head), dtype=dtype, device=device)

    return lambda: attention_hash(q, k, v, centroids, bucket_size, sm_scale)


@torch.compile
def _perf_pytorch_func(q, k, v, sm_scale):
    scores = torch.einsum('bid,bjd->bij', q, k) * sm_scale

    scores_max = torch.max(scores, dim=-1, keepdim=True)[0]
    exp_scores = torch.exp(scores - scores_max)
    exp_sum = torch.sum(exp_scores, dim=-1, keepdim=True)
    # log_sum_exp = torch.log(exp_sum) + scores_max
    attn = exp_scores / exp_sum
    # out = torch.einsum('bhgij,bhjd->bhgid', P, v)
    # attn = torch.softmax(scores, dim=-1)
    return torch.einsum('bij,bjd->bid', attn, v)


def _perf_pytorch(*, device, dtype, batch_size, n_groups, seq_len, context_len, n_buckets, bucket_size, d_head):
    sm_scale = d_head ** -0.5

    q = torch.randn((batch_size, n_groups * seq_len, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, context_len, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, context_len, d_head), dtype=dtype, device=device, requires_grad=True)

    return lambda: _perf_pytorch_func(q, k, v, sm_scale)


def sdpa_attn(q, k, v):
    q = q.permute(0, 2, 1, 3)
    k = k.permute(0, 2, 1, 3)
    v = v.permute(0, 2, 1, 3)

    from torch.nn.attention import sdpa_kernel, SDPBackend
    from torch.nn.functional import scaled_dot_product_attention

    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        attn_out = scaled_dot_product_attention(q, k, v,
                                                scale=None,
                                                is_causal=False,
                                                enable_gqa=True,
                                                )

    # For SDPA
    attn_out = attn_out.permute(0, 2, 1, 3)

    return attn_out


def _perf_sdpa(*, batch_size, n_groups, seq_len, context_len, n_buckets, bucket_size, d_head, device, dtype):
    q = torch.randn((batch_size, seq_len, n_groups, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, context_len, 1, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, context_len, 1, d_head), dtype=dtype, device=device, requires_grad=True)
    return lambda: sdpa_attn(q, k, v)


def _perf_flash(*, batch_size, n_groups, seq_len, context_len, n_buckets, bucket_size, d_head, device, dtype):
    q = torch.randn((batch_size, seq_len, n_groups, d_head), dtype=dtype, device=device, requires_grad=True)
    k = torch.randn((batch_size, context_len, 1, d_head), dtype=dtype, device=device, requires_grad=True)
    v = torch.randn((batch_size, context_len, 1, d_head), dtype=dtype, device=device, requires_grad=True)
    from flash_attn import flash_attn_func
    return lambda: flash_attn_func(q, k, v, causal=False)


def _perf_fn(name, fn, *, batch_size, n_groups, seq_len, context_len, d_head, n_buckets, bucket_size, is_bwd: bool):
    # if is_bwd:
    #     o = fn()
    #     do = torch.randn_like(o)
    #     fn = lambda: o.backward(do, retain_graph=True)
    #
    ms = triton.testing.do_bench(fn)

    flops_per_matmul = 2.0 * batch_size * n_groups * seq_len * context_len * d_head
    total_flops = 2 * flops_per_matmul
    if is_bwd:
        total_flops *= 2.5  # 2.0(bwd) + 0.5(recompute)

    tf_ps = total_flops * 1e-12 / (ms * 1e-3)
    logger.log((f'{name}', logger.Text.key), ': ', f'{ms :,.1f}ms', ' ', f'{tf_ps :,.2f}TFps')


def _test():
    device = torch.device('cuda:0')
    torch.cuda.set_device(device)

    dtype = torch.float16

    # Test correctness
    logger.log('Testing correctness...', logger.Text.title)
    _test_op(1, 1, 24, 1024, 8, 16, 16,
             dtype=torch.float32, device=device)
    _test_op(8, 4, 2040, 1023 * 60, 512, 1024, 128,
             dtype=torch.float32, device=device)
    _test_op(8, 4, 2040, 1023 * 60, 512, 1024, 128,
             dtype=torch.float32, device=device)

    # Test performance
    logger.log('Testing performance...', logger.Text.title)
    _conf = {
        'batch_size': 8,
        'n_groups': 4,
        'seq_len': 1024 * 32,
        'context_len': 64 * 1024,
        'd_head': 128,
        'n_buckets': 1024,
        'bucket_size': 1024 * 3,
    }

    for i in range(1):
        for is_bwd in [False]:
            # _perf_fn(f'flash', _perf_flash(device=device, dtype=dtype, **_conf),
            #          is_bwd=is_bwd, **_conf)
            _perf_fn(f'triton', _perf_triton_fn(device=device, dtype=dtype, **_conf),
                     is_bwd=is_bwd, **_conf)
            # _perf_fn(f'torch', _perf_pytorch(device=device, dtype=dtype, **_conf),
            #          is_bwd=is_bwd, **_conf)
            _perf_fn(f'sdpa', _perf_sdpa(device=device, dtype=dtype, **_conf),
                     is_bwd=is_bwd, **_conf)


if __name__ == "__main__":
    _test()

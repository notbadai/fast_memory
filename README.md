# Efficient Attention with Memory Modules

This work reduces the quadratic time complexity of self-attention.
It reduces per-token time complexity from $\mathcal{O}(Nd)$ to $\mathcal{O}(N^\alpha d)$ where $\alpha < 1$, $d$ is the head dimension, and $N$ is the number of tokens before it.
This is achieved through an indexing mechanism employing cross-polytope locality-sensitive hashing (LSH), which identifies a subset of keys likely to yield the highest dot-product similarities with a given query.

Our work act as memory modules. You can create a memory modules with a set of past keys, for example the keys from previous turns in a multiturn conversation. These keys will then be indexed and you can compute the dot product attention against those using the index and combine with rest of the keys using the attention output and the log-sum-exp of the attention scores from the memory module.

### Overview

Consider a single attention head with query $q \in \mathbb{R}^d$, key matrix $K \in \mathbb{R}^{N × d}$, and value matrix $V \in \mathbb{R}^{N × d}$. Standard attention is given by:

$$\text{attn}(q, K, V) = \text{softmax}\left(\frac{q K^T}{\sqrt{d}}\right) V$$

This results in a time complexity of $O(Nd)$ per query token, leading to overall quadratic $O(N^2d)$ complexity for sequences of length $N$, which becomes prohibitive for long-context applications such as large language models.

Empirical observations indicate that performance remains invariant when attention is restricted to the $M$ keys $k_i$ maximizing $q \cdot k_i$, with $M \ll N$. Naively identifying these is $O(Nd)$, negating efficiency gains.

We approximate this top-$M$ selection via an ocality-sensitive hash (LSH) index, reducing query-time cost to $O\big((C + Z)d\big)$, where $C \approx N^\alpha$ is the hash table size and $Z  \approx N^\beta$ is the bucket capacity, with $\alpha < 1$ and $\beta < 1$.

### Indexing Procedure

For a given head, keys K are indexed as follows:

1. **Projection Computation**: Compute $Y = A K^T \in \mathbb{ℝ}^{C × N}$, where $A \in \mathbb{R}^{C × d}$ is the LSH projection matrix (detailed later). <!--Each row $Y_i$ corresponds to projections onto a hyperplane.-->

2. **Bucket Assignment**: For each bucket $i \in [1, C]$, identify the top-$Z$ key indices $b_{i,1:Z} = \text{argtop}_Z(Y_{i,:})$, where $\text{argtop}_Z$ denotes the indices of the $Z$ largest values. Assign key $k_{b_{i,j}}$ to bucket $i$. This differs from standard cross-polytope LSH, which assigns to a single maximizer.

3. **Computing Attention**: For query $q$, compute $y_q = A q \in \mathbb{R}^C$ and hash $h_q = \text{argmax}_i(y_{q,i})$. Retrieve keys $k_j | j \in b_{h_q}$, apply attention with these keys.

Time complexity: $O(Cd)$ for $y_q$, $O(Zd)$ for subset attention, yielding $O\big((C + Z)d\big)$ per query. Index build: $O(NCd)$ for $Y$, plus $O(N C log Z)$ for top-$Z$ extractions (via heaps or selection algorithms), totaling $O(NCd)$, since $\log Z \ll d$.

### Construction of the Projection Matrix $A$

To enhance collision sensitivity tailored to model-specific token distributions, the projection matrix $A$ is constructed in a data-driven manner rather than using random initialization. This approach leverages embeddings from prior tokens to align projections with dense regions of the query manifold, thereby improving approximation quality compared to uniform random matrices.

The construction process comprises the following steps:

1. **Embedding Projection**: Let $X = [x_1, \dots, x_N] \in \mathbb{R}^{N \times d_m}$ represent the embeddings of $N$ prior tokens, where $d_m$ denotes the model embedding dimension. Each embedding $x_j$ is projected onto the unit sphere in query space using a function $f: \mathbb{R}^{d_m} \to S^{d-1}$, yielding projected vectors $f(m_j)$.

2. **Clustering**: Perform K-means clustering on the projected embeddings $\{f(m_j)\}_{j=1}^N$ to obtain $C$ centroids $c_1, \dots, c_C \in S^{d-1}$. The clustering employs a dot product-based distance metric, specifically minimizing $1 - c \cdot f(m_j)$ for centroid $c$, which corresponds to maximizing cosine similarity on the unit sphere.

3. **Matrix Formation**: Form the projection matrix $A \in \mathbb{R}^{C \times d}$ by stacking the transpose of the centroids: $A = [c_1^T; \dots; c_C^T]$.

The computational complexity of K-means is approximately $O(G N Cd)$, where $G$ is the number of iterations (typically small, e.g., $G = 1$ or $2$). This data-driven method ensures that the rows of $A$ are oriented toward high-density areas in the query manifold, improving approximation quality over uniform random $A$.

### Embedding Projection Function $f(\cdot)$

The function $f: \mathbb{R}^{d_m} \to S^{d-1}$ maps token embeddings to the unit sphere in query space. In various embodiments, $f(\cdot)$ may be implemented as:

- A direct projection using the model's query weights: $f(x) = \frac{W_q x}{\|W_q x\|_2}$, where $W_q$ are the query projection weights from the attention head.
- A learned linear transformation.
- A multi-layer perceptron (MLP).

Empirical results indicate that a learned $f(\cdot)$ yields superior performance. We train it by first computing the full query-key similarity matrix $QK^T$, and identifying the top-$L$ keys with the highest similarity scores for each query. Then we identify the subset of queries for which the key at $j$ ranks among the top-$L$ keys, and train $f(x_j)$ to minimize the distance between $f(x_j)$ and the L2-normalized mean of those selected queries. This alignment encourages $f(x_j)$ to approximate the average direction of queries that strongly attend to the key $k_j$.


### Memory Complexity

For a batch of $\tilde{N}$ queries, we load $\Theta(M)$ queries and iterate over $\mathbf{K}$, $\mathbf{V}$, where $M$ is the SRAM memory. This results in $\Theta(\tilde{N}dM^{-1} + C)$ passes. In each pass we load $\mathbf{K}$, $\mathbf{V}$ which is $\Theta(Zd)$ per pass. So overall it's $\Theta(\tilde{N}Zd^2M^{-1} + CZd)$.

For flash the HBM access is $\Theta(\tilde{N} N d^2 M^{-1})$.

$C$ is the number of buckets, $Z$ is the number of keys in a bucket, $N$ is the number of keys in the memory block.

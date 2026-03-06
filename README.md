# Content-Aware Image Resizing via Seam Carving

**Author:** He Tianchi

## 📖 Overview
This repository contains a high-performance Python implementation of the **Seam Carving** algorithm. Traditional image resizing techniques, such as uniform scaling or cropping, are fundamentally oblivious to image semantics—scaling introduces severe geometric distortion, while cropping indiscriminately discards peripheral context. 

Seam Carving offers a content-aware solution by iteratively identifying and removing "seams" (connected paths of pixels from top to bottom) that contribute the least to the image's overall visual importance. This optimization problem is solved using **Dynamic Programming**, effectively reducing an exponential time complexity of $O(3^H)$ to a highly efficient $O(H \cdot W)$.

## 🔬 Mathematical Foundations & Algorithm Design

### 1. Derivation of the Dual-Gradient Energy Function
To determine which pixels are "unimportant," the algorithm must quantify visual interest. Human vision is highly sensitive to edges and high-contrast textures, which correspond to regions of high spatial frequency. Therefore, we derive our cost metric using the spatial derivatives of the image.

The energy $E(x,y)$ of a pixel is defined as the $L_2$ norm (Euclidean magnitude) of its gradient vector:
$$E(x,y) = \sqrt{\left(\frac{\partial I}{\partial x}\right)^2 + \left(\frac{\partial I}{\partial y}\right)^2}$$

In a discrete pixel grid, we approximate these continuous derivatives using finite differences across all color channels ($R, G, B$):
$$\Delta x^2(x,y) = \sum_{c \in \{R,G,B\}} (I_c(x+1, y) - I_c(x-1, y))^2$$
$$\Delta y^2(x,y) = \sum_{c \in \{R,G,B\}} (I_c(x, y+1) - I_c(x, y-1))^2$$

**Border Handling:** Standard central differencing fails at the image boundaries ($x=0$ and $x=W-1$). To prevent edge artifacts and ensure strict validity in the state space, the algorithm dynamically switches to forward differencing at the left boundary and backward differencing at the right boundary.

### 2. Dynamic Programming: Optimal Substructure
Finding the minimal-energy seam is a constrained shortest-path problem on a grid graph. A greedy approach (simply picking the lowest-energy neighbor in the next row) fails because a locally optimal choice might force the path into a high-energy bottleneck later. 

Dynamic Programming resolves this by evaluating the global minimum. We define an accumulation matrix $M$, where the subproblem $M[i, j]$ represents the minimum total energy of a valid seam starting from the top row and ending specifically at pixel $(i, j)$.

* **Base Case (Top Row):** $$M[0, j] = E(0, j)$$
* **Recursive Relation:** For any subsequent row $i > 0$, the seam must arrive from a strictly connected neighbor in the row above, enforcing the structural constraint $|x(i) - x(i-1)| \le 1$:
  $$M[i, j] = E(i, j) + \min_{k \in \{j-1, j, j+1\}} M[i-1, k]$$

### 3. Backtracking: Path Reconstruction
The forward DP pass tells us the minimum possible cost, but not the path itself. To reconstruct the seam, we utilize a secondary state matrix $B$ that acts as a ledger.

1. **State Tracking:** During the calculation of $M[i, j]$, we record the directional offset ($-1, 0, \text{or} +1$) of the chosen optimal parent $k$ into $B[i, j]$.
2. **Identify Minimum Endpoint:** We scan the final row $H-1$ to find the absolute minimum cumulative energy: $j_{min} = \text{argmin}_j M[H-1, j]$.
3. **Trace Upward:** Starting from $j_{min}$, we query the $B$ matrix. If we are at column $j$ in row $i$, the parent column in row $i-1$ is simply $j + B[i, j]$. We recursively follow these offsets to the top row, yielding the exact sequence of pixels to excise.

### 4. JIT Compilation Optimization
Because the DP table construction requires evaluating millions of subproblems with strict sequential dependencies (row-by-row), native Python loops become a critical bottleneck. This implementation utilizes **Numba's Just-In-Time (JIT) compiler** (`@jit(nopython=True)`) to translate the nested dynamic programming loops into optimized C-like machine code, achieving approximately **100x speedup** and enabling near real-time processing of HD images.

# Parallel Optimization for KNN Algorithm

**Language:** C++

**Platform:** Code::Blocks

**Compiler:** GCC

## Overview

When the size of datasets grows larger and larger, the running time of the k-Nearest Neighbor (KNN) will increase dramatically. In order to improve the computational performance of the algorithm, this paper focuses on the parallel optimization of KNN from the perspective of distance calculation and sorting, and proposes a parallel version of KNN leveraged SIMD, loop unrolling, and multithreading. To reduce the latency for I/O bound performance, an appropriate reading size is set in the data loading stage to achieve parallel operation between loading and computing. In the distance calculating stage, loop unrolling, SIMD and multithreading are integrated to reduce the algorithm's running time. In the distance sorting stage, a combination of the parallel odd–even sort and merge sort is designed to obtain four times the weighted performance compared with the naive algorithm. The experimental results show that the parallel KNN algorithm can reduce the computing time and get a better overall weighted performance within an acceptable classification accuracy.

**Key Words:** KNN; Parallel Computing; SIMD; Multithreading; Sorting Algorithm
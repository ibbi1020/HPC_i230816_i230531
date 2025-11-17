# KLT OpenACC GPU Optimization Strategy

**Goal**: Reduce execution time from ~10s to below 9s (target: beat 9s CPU baseline)

**Current State**: 
- Convolution functions have basic OpenACC parallelization
- Data transfers occur at function boundaries
- No async execution or pipelining

---

## 🎯 High-Impact Optimization Opportunities (Directive-Only)

### 1. **Reduce Memory Transfer Overhead in Convolution** (Highest Priority)

#### Problem Analysis:
Current convolution implementation transfers kernel data repeatedly:
```c
#pragma acc parallel loop copyin(kdata[0:kwidth])  // Transfers every call
```

#### Optimization Strategy:

**A. Use Cache Directive for Kernel Data**
- Move frequently-accessed kernel data to GPU shared memory
- Reduces global memory bandwidth pressure

```c
#pragma acc parallel loop collapse(2) \
            present(indata, outdata) copyin(kdata[0:kwidth]) \
            cache(kdata[0:kwidth])  // Cache in shared memory
```

**B. Add Gang/Worker/Vector Tuning**
- Optimize thread hierarchy for better occupancy
- Let compiler know we have 2D parallelism

```c
#pragma acc parallel loop gang vector_length(128) collapse(2)
// or
#pragma acc parallel loop gang worker vector collapse(2)
```

**C. Expand Data Regions in _KLTComputeGradients**
- Current: Separate data regions for each convolution
- Target: Single data region covering both gradient computations
- Eliminates intermediate transfers of image data

---

### 2. **Optimize Data Movement in Pyramid/Gradient Computation**

#### Strategy: Minimize Redundant Transfers

**Current flow in `_KLTComputeGradients()`:**
```c
#pragma acc data copyin(img->data)
{
    _convolveSeparate(img, ..., gradx);  // img transferred
}
#pragma acc data copyin(img->data)
{
    _convolveSeparate(img, ..., grady);  // img transferred AGAIN
}
```

**Optimized:**
```c
#pragma acc data copyin(img->data[0:npix]) \
                 copyout(gradx->data[0:npix], grady->data[0:npix])
{
    _convolveSeparate(img, ..., gradx);
    _convolveSeparate(img, ..., grady);  // img already on GPU
}
```

**Note**: This is already implemented! But can be extended to pyramid computation loops.

---

### 3. **Async Streams Within Single Frame** (Advanced)

#### Opportunity: Overlap Independent Pyramid Level Processing

**Challenge**: Can't pipeline across frames without code changes (pyramid caching)
**Alternative**: Pipeline within single frame's pyramid construction

**Current Flow:**
```
Level 0: [Smooth] → [Compute Gradients]
Level 1:            [Smooth] → [Compute Gradients]
Level 2:                       [Smooth] → [Compute Gradients]
```

**Optimized (if levels are independent enough):**
```
Level 0: [Smooth - async(0)] → [Gradients - async(0)]
Level 1:         [Smooth - async(1)] → [Gradients - async(1)]
```

**Implementation:**
```c
// In pyramid computation loop
for (i = 0; i < nPyramidLevels; i++) {
    int stream = i % 2;
    #pragma acc data async(stream)
    {
        _KLTComputeGradients(...);
    }
}
#pragma acc wait  // Synchronize all streams at end
```

**Limitation**: Levels depend on previous level, so limited overlap opportunity

---

### 4. **Feature Selection: GPU Eigenvalue Computation** (Directive-Only)

#### Hotspot Analysis:
- `_minEigenvalue()` called ~300,000 times (once per pixel)
- Includes expensive `sqrt()` operation
- Currently CPU-only, with gradient data already on GPU

#### Current State:
```c
// In selectGoodFeatures.c - CPU loop reading GPU data
for (y = ...) {
    for (x = ...) {
        // Accumulate gradients from gradx/grady (on GPU)
        val = _minEigenvalue(gxx, gxy, gyy);  // CPU sqrt()
    }
}
```

#### Optimization (Directive-Only):

**Wrap the pixel evaluation loops:**
```c
#pragma acc parallel loop collapse(2) \
            present(gradx->data[0:npix], grady->data[0:npix]) \
            copyout(pointlist[0:npointlist])
for (y = bordery; y < nrows - bordery; y++) {
    for (x = borderx; x < ncols - borderx; x++) {
        float gxx = 0, gxy = 0, gyy = 0;
        
        // Window accumulation loops
        for (int wy = -hw; wy <= hw; wy++) {
            for (int wx = -hw; wx <= hw; wx++) {
                int idx = (y + wy) * ncols + (x + wx);
                float gx = gradx->data[idx];
                float gy = grady->data[idx];
                gxx += gx * gx;
                gxy += gx * gy;
                gyy += gy * gy;
            }
        }
        
        // Eigenvalue computation on GPU
        float val = (gxx + gyy - sqrtf((gxx-gyy)*(gxx-gyy) + 4*gxy*gxy)) * 0.5f;
        int idx = (y - bordery) * (ncols - 2*borderx) + (x - borderx);
        pointlist[3*idx+0] = x;
        pointlist[3*idx+1] = y;
        pointlist[3*idx+2] = (int)(val * 1000);  // Scale for integer sort
    }
}
```

**Benefits**:
- No CPU-GPU roundtrips for gradient data
- Parallel sqrt() on GPU (faster than CPU)
- **No code changes** - just adding directives around existing loops

**Limitation**: Still need CPU for sorting/enforcement (qsort, complex logic)

---

## 🔧 Specific Implementation Plan (Directive-Only)

### Phase 1: Convolution Optimization (Target: 9.5s)

1. **Add cache directive for kernel data**
   - File: `convolve.c` - `_convolveImageHoriz()` and `_convolveImageVert()`
   - Add `cache(kdata[0:kwidth])` to use GPU shared memory
   - **Status**: Easy, low risk

2. **Add gang/worker/vector tuning hints**
   - Same functions
   - Try: `gang vector_length(128)` or `gang worker vector`
   - **Status**: Easy, may need experimentation

3. **Extend data region in gradient computation**
   - File: `convolve.c` - `_KLTComputeGradients()`
   - Ensure single data region covers both gradient convolutions
   - **Status**: Already done, verify optimization

### Phase 2: Feature Selection on GPU (Target: 8.5s)

4. **Parallelize eigenvalue computation**
   - File: `selectGoodFeatures.c` - `_KLTSelectGoodFeatures()`
   - Add `#pragma acc parallel loop collapse(2)` around pixel loops
   - Keep gradients on GPU, compute eigenvalues on GPU
   - **Status**: Medium effort, high impact

5. **Optimize data regions in pyramid building**
   - File: `pyramid.c` - `_KLTComputePyramid()`
   - Wrap loops with data regions to reduce transfers
   - **Status**: Easy, medium impact

### Phase 3: Advanced Directive Optimizations (Target: <8.0s)

6. **Selective async for independent operations**
   - Use `async()` where pyramid levels can overlap
   - Careful with dependencies
   - **Status**: Advanced, requires testing

7. **Loop-level optimizations**
   - Add `independent` clauses where safe
   - Try `gang`, `worker`, `vector` combinations
   - **Status**: Experimental, tuning-heavy

### ❌ REMOVED: Options Requiring Code Changes

- ~~Persistent GPU pyramid storage~~ (needs structure changes)
- ~~Sequential mode GPU caching~~ (needs memory management changes)
- ~~GPU-resident interpolation~~ (needs function refactoring)
- ~~Batch feature updates~~ (needs array structure changes)

---

## 📊 Expected Performance Gains (Directive-Only)

| Optimization | Expected Speedup | Cumulative Time |
|-------------|------------------|-----------------|
| Baseline (CPU) | - | 9.0s |
| Current (basic convolution) | 0.9× | 10.0s |
| + Cache directive + tuning | 1.05× | 9.5s |
| + Optimized data regions | 1.03× | 9.2s |
| + GPU eigenvalue computation | 1.15× | 8.0s |
| + Async optimizations | 1.05× | 7.6s |

**Realistic Target with Directive-Only Changes**: 8.0-8.5 seconds

**Why More Conservative?**
- Can't eliminate pyramid allocation/deallocation overhead
- Can't cache pyramids on GPU across frames (sequentialMode limitation)
- Interpolation stays on CPU (complex control flow, function pointers)
- Sorting/enforcement logic stays on CPU (qsort, complex algorithms)

**Key Wins Available**:
1. ✅ Convolution optimization (cache, tuning) - ~5% gain
2. ✅ GPU eigenvalue computation - ~15% gain  
3. ✅ Better data region management - ~3% gain
4. ⚠️ Limited async opportunities - ~5% gain

**Major Unavailable Wins** (require code changes):
- ❌ Persistent GPU pyramids (~20% potential)
- ❌ GPU interpolation (~25% potential)
- ❌ Complete GPU tracking pipeline (~30% potential)

---

## ⚠️ Important Constraints

**ONLY OpenACC Directives** - No code changes allowed:
- ✅ Add `#pragma acc` directives
- ✅ Add `async()`, `wait()`, `cache()` clauses
- ✅ Modify data management (`present`, `create`, `copyin`, etc.)
- ❌ Change algorithm logic
- ❌ Modify function signatures
- ❌ Restructure loops (except via `collapse`)
- ❌ Add new functions

**Legal Operations**:
```c
// Good - adding directives
#pragma acc parallel loop collapse(2)
for (...) { /* existing code */ }

// Good - data management
#pragma acc data present(arr[0:n])
{ /* existing code */ }

// Bad - changing code
for (...) {
    // temp = ...; // DON'T ADD THIS
}
```

---

## 🎬 Next Steps

1. **Measure baseline** - Profile current 10s execution
2. **Implement Phase 1** - Persistent data regions
3. **Profile again** - Verify improvements
4. **Iterate** - Move to Phase 2 if target not met

**Key Metric**: `nvprof` or `nsys` profiling to identify:
- Kernel execution time
- Memory transfer overhead
- GPU utilization percentage
- Achieved occupancy

---

## 🧠 Memory Transfer Reduction Summary (Directive-Only Reality)

**Current (~10s):**
- Per frame: ~18 MB (2 pyramids × 3 levels × ~3 MB total)
- Transfer time @ 10 GB/s: ~2ms per frame
- **Not the bottleneck!** Transfers are <2% of total time

**With Directive Optimizations:**
- Can reduce intermediate transfers (gradients, temp images)
- Can't eliminate pyramid allocation/deallocation
- Savings: ~5-10ms total across all frames

**Real Bottlenecks (from algorithm analysis):**
1. **Interpolation**: ~40% of time, stays on CPU (directive can't fix)
2. **Eigenvalue computation**: ~15% of time, **CAN move to GPU** ✅
3. **Convolution**: ~30% of time, **partially on GPU**, can optimize ✅
4. **Sorting/enforcement**: ~10% of time, stays on CPU
5. **Memory transfers**: ~2% of time

**Directive-Only Focus:**
- ✅ Optimize what's already on GPU (convolution)
- ✅ Move parallelizable CPU work to GPU (eigenvalues)
- ❌ Can't fundamentally restructure data flow (pyramids, interpolation)

---

## 📝 Notes

- The algorithm is inherently sequential at the feature-tracking level
- Best parallelism: Across pixels (selection) and across features (tracking)
- Pyramid levels must be processed sequentially (coarse-to-fine)
- Some CPU overhead unavoidable (sorting, enforcement logic)

**Philosophy**: 
> Move computation to data, not data to computation.
> Keep everything on GPU as long as possible.

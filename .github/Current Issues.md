hort diagnosis from your Nsight data:

Micro-kernels dominate: textureInterpolateKernel

84,134 launches, avg 2.04 µs each → extreme launch fragmentation.
88.3% of GPU kernel time is in these tiny kernels.
Host API time is the real bottleneck (heavy per-call setup)

cudaMemcpy2DToArray: 63.8% of CUDA API time (84,134 calls, ~0.42 ms each)
cudaMallocArray + cudaFreeArray: 22.6% combined (84,134 + 84,134 calls)
cudaCreateTextureObject + cudaDestroyTextureObject: 1.4% + 0.3% (84,134 calls)
cudaMemcpy (coords/results): 5.8% (168,877 calls)
What this means

You’re creating/destroying a CUDA array and texture object and copying the image into the array for almost every interpolation call. The compute per call (2 µs) is dwarfed by setup (hundreds of µs).
Convolution kernels are fine (303 calls, ~36–45 µs each), eigenvalue negligible.
Action plan (highest impact first)

Persist the texture per frame (kill per-call cudaMallocArray/cudaMemcpy2DToArray/cudaFreeArray)

Create/bind a single cudaArray or pitched devPtr + texture object once per image.
Reuse the same texture object for all interpolation calls in that frame.
Tear down at end-of-frame.
Expected: Remove ~63.8% + 22.6% + 1.4% + 0.3% of CUDA API overhead.
Reuse device buffers for coords/results (avoid alloc/copy churn)

Keep d_coords/d_results in the tracking context; grow-capacity if needed.
Use cudaMemcpyAsync for coords/results and overlap if possible.
Expected: Cut the 5.8% memcpy API cost and reduce syncs.
Batch more work per launch (reduce 84k kernel instances)

Aggregate all interpolation points for a step/level into one call.
If helper functions are the source, add a small staging buffer to collect queries and launch once.
Expected: 10–100× fewer launches; improved occupancy and SM utilization.
Only if needed: raise workload size

More features or bigger patches help only after you implement 1–3.
More images alone won’t fix per-call setup overhead.
Quick sanity checks you can run now

Confirm the per-call array churn is real:
In your code path that does interpolation, search for any cudaMallocArray/cudaMemcpy2DToArray/cudaFreeArray/cudaCreateTextureObject calls inside hot loops. They must move to per-frame setup.
Validate kernel batching:
Count how often interpolation is invoked per feature/iteration; aim to pack all queries into one call per iteration/level.
Minimal implementation guidance

Per-frame texture:
At frame load: allocate cudaArray (or use pitched devPtr), one cudaMemcpy2D to it, create one cudaTextureObject.
Store texObj in the tracking context; pass it to all interpolation launches.
At frame end: destroy/free once.
Persistent coords/results:
Keep d_coords/d_results on the device; only resize when capacity < needed.
Copy coords in, launch once, copy results out.
Why this matters

Your kernel does ~2 µs of work, but you pay ~0.5–0.6 ms of setup per call. That’s ~250× overhead. Persisting the texture and batching will flip the ratio so compute dominates, not API overhead.
If you want, I can draft the per-frame “begin/end” texture API and the reusable coords/results buffers so you can wire them into trackFeatures.c with minimal changes.


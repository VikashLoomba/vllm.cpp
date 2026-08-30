measure(LTX25-CONNECTOR-GEMM): the connector is attention, not GEMM

#2354 put 224.882 s -- 43.52% of an LTX-2.5 render -- on host f32 GEMM at about
37 GFLOP/s, and named the first traceable step as determining whether
`vt::MatmulBT` reaches the specialized elementwise kernel on those shapes or
falls back to `MatmulOneChunkRef`. It reaches the specialized kernel. That is
not the answer.

Two instruments sharing no code agree that `Ltx2ConnectorForward` is dominated
by `vt::AttentionCross`, not by the GEMM. A `perf` profile of the function
itself puts `LoadF32` at 23.09%, `vt::SizeOf` at 20.49% and the
`AttentionCrossKernel` lambda at 10.74%, against 31.0% in the AVX-512 GEMM
micro-kernels. Wall-clock timing of the same shapes puts attention at 65.9% of a
video layer. The decomposition closes: attention plus GEMM is 93.54 s against
the 91.98 s measured for eight layers of both streams, 1.7% apart, so there is
no third term of any size.

So the 37 GFLOP/s was never the GEMM's rate. It is leaf seconds divided by GEMM
flops, which equals the GEMM's rate only if the leaf is the GEMM. On this box
the same construction reads 50.7 GFLOP/s for a layer whose GEMM runs at 153.3.
#2354's predicted 34 and its measured 37 agreed because both divided the whole
leaf by the same flops, whatever else was inside it.

The attention performs 4.2% of the layer's arithmetic and takes 65.9% of its
time, because `AttentionCrossKernel` resolves the element dtype per element
through `vt::SizeOf`, which is out-of-line in `src/vt/dtype.cpp` and the build
enables no LTO. The probe prices the repair: a hoisted-dtype reference on ONE
thread beats the shipped kernel on twenty, byte-equal on both streams.

It is not landed here. `AttentionCrossKernel` is a `vt` seam every model
reaches, `AttentionKernel` beside it carries the same defect, and this devbox's
disk cannot hold a build tree to gate a change to either. That is a row of its
own and the spec's `## Owed` names it.

One record is qualified rather than corrected. `include/vt/quant.h` states the
elementwise `[N,K]`->`[K,N]` repack as "1.16x to 1.30x on dgx". Byte-identity
reproduces exactly at the connector's shapes; the speed does not -- here the
repack is 2.06x SLOWER, outside its own same-arm control on every large shape.
The aarch64 run that would say whether that sentence needs a scope note or a
correction is queued behind four other jobs, so the header is left alone.

What this does not claim: every number is x86-64 AVX-512 on a devbox shared with
other agents at loadavg 10 to 40, so the absolute rates are lower bounds and the
ratios are what the finding rests on, each with a same-arm control beside it. No
render ran, because nothing changed.

NO ISSUE. `gh api user` returns `Your account is suspended`, so no issue could be
opened. This row takes up the first `## Owed` item of
`.agents/specs/ltx25-text-cond-device.md`, which records the same block and the
same reason. REMOTE_UNVERIFIED at the tracker.

FOLLOWING_AGENTS_PROTOCOL

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:claude-opus-5 [claude-code]

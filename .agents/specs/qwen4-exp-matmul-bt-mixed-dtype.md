# `qwen4_exp`: the `(f32, bf16)` `matmul_bt` a CUDA forward stops on

Row: `MODEL-MM-QWEN4-EXP`
Issue: [#2452](https://github.com/mudler/vllm.cpp/issues/2452)
Campaign: [#1978](https://github.com/mudler/vllm.cpp/issues/1978), spec
[`qwen4-exp-flash-next.md`](qwen4-exp-flash-next.md)

## Scope

One question and one fix. `ModelRegistry::Forward` on a CUDA queue over the tiny
`qwen4exp` GGUF fixture stops with

```text
vt cuda: matmul_bt: unsupported dtype combo (f32,bf16)->f32;
    supported: (bf16,bf16)->f32|bf16, (f32,f32)->f32|bf16
```

thrown from `src/vt/cuda/cuda_matmul.cu:425`. `ComboName(a, b, out)` names the
ACTIVATION first, so the operands are an **f32 activation** against a **bf16
weight**, producing f32. #2452 measured that stop and deliberately did not
localize it.

In scope: naming the call site, deciding which side is wrong, and the smallest
change that makes a CUDA forward pass this point. Out of scope: `qwen4_exp`'s
remaining walls, which are other waves' files —
`qwen4_exp_forward.cpp` (#2453), `qwen4_exp_qsa_block.cpp` (#2422),
`cuda_quant_dot.cu` (#2423), `gguf_keep_quant.cpp` / `model_loader.cpp` (#2397).

## The question this spec exists to answer

`AGENTS.md` "vLLM is the reference" states the polarity: vLLM resolves one model
dtype and every layer inherits it, and an `f32` value is a rare annotated
exception. A mixed `(f32, bf16)` GEMM therefore reports that one side did not
inherit. There are three shapes it could be, and they take three different fixes:

1. **An activation was widened to f32 that should have stayed bf16.** The fix is
   at the producer and the CUDA combo stays unsupported. A token gate cannot see
   this: the tokens still match while the path moves twice the bytes.
2. **The reference genuinely runs a mixed-precision GEMM here.** Then mirror it,
   cite the `file:line`, and match the accumulate and output dtypes.
3. **The CPU arm is merely laxer.** `vt::MatmulBT` (`src/vt/ops.cpp`) checks only
   `IsFloat(a) && IsFloat(b)`, and `MatmulBTKernel` on CPU reads both operands
   through a dtype-generic getter, so the mix runs there and answers. That makes
   CPU a value oracle and never a dtype-policy oracle.

## The oracle

vLLM registers no `qwen4_exp` at its pin, so the algorithm oracle is the row's
accepted lane pin, `transformers` **5.16.0**
(`.agents/oracles/transformers.md`), at
`src/transformers/models/qwen4_exp/modeling_qwen4_exp.py`. Read at that tag:

* `Qwen4ExpTextRMSNorm.forward` (`:173-178`) computes in `float()` and returns
  `output.type_as(x)`.
* `Qwen4ExpTextRMSNormGated.forward` (`:192-201`) records `input_dtype`, upcasts,
  and returns `hidden_states.to(input_dtype)`.
* `Qwen4ExpTextTopKRouter.forward` (`:907-916`) runs `F.linear` at model dtype
  and upcasts only the SOFTMAX (`dtype=torch.float`), then casts the top-k values
  back with `router_top_value.to(router_logits.dtype)`.
* `Qwen4ExpTextGatedResidual.forward` (`:955-969`) applies `hc_norm` and feeds
  `hyper_input_normed` — a `type_as(x)` result — to all three `nn.Linear`s.

Every f32 in the reference is an interior accumulation that is cast back before
the next parameterized layer. **No `nn.Linear` in this architecture ever receives
an f32 activation while its weight is model dtype.** Shape 2 is therefore ruled
out by the oracle rather than by preference.

## Method

The refusal is CUDA-only, but the composition that produces the operand dtypes is
not: `vt::MatmulBT` is device-agnostic and the CPU arm runs the same call. So the
call site is localized on the CPU tier, where a full `LoadedEngine` forward over
the same fixture already completes
(`tests/vllm/models/test_qwen4_exp_runner.cpp`, case 5), by an uncommitted
instrumentation in `vt::MatmulBT` that prints a backtrace for every
`a.dtype != b.dtype` dispatch. That names the site without a lease. The FIX is
then gated on a GPU, because the CPU tier cannot gate this refusal at all.

## Design

Filled in once the call site is measured; the fix follows the shape the
measurement selects, and this section records which one it was and why.

## Risks

* **Adding the CUDA combo would close the symptom and keep the defect.** If the
  activation was wrongly widened, the combo makes the forward run while the path
  keeps moving twice the bytes, and no token gate would ever see it.
* **A CPU-green gate proves nothing here.** CI has no GPU, and a sibling wave
  measured a mutation that left all four CPU suites green with identical logits.
* **A mutation kill against a red baseline, or one whose build failed, is void.**
  Applied-ness is proven by a counted property, not by a patch exit code.

## Tests

Red first, through the production entry point (`ModelRegistry::Forward` /
`LoadedEngine`) on a CUDA queue. A unit test that constructs a GEMM by hand
proves the kernel works and never that anything reaches it. Reachability is
proven by deleting the production call site in a scratch copy and showing the
focused gate go red.

## Gates

* The focused `qwen4_exp` device case, on a leased GPU, red before and green
  after.
* CPU-vs-CUDA parity on the localized GEMM's output, reported as `max|diff|`.
* `scripts/agent-preflight.sh --staged`.

## Owed

Recorded when the measurement lands.

## Stop conditions

* The forward stops at a NEW wall that belongs to another wave's files. Report
  the new stop point precisely and do not chase it.
* No fleet device is leasable. Then the fix is unproven on a device and says so.

## Now

`ACTIVE` — localizing the call site.

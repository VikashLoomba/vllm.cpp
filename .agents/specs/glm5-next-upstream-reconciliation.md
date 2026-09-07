# Reconcile GLM-5.3-Flash onto vLLM

Row: `MODEL-MM-GLM53-FLASH`

Issue: [#3045](https://github.com/mudler/vllm.cpp/issues/3045).
Campaign: [glm5-next-flash.md](glm5-next-flash.md).
Base: `e2fb2f06d9944c4bbe66531034479d370df67815`.

## Scope

Reconcile the campaign records after vLLM registered GLM-5.3-Flash.
Expire the transformers algorithm exception under its stated condition.
Preserve its pin and historical evidence. Record the upstream surfaces that
the next device implementation must reconcile.

This change edits this spec, the campaign spec, and the transformers oracle
record. All three are documentation. It changes no product, test, script,
policy, checker, generated file, or continuous integration configuration.
The campaign lifecycle remains `ACTIVE`.

## Upstream anchors

vLLM [PR #53906](https://github.com/vllm-project/vllm/pull/53906) merged
on 3 September 2026 at `98ed0856f31fa3aaf5e27464e2b4ef5a8ee6b2f5`.
The model registry at that object registers `Glm5NextForCausalLM`,
`Glm5NextForConditionalGeneration`, and `Glm5NextMTPModel`.
The global parity pin remains
`e126687a9a828d513c01a07cd69f025f27d63280`, which lacks these registrations.
The merged revision is a fixed source reference ahead of that pin. It is not
an accepted parity denominator or an advance of the global pin.

## Design

Add an explicit current reconciliation before the campaign's original oracle
survey. Label the survey as historical. Update its current `## Now` and
preserve the preceding status as dated history.

Add expiry fields to the `glm5_next` lane and retain its original acceptance,
scope, pin, and evidence. Follow the `qwen4_exp` expiry record's shape.
The expiry changes algorithm authority to vLLM. It does not erase earlier
component comparisons or establish model gateability.

Record source anchors for configuration, model composition, mixture of
experts (MoE), attention, KDA, k-pool caches, and their AMD dispatch.
Name applicable upstream tests and the remaining device-port obligations.

## Risks

Registration does not prove that the model builds or runs. Keep the model
gate `PENDING` until a source build runs the real checkpoint.
Do not infer the resolved runtime dtype, backend, or memory fit from source.
Do not turn historical memory measurements into a permanent hardware ceiling.
The device implementation must reconcile defaults and errors before claiming
equivalence. Numerical agreement alone cannot establish matching memory formats.

## Tests and gates

The focused documentation check searches the oracle record for
`expired_by = vllm-project/vllm 98ed0856`.
It must fail before the edit and pass afterward. Removing that line in a
scratch copy must restore the failure. No new persistent checker is needed.

Read the merged registry and the parity-pin registry with the same `git grep`.
Use `Glm4MoeForCausalLM` as the positive control for the pinned registry.
Run `git diff --check`, the record and oracle checks, and the full
`scripts/agent-preflight.sh --staged` before handoff. Report skips separately.
Verify that the global pin and unrelated records remain byte-identical.

No model, GPU, throughput, or latency gate applies to this documentation edit.
Those gates remain owed by the campaign, without a waiver.

## Evidence

On 7 September 2026, `gh pr view 53906 --repo vllm-project/vllm` returned
`MERGED`, `mergedAt=2026-09-03T16:40:36Z`, and the source object above.
`git log -S Glm5Next` on the upstream registry identifies that same commit.
The focused expiry search returned exit 1 before edits.

## Stop conditions

Do not change the global pin, product behavior, unrelated rows, or benchmark
claims. Report any required expansion to the operator.
No push, merge, GPU work, or large download belongs to this change.

## Owed

The campaign issue [#1998](https://github.com/mudler/vllm.cpp/issues/1998)
owns the real-model oracle build and execution. Device wiring remains under
[#2410](https://github.com/mudler/vllm.cpp/issues/2410).
The reconciliation issue closes only when these documentation changes land.

## Now

Specification committed before the record edits. Implementation and its full
gate remain pending until the record changes are applied and reviewed.

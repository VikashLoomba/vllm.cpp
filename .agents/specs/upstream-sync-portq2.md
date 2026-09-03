# Sync cycle `e126687a9a`, wave PORTQ-2

Row: `UPSTREAM-SYNC-HEADPIN` — inherited from the predecessor waves' specs.
**It is not a matrix row**: zero hits in `roadmap_v1.md` and in every
`*-matrix.md`, as [`upstream-sync-portq1.md`](upstream-sync-portq1.md) records.
The issues this wave cites are carried under `## Owed` below, which is the route
AGENTS.md gives when no row owns the work.
Issue: [#2646](https://github.com/mudler/vllm.cpp/issues/2646).
Row spec: [`upstream-sync-headpin.md`](upstream-sync-headpin.md).
Predecessor: [`upstream-sync-portq1.md`](upstream-sync-portq1.md) (#2632, landed
as #2642), which did entries 1-40 of the same queue.
Parent: [#2611](https://github.com/mudler/vllm.cpp/issues/2611), which owns the
290-entry queue.

## Now

PENDING — filled in when the tranche is read.

## 1. Scope

**One question, asked forty times.** For each of PORT-NOW entries **41 through
80** of `5559679229..e126687a9a` in upstream commit order, what is true **of this
tree**?

Advancing the parity pin is the last unmet clause of the cycle, and AGENTS.md
permits it only "after you reconcile every affected row and gate". The queue is
what remains to reconcile. PORTQ-1 took the first 40; this wave takes the next 40
and does not attempt more. A sibling wave, PORTQ-3, takes 81-120 concurrently and
this wave stays off its range and its record surfaces.

In scope:

1. Derive the tranche reproducibly, and show the derivation reproduces the
   recorded 290 / 1465 counts before using it.
2. One label per entry against the tree at `origin/main`, each carrying a
   verified `path:line` or the searches that found nothing.
3. Every disagreement with the recorded disposition, stated as a disagreement.
4. One issue per real gap, naming an owning row **that exists**.

Out of scope:

- **Porting.** Classification is the deliverable.
- **Advancing the pin**, and the remaining 210 entries.
- Any throughput, latency or memory number. Nothing here is executed, so nothing
  here may be quoted as a speed axis.

## 2. Design

### 2.1 The tranche is derived, not chosen

```console
$ git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u > range.txt   # 1465
$ comm -12 <(sort portnow.txt) range.txt | wc -l                          # 290
$ git rev-list --reverse 5559679229..e126687a9a | cut -c1-10 \
    | grep -x -F -f portnow_range.txt | sed -n '41,80p'                   # the tranche
```

`portnow.txt` is the 315 SHAs of
[`../sync/2026-09-01-cdefd9d.md`](../sync/2026-09-01-cdefd9d.md) §4. Both counts
are the ones `../sync/2026-09-02-e126687.md` §4 publishes, so the extraction is
verified against a committed number before anything is read off it. The
tranche's first and 40th entries reproduce PORTQ-1's published endpoints
(`3f1d40960f`, `b49eaf205a`) at offsets 1 and 40, which is the second check that
the ordering is the same one PORTQ-1 used.

### 2.2 Three labels, and a named sub-shape for the third

Identical to PORTQ-1 §2.2, deliberately, so the two tranches compose:
`ALREADY_SATISFIED`, `REAL_GAP`, `NOT_APPLICABLE` split into `surface-absent`
(discarded work) and `inert:<gate>` (deferred work, gate named).

### 2.3 Upstream is read at a revision, never in a working tree

Every read is `git -C /home/mudler/_git/vllm show <sha>:<path>` or
`git show <sha>`. An anchor read from a working tree is an anchor at whatever
that tree is checked out to.

### 2.4 Absence is never concluded from one grep, and the instrument is different

PORTQ-1's retraction is this wave's binding lesson. Re-reading a citation cannot
catch a false claim of absence, because absence has no citation to re-read. So a
missing symbol is reported only after at least two spellings and two locations,
**searched for the concept in this tree's vocabulary rather than upstream's**,
and the report says which searches were tried and returned nothing.

### 2.5 Git is asked before novelty is claimed

`git log --oneline --grep '<sha>'` and `--grep 'vllm#<PR>'`, plus a grep of
`.agents/` for both, before any entry is called newly read. PORTQ-1 found three
of its forty already triaged by `b7e414cc4`, one of those readings stale. This
tranche's first pass found four entries already re-derived against the tree by
#2524's stratified sample, recorded in `../sync/2026-09-01-cdefd9d.md` §13; each
is re-verified at this tree rather than carried.

### 2.6 The reading is delegated, and its citations are re-checked

Four fresh readers take ten entries each from one shared brief. Citations are
printed again by this wave's operator before they are written down.

## 3. Risks

- **A carried label is anchoring.** Mitigated by treating disagreement as the
  expected result.
- **`inert` is a judgement about reachability, not a grep.** Mitigated by naming
  the gate so a later reader can falsify it.
- **A prior REAL_GAP can be stale in either direction.** #2524's §13 anchors were
  taken at `63889449c`; the tree has moved. Each is re-verified here.
- **Forty is not 290.** Any rate computed here is an extrapolation from a
  non-random contiguous slice, recorded as an estimate beside the number.
- **Disk.** The host is at 95%. Nothing in this wave builds; a worktree that
  compiled would be the defect.

## 4. Gates

- `git rev-list 5559679229..e126687a9a | cut -c1-10 | sort -u | wc -l` prints
  `1465` and the PORT-NOW intersection prints `290`. The derivation is not used
  until both reproduce.
- Every `path:line` in the report resolves in the tree at the merge commit.
- Every row named as an owner resolves in `.agents/roadmap_v1.md` or a
  `.agents/*-matrix.md`.
- `scripts/agent-preflight.sh --staged`, read by grepping its output for
  `gate(s) failed` and `NOT a green` rather than by its exit code.
- `python3 scripts/check-pr-size.py --base origin/main --head HEAD`, which
  preflight skips.
- `python3 scripts/agent-pr-body.py --pr <N>` before the body becomes the commit.

## 5. Stop conditions

- Stop at entry 80. Entries 81-120 belong to PORTQ-3.
- Stop before porting. A real gap ends at an issue with an owning row.
- Stop before the pin. Nothing read here is a reason to move it.

## Owed

- The remaining 210 PORT-NOW entries of `5559679229..e126687a9a`
  ([#2611](https://github.com/mudler/vllm.cpp/issues/2611)).
- Porting the real gaps this wave names. Each carries its own issue.
- [#2524](https://github.com/mudler/vllm.cpp/issues/2524), whose §13 worked list
  this wave re-verifies for the four items inside this tranche and does not
  discharge.

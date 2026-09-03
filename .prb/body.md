Forty entries re-derived: **11 ALREADY_SATISFIED, 11 REAL_GAP, 18 NOT_APPLICABLE**
(14 surface-absent, 4 inert). **Twenty-nine of the forty disagree with the
recorded PORT-NOW disposition**, and the record was accurate about upstream in all
forty -- it simply never asked what is already here. Across the four landed
tranches that is now 126 of 160.

**Nothing here was executed: no build, no test run, no GPU, no lease.** Every
label is a static reading of source. An ALREADY_SATISFIED means the code
implements the behaviour, not that a gate observed it doing so; a REAL_GAP means
the behaviour is absent from the source, not that a failing test was seen.

**One of the eleven gaps is a label this wave got wrong and a fresh review
caught.** c76a425278 was published as NOT_APPLICABLE surface-absent on evidence
that was inverted. The commit rewrites two kernels; the report matched it to one
symbol instead of to the files it edits. cp_gather_cache really is the fp8/DCP
sibling this tree does not implement, but the other half rewrites
gather_and_maybe_dequant_cache, which this tree ports 1:1 and says so in its own
header at include/vt/ops.h:5225-5227, and which it still carries in the pre-fix
shape: GatherMlaCacheKernel still takes token_to_seq and still launches
grid = num_tokens, reached in production through BuildMlaChunkedContext. The
closing sentence had offered grid = num_tokens as proof the optimization was
present, and that line IS upstream's deleted launch. Section 6.10 records the
mechanism rather than only the fix, because a citation re-read confirms an anchor
exists and cannot see that it supports the opposite conclusion -- the same
structural hole section 2.3 names for claims of absence. The gap is #2706.

Three citations are corrected with it: 132's DCP gate is at :311-313, 128's 400
is at api_server.cpp:318-330, and 134 has five MambaSpec constructions not six.
All substance stands; these are the citation-drift shape every tranche has
produced.

Nine issues carry the eleven gaps -- #2683, #2684, #2685 (two entries), #2688,
#2689, #2693, #2694, #2696, #2706 -- plus the pre-existing #2650, which already
owns 608c12473f. Four inert entries and one inert half are collected on #2700,
and a pre-pin correctness hole found on the way is #2697. Every owner row was
grepped out of its matrix by hand, because check-agent-record.py passes on a Row:
line whether or not the row resolves; #2700 carries Row: - , so the spec's Owed
now lists what it owns, which is the obligation that line exists to carry.

Two shapes are worth the next tranche's attention. Two of the gaps are PRE-PIN
holes a commit-range queue structurally cannot see: upstream already carried the
drafting-slot reservation AT 5559679229, so 0914ed2e81 and 2ac1f683f1 are one
unit of work rather than two commits to apply. Eight entries in all turned out to
be pre-pin, the largest count of the four tranches. And f3c1638927 is a shape no
tranche has met before -- half of a NET-ZERO PAIR, both introduced and removed
after the pin, so the correct action is to skip both rather than port and unport.

608c12473f is the trailing half of a gap PORTQ-3 already filed as #2650 and must
not be ported first: the workspace floor it deletes is load-bearing here, and
removing it before the per-request packer lands turns an over-allocation into a
runtime_error on any batch with many with-context prefills.

Two source records are corrected in place, annotated rather than rewritten.
2026-09-01-cdefd9d.md section 13 cites a path that does not exist, and this is the
SECOND finding of that same wrong path in that same file -- PORTQ-3 corrected it
for a different entry and repaired it only inside its own report, so the source
went on to mislead a fourth reader. 2026-09-02-db92053.md section 4 is marked
SUPERSEDED BY THE TARGET ADVANCE, not falsified: its staging warning is correct at
its own target and its conclusion is discharged at this one. The section 4 SHA
extraction was re-run after both annotations and still yields 315, byte-identical
to the pre-annotation list.

A third proposed correction was WITHDRAWN after the upstream diff was re-read. A
reader reported that the queue's entry for 3e174bb73c presents a rename as a
behaviour change. The rename half is right and was verified independently, but the
record's sentence describes which value goes where and is accurate both before and
after the commit. Annotating a true sentence as false would have put a wrong
correction into a permanent record.

Section 2.3 gains a seventh false-zero mechanism, hit live during review: an
unquoted path variable that zsh declines to word-split, so grep received one
nonexistent path and every probe in the batch returned 0 at rc 0 with stderr
suppressed. Only the mandatory positive controls caught it. That is now three
distinct shell-level mechanisms.

**One extrapolation, and it is an estimate rather than a count.** Eleven of forty
is 27.5%; the four tranches read 15%, 22.5%, 20%, 27.5% and #2524's stratified
sample of 63 gave 17.5%. Applied to the 290 that is **roughly 58-68 real gaps,
estimated from 223 entries, not counted**. Five reasons not to treat 27.5% as the
rate, stated here rather than behind a link: this is a contiguous slice of
history, so a period effect and a sample rate are not separable; two of the eleven
are pre-pin and one is a re-finding, so the figure for gaps this range introduces
is 8 of 40, or 20%, squarely on the earlier tranches; Shape D is what is growing,
not the gap rate, because the queue cannot represent a pre-pin hole; eleven of
forty is small enough that two entries either way moves it from 22.5% to 32.5%,
and 132 moving is exactly what happened; and the tranches cluster by subsystem
differently. What IS a count: these forty contain eleven real gaps and twenty-nine
labels that contradict the record.

The pin did NOT advance and nothing read here is a reason to move it. It remains
5559679229bc961848b121ccdeaa8fa5d79bec98.

FOLLOWING_AGENTS_PROTOCOL

Closes #2680

Following-Agents-Protocol: true
AI-Assisted: true
Assisted-by: AGENT:claude-opus-5-1m [claude-code]

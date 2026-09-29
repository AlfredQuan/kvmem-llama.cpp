# Selective KV refresh (experimental)

Add `--kvmem-blend-ratio 0.1` to an existing retrieval-mode server or CLI
command to refresh about 10% of the retrieved history before Query Replay.
The default is `0` (disabled). This currently requires text-only history and
FlashAttention. The supported ratio range is `[0, 1]`.

The refresh runs when retrieval changes the attention view and the caller
replays the query. The server's `all_resident`, `unchanged_selection`, and
same-query continuation fast paths do not refresh history unnecessarily.

1. Reuse retrieval scores to select whole resident blocks, highest score first;
   ties prefer earlier historical positions. The ratio denominator is the
   resident history ending before the query boundary. Exclude the unchanged
   contiguous prefix and every block overlapping the query/replay suffix.
   Round down to whole blocks; a small ratio can select zero tokens.
2. Process selected tokens in historical order. Preserve their original RoPE
   positions, logical row IDs, and causal attention mask. Attention reads all
   available earlier KV, including refreshed earlier rows and unchanged cached
   rows. M-RoPE text models can batch nonconsecutive selected blocks, up to the
   configured batch size. Other models use consecutive runs.
3. Keep the refreshed GPU rows for Query Replay and generation. Keep CPU packed
   KV and the mean-K index unchanged. Restore original packed rows before
   reselecting, spilling, removing, or truncating those rows. This reuses the
   existing CPU store, with no second GPU KV backup.

For hybrid models, `--kvmem-blend-state carry` (default) advances the current
recurrent/conv state across selected tokens. Query Replay continues from the
resulting state. `--kvmem-blend-state reset-restore` instead checkpoints the
query-boundary recurrent/conv state, clears it **once** before refresh, advances
the temporary state across all selected spans, then restores the checkpoint.
Refreshed attention KV stays in place. This follows qw3's GDN state policy;
it does not change positions, batching, or retrieval. Both policies are
approximations, not equivalent to replaying the entire retrieved window or a
dense causal prefill. MTP draft state is not advanced by history refresh;
normal Query Replay updates its query suffix and hidden-state carry.

`KVMEM_BLEND` records tokens, blocks, consecutive runs, actual forward calls,
and elapsed milliseconds when diagnostics are enabled (`--kvmem-trace`). The
measurement includes selection, source preparation, and synchronized forwards.
`prepare_ms` includes source preparation and any recurrent checkpoint;
`forward_ms` measures model work. `state_ms` separately reports the combined
checkpoint, clear, and restore overhead (included in the total), and
`state_bytes` records the host checkpoint size.
It excludes later Query Replay and generation. Deferred source restoration is
logged separately as `KVMEM_BLEND_RESTORE`.

Run the model-backed regression with:

```sh
build/bin/kvmem-mtp-kv-test /path/to/model.gguf --blend-only
```

It checks disabled identity, selected versus untouched GPU rows, immutable CPU
KV/index, source restoration and reselection, live recurrent-state updates,
future-row masking, and first-attention drift versus consecutive-run execution
(0.5% relative RMS tolerance). It also reports full-model KV differences:
changing batch shape changes recurrent and attention rounding, which IQ3
activation quantization can amplify through later layers.
Batching is not bitwise equivalent to consecutive-run execution.
It uses Q8_0 and F16 KV with snapshots, and Q8_0 with ReplaySSM, on Qwen 27B.
Model weights are not downloaded by the test.

## Experimental weighting and original-history neighbors

`--kvmem-blend-alpha 0.25` keeps 75% original KV and 25% refreshed KV
(`0.5` keeps equal weights; the default `1` directly replaces KV). Both K and
V are mixed **after all refresh forwards**, so alpha does not change the
refreshed rows' forward trajectory or the recurrent-state update. `alpha=0`
still runs refresh, then copies the original KV bytes back exactly. In the
default `carry` mode it also advances recurrent state, so it is a diagnostic
control, not equivalent to ratio `0`.
Weighted refresh currently supports single-GPU CUDA F16/Q8_0 caches. A
multi-GPU cache with `alpha < 1` is rejected before refresh because the mixing
scratch is device-local. Q8 operands are
dequantized numerically, interpolated, and requantized, never mixed as bytes.
The existing transfer slab supplies original CPU bytes, without a second GPU
backup or a GPU-to-CPU readback of refreshed KV.

`--kvmem-blend-ratio 0.04 --kvmem-blend-neighbors` selects about 4% core
tokens by retrieval score and refreshes each core block's original predecessor
and successor too. Blocks must actually abut in original history. Duplicates,
the unchanged prefix, partial blocks, and query-overlapping blocks are excluded.
The unique recomputation fraction is at most three times the core ratio.
Missing neighbors replace the lowest-score resident full blocks, preserving
sink/recent/mandatory rows and the original window capacity. An infeasible
expansion fails explicitly instead of growing the window. This can change
retrieval coverage as well as refresh coverage; compare both in quality tests.

`KVMEM_BLEND_PLAN` logs core/refreshed/added/evicted original block IDs.
`KVMEM_BLEND` also reports alpha, actual fraction, core block count, stage-in
count and mixing time. The total includes neighbor assembly and mixing;
`forward_ms` excludes mixing. The host policy test checks original adjacency,
deduplication, budget and protected boundaries. The model-backed test also
checks weighted K/V against a numerical reference, exact alpha-zero restoration,
identical post-refresh GDN state across weights, and missing-neighbor stage-in.

The reset/restore regression additionally checks an independent zero-state
reference, byte-exact query-boundary restoration, preserved refreshed attention
KV, immutable CPU KV/index, and successful target/MTP Query Replay.

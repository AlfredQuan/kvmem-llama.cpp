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

For hybrid models, refresh **advances the current recurrent/conv state** across
selected tokens. It neither clears that state nor restores it afterwards.
Query Replay continues from the resulting state. The pre-existing query replay
checkpoint is still restored before refresh starts. This is an approximation:
it is not equivalent to replaying the entire retrieved window, nor to a dense
causal prefill. MTP draft state is not advanced by history refresh; normal Query
Replay updates its query suffix and hidden-state carry.

`KVMEM_BLEND` records tokens, blocks, consecutive runs, actual forward calls,
and elapsed milliseconds when diagnostics are enabled (`--kvmem-trace`). The
measurement includes selection, source preparation, and synchronized forwards.
`prepare_ms` and `forward_ms` split that total into preparation and model work.
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

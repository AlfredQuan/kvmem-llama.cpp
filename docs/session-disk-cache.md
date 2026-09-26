# Session KV cache in RAM and on NVMe

This extends the session ownership and matching introduced by PR #45. There
is still one active inference slot and one GPU working set. Inactive sessions
can retain their host KV in RAM or move their large payloads to an SSD/NVMe
directory. Requests continue to use the normal chat API, optionally with
`"kvmem": {"conversation_id": "my-chat"}`.

Example server arguments (in addition to your model/context/KV options):

```text
--kvmem-conversations 3
--kvmem-session-ram-gb 12
--kvmem-session-nvme-gb 40
--kvmem-session-cache-dir D:/KVMem/session-cache
```

The `GB` options use GiB (1024³ bytes) and accept fractions. `12` and `40` are
example limits, not a guarantee that every model's 200k context fits. The RAM
requirement depends on KV types, attention dimensions, recurrent checkpoints,
MTP and the requested output length.

## Capacity and LRU

- `--kvmem-conversations` limits the total number of cached sessions across
  RAM and disk, including the active session. Exceeding it discards the least
  recently used inactive session.
- `--kvmem-session-ram-gb` aliases `--kvmem-conversations-gb`. With session
  disk caching enabled, the server reserves capacity **before** a request,
  accounting for its prompt, maximum output, MTP lookahead, packed KV and
  recurrent checkpoints. An individual request larger than the RAM allowance
  returns HTTP 400 before changing the cache.
- To free RAM, the least recently used inactive RAM session moves to disk.
  If disk space/quota is insufficient, disk sessions are discarded in LRU
  order. A session larger than the disk quota is discarded when it must leave
  RAM. A failed write keeps the original RAM state until ordinary LRU eviction
  can discard it; a failed deletion keeps its quota charge.
- File size includes metadata, checksum and temporary writes. Space is reserved
  before writing. RAM includes retained KV, means, block/position metadata,
  token indexes, query state and recurrent/MTP checkpoints. Cold entries still
  need a small RAM index, which also counts toward the cap.
- Touching a session on an accepted request updates its LRU timestamp. Moving
  it between tiers does not. The target and the active request are protected
  from eviction.

The RAM limit describes session storage, **not process RSS**. Model weights,
the GPU working set, backend transfer/compute buffers, the current HTTP request
and media encoder buffers, allocator overhead, and the OS file page cache need
additional memory. Admission is conservative and may reject a request whose
eventual short generation would have fitted.

Disk mode requires KVMem, flash attention, `--kvmem-conversations > 1`, a
positive RAM cap, and an explicit cache directory. Leave `--kvmem-cpu-gb` and
`--kvmem-nvme-gb` at zero and do not enable `--kvmem-raw-k-nvme`: those older
options allocate separate block-tier arenas per session. This session file
cache works on Windows as well as POSIX and does not depend on the older
POSIX NVMe tier or `KVMEM_ENABLE_NVME`.

Without `--kvmem-session-nvme-gb`, the existing RAM-only PR #45 policy remains:
the original RAM cap is a soft retention cap and allows a larger active store.

## Save and restore

The adapter drains pending GPU transfers before detaching a session, then
moves ownership of the existing host store. It does not make another complete
KV copy. The file writer streams existing packed main KV, MTP KV, exact F32
mean sums, recurrent states and speculative carry. Shared checkpoints are
stored once. Each raw block has a section length; format/configuration, vector
bounds, token counts, file length and a whole-file checksum are checked.

A switch to a cold session first parks the outgoing session in a valid empty
execution context. RAM is freed before the incoming payload is read. Loading
uses bounded I/O chunks and allocates the destination store directly. Failed
or corrupt snapshots become cache misses, so the prompt is recomputed instead
of using incomplete KV. The stored media index uses llama.cpp's placeholder
chunks; incoming request data supplies media if a span must be replayed.

These files are **only valid within the same server run**. Their small in-memory
indexes include model/runtime identity. Each run creates a unique subdirectory
and removes only its own tracked files on graceful exit. A forcibly terminated
process can leave an orphan run directory; it is not loaded or swept by a later
server. Remove obsolete run directories yourself while those servers are
stopped. Quotas apply to each server run, not all servers sharing a drive.

The active session is loaded entirely into RAM. Restart persistence and
per-block paging of an active session are outside this implementation.

## Diagnostics and checks

`/slots` → `kvmem.conversations` reports RAM `bytes`/`bytes_max`,
`disk_bytes`/`disk_bytes_max`, `spills`, `restores`, `disk_errors`, and the existing
session/eviction counters. `--kvmem-trace` adds `session_select`, `session_spill`
and `session_restore` events.

Portable tests (no model required):

```text
cmake -S . -B build-session-host -DKVMEM_BUILD_LLAMA=OFF
cmake --build build-session-host --target kvmem-session-snapshot-test kvmem-conversation-store-test
ctest --test-dir build-session-host --output-on-failure -R "kvmem-(session-snapshot|conversation-store)-test"
```

The snapshot test also accepts `--large-file` for an explicit >4 GiB streaming
roundtrip using a 1 MiB buffer. It temporarily needs just over 4 GiB of free
space and removes its files when it finishes.

Real model regression, including interleaved A/B/C histories, exact answers
against a RAM reference, prefix cache hits, corruption fallback and eviction:

```text
python scripts/test_server_session_disk.py --server PATH/llama-kvmem-server --model PATH/model.gguf --output artifacts/session-disk
```

Add `--mtp` for a model with an MTP head. The script starts only its own server
processes on unused ports and downloads no models.

### Local verification (Windows, 2026-09-26)

- Built with MSVC 19.44 and CUDA 12.9.86 for the local RTX 5050/5060 Ti.
- Snapshot, conversation-policy and server-options CTest targets passed.
- A >4 GiB file roundtrip passed with a 1 MiB I/O buffer.
- Qwen3.5-0.8B Q8_0: interleaved A/B/C disk restores preserved answers and
  resumed 5,579–5,580 prefix tokens. Capacity rejection, corrupt-file fallback
  and quota eviction passed.
- Qwen3.8-27B IQ3_S with MTP: disk restores matched the RAM reference exactly,
  resuming 6,106–6,108 tokens. The one-file disk-quota case retained the more
  recent B session and discarded A. Corruption and undersized-disk cases passed.
  The roundtrip fixture used a 1,563,079,285-byte RAM allowance; retained session
  bytes stayed below 882,029,335, with two cold files totaling up to 1,742,224,146
  bytes. These are session counters, not RSS.
- The original RAM-only regression passed, including count LRU, byte-cap
  eviction and rejected-request isolation. Its fixed 90% hit assertion was
  corrected to allow the existing non-MTP query-checkpoint replay (512 tokens
  plus template slack); the test still requires a substantial cached prefix.

Real-model fixtures use roughly 6k-token histories. This is not a 200k-token
inference or disk-throughput benchmark.

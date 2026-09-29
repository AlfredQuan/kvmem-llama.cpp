#include "llama-kvmem-blend.h"
#include <cstdio>

static void check(bool value) { if (!value) throw std::runtime_error("blend policy regression"); }
int main() {
    kvmem::KvMemStoreConfig cfg;
    cfg.block_tokens = 32; cfg.select_budget = 32*10; cfg.sink_blocks = 1; cfg.recent_blocks = 1;
    kvmem::KvMemStore s(cfg);
    s.register_append(32*30);
    const std::vector<uint32_t> resident{0, 3, 5, 6, 9, 12, 15, 18, 22, 29};
    for (uint32_t id = 0; id < 30; ++id) s.set_block_gpu_slot(id, -1);
    for (size_t i = 0; i < resident.size(); ++i) s.set_block_gpu_slot(resident[i], i);
    std::vector<double> scores(30, 0); scores[5] = scores[6] = 2; scores[9] = 1;
    s.set_retrieval_scores(scores);
    auto plain = kvmem_plan_blend(s, 29*32, .34f, false, {29});
    check(plain.core == std::vector<uint32_t>({5,6,9}));
    check(plain.refresh == plain.core && plain.tokens == 96 && plain.window_tokens == 288);
    check(plain.resident == resident && plain.added.empty());
    auto neighbors = kvmem_plan_blend(s, 29*32, .23f, true, {22,29});
    check(neighbors.core == std::vector<uint32_t>({5,6}));
    check(neighbors.refresh == std::vector<uint32_t>({4,5,6,7})); // not sparse neighbors 3/9
    check(neighbors.added == std::vector<uint32_t>({4,7}));
    check(neighbors.evicted == std::vector<uint32_t>({18,15})); // protect mandatory 22, recent 29, sink 0
    check(neighbors.resident.size() == resident.size() && neighbors.tokens == 128);
    check(kvmem_plan_blend(s, 29*32, 0, true, {}).refresh.empty());
    // The right neighbor overlaps the query, while the left one is valid.
    scores.assign(30, 0); scores[22] = 3; s.set_retrieval_scores(scores);
    auto edge = kvmem_plan_blend(s, 23*32+1, .12f, true, {29});
    check(edge.core == std::vector<uint32_t>({22}));
    check(edge.refresh == std::vector<uint32_t>({21,22}));
    // All unselected rows protected: fail without silently growing the window.
    bool failed = false;
    try { kvmem_plan_blend(s, 29*32, .12f, true, resident); }
    catch (const std::runtime_error &) { failed = true; }
    check(failed);
    std::puts("blend policy: original adjacency, dedup, exact capacity, protected/query boundaries passed");
}

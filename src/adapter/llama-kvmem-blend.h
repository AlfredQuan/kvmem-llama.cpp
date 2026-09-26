#pragma once

#include "kvmem/kvmem_store.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

struct kvmem_blend_plan {
    std::vector<uint32_t> core, refresh, resident, added, evicted;
    uint32_t window_tokens = 0, tokens = 0;
};

// Host-only policy. Neighbors refer to original history, never sparse-window
// adjacency. Any missing neighbors replace low-score, unprotected residents.
inline kvmem_blend_plan kvmem_plan_blend(const kvmem::KvMemStore & store,
        uint32_t boundary, float ratio, bool neighbors, const std::vector<uint32_t> & mandatory) {
    kvmem_blend_plan p;
    const auto & blocks = store.blocks();
    uint32_t prefix_end = 0;
    for (const auto & b : blocks) {
        if (b.gpu_slot < 0) continue;
        p.resident.push_back(b.block_id);
        if (b.orig_pos_end() > boundary) continue;
        if (b.orig_pos_start == prefix_end) prefix_end = b.orig_pos_end();
        else if (!neighbors || b.n_tokens == store.config().block_tokens) p.core.push_back(b.block_id);
        p.window_tokens += b.n_tokens;
    }
    auto score = [&](uint32_t id) { return std::isfinite(blocks[id].retrieval_score) ? blocks[id].retrieval_score : 0.0; };
    auto better = [&](uint32_t a, uint32_t b) { return score(a) != score(b) ? score(a) > score(b) : a < b; };
    std::sort(p.core.begin(), p.core.end(), better);
    const uint32_t budget = (uint32_t) std::floor(p.window_tokens * (double) ratio);
    uint32_t core_tokens = 0;
    p.core.erase(std::remove_if(p.core.begin(), p.core.end(), [&](uint32_t id) {
        if (blocks[id].n_tokens > budget-core_tokens) return true;
        core_tokens += blocks[id].n_tokens;
        return false;
    }), p.core.end());
    p.refresh = p.core;
    if (neighbors && !p.core.empty()) {
        for (uint32_t id : p.core) for (int delta : {-1, 1}) {
            const int64_t next = (int64_t) id + delta;
            if (next < 0 || next >= (int64_t) blocks.size()) continue;
            const auto & b = blocks[next];
            if (b.orig_pos_start < prefix_end || b.orig_pos_end() > boundary ||
                    b.n_tokens != store.config().block_tokens) continue;
            if ((delta < 0 && b.orig_pos_end() != blocks[id].orig_pos_start) ||
                (delta > 0 && b.orig_pos_start != blocks[id].orig_pos_end())) continue;
            p.refresh.push_back((uint32_t) next);
        }
    }
    std::sort(p.refresh.begin(), p.refresh.end());
    p.refresh.erase(std::unique(p.refresh.begin(), p.refresh.end()), p.refresh.end());
    for (uint32_t id : p.refresh) {
        p.tokens += blocks[id].n_tokens;
        if (blocks[id].gpu_slot < 0) p.added.push_back(id);
    }
    if (!p.added.empty()) {
        std::vector<uint32_t> victims;
        const uint32_t recent_begin = blocks.size() - std::min<size_t>(blocks.size(), store.config().recent_blocks);
        for (uint32_t id : p.resident) {
            const auto & b = blocks[id];
            if (b.orig_pos_start < prefix_end || id < store.config().sink_blocks || id >= recent_begin ||
                    b.orig_pos_end() > boundary || b.n_tokens != store.config().block_tokens ||
                    std::binary_search(p.refresh.begin(), p.refresh.end(), id) ||
                    std::find(mandatory.begin(), mandatory.end(), id) != mandatory.end()) continue;
            victims.push_back(id);
        }
        std::sort(victims.begin(), victims.end(), [&](uint32_t a, uint32_t b) { return better(b, a); });
        if (victims.size() < p.added.size()) throw std::runtime_error("KVMem blend neighbors exceed unprotected history capacity");
        p.evicted.assign(victims.begin(), victims.begin()+p.added.size());
        p.resident.erase(std::remove_if(p.resident.begin(), p.resident.end(), [&](uint32_t id) {
            return std::find(p.evicted.begin(), p.evicted.end(), id) != p.evicted.end();
        }), p.resident.end());
        p.resident.insert(p.resident.end(), p.added.begin(), p.added.end());
        std::sort(p.resident.begin(), p.resident.end());
    }
    return p;
}

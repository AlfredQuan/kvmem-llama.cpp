#pragma once

// Included after ServerState and conversation bookkeeping. The cold entry
// keeps only selection metadata; packed main/MTP KV and recurrent checkpoints
// are streamed. No second whole-session serialization buffer is allocated.
static void session_write(ServerState & st, int id, kvmem::SnapshotWriter & out) {
    const auto & conv = st.conv.at(id);
    out.scalar(uint64_t(0x314e4f4953534553ull)); // SESSION1, process-local version
    out.scalar(int32_t(id)); out.scalar(conv.stored);
    std::vector<std::shared_ptr<const MultimodalCheckpointData>> unique;
    auto index = [&](const std::shared_ptr<const MultimodalCheckpointData> & p) -> uint32_t {
        if (!p) return UINT32_MAX;
        auto it = std::find(unique.begin(), unique.end(), p);
        if (it == unique.end()) { unique.push_back(p); return uint32_t(unique.size()-1); }
        return uint32_t(it - unique.begin());
    };
    std::vector<uint32_t> indices;
    for (const auto & c : conv.mm_checkpoints) indices.push_back(index(c.data));
    const uint32_t live = index(conv.mm_live_checkpoint);
    out.scalar(uint32_t(unique.size()));
    for (const auto & p : unique) {
        out.vector(p->recurrent); out.vector(p->draft_carry); out.vector(p->tail_mean);
    }
    out.vector(indices); out.scalar(live);
    out.vector(conv.gdn_ckpt); out.vector(conv.gdn_carry);
    out.vector(conv.gdn_query_carry); out.vector(conv.gdn_ckpt_query);
    llama_kvmem_store_snapshot_write(st.conv_table.find(id)->store_id, out);
}

static bool session_spill(ServerState & st, int id, int protected_id) {
    const auto started = std::chrono::steady_clock::now();
    if (id == st.conv_active || id == protected_id || st.conv.at(id).cold) return false;
    auto & conv = st.conv.at(id);
    if (!conv.stored && conv.cached_tokens.empty()) return false; // no reusable payload to save
    const int32_t store = st.conv_table.find(id)->store_id;
    try {
        // A previously restored file whose removal failed stays charged until
        // it can be removed. Never overwrite it outside the quota.
        if (!st.session_files->erase(id)) throw std::runtime_error("cannot remove previous session snapshot");
        kvmem::SnapshotWriter size;
        session_write(st, id, size);
        if (size.bytes() + 8 > st.session_files->limit()) return false;
        for (int victim : st.conv_table.lru_order()) {
            if (size.bytes() + 8 <= st.session_files->limit() - st.session_files->bytes() &&
                size.bytes() + 8 <= st.session_files->available()) break;
            if (victim != protected_id && victim != st.conv_active && victim != id && st.session_files->contains(victim))
                conversation_evict(st, victim, "disk_lru");
        }
        conv.restore_bytes = std::max(conversation_bytes(st, id), llama_kvmem_store_capacity(conv.stored));
        st.session_files->save(id, size.bytes(), [&](kvmem::SnapshotWriter & out) { session_write(st, id, out); });
        // The file is complete before any RAM payload is released.
        llama_kvmem_store_release_payload(store);
        conv.disk_gen = !conv.gdn_ckpt.empty(); conv.disk_query = !conv.gdn_ckpt_query.empty();
        for (auto & checkpoint : conv.mm_checkpoints) checkpoint.data.reset();
        conv.mm_live_checkpoint.reset();
        std::vector<uint8_t>().swap(conv.gdn_ckpt); std::vector<uint8_t>().swap(conv.gdn_carry);
        std::vector<uint8_t>().swap(conv.gdn_query_carry); std::vector<uint8_t>().swap(conv.gdn_ckpt_query);
        conv.cold = true;
        st.conv_table.set_bytes(id, conversation_bytes(st, id));
        ++st.conv_counts.spills;
        kvmem_diag("KVMEM_TRACE session_spill id=%d disk_bytes=%llu ram_bytes=%llu ms=%.2f\n", id,
            (unsigned long long)(size.bytes()+8), (unsigned long long)st.conv_table.find(id)->bytes,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
        return true;
    } catch (const std::exception & e) {
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM session spill id=%d failed: %s\n", id, e.what());
        return false;
    }
}

static bool session_restore(ServerState & st, int id) {
    const auto started = std::chrono::steady_clock::now();
    auto & conv = st.conv.at(id);
    const int32_t store = st.conv_table.find(id)->store_id;
    std::vector<MultimodalCheckpoint> checkpoints = conv.mm_checkpoints;
    std::shared_ptr<const MultimodalCheckpointData> live;
    std::vector<uint8_t> gen, carry, query_carry, query;
    try {
        st.session_files->load(id, [&](kvmem::SnapshotReader & in) {
            in.expect(uint64_t(0x314e4f4953534553ull)); in.expect(int32_t(id)); in.expect(conv.stored);
            const uint32_t n = in.scalar<uint32_t>();
            if (n > checkpoints.size()+1) throw std::runtime_error("invalid session checkpoint count");
            std::vector<std::shared_ptr<const MultimodalCheckpointData>> unique;
            for (uint32_t i = 0; i < n; ++i) {
                auto p = std::make_shared<MultimodalCheckpointData>();
                p->recurrent = in.vector<uint8_t>(conv.restore_bytes);
                p->draft_carry = in.vector<uint8_t>(conv.restore_bytes);
                p->tail_mean = in.vector<float>(conv.restore_bytes/sizeof(float));
                p->accounting = st.mm_checkpoint_accounting;
                p->accounting->live_bytes += p->bytes();
                p->accounting->peak_bytes = std::max(p->accounting->peak_bytes, p->accounting->live_bytes);
                unique.push_back(std::move(p));
            }
            auto get = [&](uint32_t index) -> std::shared_ptr<const MultimodalCheckpointData> {
                if (index == UINT32_MAX) return {};
                if (index >= unique.size()) throw std::runtime_error("invalid session checkpoint reference");
                return unique[index];
            };
            const auto indices = in.vector<uint32_t>(checkpoints.size());
            if (indices.size() != checkpoints.size()) throw std::runtime_error("session checkpoint count changed");
            for (size_t i = 0; i < indices.size(); ++i) checkpoints[i].data = get(indices[i]);
            live = get(in.scalar<uint32_t>());
            gen = in.vector<uint8_t>(conv.restore_bytes); carry = in.vector<uint8_t>(conv.restore_bytes);
            query_carry = in.vector<uint8_t>(conv.restore_bytes); query = in.vector<uint8_t>(conv.restore_bytes);
            llama_kvmem_store_snapshot_read(store, in);
        });
        conv.mm_checkpoints = std::move(checkpoints); conv.mm_live_checkpoint = std::move(live);
        conv.gdn_ckpt = std::move(gen); conv.gdn_carry = std::move(carry);
        conv.gdn_query_carry = std::move(query_carry); conv.gdn_ckpt_query = std::move(query);
        conv.cold = false;
        if (!st.session_files->erase(id)) ++st.conv_counts.disk_errors;
        st.conv_table.set_bytes(id, conversation_bytes(st, id));
        ++st.conv_counts.restores;
        kvmem_diag("KVMEM_TRACE session_restore id=%d rows=%u ms=%.2f\n", id, conv.stored,
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now()-started).count());
        return true;
    } catch (const std::exception & e) {
        llama_kvmem_store_release_payload(store);
        ++st.conv_counts.disk_errors;
        LOG_WRN("srv    KVMEM session restore id=%d failed: %s; cache miss\n", id, e.what());
        return false;
    }
}

// Reserve the entire target before allocating/reading it. Demotion never
// changes LRU timestamps. Cold metadata counts against the RAM budget too.
static bool session_make_room(ServerState & st, int target, uint64_t reserve) {
    if (reserve > st.conv_limits.max_bytes) return false;
    auto fits = [&] {
        uint64_t other = 0;
        for (const auto & entry : st.conv_table.entries()) if (entry.id != target) other += entry.bytes;
        return other <= st.conv_limits.max_bytes - reserve;
    };
    for (int victim : st.conv_table.lru_order()) {
        if (fits()) return true;
        if (victim == target || victim == st.conv_active || !st.conv.count(victim)) continue;
        if (!st.conv.at(victim).cold && !session_spill(st, victim, target))
            conversation_evict(st, victim, "ram_lru");
    }
    for (int victim : st.conv_table.lru_order()) {
        if (fits()) return true;
        if (victim != target && victim != st.conv_active) conversation_evict(st, victim, "metadata_lru");
    }
    return fits();
}

static void session_begin_request(ServerState & st, const kvmem_prompt & prompt,
                                   const std::string & client_id, int predict) {
    // Model buffers, GPU working set, request/media decoding and file-system
    // page cache are outside this retained-session budget.
    const uint64_t rows = uint64_t(prompt.tokens.size()) + uint64_t(std::max(0, predict)) +
        (st.spec.ok ? uint64_t(std::max(0, st.spec_n_max)) + 1 : 0);
    if (rows > UINT32_MAX) throw std::invalid_argument("session token count overflow");
    uint64_t reserve = llama_kvmem_store_capacity(uint32_t(rows));
    std::vector<kvmem_store_match> matches;
    for (const auto & entry : st.conv_table.entries()) matches.push_back(conversation_match(st, entry.id, prompt));
    auto limits = st.conv_limits; limits.max_bytes = 0; limits.attached = st.conv_active;
    const auto plan = kvmem_store_select(matches, int(prompt.tokens.size())-(st.spec.ok ? 1 : 0), st.spec.ok, client_id, limits);
    int target = plan.id;
    if (target >= 0) reserve = std::max({reserve, st.conv_table.find(target)->bytes, st.conv.at(target).restore_bytes});
    if (reserve > st.conv_limits.max_bytes)
        throw std::invalid_argument("request exceeds session RAM capacity (requires " + std::to_string(reserve) +
            " bytes, cap " + std::to_string(st.conv_limits.max_bytes) +
            "); increase --kvmem-session-ram-gb or reduce context/max_tokens");
    if (target != st.conv_active && st.conv_active >= 0) {
        if (!st.mm_committed || st.conv_table.find(st.conv_active)->store_id != llama_kvmem_store_current())
            throw std::runtime_error("session switch requires a committed active conversation");
        const int outgoing = st.conv_active;
        conversation_swap(st, st.conv.at(outgoing));
        if (!llama_kvmem_store_park()) {
            conversation_swap(st, st.conv.at(outgoing));
            if (!llama_kvmem_store_n_tokens() && !st.cached_tokens.empty()) memory_clear_all(st);
            throw std::runtime_error("could not park active session");
        }
        st.conv_active = -1;
        st.mm_live_checkpoint.reset(); st.mm_rollback.reset(); st.mm_rollback_prompt.reset();
        st.mm_pending_query.reset();
        if (!llama_kvmem_store_rows(st.conv_table.find(outgoing)->store_id)) conversation_drop_payload(st.conv.at(outgoing));
        st.conv_table.set_bytes(outgoing, conversation_bytes(st, outgoing));
    }
    // Enforce the count limit after parking, so the previous active session is
    // eligible too. Never destroy the selected target.
    for (int victim : st.conv_table.lru_order()) {
        if (st.conv_table.count() + (target < 0 ? 1 : 0) <= st.conv_limits.max_stores) break;
        if (victim != target && victim != st.conv_active) conversation_evict(st, victim, "lru");
    }
    if (st.conv_table.count() + (target < 0 ? 1 : 0) > st.conv_limits.max_stores ||
        !session_make_room(st, target, reserve)) throw std::runtime_error("cannot free session cache capacity");
    if (target >= 0 && st.conv.at(target).cold && !session_restore(st, target)) {
        if (!conversation_evict(st, target, "invalid_snapshot")) throw std::runtime_error("cannot remove invalid session snapshot");
        target = -1;
    }
    if (target < 0) {
        const int32_t store = llama_kvmem_store_create();
        if (store < 0) throw std::runtime_error("cannot allocate session store");
        target = st.conv_table.add(store); st.conv.emplace(target, kvmem_conversation{});
    }
    if (target != st.conv_active) {
        const int32_t store = st.conv_table.find(target)->store_id;
        const bool restored = llama_kvmem_store_switch(store);
        if (llama_kvmem_store_current() != store) throw std::runtime_error("cannot attach session store");
        conversation_swap(st, st.conv.at(target)); st.conv_active = target;
        st.mm_live_checkpoint.reset();
        if (!restored) memory_clear_all(st);
        ++st.conv_counts.switches;
    }
    st.conv_table.touch(target, ++st.conv_clock);
    if (plan.id == target) ++st.conv_counts.extends; else ++st.conv_counts.forks;
    conversation_publish(st);
    kvmem_diag("KVMEM_TRACE session_select id=%d keep=%d reserve=%llu\n", target,
        plan.id == target ? plan.keep : 0, (unsigned long long)reserve);
}

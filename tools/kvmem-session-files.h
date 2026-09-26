#pragma once

#include "kvmem/snapshot.hpp"
#include <chrono>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <string>

// Private to one server run. Only tracked files are removed; other runs and
// user files in the configured directory are never swept. Temporary files are
// charged before the first write and remain charged if removal fails.
class kvmem_session_files {
public:
    kvmem_session_files(const std::filesystem::path & root, uint64_t limit) : limit_(limit) {
        std::filesystem::create_directories(root);
        for (int attempt = 0; attempt < 32; ++attempt) {
            dir_ = root / ("run-" + std::to_string(std::chrono::high_resolution_clock::now().time_since_epoch().count()) +
                           "-" + std::to_string(std::random_device{}()));
            if (std::filesystem::create_directory(dir_)) {
                std::filesystem::permissions(dir_, std::filesystem::perms::owner_all,
                                             std::filesystem::perm_options::replace);
                return;
            }
        }
        throw std::runtime_error("cannot create private session cache directory");
    }
    ~kvmem_session_files() {
        for (const auto & entry : files_) { std::error_code ec; std::filesystem::remove(entry.second.path, ec); }
        std::error_code ec; std::filesystem::remove(dir_, ec);
    }
    kvmem_session_files(const kvmem_session_files &) = delete;
    kvmem_session_files & operator=(const kvmem_session_files &) = delete;
    uint64_t bytes() const { return bytes_; }
    uint64_t limit() const { return limit_; }
    uint64_t available() const { return std::filesystem::space(dir_).available; }
    const std::filesystem::path & directory() const { return dir_; }
    std::filesystem::path path(int id) const { return files_.at(id).path; }
    bool contains(int id) const { return files_.count(id) != 0; }
    bool erase(int id) {
        const auto it = files_.find(id);
        if (it == files_.end()) return true;
        std::error_code ec;
        std::filesystem::remove(it->second.path, ec);
        if (ec) return false;
        bytes_ -= it->second.bytes; files_.erase(it); return true;
    }
    template<class Write> void save(int id, uint64_t payload_bytes, const Write & write) {
        if (contains(id) || payload_bytes > UINT64_MAX - 8 || payload_bytes + 8 > limit_ - bytes_)
            throw std::runtime_error("session disk quota exceeded");
        const auto total = payload_bytes + 8;
        if (available() < total)
            throw std::runtime_error("insufficient free space for session snapshot");
        const auto temp = dir_ / (std::to_string(id) + ".tmp");
        auto final = dir_ / (std::to_string(id) + ".kv");
        files_.emplace(id, record{temp, total}); bytes_ += total;
        try {
            std::ofstream file(temp, std::ios::binary | std::ios::trunc);
            file.exceptions(std::ios::badbit | std::ios::failbit);
            kvmem::SnapshotWriter out([&](const void * p, size_t n) { file.write((const char *)p, std::streamsize(n)); });
            write(out);
            if (out.bytes() != payload_bytes) throw std::runtime_error("session changed during snapshot");
            const uint64_t hash = out.hash(); file.write((const char *)&hash, sizeof(hash));
            file.flush(); file.close();
            std::filesystem::rename(temp, final);
            files_.at(id).path.swap(final); // noexcept: never orphan a renamed file
        } catch (...) { erase(id); throw; }
    }
    template<class Read> void load(int id, const Read & read) const {
        const auto & record = files_.at(id);
        if (std::filesystem::file_size(record.path) != record.bytes || record.bytes < 8)
            throw std::runtime_error("truncated session snapshot");
        std::ifstream file(record.path, std::ios::binary);
        file.exceptions(std::ios::badbit | std::ios::failbit);
        kvmem::SnapshotReader in([&](void * p, size_t n) { file.read((char *)p, std::streamsize(n)); }, record.bytes - 8);
        read(in);
        uint64_t hash = 0; file.read((char *)&hash, sizeof(hash));
        if (in.remaining() || hash != in.hash()) throw std::runtime_error("session snapshot checksum mismatch");
    }
private:
    struct record { std::filesystem::path path; uint64_t bytes; };
    std::filesystem::path dir_;
    std::map<int, record> files_;
    uint64_t bytes_ = 0, limit_;
};

/*
    This file is part of Leela Zero.
    Copyright (C) 2017-2019 Michael O and contributors

    Leela Zero is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    Leela Zero is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with Leela Zero.  If not, see <http://www.gnu.org/licenses/>.

    Additional permission under GNU GPL version 3 section 7

    If you modify this Program, or any covered work, by linking or
    combining it with NVIDIA Corporation's libraries from the
    NVIDIA CUDA Toolkit and/or the NVIDIA CUDA Deep Neural
    Network library and/or the NVIDIA TensorRT inference library
    (or a modified version of those libraries), containing parts covered
    by the terms of the respective license agreement, the licensors of
    this Program grant you additional permission to convey the resulting
    work.
*/

#ifndef NNCACHE_H_INCLUDED
#define NNCACHE_H_INCLUDED

#include "config.h"

#include <array>
#include <atomic>
#include <deque>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

class UCTNode;
class GameState;
struct BackupData {
    struct NodeFactor {
        UCTNode* node;
        float factor;
        NodeFactor(UCTNode* node, float factor) : node(node), factor(factor) {}
    };
    float eval{ -1.0f };
    std::vector<NodeFactor> path;
    int symmetry;
    std::unique_ptr<GameState> state;
    std::atomic<unsigned>* pending_counter;
    //int multiplicity{1};
};

class NNCache {
public:

    // Maximum size of the cache in number of items.
    static constexpr int MAX_CACHE_COUNT = 150'000;

    // Minimum size of the cache in number of items.
    static constexpr int MIN_CACHE_COUNT = 1;

    struct Netresult {
        // 19x19 board positions
        std::array<float, NUM_INTERSECTIONS> policy;

        // pass
        float policy_pass;

        // winrate
        float winrate;

        Netresult() : policy_pass(0.0f), winrate(-1.0f) {
            //policy.fill(0.0f);
        }
    };

    NNCache(int size = MAX_CACHE_COUNT);  // ~ 208MiB

    // Set a reasonable size gives max number of playouts
    void set_size_from_playouts(int max_playouts);

    // Resize NNCache
    void resize(int size);
    void clear();

    // Return the hit rate ratio.
    std::pair<int, int> hit_rate() const {
        return {m_hits, m_lookups};
    }

    void clear_stats();
    void dump_stats();

    // Return the estimated memory consumption of the cache.
    size_t get_estimated_size();

    struct Entry {
        //Entry(const Netresult& r) : result(r) {}
        std::atomic_flag ready{ ATOMIC_FLAG_INIT };
        Netresult result;  // ~ 1.4KiB
        std::vector<BackupData> backup_obligations;
    };

    static constexpr size_t ENTRY_SIZE =
        sizeof(Entry)
        + sizeof(std::uint64_t)
        + sizeof(std::shared_ptr<Entry>);

    std::shared_ptr<NNCache::Entry> lookup_and_insert(BackupData& bd, bool& ready, bool& first_visit);
private:
    
    size_t m_size;
    unsigned m_parts;

    // Statistics
    std::atomic<unsigned> m_hits{0};
    std::atomic<unsigned> m_lookups{0};
    std::atomic<unsigned> m_inserts{0};

    class CachePartition {
    private:
        // Map from hash to {features, result}
        std::unordered_map<std::uint64_t, std::shared_ptr<Entry>> m_cache;
        // Order entries were added to the map.
        std::deque<size_t> m_order;
        std::mutex m_mutex;
        //std::atomic<uint8_t> m_lock;
    public:
        size_t get_size() {
            return m_cache.size();
        }
        void resize(int size) {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_cache.reserve(size);
            while (m_order.size() > size) {
                m_cache.erase(m_order.front());
                m_order.pop_front();
            }
        }
        void clear() {
            std::lock_guard<std::mutex> lk(m_mutex);
            m_cache.clear();
            m_order.clear();
        }
        std::shared_ptr<NNCache::Entry> lookup(std::uint64_t hash, BackupData& bd, bool& ready) {
            std::lock_guard<std::mutex> lk(m_mutex);
            auto iter = m_cache.find(hash);
            if (iter != m_cache.end()) {
                auto result = iter->second;
                if (result->ready.test_and_set())
                    ready = true;
                else {
                    result->backup_obligations.emplace_back(std::move(bd));
                    result->ready.clear();
                }
                return result;
            }
            return nullptr;
        }
        std::shared_ptr<NNCache::Entry> insert(std::uint64_t hash, size_t sz, BackupData& bd) {
            std::lock_guard<std::mutex> lk(m_mutex);
            // If the cache is too large, remove the oldest entry.
            if (m_order.size() >= sz) {
                m_cache.erase(m_order.front());
                m_order.pop_front();
            }
            auto result = std::make_shared<Entry>();
            result->backup_obligations.emplace_back(std::move(bd));

            m_cache.emplace(hash, result);
            m_order.push_back(hash);
            return result;
        }
    };
    std::vector<CachePartition*> m_partitions;
};

using Netresult_ptr = std::shared_ptr<NNCache::Entry>;

#endif

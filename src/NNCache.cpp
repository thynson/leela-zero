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

#include "config.h"
#include <functional>
#include <memory>

#include "NNCache.h"
#include "Utils.h"
#include "UCTSearch.h"
#include "GTP.h"

const int NNCache::MAX_CACHE_COUNT;
const int NNCache::MIN_CACHE_COUNT;
const size_t NNCache::ENTRY_SIZE;

NNCache::NNCache(int size) {
    m_parts = 1 << cfg_nncache_exp;
    m_partitions.resize(m_parts--);
    for (auto& p : m_partitions) {
        p = new CachePartition();
    }
    resize(size);
}

std::shared_ptr<NNCache::Entry> NNCache::lookup_and_insert(BackupData& bd, bool& ready, bool& first_visit) {
    auto state = bd.state.get();
    const auto hash = state->board.get_hash();
    /*
    ++m_lookups;
    bd.symmetry = 0;
    auto result = m_partitions[hash & m_parts]->lookup(hash, bd, ready);
    if (result) {
        ++m_hits;
        return result;
    }*/
    // If we are not generating a self-play game, try to find
    // symmetries if we are in the early opening.
    if (!cfg_noise && !cfg_random_cnt
        && state->get_movenum()
        < (state->get_timecontrol().opening_moves(BOARD_SIZE) / 2)) {
        // See if we already have this in the cache.
        for (auto sym = 0; sym < Network::NUM_SYMMETRIES; ++sym) {
            ++m_lookups;
            const auto hash = state->get_symmetry_hash(sym);
            bd.symmetry = sym;
            auto result = m_partitions[hash & m_parts]->lookup(hash, bd, ready);
            if (result) {
                ++m_hits;
                return result;
            }
        }
    }
    bd.symmetry = Network::IDENTITY_SYMMETRY; // = 0
    ++m_inserts;
    first_visit = true;
    return m_partitions[hash & m_parts]->insert(hash, m_size, bd);
}

void NNCache::resize(int size) {
    m_size = size / m_partitions.size();
    for (auto& p : m_partitions) {
        p->resize(m_size);
    }
}

void NNCache::clear() {
    for (auto& p : m_partitions) {
        p->clear();
    }
}

void NNCache::set_size_from_playouts(int max_playouts) {
    // cache hits are generally from last several moves so setting cache
    // size based on playouts increases the hit rate while balancing memory
    // usage for low playout instances. 150'000 cache entries is ~208 MiB
    constexpr auto num_cache_moves = 3;
    auto max_playouts_per_move =
        std::min(max_playouts,
                 UCTSearch::UNLIMITED_PLAYOUTS / num_cache_moves);
    auto max_size = num_cache_moves * max_playouts_per_move;
    max_size = std::min(MAX_CACHE_COUNT, std::max(MIN_CACHE_COUNT, max_size));
    resize(max_size);
}

void NNCache::clear_stats() {
    m_hits = m_lookups = m_inserts = 0;
}

void NNCache::dump_stats() {
    auto hits = m_hits.load(), lookups = m_lookups.load();
    Utils::myprintf(
        "NNCache: %d/%d hits/lookups = %.1f%% hitrate, %d inserts, size: ",
        hits, lookups, 100. * hits / (lookups + 1),
        m_inserts.load());
    for (auto& p : m_partitions) {
        Utils::myprintf("%u, ", p->get_size());
    }
    Utils::myprintf("\n");
}

size_t NNCache::get_estimated_size() {
    unsigned cnt = 0;
    for (auto& p : m_partitions) {
        cnt += p->get_size();
    }
    return cnt * NNCache::ENTRY_SIZE;
}

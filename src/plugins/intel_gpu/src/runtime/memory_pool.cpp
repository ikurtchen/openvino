// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <algorithm>
#include <fstream>
#include <vector>

#include "intel_gpu/runtime/memory.hpp"
#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/memory_pool.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <list>
#include <string>
#include <utility>
#include <set>
#include <stdexcept>

namespace cldnn {
memory_record::memory_record(memory_set users,
                             std::shared_ptr<memory>& memory,
                             uint32_t net_id,
                             allocation_type type)
    : _users(users), _memory(memory), _network_id(net_id), _type(type) {}

memory::ptr memory_pool::alloc_memory(const layout& layout, allocation_type type, bool reset) {
    return _engine->allocate_memory(layout, type, reset);
}

memory_pool::~memory_pool() {}

std::vector<primitive_id> memory_pool::get_conflicts(const memory_set& a,
                               const std::set<primitive_id>& b,
                               uint32_t b_network_id) {
    std::set<primitive_id> a_same_network;
    for (auto const& mem_usr : a) {
        if (mem_usr._network_id == b_network_id) {
            a_same_network.insert(mem_usr._id);
        }
    }
    std::vector<primitive_id> intersection;
    intersection.reserve(std::min(a_same_network.size(), b.size()));
    set_intersection(a_same_network.begin(),
                     a_same_network.end(),
                     b.begin(),
                     b.end(),
                     std::back_inserter(intersection));
    return intersection;
}

void memory_pool::release_memory(memory* mem, const primitive_id& id, uint32_t network_id) {
    // check nonpadded pool first
    auto _layout = mem->get_layout();
    auto type = mem->get_allocation_type();

    {
        auto range = _non_padded_pool.equal_range(_layout.bytes_count());
        auto it = range.first;

        while (it != range.second && it != _non_padded_pool.end()) {
            if (it->second._network_id == network_id &&
                it->second._type == type &&
                it->second._memory.get() == mem) {
                auto user_it = it->second._users.begin();
                for (; user_it != it->second._users.end(); user_it ++) {
                    if (user_it->_id == id && user_it->_network_id == network_id)
                        break;
                }

                // normally there should be only one entry
                if (user_it != it->second._users.end()) {
                    user_it = it->second._users.erase(user_it);
                }
                if (it->second._users.empty()) {
                    // if this was the only user of the memory, then free it up
                    it = _non_padded_pool.erase(it);
                }

                //entry found and processed - so return
                return;
            } else {
                ++it;
            }
        }
    }
    {
        auto itr = _padded_pool.find(_layout);

        if (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();

            while (list_itr != list.end()) {
                if (list_itr->_memory.get() == mem &&
                    list_itr->_network_id == network_id &&
                    list_itr->_type == type) {
                    auto user_it = list_itr->_users.begin();
                    for (; user_it != list_itr->_users.end(); user_it ++) {
                        if (user_it->_id == id && user_it->_network_id == network_id)
                            break;
                    }

                    // normally there should be only one entry
                    if (user_it != list_itr->_users.end()) {
                        user_it = list_itr->_users.erase(user_it);
                    }
                    if (list_itr->_users.empty()) {
                        // if this was the only user of the memory, then free it up
                        list.erase(list_itr);
                    }

                    //entry found and processed - so return
                    break;
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
                _padded_pool.erase(itr);
            }
        }
    }
}

memory::ptr memory_pool::get_from_non_padded_pool(const layout& layout,
                                                  const primitive_id& id,
                                                  uint32_t network_id,
                                                  const std::set<primitive_id>& restrictions,
                                                  allocation_type type) {
    GPU_DEBUG_GET_INSTANCE(debug_config);
    auto it = _non_padded_pool.lower_bound(layout.bytes_count());
    while (it != _non_padded_pool.end()) {
        auto conflicts = get_conflicts(it->second._users, restrictions, network_id);
        bool may_reuse = (it->second._network_id == network_id) && it->second._type == type &&
                            it->second._memory->get_layout().format != format::fs_b_yx_fsv32 &&
                            layout.format != format::fs_b_yx_fsv32 &&
                            ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
                            (layout.feature() % 32 == 0));

        if (may_reuse && conflicts.empty()) {//no conflict, reuse directly
            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                if (type == allocation_type::usm_device) {
                    GPU_DEBUG_COUT << id << "(" << layout.bytes_count() << ")" << "reuse memory (" << it->second._memory << ") with size:"<< it->second._memory->get_layout().bytes_count() << std::endl;
                }
            }

            it->second._users.insert(memory_user(id, network_id, layout.bytes_count(), 0));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
            return ret_mem;
        }
        else if (may_reuse) {//may resue, need to figure out whether it has available slot
            std::vector<std::vector<size_t>> intervals;
            for (auto conflict : conflicts) {
                for (auto &user : it->second._users) {
                    if (user._id == conflict) {
                        intervals.push_back({user._offset, user._offset + user._size});
                    }
                }
            }
            //merge overlapped intervals
            sort(intervals.begin(), intervals.end());
            std::vector<std::vector<size_t>> res = {intervals[0]};
            for (size_t i = 1; i < intervals.size(); i++) {
                if (res.back()[1] >= intervals[i][0]) {
                    res.back()[1] = std::max(res.back()[1], intervals[i][1]);
                    continue;
                } else {
                    res.push_back(intervals[i]);
                }
            }

            std::vector<std::vector<size_t>> availables = {};
            if (res[0][0] > 0)
                availables.push_back({res[0][0], 0 });
            for (size_t i = 0; i < res.size() - 1; i++) {
                availables.push_back({res[i+1][0] - res[i][1], res[i][1]});
            }
            if (res.back()[1] < it->second._memory->get_layout().bytes_count()) {
                availables.push_back({it->second._memory->get_layout().bytes_count() - res.back()[1], res.back()[1]});
            }

            sort(availables.begin(), availables.end());

            size_t offset = 0; size_t i = 0;
            for (; i < availables.size(); i ++) {
                if (availables[i][0] >= layout.bytes_count()) {
                    offset = availables[i][1];
                    break;
                }
            }

            if ( i == availables.size()) {
                ++it;
                continue;
            }

            GPU_DEBUG_IF(debug_config->verbose >= 2) {
                if (type == allocation_type::usm_device) {
                    GPU_DEBUG_COUT << id << "(" << layout.bytes_count() << ")" << "reuse memory (ptr:" << it->second._memory <<", size" << it->second._memory->get_layout().bytes_count() << ") with offset:"<< offset << std::endl;
                }
            }

            it->second._users.insert(memory_user(id, network_id, layout.bytes_count(), offset));
            auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout, offset);
            return ret_mem;

        }
        else { //impossible to resue the memory
            ++it;
        }
    }

    //TODO: temporary workround for insufficient memory by Renzhi
    auto allocated_memory = _engine->get_used_device_memory(type);
    if (type == allocation_type::usm_device &&
         allocated_memory + layout.bytes_count() > _engine->get_device_info().max_global_mem_size) {
            GPU_DEBUG_COUT << "Warning: No available device memory for " << id << ", will use system memory instead." << std::endl;
        return nullptr;
    }
    // didn't find anything for you? create new resource
    auto mem = alloc_memory(layout, type);
    {
        _non_padded_pool.emplace(layout.bytes_count(),
                                 memory_record({{id, network_id, layout.bytes_count(), 0}}, mem, network_id, type));
    }

    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "[non-padded, " << id  << "(mem:" << mem  << ",type:" << type << ")"<< ": output]" << std::endl;
    }
    return mem;
}

memory::ptr memory_pool::get_from_padded_pool(const layout& layout,
                                              const primitive_id& id,
                                              uint32_t network_id,
                                              const std::set<primitive_id>& restrictions,
                                              allocation_type type) {
    auto first_level_cache = _padded_pool.find(layout);

    if (first_level_cache != _padded_pool.end()) {
        for (auto& rec_list : first_level_cache->second) {
            auto conflicts = get_conflicts(rec_list._users, restrictions, network_id);
            if (rec_list._network_id == network_id &&
                rec_list._type == type &&
                ((layout.format != format::b_fs_yx_fsv32 && layout.format != format::b_fs_zyx_fsv32) ||
                 (layout.feature() % 32 == 0)) &&
                // TODO: check if this condition always correct
                layout.feature() <= rec_list._memory->get_layout().feature() &&
                layout.batch() <= rec_list._memory->get_layout().batch() &&
                rec_list._memory->get_layout().format != format::fs_b_yx_fsv32 &&
                layout.format != format::fs_b_yx_fsv32 && conflicts.empty()) {
                rec_list._users.insert({id, network_id, layout.bytes_count(), 0});
                auto ret_mem = _engine->reinterpret_buffer(*(rec_list._memory), layout);
                return ret_mem;
            }
        }

        //TODO: temporary workround for insufficient memory by Renzhi
        auto allocated_memory = _engine->get_used_device_memory(type);
        if (type == allocation_type::usm_device &&
            allocated_memory + layout.bytes_count() > _engine->get_device_info().max_global_mem_size) {
            return nullptr;
        }

        auto mem = alloc_memory(layout, type);
        first_level_cache->second.emplace_back(
            memory_record({{id, network_id, layout.bytes_count(), 0}}, mem, network_id, type));
        return mem;
    }
    GPU_DEBUG_GET_INSTANCE(debug_config);
    GPU_DEBUG_IF(debug_config->verbose >= 2) {
        GPU_DEBUG_COUT << "[padded, " << id << ": output]" << std::endl;
    }

    //TODO: temporary workround for insufficient memory by Renzhi
    auto allocated_memory = _engine->get_used_device_memory(type);
    if (type == allocation_type::usm_device &&
        allocated_memory + layout.bytes_count() > _engine->get_device_info().max_global_mem_size) {
        return nullptr;
    }

    auto mem = alloc_memory(layout, type);
    std::list<memory_record> list = {memory_record({{id, network_id, layout.bytes_count(), 0}}, mem, network_id, type)};
    _padded_pool.emplace(layout, std::move(list));
    return mem;
}

/*
        This is not reusable within one network or it's internal micronetworks. But we can use this memory records
   between networks.
    */
memory::ptr memory_pool::get_from_across_networks_pool(const layout& layout,
                                                       const primitive_id& id,
                                                       uint32_t network_id,
                                                       allocation_type type) {
    auto it = _no_reusable_pool.lower_bound(layout.bytes_count());

    while (it != _no_reusable_pool.end()) {
        auto conflicts = get_conflicts(it->second._users, {}, network_id);
        if (it->second._network_id != network_id &&
            it->second._type == type) {  // don't use non reusable resources within the same network
            if (conflicts.empty()) {
                it->second._users.insert(memory_user(id, network_id, layout.bytes_count(), 0));
                auto ret_mem = _engine->reinterpret_buffer(*it->second._memory, layout);
                return ret_mem;
            }
        }
        ++it;
    }
    auto mem = alloc_memory(layout, type);
    {
        _no_reusable_pool.emplace(layout.bytes_count(),
                                  memory_record({{id, network_id, layout.bytes_count(), 0}}, mem, network_id, type));
    }
    return mem;
}

memory::ptr memory_pool::get_memory(const layout& layout, allocation_type type, bool reset) {
    return alloc_memory(layout, type, reset);
}

memory::ptr memory_pool::get_memory(const layout& layout,
                                    const primitive_id& id,
                                    uint32_t network_id,
                                    const std::set<primitive_id>& restrictions,
                                    allocation_type type,
                                    bool reusable_across_network) {
    if (reusable_across_network) {
        // reusable within the same network
        if (!layout.format.is_image() && layout.data_padding == padding{{0, 0, 0, 0}, 0}) {
            // non-padded buffers
            return get_from_non_padded_pool(layout, id, network_id, restrictions, type);
        } else if (!layout.format.is_image()) {
            // padded buffers
            return get_from_padded_pool(layout, id, network_id, restrictions, type);
        } else {
            // images (reuse not yet implemented)
            return alloc_memory(layout, type);
        }
    } else {
        return alloc_memory(layout, type);
    }
}

void memory_pool::clear_pool() { _non_padded_pool.clear(); }

void memory_pool::clear_pool_for_network(uint32_t network_id) {
    // free up _non_padded_pool for this network
    {
        auto itr = _non_padded_pool.begin();

        while (itr != _non_padded_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
                itr = _non_padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

    // free up _padded_pool for this network
    {
        auto itr = _padded_pool.begin();

        while (itr != _padded_pool.end()) {
            auto& list = itr->second;
            auto list_itr = list.begin();

            while (list_itr != list.end()) {
                if (list_itr->_network_id == network_id) {
                    list_itr = list.erase(list_itr);
                } else {
                    list_itr++;
                }
            }

            if (list.empty()) {
                itr = _padded_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }

    // free up _no_reusable_pool for this network
    {
        auto itr = _no_reusable_pool.begin();

        while (itr != _no_reusable_pool.end()) {
            auto& record = itr->second;

            if (record._network_id == network_id) {
                itr = _no_reusable_pool.erase(itr);
            } else {
                itr++;
            }
        }
    }
}

memory_pool::memory_pool(engine& engine) : _engine(&engine) { }

}  // namespace cldnn

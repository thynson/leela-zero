/*
    This file is part of Leela Zero.
    Copyright (C) 2018 Junhee Yoo and contributors

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
*/
#include "config.h"

#ifdef USE_OPENCL

#include "GTP.h"
#include "Random.h"
#include "Network.h"
#include "Utils.h"
#include "OpenCLScheduler.h"

using Utils::ceilMultiple;
using Utils::myprintf;

class from_float{
public:
    from_float(const std::vector<float> & f) : m_f(f) {}

    operator const std::vector<float>&() {
        return m_f;
    }

    operator std::vector<half_float::half>() {
        auto ret = std::vector<half_float::half>(m_f.size());
        std::copy(cbegin(m_f), cend(m_f), begin(ret));
        return ret;
    }
private:
    const std::vector<float>& m_f;
};

template <typename T>
static std::vector<T> zeropad_U(const std::vector<float>& U,
                                const int outputs, const int channels,
                                const int outputs_pad,
                                const int channels_pad) {
    // Fill with zeroes
    auto Upad =
        std::vector<T>(WINOGRAD_TILE * outputs_pad * channels_pad);

    for (auto xi = 0; xi < WINOGRAD_ALPHA; xi++){
        for (auto nu = 0; nu < WINOGRAD_ALPHA; nu++) {
            for (auto c = 0; c < channels; c++) {
                for (auto o = 0; o < outputs; o++) {
                    Upad[xi * (WINOGRAD_ALPHA * outputs_pad * channels_pad)
                         + nu * (outputs_pad * channels_pad)
                         + c * outputs_pad +
                          o] =
                    U[xi * (WINOGRAD_ALPHA * outputs * channels)
                      + nu * (outputs * channels)
                      + c * outputs
                      + o];
                }
            }
        }
    }

    return Upad;
}

template <typename net_t>
OpenCLScheduler<net_t>::OpenCLScheduler() {
    // multi-gpu?
    auto gpus = cfg_gpus;

    // An empty GPU list from the command line represents autodetect.
    // Put a minus one GPU index here.
    if (gpus.empty()) {
        gpus = {-1};
    }

    auto silent{false};

    for (auto gpu : gpus) {
        auto opencl = std::make_unique<OpenCL<net_t>>(gpu, silent);
        auto net = std::make_unique<OpenCL_Network<net_t>>(*opencl);
        m_opencl.push_back(std::move(opencl));
        m_networks.push_back(std::move(net));

        // Starting next GPU, let's not dump full list of GPUs.
        silent = true;
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::initialize(const int channels) {
    // Launch the worker threads.  Minimum 1 worker per GPU, but use enough threads
    // so that we can at least concurrently schedule something to the GPU.

    auto num_gpus = m_opencl.size();

    // number of worker threads
    if (cfg_workers.empty()) {
        cfg_workers.assign(num_gpus, 2); 
    }
    else {
        while (cfg_workers.size() < num_gpus) {
            cfg_workers.push_back(cfg_workers.back());
        }
    }

    // batch sizes
    if (cfg_batch_size.empty()) {
        cfg_batch_size.assign(num_gpus, 1);
    }
    else {
        while (cfg_batch_size.size() < num_gpus) {
            cfg_batch_size.push_back(cfg_batch_size.back());
        }
    }

    auto queue_size = 1;
    auto gnum = 0;
    for (auto & opencl : m_opencl)
        queue_size += cfg_workers[gnum++];
    empty_workers.resize(queue_size);
    unfull_workers.resize(queue_size);

    auto num_workers = cfg_workers;
    auto count = 1;
    while (count < queue_size) {
        auto gnum = 0;
        for (auto& opencl : m_opencl) {
            auto& num = num_workers[gnum];
            if (num)
                empty_workers.push_back({ gnum,--num }), count++;
            gnum++;
        }
    }
    empty_workers_writing = empty_workers_written = queue_size - 1;

    constexpr auto in_size = Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    inputs.resize(num_gpus);
    backup_entries.resize(num_gpus);
    writing_location.resize(num_gpus);
    written_location.resize(num_gpus);
    batch_stats.resize(num_gpus);

    gnum = 0;
    for (auto & opencl : m_opencl) {
        auto batchsize = cfg_batch_size[gnum];
        auto num_workers = cfg_workers[gnum];
        opencl->initialize(channels, batchsize);

        inputs[gnum].resize(num_workers);
        backup_entries[gnum].resize(num_workers);
        writing_location[gnum] = new std::atomic<int>[num_workers]();
        written_location[gnum] = new std::atomic<int>[num_workers]();
        cv[gnum] = new std::condition_variable[num_workers];

        batch_stats[gnum] = new std::atomic<int>[batchsize]();

        for (auto i = unsigned{0}; i < num_workers; i++) {
            inputs[gnum][i] = new net_t[in_size * batchsize];
            backup_entries[gnum][i] = new BackupEntry[batchsize];
            auto t = std::thread(&OpenCLScheduler<net_t>::batch_worker, this, gnum, i);
            m_worker_threads.push_back(std::move(t));
        }
        gnum++;
    }

    printf("max queue size: %d\n", m_max_queue_size.load());

    // Exit immediately after tuning.  We should exit here because we skipped
    // initializing rest of the kernels due to some NVIDIA drivers crashing.
    if (cfg_tune_only) {
        exit(EXIT_SUCCESS);
    }
}

template <typename net_t>
OpenCLScheduler<net_t>::~OpenCLScheduler() {
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_running = false;
    }
    m_cv.notify_all();
    for (auto & x : m_worker_threads) {
        x.join();
    }
}

template<typename net_t>
bool OpenCLScheduler<net_t>::needs_autodetect() {
    for (auto& opencl : m_opencl) {
        // If any card has no native fp16 compute, we'll have to benchmark.
        if (!opencl->has_fp16_compute()) {
            return true;
        }
    }
    return false;
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_input_convolution(
    unsigned int filter_size,
    unsigned int channels,
    unsigned int outputs,
    const std::vector<float>& weights,
    const std::vector<float>& means,
    const std::vector<float>& variances) {

    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto kwg = tuners[2];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto k_ceil = ceilMultiple(ceilMultiple(channels, kwg), vwm);

        const auto Upad = zeropad_U<net_t>(weights,
                                           outputs, channels,
                                           m_ceil, k_ceil);
        opencl_net->push_input_convolution(
            filter_size, channels, outputs,
            Upad, from_float(means), from_float(variances)
        );
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_residual(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights_1,
                                           const std::vector<float>& means_1,
                                           const std::vector<float>& variances_1,
                                           const std::vector<float>& weights_2,
                                           const std::vector<float>& means_2,
                                           const std::vector<float>& variances_2) {
    for (const auto& opencl_net : m_networks) {
        const auto tuners = opencl_net->getOpenCL().get_sgemm_tuners();

        const auto mwg = tuners[0];
        const auto vwm = tuners[3];

        const auto m_ceil = ceilMultiple(ceilMultiple(outputs, mwg), vwm);
        const auto Upad1 = zeropad_U<net_t>(weights_1,
                                            outputs, outputs,
                                            m_ceil, m_ceil);
        const auto Upad2 = zeropad_U<net_t>(weights_2,
                                            outputs, outputs,
                                            m_ceil, m_ceil);
        opencl_net->push_residual(filter_size, channels, outputs,
                                  Upad1,
                                  from_float(means_1),
                                  from_float(variances_1),
                                  Upad2,
                                  from_float(means_2),
                                  from_float(variances_2));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_convolve(unsigned int filter_size,
                                           unsigned int channels,
                                           unsigned int outputs,
                                           const std::vector<float>& weights) {
    for (const auto & opencl_net : m_networks) {
        opencl_net->push_convolve(filter_size, channels, outputs,
                                  from_float(weights));
    }
}

template <typename net_t>
void OpenCLScheduler<net_t>::push_weights(
    unsigned int filter_size,
    unsigned int channels,
    unsigned int outputs,
    std::shared_ptr<const ForwardPipeWeights> weights) {

    auto weight_index = size_t{0};

    // Winograd filter transformation changes filter size to 4x4
    push_input_convolution(filter_size, channels, outputs,
                           weights->m_conv_weights[weight_index],
                           weights->m_batchnorm_means[weight_index],
                           weights->m_batchnorm_stddevs[weight_index]);
    weight_index++;

    // residual blocks : except the first entry,
    // the second ~ last entry is all on residual topwer
    for (auto i = size_t{0}; i < weights->m_conv_weights.size()/2; i++) {
        push_residual(filter_size, outputs, outputs,
                      weights->m_conv_weights[weight_index],
                      weights->m_batchnorm_means[weight_index],
                      weights->m_batchnorm_stddevs[weight_index],
                      weights->m_conv_weights[weight_index + 1],
                      weights->m_batchnorm_means[weight_index + 1],
                      weights->m_batchnorm_stddevs[weight_index + 1]);
        weight_index += 2;
    }

    // Output head convolutions
    push_convolve(1, outputs, Network::OUTPUTS_POLICY, weights->m_conv_pol_w);
    push_convolve(1, outputs, Network::OUTPUTS_VALUE, weights->m_conv_val_w);
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward(const std::vector<float>& input,
                                     std::vector<float>& output_pol,
                                     std::vector<float>& output_val) {
    auto entry = std::make_shared<ForwardQueueEntry>(input, output_pol, output_val);
    std::unique_lock<std::mutex> lk(entry->mutex);
    {
        std::unique_lock<std::mutex> lk(m_mutex);
        m_forward_queue.push_back(entry);

        if (m_single_eval_in_progress.load()) {
            m_waittime += 2;
        }
    }
    m_cv.notify_one();
    entry->cv.wait(lk);
}

template <typename net_t>
void OpenCLScheduler<net_t>::forward0(std::unique_ptr<const std::vector<float>> input,
                                      const int tomove,
                                      const int symmetry,
                                      Netresult_ptr result) {
    // auto max_size = cfg_batch_size * m_opencl.size() * 2;
    m_search->m_positions++;
    std::unique_lock<std::mutex> lk(m_mutex);
    m_forward_queue0.push_back(std::make_unique<ForwardQueueEntry0>(
        std::move(input), tomove, symmetry, result));
    m_cv.notify_one();
    /*
    if (m_search->m_run && (int)m_forward_queue0.size() >= m_max_queue_size.load()) {
        m_cv0.wait(lk, [&] { return (int)m_forward_queue0.size() < m_max_queue_size.load()
            || !m_search->m_run; });
        //lk.unlock();
        //m_search->backup();
    }
    */
}

template <typename net_t>
void OpenCLScheduler<net_t>::batch_worker(const size_t gnum, const size_t i) {

    constexpr auto in_size = Network::INPUT_CHANNELS * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_pol_size = Network::OUTPUTS_POLICY * BOARD_SIZE * BOARD_SIZE;
    constexpr auto out_val_size = Network::OUTPUTS_VALUE * BOARD_SIZE * BOARD_SIZE;

    OpenCLContext context;

    auto batch_input = std::vector<net_t>(in_size * cfg_batch_size[gnum]);
    auto batch_output_pol = std::vector<float>(out_pol_size * cfg_batch_size[gnum]);
    auto batch_output_val = std::vector<float>(out_val_size * cfg_batch_size[gnum]);

    auto pickup_task = [this, gnum, in_size, i](std::vector<net_t>&batch_in) {
        std::vector<std::unique_ptr<ForwardQueueEntry0>> inputs;
        inputs.reserve(cfg_batch_size[gnum]);
        auto index = 0;
        int count = 0;
        int remaining = cfg_batch_size[gnum];

        std::unique_lock<std::mutex> lk(m_mutex, std::defer_lock);
        while (remaining) {
            bool idle = !(m_networks[gnum]->m_occupied.load()) && inputs.size() > 0;
            if (idle || !m_running) break;
            lk.lock();
            int queue_size = m_forward_queue0.size();
            if (!queue_size) {
                m_cv.wait(lk,
                    [this, gnum, &queue_size, &idle, &inputs] {
                    queue_size = m_forward_queue0.size();
                    idle = !(m_networks[gnum]->m_occupied.load()) && inputs.size() > 0;
                    return !m_running || queue_size > 0 || idle; });
            }
            if (idle || !m_running) break;

            count = std::min(queue_size, remaining);
            auto end = begin(m_forward_queue0);
            std::advance(end, count);
            std::move(begin(m_forward_queue0), end, std::back_inserter(inputs));
            m_forward_queue0.erase(begin(m_forward_queue0), end);
            lk.unlock();
            //if (count) { (*pickup_stats[count - 1])++; }

            m_max_queue_size -= count;
            remaining -= count;

            while (index < inputs.size()) {
                std::transform(inputs[index]->in->begin(), inputs[index]->in->end(), std::back_inserter(batch_in),
                    [](float x) {return (net_t)x; });
                ++index;
            }
        }
        ++(m_networks[gnum]->m_occupied);
        //myprintf("max queue size: %d - worker %d picking up\n", m_max_queue_size.load(), i);
        m_max_queue_size -= remaining;
        //m_search->m_pending_netresults += remaining;
        //myprintf("max queue size: %d - worker %d pickup finished\n", m_max_queue_size.load(), i);
        return inputs;
    };

    while (true) {

        batch_input.resize(0);
        auto inputs = pickup_task(batch_input);
        //m_cv0.notify_all();
        auto count = inputs.size();
        if (count) { (*batch_stats[count - 1])++; }

        /*
        for (auto count : batch_stats) {
        myprintf("%d, ", count->load());
        }
        myprintf("\n");
        */

        if (!m_running) return;

        batch_input.resize(in_size * count);
        batch_output_pol.resize(out_pol_size * count);
        batch_output_val.resize(out_val_size * count);

        {
            m_networks[gnum]->forward(
                batch_input.data(), batch_output_pol, batch_output_val, context, m_cv, count);
        }
        //std::unique_lock<std::mutex> lk(m_mutex);
        //m_cv.notify_all();
        //lk.unlock();
        /*{
            auto t = std::thread([=](std::vector<std::unique_ptr<ForwardQueueEntry0>> inputs_) {
                auto index = 0;
                for (auto it = inputs_.begin(); it != inputs_.end(); ++it) {
                    std::vector<float> out_p(begin(batch_output_pol) + out_pol_size * index,
                        begin(batch_output_pol) + out_pol_size * (index + 1));
                    std::vector<float> out_v(begin(batch_output_val) + out_val_size * index,
                        begin(batch_output_val) + out_val_size * (index + 1));
                    index++;
                    m_network->process_output(out_p, out_v, (*it)->tomove, (*it)->symmetry, (*it)->result);
                }
                //m_search->m_pending_netresults += cfg_batch_size[gnum];
                m_search->m_cv.notify_all();
            }, std::move(inputs));
            t.detach();
        }*/
        {
            for (auto index = 0; index < inputs.size(); ) {
                std::vector<float> out_p(begin(batch_output_pol) + out_pol_size * index,
                    begin(batch_output_pol) + out_pol_size * (index + 1));
                std::vector<float> out_v(begin(batch_output_val) + out_val_size * index,
                    begin(batch_output_val) + out_val_size * (index + 1));
                m_network->process_output(out_p, out_v, inputs[index]->tomove, inputs[index]->symmetry, inputs[index]->result);
                index++;
            }
            //m_search->m_pending_netresults += cfg_batch_size[gnum];
            m_search->m_cv.notify_all();
        }
        /*
        {
            std::vector<std::thread> backup_threads;
            auto index = 0;
            for (auto it = begin(inputs); it != end(inputs); ++it) {
                std::vector<float> out_p(begin(batch_output_pol) + out_pol_size * index,
                                         begin(batch_output_pol) + out_pol_size * (index + 1));
                std::vector<float> out_v(begin(batch_output_val) + out_val_size * index,
                                         begin(batch_output_val) + out_val_size * (index + 1));
                index++;
                
                auto t = std::thread([=](std::vector<float>& p, std::vector<float>& v,
                    const int tomove,
                    const int symmetry,
                    Netresult_ptr result) {
                    m_network->process_output(p, v, tomove, symmetry, result); }, out_p, out_v, 
                    (*it)->tomove, (*it)->symmetry, (*it)->result);
                t.detach(); // can't control any more, but no harm even after !m_run, since won't be able to back up anything.
                
                //backup_threads.emplace_back(std::thread([=](std::vector<float>& p, std::vector<float>& v) { 
                  //  m_network->process_output(p, v,
                    //(*it)->tomove, (*it)->symmetry, (*it)->result); }, out_p, out_v));
                    
                //m_network->process_output(out_p, out_v, (*it)->tomove, (*it)->symmetry, (*it)->result);
            }
            for (auto iter = backup_threads.begin(); iter != backup_threads.end(); iter++) {
            //    iter->join();
            }
        }
        */
        //myprintf("%d ", m_max_queue_size.load());
        m_max_queue_size += cfg_batch_size[gnum];
        //myprintf("max queue size: %d - worker %d\n", m_max_queue_size.load(), i);
        //lk.lock();
        m_cv0.notify_all();
        //m_search->backup();
        //m_search->m_cv.notify_all();
    }
}

template class OpenCLScheduler<float>;
#ifdef USE_HALF
template class OpenCLScheduler<half_float::half>;
#endif

#endif

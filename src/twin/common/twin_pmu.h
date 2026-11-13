// Core cycles and instructions for twin benches (docs/twin/harness.md "Measuring cycles").
//
// Why: on the workstation VM, TSC x a measured clock is biased (vCPU steal during the clock loop,
// frequency changes), by up to 2x. The PMU's user-mode core cycles are exact (checked: 200.2M cycles
// for 200M dependent adds), but rdpmc traps to the hypervisor in this VM (~1-2 us), so read the
// counters only a few times per run, never per call. Reads follow the mmap page's seqlock (rdpmc +
// offset): a context switch between two reads is safe. If another perf user multiplexes the counters,
// FullyScheduled() turns false and the counts are partial: rerun.
//
// Use (Linux x86; ok() false elsewhere or without perf access):
//   tw::bench::Pmu pmu;                       // opens cycles + instructions for this thread
//   const auto a = pmu.Read();                // before the run (the thread stays on one core: taskset)
//   std::uint64_t t0 = __rdtsc(), part = 0;   // TSC around the part you want, summed inside the run
//   ... run; part += __rdtsc() - t; ...
//   const auto b = pmu.Read();
//   double runCycles = tw::bench::Pmu::Delta(a, b).cycles;
//   double partCycles = runCycles * double(part) / double(__rdtsc() - t0);   // split by TSC share
// Report the min of 5 runs. Instructions of a part can be counted exactly with Read() around each call
// in a separate, untimed run (the traps distort its cycles, not its instruction count).
#ifndef TW_TWIN_PMU_H
#define TW_TWIN_PMU_H

#include <cstdint>
#include <cstring>

#if defined(__linux__) && (defined(__x86_64__) || defined(__i386__))
#include <linux/perf_event.h>
#include <sys/mman.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <x86intrin.h>
#define TW_PMU_AVAILABLE 1
#else
#define TW_PMU_AVAILABLE 0
#endif

namespace tw {
namespace bench {

class Pmu {
public:
    struct Sample { std::uint64_t cycles = 0, instructions = 0; };

    Pmu() {
#if TW_PMU_AVAILABLE
        for (int i = 0; i < 2; ++i) {
            perf_event_attr a;
            std::memset(&a, 0, sizeof a);
            a.size = sizeof a;
            a.type = PERF_TYPE_HARDWARE;
            a.config = i == 0 ? PERF_COUNT_HW_CPU_CYCLES : PERF_COUNT_HW_INSTRUCTIONS;
            a.exclude_kernel = 1;
            a.exclude_hv = 1;
            fd_[i] = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
            if (fd_[i] < 0) continue;
            void *m = mmap(nullptr, static_cast<std::size_t>(sysconf(_SC_PAGESIZE)), PROT_READ, MAP_SHARED, fd_[i], 0);
            page_[i] = m == MAP_FAILED ? nullptr : static_cast<perf_event_mmap_page *>(m);
        }
#endif
    }
    ~Pmu() {
#if TW_PMU_AVAILABLE
        for (int i = 0; i < 2; ++i) {
            if (page_[i]) munmap(page_[i], static_cast<std::size_t>(sysconf(_SC_PAGESIZE)));
            if (fd_[i] >= 0) close(fd_[i]);
        }
#endif
    }
    Pmu(const Pmu &) = delete;
    Pmu &operator=(const Pmu &) = delete;

    bool ok() const {
#if TW_PMU_AVAILABLE
        return page_[0] && page_[1] && page_[0]->cap_user_rdpmc && page_[1]->cap_user_rdpmc && page_[0]->index && page_[1]->index;
#else
        return false;
#endif
    }
    // The events' 64-bit counts (user mode), read with the perf mmap page's seqlock: rdpmc + the kernel's offset,
    // so a context switch or a counter reschedule between two reads does not corrupt the difference.
    Sample Read() const {
        Sample s;
#if TW_PMU_AVAILABLE
        if (ok()) {
            s.cycles = Count(page_[0]);
            s.instructions = Count(page_[1]);
        }
#endif
        return s;
    }
    static Sample Delta(const Sample &a, const Sample &b) { return Sample{b.cycles - a.cycles, b.instructions - a.instructions}; }
    // True when both events ran all the time they were enabled (no multiplexing with other perf users).
    bool FullyScheduled() const {
#if TW_PMU_AVAILABLE
        if (!ok()) return false;
        for (int i = 0; i < 2; ++i) {
            const volatile perf_event_mmap_page *pc = page_[i];
            if (pc->time_running != pc->time_enabled) return false;
        }
        return true;
#else
        return false;
#endif
    }

private:
#if TW_PMU_AVAILABLE
    static std::uint64_t Count(const perf_event_mmap_page *page) {
        const volatile perf_event_mmap_page *pc = page;
        std::uint32_t seq, idx;
        std::int64_t count;
        do {
            seq = pc->lock;
            __asm__ __volatile__("" ::: "memory");
            idx = pc->index;
            count = pc->offset;
            if (pc->cap_user_rdpmc && idx) {
                const int width = static_cast<int>(pc->pmc_width);
                std::int64_t raw = static_cast<std::int64_t>(__rdpmc(static_cast<int>(idx - 1)));
                raw = static_cast<std::int64_t>(static_cast<std::uint64_t>(raw) << (64 - width)) >> (64 - width);   // sign-extend
                count += raw;
            }
            __asm__ __volatile__("" ::: "memory");
        } while (pc->lock != seq);
        return static_cast<std::uint64_t>(count);
    }
    int fd_[2] = {-1, -1};
    perf_event_mmap_page *page_[2] = {nullptr, nullptr};
#endif
};

}  // namespace bench
}  // namespace tw

#endif

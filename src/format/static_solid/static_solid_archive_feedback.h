#ifndef TMNF_STATIC_SOLID_ARCHIVE_FEEDBACK_H
#define TMNF_STATIC_SOLID_ARCHIVE_FEEDBACK_H

#include "engine/core/engine_types.h"
#include <cstddef>
#include <vector>


class CClassicBufferCrypted;

struct CGameCtnReplayStaticSolidArchiveFeedback {
    int ApplyValue(
            CClassicBufferCrypted *reader,
            u32 value,
            int nested,
            int deferUntilCompressedPageRead);
    int FlushTo(CClassicBufferCrypted *reader);

    u32 RootAppliedFlag() const;
    u32 AppliedCount() const;
    u32 NestedAppliedCount() const;

private:
    static constexpr size_t QueuedValueLimit = 128u;

    std::vector<u32> queuedValues;
    u32 appliedCount = 0u;
    u32 nestedAppliedCount = 0u;

    int QueueDeferred(u32 value, int nested);
    void RecordDirect(int nested);
};

#endif

#ifndef EOG_MODEL_CONFIG_HPP
#define EOG_MODEL_CONFIG_HPP

#include <cstddef>
#include <cstdint>

namespace eog
{
static constexpr size_t kInputFrames = 256;
static constexpr size_t kInputChannels = 2;
static constexpr size_t kInputElementCount = kInputFrames * kInputChannels;
static constexpr size_t kClassCount = 5;

static constexpr float kInputScale = 0.03121989034116268f;
static constexpr int32_t kInputZeroPoint = -24;
static constexpr float kOutputScale = 0.00390625f;
static constexpr int32_t kOutputZeroPoint = -128;

static constexpr const char* kLabels[kClassCount] = {
    "down",
    "forward",
    "left",
    "right",
    "up",
};
} /* namespace eog */

#endif /* EOG_MODEL_CONFIG_HPP */

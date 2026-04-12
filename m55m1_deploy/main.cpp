/**************************************************************************//**
 * @file     main.cpp
 * @version  V1.00
 * @brief    EOG 5-class inference sample for NuMaker-M55M1
 ******************************************************************************/
#include "BoardInit.hpp"
#include "BufAttributes.hpp"
#include "EogModelConfig.hpp"
#include "NNModel.hpp"

#undef PI
#include "NuMicro.h"

#include <algorithm>
#include <cstdint>
#include <vector>

#define LOG_LEVEL 2
#include "log_macros.h"

#undef ACTIVATION_BUF_SZ
#define ACTIVATION_BUF_SZ (256 * 1024)

namespace arm
{
namespace app
{
static uint8_t tensorArena[ACTIVATION_BUF_SZ] ACTIVATION_BUF_ATTRIBUTE;

namespace nn
{
extern const uint8_t* GetModelPointer();
extern size_t GetModelLen();
} /* namespace nn */
} /* namespace app */
} /* namespace arm */

namespace
{
static float g_testWindow[eog::kInputElementCount] = {0.0f};

inline int8_t QuantizeFloatToInt8(float value, float scale, int32_t zeroPoint)
{
    const float q = (value / scale) + static_cast<float>(zeroPoint);
    const int32_t rounded = static_cast<int32_t>(q + (q >= 0.0f ? 0.5f : -0.5f));
    const int32_t clamped = std::min<int32_t>(INT8_MAX, std::max<int32_t>(INT8_MIN, rounded));
    return static_cast<int8_t>(clamped);
}

void FillInputTensorFromFloatWindow(TfLiteTensor* inputTensor)
{
    auto* dst = inputTensor->data.int8;
    for (size_t i = 0; i < eog::kInputElementCount; ++i)
    {
        dst[i] = QuantizeFloatToInt8(g_testWindow[i], eog::kInputScale, eog::kInputZeroPoint);
    }
}

float DequantizeInt8(int8_t value, float scale, int32_t zeroPoint)
{
    return (static_cast<int32_t>(value) - zeroPoint) * scale;
}
} /* namespace */

int main()
{
    BoardInit();

    arm::app::NNModel model;
    if (!model.Init(arm::app::tensorArena,
                    sizeof(arm::app::tensorArena),
                    arm::app::nn::GetModelPointer(),
                    arm::app::nn::GetModelLen()))
    {
        printf_err("Failed to initialise EOG model.\n");
        return 1;
    }

    info("Tensor arena cache policy: WTRA\n");
    const std::vector<ARM_MPU_Region_t> mpuConfig = {
        {
            ARM_MPU_RBAR(((unsigned int)arm::app::tensorArena), ARM_MPU_SH_NON, 0, 1, 1),
            ARM_MPU_RLAR((((unsigned int)arm::app::tensorArena) + ACTIVATION_BUF_SZ - 1),
                         eMPU_ATTR_CACHEABLE_WTRA),
        },
    };
    InitPreDefMPURegion(&mpuConfig[0], mpuConfig.size());

    TfLiteTensor* inputTensor = model.GetInputTensor(0);
    TfLiteTensor* outputTensor = model.GetOutputTensor(0);

    printf("\n=== EOG 1D-CNN on M55M1 ===\n");
    printf("Model size : %lu bytes\n", (unsigned long)arm::app::nn::GetModelLen());
    printf("Input shape: [1, %u, %u]\n", (unsigned)eog::kInputFrames, (unsigned)eog::kInputChannels);
    printf("Output size: %lu bytes\n", (unsigned long)outputTensor->bytes);
    printf("Mode       : %s\n", 
#if defined(EOG_USE_ETHOS_U)
           "Ethos-U / Vela model"
#else
           "TFLM CPU model"
#endif
    );

    FillInputTensorFromFloatWindow(inputTensor);

    if (!model.RunInference())
    {
        printf_err("Inference failed.\n");
        return 2;
    }

    outputTensor = model.GetOutputTensor(0);

    size_t bestIdx = 0;
    float bestScore = -1.0f;

    printf("\nPrediction scores:\n");
    for (size_t i = 0; i < eog::kClassCount; ++i)
    {
        const int8_t q = outputTensor->data.int8[i];
        const float score = DequantizeInt8(q, eog::kOutputScale, eog::kOutputZeroPoint);
        printf("  %-8s : %.4f (q=%d)\n", eog::kLabels[i], (double)score, (int)q);
        if (score > bestScore)
        {
            bestScore = score;
            bestIdx = i;
        }
    }

    printf("\nPredicted class: %s (%.4f)\n", eog::kLabels[bestIdx], (double)bestScore);
    printf("Replace g_testWindow[] with one preprocessed EOG window of shape [256, 2].\n");

    while (1)
    {
        __WFI();
    }
}

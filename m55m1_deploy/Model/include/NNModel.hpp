/**************************************************************************//**
 * @file     NNModel.hpp
 * @version  V1.00
 * @brief    EOG model header file
 ******************************************************************************/
#ifndef NN_MODEL_HPP
#define NN_MODEL_HPP

#include "Model.hpp"

namespace arm
{
namespace app
{

class NNModel : public Model
{
public:
    static constexpr uint32_t ms_inputRowsIdx = 1;
    static constexpr uint32_t ms_inputColsIdx = 2;
    static constexpr uint32_t ms_inputChannelsIdx = 3;

protected:
    const tflite::MicroOpResolver& GetOpResolver() override;
    bool EnlistOperations() override;

private:
    static constexpr int ms_maxOpCnt = 10;
    tflite::MicroMutableOpResolver<ms_maxOpCnt> m_opResolver;
};

} /* namespace app */
} /* namespace arm */

#endif /* NN_MODEL_HPP */

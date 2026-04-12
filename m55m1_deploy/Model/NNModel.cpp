/**************************************************************************//**
 * @file     NNModel.cpp
 * @version  V1.00
 * @brief    EOG model source file
 ******************************************************************************/
#include "NNModel.hpp"

const tflite::MicroOpResolver& arm::app::NNModel::GetOpResolver()
{
    return this->m_opResolver;
}

bool arm::app::NNModel::EnlistOperations()
{
#if defined(EOG_USE_ETHOS_U)
    return this->m_opResolver.AddEthosU() == kTfLiteOk;
#else
    return this->m_opResolver.AddExpandDims() == kTfLiteOk &&
           this->m_opResolver.AddConv2D() == kTfLiteOk &&
           this->m_opResolver.AddMul() == kTfLiteOk &&
           this->m_opResolver.AddAdd() == kTfLiteOk &&
           this->m_opResolver.AddReshape() == kTfLiteOk &&
           this->m_opResolver.AddMaxPool2D() == kTfLiteOk &&
           this->m_opResolver.AddMean() == kTfLiteOk &&
           this->m_opResolver.AddFullyConnected() == kTfLiteOk &&
           this->m_opResolver.AddSoftmax() == kTfLiteOk;
#endif
}

/**************************************************************************//**
 * @file     BoardInit.cpp
 * @version  V1.00
 * @brief    Target board initialise function for EOG deployment
 ******************************************************************************/
#include "NuMicro.h"
#include "log_macros.h"

#if defined(ARM_NPU)
#include "ethosu_npu_init.h"
#endif

#define DESIGN_NAME "M55M1"

static void SYS_Init(void)
{
    CLK_EnableXtalRC(CLK_SRCCTL_HIRCEN_Msk);
    CLK_WaitClockReady(CLK_STATUS_HIRCSTB_Msk);

    CLK_EnableXtalRC(CLK_SRCCTL_HXTEN_Msk);
    CLK_WaitClockReady(CLK_STATUS_HXTSTB_Msk);

    CLK_SetBusClock(CLK_SCLKSEL_SCLKSEL_APLL0, CLK_APLLCTL_APLLSRC_HXT, FREQ_220MHZ);
    SystemCoreClockUpdate();

    CLK_EnableModuleClock(GPIOA_MODULE);
    CLK_EnableModuleClock(GPIOB_MODULE);
    CLK_EnableModuleClock(GPIOC_MODULE);
    CLK_EnableModuleClock(GPIOD_MODULE);
    CLK_EnableModuleClock(GPIOE_MODULE);
    CLK_EnableModuleClock(GPIOF_MODULE);
    CLK_EnableModuleClock(GPIOG_MODULE);
    CLK_EnableModuleClock(GPIOH_MODULE);
    CLK_EnableModuleClock(GPIOI_MODULE);
    CLK_EnableModuleClock(GPIOJ_MODULE);

#if defined(ARM_NPU)
    CLK_EnableModuleClock(FMC0_MODULE);
    CLK_EnableModuleClock(NPU0_MODULE);
#endif

    SetDebugUartCLK();
    SetDebugUartMFP();
}

int BoardInit(void)
{
    SYS_UnlockReg();
    SYS_Init();
    InitDebugUart();
    SYS_LockReg();

    info("%s: complete\n", __FUNCTION__);

#if defined(ARM_NPU)
    int state = arm_ethosu_npu_init();
    if (state != 0)
    {
        return state;
    }
#endif

    info("Target system: %s\n", DESIGN_NAME);
    return 0;
}

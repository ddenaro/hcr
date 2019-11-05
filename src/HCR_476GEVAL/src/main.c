/**
 ******************************************************************************
 * @file    main.c
 * @author  Ac6
 * @version V1.0
 * @date    01-December-2013
 * @brief   Default main function.
 ******************************************************************************
 */

#include <stdio.h>

#include "stm32l4xx.h"
#include "stm32l476g_eval.h"

static void SystemClock_Config(void);
UART_HandleTypeDef g_hlpuart1;
void LPUART1_UART_Init(void);
void HAL_UART_MspInit(UART_HandleTypeDef* huart);
void HAL_MspInit(void);
void Application(void);

int main(void)
{
	HAL_Init();
	SystemClock_Config();

	LPUART1_UART_Init();
	printf("Board setup done.\r\n");

	Application();
}

/**
 * @brief  Retargets the C library printf function to the USART.
 * @param  None
 * @retval None
 */
int __io_putchar(int ch)
{
	/* Place your implementation of fputc here */
	/* e.g. write a character to the EVAL_COM1 and Loop until the end of transmission */
	HAL_UART_Transmit(&g_hlpuart1, (uint8_t *) &ch, 1, 0xFFFF);

	return ch;
}

void LPUART1_UART_Init(void)
{
	g_hlpuart1.Instance = LPUART1;
	g_hlpuart1.Init.BaudRate = 115000;
	g_hlpuart1.Init.WordLength = UART_WORDLENGTH_8B;
	g_hlpuart1.Init.StopBits = UART_STOPBITS_1;
	g_hlpuart1.Init.Parity = UART_PARITY_NONE;
	g_hlpuart1.Init.Mode = UART_MODE_TX_RX;
	g_hlpuart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
	g_hlpuart1.Init.OneBitSampling = UART_ONE_BIT_SAMPLE_DISABLE;
	g_hlpuart1.AdvancedInit.AdvFeatureInit = UART_ADVFEATURE_NO_INIT;

	if (HAL_UART_Init(&g_hlpuart1) != HAL_OK)
	{
		while (1);
	}
}

#define LPUART_TX_Pin GPIO_PIN_7
#define LPUART_TX_GPIO_Port GPIOG
#define LPUART_RX_3V3_Pin GPIO_PIN_8
#define LPUART_RX_3V3_GPIO_Port GPIOG
void HAL_UART_MspInit(UART_HandleTypeDef* huart)
{
	GPIO_InitTypeDef GPIO_InitStruct =
	{ 0 };

	if (huart->Instance != LPUART1)
		return;

	__HAL_RCC_LPUART1_CLK_ENABLE()
	;
	__HAL_RCC_GPIOG_CLK_ENABLE()
	;
	HAL_PWREx_EnableVddIO2();
	/**LPUART1 GPIO Configuration
	 PG7     ------> LPUART1_TX
	 PG8     ------> LPUART1_RX
	 */
	GPIO_InitStruct.Pin = LPUART_TX_Pin | LPUART_RX_3V3_Pin;
	GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
	GPIO_InitStruct.Pull = GPIO_PULLUP;
	GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
	GPIO_InitStruct.Alternate = GPIO_AF8_LPUART1;
	HAL_GPIO_Init(GPIOG, &GPIO_InitStruct);
}

void HAL_MspInit(void)
{
	__HAL_RCC_SYSCFG_CLK_ENABLE()
	;
	__HAL_RCC_PWR_CLK_ENABLE()
	;
}

/**
 * @brief  System Clock Configuration
 *         The system Clock is configured as follows :
 *            System Clock source            = PLL (HSE)
 *            SYSCLK(Hz)                     = 80000000
 *            HCLK(Hz)                       = 80000000
 #if defined(USE_STM32L476G_EVAL_REVA)
 * @ note REVA depency, need AHBCLK div 2 to perform correctly LCD access
 *            AHB Prescaler                  = 2
 #elif defined(USE_STM32L476G_EVAL_REVB)
 *            AHB Prescaler                  = 1
 #endif
 *            APB1 Prescaler                 = 1
 *            APB2 Prescaler                 = 1
 *            HSE Frequency(Hz)              = 8000000
 *            PLL_M                          = 1
 *            PLL_N                          = 20
 *            PLL_P                          = 7
 *            PLL_Q                          = 4
 *            PLL_R                          = 2
 *            Flash Latency(WS)              = 4
 * @param  None
 * @retval None
 */
void SystemClock_Config(void)
{
	RCC_ClkInitTypeDef RCC_ClkInitStruct =
	{ 0 };
	RCC_OscInitTypeDef RCC_OscInitStruct =
	{ 0 };

	/* Enable HSE Oscillator and activate PLL with HSE as source   */
	/* (Default MSI Oscillator enabled at system reset remains ON) */
	RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSE;
	RCC_OscInitStruct.HSEState = RCC_HSE_ON;
	RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
	RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSE;
	RCC_OscInitStruct.PLL.PLLM = 1;
	RCC_OscInitStruct.PLL.PLLN = 20;
	RCC_OscInitStruct.PLL.PLLR = 2;
	RCC_OscInitStruct.PLL.PLLP = 7;
	RCC_OscInitStruct.PLL.PLLQ = 4;
	if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
	{
		/* Initialization Error */
		while (1)
			;
	}

	/* Select PLL as system clock source and configure the HCLK, PCLK1 and PCLK2
	 clocks dividers */
	RCC_ClkInitStruct.ClockType = (RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_PCLK1 | RCC_CLOCKTYPE_PCLK2);
	RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
	RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV2;
	RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;
	RCC_ClkInitStruct.APB2CLKDivider = RCC_HCLK_DIV1;
	if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_4) != HAL_OK)
	{
		/* Initialization Error */
		while (1)
			;
	}
}

/*
 * app.c
 *
 *  Created on: 14 feb 2019
 *      Author: denarod
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "stm32l496g_discovery.h"
#include "stm32l496g_discovery_lcd.h"
#include "stm32l496g_discovery_ts.h"

#include "phone565.bmp.h"
#include "mail565.bmp.h"
#include "chat565.bmp.h"
#include "play565.bmp.h"

#include "hcr_nn.h"
#include "hcr_nn_data.h"
#define USE_QUANTIZED_MODEL 0

static ai_handle g_hcr_network = AI_HANDLE_NULL;
static uint8_t g_net_activations[AI_HCR_NN_DATA_ACTIVATIONS_SIZE];
static float g_ai_output[AI_HCR_NN_OUT_1_SIZE];
#if USE_QUANTIZED_MODEL
static ai_u8 g_ai_input[AI_HCR_NN_IN_1_SIZE];
#else
static ai_float g_ai_input[AI_HCR_NN_IN_1_SIZE];
#endif
static ai_buffer g_net_in[AI_HCR_NN_IN_NUM]  = AI_HCR_NN_IN;
static ai_buffer g_net_out[AI_HCR_NN_OUT_NUM] = AI_HCR_NN_OUT;
static char AI_Process(float *prob,bool digit);

#define PEN_POINT_SIZE 9
#define TOUCH_TIMEOUT 700

static void fsm(void);
#define FSM_STATE_READY 1
#define FSM_STATE_DRAW 2

#define PATCH_SIZE 28
static uint16_t g_patch565[PATCH_SIZE*PATCH_SIZE];

void TextComposition(char new_char);





void Application(void)
{
    BSP_LCD_Init();
	BSP_TS_Init(ST7789H2_LCD_PIXEL_WIDTH,ST7789H2_LCD_PIXEL_HEIGHT);
    BSP_JOY_Init(JOY_MODE_GPIO);

    memset(g_patch565,0,sizeof(g_patch565));
    __HAL_RCC_CRC_CLK_ENABLE();

    ai_hcr_nn_create(&g_hcr_network,(const ai_buffer*)AI_HCR_NN_DATA_CONFIG);
    ai_network_params hcr_net_params = AI_NETWORK_PARAMS_INIT(AI_HCR_NN_DATA_WEIGHTS(ai_hcr_nn_data_weights_get()),AI_HCR_NN_DATA_ACTIVATIONS(g_net_activations));
    ai_hcr_nn_init(g_hcr_network,&hcr_net_params);


    g_net_in[0].data = AI_HANDLE_PTR(g_ai_input);
    g_net_in[0].n_batches = 1;
    g_net_out[0].n_batches = 1;
    g_net_out[0].data = AI_HANDLE_PTR(g_ai_output);

    TS_StateTypeDef ts_state;
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
    BSP_LCD_SetBackColor(LCD_COLOR_BLACK);
    BSP_LCD_SetFont(&Font24);
    BSP_LCD_DisplayStringAt(1,96,(uint8_t*)"Handwriting",CENTER_MODE);
    BSP_LCD_DisplayStringAt(1,124,(uint8_t*)"recognition",CENTER_MODE);
    BSP_LCD_SetFont(&Font16);
    BSP_LCD_DisplayStringAt(1,154,(uint8_t*)"tap to start",CENTER_MODE);
    while(1)
    {
        if( (BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.touchDetected > 0) )
        {
            while((BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.touchDetected > 0));
            break;
        }
    }


    while(1) fsm();
}

void fsm(void)
{
    static bool g_is_digit = false;
    static bool g_run_inference = false;
    static int g_fsm_state = FSM_STATE_READY;
    static uint32_t touch_time;
    TS_StateTypeDef ts_state;
    int ii;
    char prediction;
    float prob=0.0F;
    char msg[30];

    //printf("g_fsm_state %d\r\n", g_fsm_state);
    switch(g_fsm_state)
    {

        case FSM_STATE_READY:
            BSP_LCD_Clear(LCD_COLOR_BLACK);
            BSP_LCD_DrawHLine(1,216,239);
            BSP_LCD_SetFont(&Font16);
            BSP_LCD_DisplayStringAt(1,223,(g_is_digit)?(uint8_t*)"Digits":(uint8_t*)"Letters",CENTER_MODE);

            if( g_run_inference )
            {
                // Draw mini patch
                ii=0;
                for(uint16_t y=0;y<PATCH_SIZE;y+=1)
                {
                    for(uint16_t x=0;x<PATCH_SIZE;x+=1)
                    {
                        BSP_LCD_DrawPixel(x,y,g_patch565[ii++]);
                    }
                }

                // run inference
                touch_time = HAL_GetTick();
                prediction = AI_Process(&prob,g_is_digit);
                touch_time = HAL_GetTick() - touch_time;
                g_run_inference = false;

                // show prediction
                sprintf(msg,"= %c",prediction);
				BSP_LCD_DisplayStringAt(38,4,(uint8_t*)msg,LEFT_MODE);
				sprintf(msg,"(%d%% %dms)",(int)(prob*100),touch_time);
				BSP_LCD_DisplayStringAt(98,8,(uint8_t*)msg,LEFT_MODE);

				// show text composition
				TextComposition(prediction);

            }

            while(1)
            {
                if( (BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.touchDetected > 0) )
                {
                    g_fsm_state = FSM_STATE_DRAW;
                    BSP_LCD_Clear(LCD_COLOR_BLACK);
                    touch_time = HAL_GetTick();
                    break;
                }
                if( BSP_JOY_GetState() != JOY_NONE )
                {
                    g_is_digit = !g_is_digit;
                    while( BSP_JOY_GetState() != JOY_NONE );
                    break;
                }
            }
            break;

        case FSM_STATE_DRAW:

            while(1)
            {
                if( (BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.touchDetected > 0) )
                {
                    BSP_LCD_FillCircle(ts_state.touchX[0],ts_state.touchY[0],PEN_POINT_SIZE);
                    touch_time = HAL_GetTick();
                }

                if(HAL_GetTick() - touch_time > TOUCH_TIMEOUT)
                {
                    // Grab data
                    ii = 0;
                    for(uint16_t y=16;y<ST7789H2_LCD_PIXEL_HEIGHT;y+=8)
                    {
                        for(uint16_t x=16;x<ST7789H2_LCD_PIXEL_HEIGHT;x+=8)
                        {
                            g_patch565[ii] = BSP_LCD_ReadPixel(x,y);
#if USE_QUANTIZED_MODEL
                            g_ai_input[ii] = (g_patch565[ii] > 0)?255:0;
#else
                            g_ai_input[ii] = (g_patch565[ii] > 0)?1.0F:0.0F;
#endif
                            ii++;
                        }
                    }
                    g_fsm_state = FSM_STATE_READY;
                    g_run_inference = true;
                    break;
                }
            }
            break;

        default:
            break;
    }

}




void TextComposition(char new_char)
{
  static uint8_t txt[13];
  static uint8_t txt_i = 0;

  	if( new_char != '?' )
  	{
		txt[txt_i++] = new_char;
		if(txt_i == 13) { txt_i = 1; txt[0] = new_char; }
		txt[txt_i] = 0;
  	}

	if(strcmp(txt,"CALL")==0)
	{
		txt[0] = 0; txt_i = 0;
		BSP_LCD_DrawBitmap(72,72,(uint8_t*)phone565_bmp);
	}
	else if(strcmp(txt,"MAIL")==0)
	{
		txt[0] = 0; txt_i = 0;
		BSP_LCD_DrawBitmap(72,72,(uint8_t*)mail565_bmp);
	}
	else if(strcmp(txt,"CHAT")==0)
	{
		txt[0] = 0; txt_i = 0;
		BSP_LCD_DrawBitmap(72,72,(uint8_t*)chat565_bmp);
	}
	else if(strcmp(txt,"PLAY")==0)
	{
		txt[0] = 0; txt_i = 0;
		BSP_LCD_DrawBitmap(72,72,(uint8_t*)play565_bmp);
	}
	else
	{
		BSP_LCD_SetFont(&Font24);
		BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
		BSP_LCD_DisplayStringAt(1,120,txt,CENTER_MODE);
		BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
	}
}




char AI_Process(float *prob,bool digit)
{
  char prediction;
  float max = 0.0F;
  int32_t imax = -1;

  	ai_hcr_nn_run(g_hcr_network, &g_net_in[0], &g_net_out[0]);
  	for(int ii=0;ii<AI_HCR_NN_OUT_1_SIZE;ii++)
  	{
  		if( g_ai_output[ii] > max ) { max = g_ai_output[ii]; imax = ii; }
  	}

    if( digit )
    {
        if(imax == 24) imax = 0;      // O -> 0
        else if(imax == 18) imax = 1; // I -> 1
        else if(imax == 16) imax = 6; // G -> 6
        else if(imax == 28) imax = 5; // S -> 5
        else if(imax == 35) imax = 2; // Z -> 2
        else if(imax>9) max = 0.0F;
    }
    else
    {
        if(imax == 0) imax = 24;      // 0 -> O
        else if(imax == 1) imax = 18; // 1 -> I
        else if(imax == 6) imax = 16; // 6 -> G
        else if(imax == 5) imax = 28; // 5 -> S
        else if(imax == 2) imax = 35; // 2 -> Z
        else if(imax<10) max = 0.0F;
    }

    if(imax >= 0 && max > 0.5F) prediction = (imax<10)?imax+48:imax+55;
    else prediction = '?';
    *prob = max;
    return prediction;
}




/*
 * app.c
 *
 *  Created on: 14 feb 2019
 *      Author: denarod
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include "stm32l476g_eval.h"
#include "stm32l476g_eval_ts.h"
#include "stm32l476g_eval_lcd.h"

#include "hcr_float.h"
#include "hcr_float_data.h"

#define LCD_XSIZE 320
#define LCD_YSIZE 240

#define PEN_POINT_SIZE 7
#define TOUCH_TIMEOUT 400

static void fsm(void);
#define FSM_STATE_READY 1
#define FSM_STATE_DRAW 2

#define PATCH_SIZE 28

uint8_t g_framebuffer[160*120];
void DrawUI();
void MemDrawVLine(uint16_t Xpos, uint16_t Ypos, uint16_t Length);
void MemFillCircle(int x, int y, int radius);
char AI_Process(float *prob);
void TextComposition(char new_char);

static ai_handle g_hcr_network = AI_HANDLE_NULL;
static uint8_t g_net_activations[AI_HCR_FLOAT_DATA_ACTIVATIONS_SIZE];
static float g_ai_output[AI_HCR_FLOAT_OUT_1_SIZE];
static float g_ai_input[AI_HCR_FLOAT_IN_1_SIZE];
static ai_buffer g_net_in[AI_HCR_FLOAT_IN_NUM]  = AI_HCR_FLOAT_IN;
static ai_buffer g_net_out[AI_HCR_FLOAT_OUT_NUM] = AI_HCR_FLOAT_OUT;




void Application(void)
{
    BSP_LCD_Init();
    BSP_TS_Init(BSP_LCD_GetXSize(), BSP_LCD_GetYSize());

    __HAL_RCC_CRC_CLK_ENABLE();

    TS_StateTypeDef ts_state;
    BSP_LCD_Clear(LCD_COLOR_BLACK);
    BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
    BSP_LCD_SetBackColor(LCD_COLOR_BLACK);
    BSP_LCD_SetFont(&Font24);
    BSP_LCD_DisplayStringAt(1,96,(uint8_t*)"Handwriting",CENTER_MODE);
    BSP_LCD_DisplayStringAt(1,124,(uint8_t*)"recognition",CENTER_MODE);
    BSP_LCD_SetFont(&Font16);
    BSP_LCD_DisplayStringAt(1,154,(uint8_t*)"tap to start",CENTER_MODE);

    memset(g_framebuffer,0,sizeof(g_framebuffer));

    ai_hcr_float_create(&g_hcr_network,(const ai_buffer*)AI_HCR_FLOAT_DATA_CONFIG);
    ai_network_params hcr_net_params = AI_NETWORK_PARAMS_INIT(AI_HCR_FLOAT_DATA_WEIGHTS(ai_hcr_float_data_weights_get()),AI_HCR_FLOAT_DATA_ACTIVATIONS(g_net_activations));
    ai_hcr_float_init(g_hcr_network,&hcr_net_params);


    g_net_in[0].data = AI_HANDLE_PTR(g_ai_input);
    g_net_in[0].n_batches = 1;
    g_net_out[0].n_batches = 1;
    g_net_out[0].data = AI_HANDLE_PTR(g_ai_output);


    while(1)
    {
    	ts_state.TouchDetected = 0;
    	BSP_TS_GetState(&ts_state);
        if( ts_state.TouchDetected > 0 )
        {
            break;
        }
    }

    while(1) fsm();
}


void fsm(void)
{
    static bool g_run_inference = false;
    static int g_fsm_state = FSM_STATE_READY;
    static uint32_t touch_time;
    TS_StateTypeDef ts_state;
    int ii;
    char prediction;
    float prob=0.0F;
    char msg[30];

    switch(g_fsm_state)
    {

        case FSM_STATE_READY:
        	DrawUI();

            if( g_run_inference )
            {
            	DrawUI();

                // Draw mini patch
                ii=0;
                for(uint16_t y=0;y<PATCH_SIZE;y+=1)
                {
                    for(uint16_t x=0;x<PATCH_SIZE;x+=1)
                    {
                        BSP_LCD_DrawPixel(y,x,(g_ai_input[ii++]>0.0F)?0xFFFF:0x0000);
                    }
                }

                // run inference
                touch_time = HAL_GetTick();
                prediction = AI_Process(&prob);
                touch_time = HAL_GetTick() - touch_time;
                g_run_inference = false;

                // show prediction
                BSP_LCD_SetFont(&Font16);
                sprintf(msg,"= %c",prediction);
				BSP_LCD_DisplayStringAt(38,4,(uint8_t*)msg,LEFT_MODE);
				sprintf(msg,"(%d%% %dms)",(int)(prob*100),touch_time);
				BSP_LCD_DisplayStringAt(98,4,(uint8_t*)msg,LEFT_MODE);

				// show text composition
				TextComposition(prediction);

            }

            while(1)
            {
                ts_state.TouchDetected = 0;
                if( (BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.TouchDetected > 0) )
                {
                    g_fsm_state = FSM_STATE_DRAW;
                	DrawUI();
                    memset(g_framebuffer,0,sizeof(g_framebuffer));
                    touch_time = HAL_GetTick();
                    break;
                }
            }
            break;

        case FSM_STATE_DRAW:

            while(1)
            {
            	ts_state.TouchDetected = 0;
                if( (BSP_TS_GetState(&ts_state) == TS_OK )&&( ts_state.TouchDetected > 0) )
                {
                    BSP_LCD_FillCircle(ts_state.x,ts_state.y,PEN_POINT_SIZE);
                    MemFillCircle(ts_state.x,ts_state.y,3);
                    touch_time = HAL_GetTick();
                }

                if(HAL_GetTick() - touch_time > TOUCH_TIMEOUT)
                {
                    // Grab data
                    ii = 0;
                    for(uint16_t y=36;y<120;y+=3)
                    {
                        for(uint16_t x=38;x<160-38;x+=3)
                        {
                        	g_ai_input[ii] = (float)g_framebuffer[(y*160) + x] * 1.0F;
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


char AI_Process(float *prob)
{
  char prediction;
  float max = 0.0F;
  int32_t imax = -1;
  int softmax;

  	ai_hcr_float_run(g_hcr_network, &g_net_in[0], &g_net_out[0]);

  	softmax = AI_HCR_FLOAT_OUT_1_SIZE;
  	for(int ii=0;ii<softmax;ii++)
  	{
  		if( g_ai_output[ii] > max ) { max = g_ai_output[ii]; imax = ii; }
  	}


    if(imax >= 0 && max > 0.5F) prediction = (imax<10)?imax+48:imax+55;
    else prediction = '?';
    *prob = max;
    return prediction;
}







/*****************************************************************************
 *
 *
 * User Interface
 *
 *
 *****************************************************************************/

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

	BSP_LCD_SetFont(&Font16);
	BSP_LCD_SetTextColor(LCD_COLOR_YELLOW);
	BSP_LCD_DisplayStringAt(1,LINE(3),txt,CENTER_MODE);
	BSP_LCD_SetTextColor(LCD_COLOR_WHITE);
}

void DrawUI()
{
    BSP_LCD_Clear(LCD_COLOR_BLACK);
	BSP_LCD_DrawRect(76,71,168,168);
}

void MemFillCircle(int Xpos, int Ypos, int radius)
{
	int32_t  decision;      /* Decision Variable */
	uint32_t  curx;    		/* Current X Value */
	uint32_t  cury;    		/* Current Y Value */

	decision = 3 - (radius << 1);

	curx = 0;
	cury = radius;

	while (curx <= cury)
	{
		if(cury > 0)
		{
			MemDrawVLine(Xpos + curx, Ypos - cury, 2*cury);
			MemDrawVLine(Xpos - curx, Ypos - cury, 2*cury);
		}

		if(curx > 0)
		{
			MemDrawVLine(Xpos - cury, Ypos - curx, 2*curx);
			MemDrawVLine(Xpos + cury, Ypos - curx, 2*curx);
		}
		if (decision < 0)
		{
		  decision += (curx << 2) + 6;
		}
		else
		{
		  decision += ((curx - cury) << 2) + 10;
		  cury--;
		}
		curx++;
	}

}

void MemDrawVLine(uint16_t Xpos, uint16_t Ypos, uint16_t Length)
{
  uint32_t index = 0;
  uint32_t i = 0;

  	Xpos /= 2;
  	Ypos /= 2;
  	Length /=2;
	for(i = 0; i < Length; i++)
	{
		index = ((Ypos + i) * 160) + Xpos;
		if(index >= (160*120)) continue;
		g_framebuffer[ index ] = 1;
	}
}


/**
  ******************************************************************************
  * @file    hcr_float.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Tue Oct  1 13:31:40 2019
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */



#include "hcr_float.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 4
#define AI_TOOLS_VERSION_MINOR 1
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_hcr_float
 
#undef AI_HCR_FLOAT_MODEL_SIGNATURE
#define AI_HCR_FLOAT_MODEL_SIGNATURE     "e59505f779e8eadda6a6d7a6c55d2905"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Tue Oct  1 13:31:40 2019"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_HCR_FLOAT_N_BATCHES
#define AI_HCR_FLOAT_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2d_4_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv2d_2_scratch0_array;   /* Array #1 */
AI_STATIC ai_array dense_2_bias_array;   /* Array #2 */
AI_STATIC ai_array dense_2_weights_array;   /* Array #3 */
AI_STATIC ai_array dense_1_bias_array;   /* Array #4 */
AI_STATIC ai_array dense_1_weights_array;   /* Array #5 */
AI_STATIC ai_array conv2d_4_bias_array;   /* Array #6 */
AI_STATIC ai_array conv2d_4_weights_array;   /* Array #7 */
AI_STATIC ai_array conv2d_3_bias_array;   /* Array #8 */
AI_STATIC ai_array conv2d_3_weights_array;   /* Array #9 */
AI_STATIC ai_array conv2d_2_bias_array;   /* Array #10 */
AI_STATIC ai_array conv2d_2_weights_array;   /* Array #11 */
AI_STATIC ai_array conv2d_1_bias_array;   /* Array #12 */
AI_STATIC ai_array conv2d_1_weights_array;   /* Array #13 */
AI_STATIC ai_array input_0_output_array;   /* Array #14 */
AI_STATIC ai_array conv2d_1_output_array;   /* Array #15 */
AI_STATIC ai_array conv2d_2_output_array;   /* Array #16 */
AI_STATIC ai_array conv2d_3_output_array;   /* Array #17 */
AI_STATIC ai_array conv2d_4_output_array;   /* Array #18 */
AI_STATIC ai_array dense_1_output_array;   /* Array #19 */
AI_STATIC ai_array dense_1_nl_output_array;   /* Array #20 */
AI_STATIC ai_array dense_2_output_array;   /* Array #21 */
AI_STATIC ai_array dense_2_nl_output_array;   /* Array #22 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2d_4_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv2d_2_scratch0;   /* Tensor #1 */
AI_STATIC ai_tensor dense_2_bias;   /* Tensor #2 */
AI_STATIC ai_tensor dense_2_weights;   /* Tensor #3 */
AI_STATIC ai_tensor dense_1_bias;   /* Tensor #4 */
AI_STATIC ai_tensor dense_1_weights;   /* Tensor #5 */
AI_STATIC ai_tensor conv2d_4_bias;   /* Tensor #6 */
AI_STATIC ai_tensor conv2d_4_weights;   /* Tensor #7 */
AI_STATIC ai_tensor conv2d_3_bias;   /* Tensor #8 */
AI_STATIC ai_tensor conv2d_3_weights;   /* Tensor #9 */
AI_STATIC ai_tensor conv2d_2_bias;   /* Tensor #10 */
AI_STATIC ai_tensor conv2d_2_weights;   /* Tensor #11 */
AI_STATIC ai_tensor conv2d_1_bias;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_1_weights;   /* Tensor #13 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #14 */
AI_STATIC ai_tensor conv2d_1_output;   /* Tensor #15 */
AI_STATIC ai_tensor conv2d_2_output;   /* Tensor #16 */
AI_STATIC ai_tensor conv2d_3_output;   /* Tensor #17 */
AI_STATIC ai_tensor conv2d_4_output;   /* Tensor #18 */
AI_STATIC ai_tensor conv2d_4_output0;   /* Tensor #19 */
AI_STATIC ai_tensor dense_1_output;   /* Tensor #20 */
AI_STATIC ai_tensor dense_1_nl_output;   /* Tensor #21 */
AI_STATIC ai_tensor dense_2_output;   /* Tensor #22 */
AI_STATIC ai_tensor dense_2_nl_output;   /* Tensor #23 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv2d_1_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2d_2_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv2d_3_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain conv2d_4_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain dense_1_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain dense_1_nl_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain dense_2_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain dense_2_nl_chain;   /* Chain #7 */


/**  Subgraph network operator tensor chain declarations  *********************/


/**  Subgraph network operator declarations  *********************************/


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d conv2d_1_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_2_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d conv2d_3_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_4_layer; /* Layer #3 */
AI_STATIC ai_layer_dense dense_1_layer; /* Layer #4 */
AI_STATIC ai_layer_nl dense_1_nl_layer; /* Layer #5 */
AI_STATIC ai_layer_dense dense_2_layer; /* Layer #6 */
AI_STATIC ai_layer_nl dense_2_nl_layer; /* Layer #7 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 144,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 6480,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 34560,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1296,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 12,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 324,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 81,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 3,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 27,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 784,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 2028,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 432,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 192,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_1_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_1_nl_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 180,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_2_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 36,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  dense_2_nl_output_array, AI_ARRAY_FORMAT_FLOAT|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 36,
  AI_STATIC)




/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 12, 8, 2), AI_STRIDE_INIT(4, 4, 4, 48, 384),
  1, &conv2d_4_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 24, 2), AI_STRIDE_INIT(4, 4, 4, 12, 288),
  1, &conv2d_2_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &dense_2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 180, 36, 1, 1), AI_STRIDE_INIT(4, 4, 720, 25920, 25920),
  1, &dense_2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 180, 1, 1), AI_STRIDE_INIT(4, 4, 4, 720, 720),
  1, &dense_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 192, 180, 1, 1), AI_STRIDE_INIT(4, 4, 768, 138240, 138240),
  1, &dense_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &conv2d_4_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 12, 3, 3, 12), AI_STRIDE_INIT(4, 4, 48, 144, 432),
  1, &conv2d_4_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 12, 1, 1), AI_STRIDE_INIT(4, 4, 4, 48, 48),
  1, &conv2d_3_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 12), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv2d_3_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_2_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 3), AI_STRIDE_INIT(4, 4, 12, 36, 108),
  1, &conv2d_2_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 1, 1), AI_STRIDE_INIT(4, 4, 4, 12, 12),
  1, &conv2d_1_bias_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 3, 3), AI_STRIDE_INIT(4, 4, 4, 12, 36),
  1, &conv2d_1_weights_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1, 28, 28), AI_STRIDE_INIT(4, 4, 4, 4, 112),
  1, &input_0_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 26, 26), AI_STRIDE_INIT(4, 4, 4, 12, 312),
  1, &conv2d_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 12, 12), AI_STRIDE_INIT(4, 4, 4, 12, 144),
  1, &conv2d_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 12, 10, 10), AI_STRIDE_INIT(4, 4, 4, 48, 480),
  1, &conv2d_3_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 12, 4, 4), AI_STRIDE_INIT(4, 4, 4, 48, 192),
  1, &conv2d_4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 192, 1, 1), AI_STRIDE_INIT(4, 4, 4, 768, 768),
  1, &conv2d_4_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 180, 1, 1), AI_STRIDE_INIT(4, 4, 4, 720, 720),
  1, &dense_1_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_1_nl_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 180, 1, 1), AI_STRIDE_INIT(4, 4, 4, 720, 720),
  1, &dense_1_nl_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &dense_2_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  dense_2_nl_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 36, 1, 1), AI_STRIDE_INIT(4, 4, 4, 144, 144),
  1, &dense_2_nl_output_array, NULL)


/**  Layer declarations section  **********************************************/



AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 0,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &conv2d_2_layer, AI_STATIC,
  .tensors = &conv2d_1_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 1,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &conv2d_3_layer, AI_STATIC,
  .tensors = &conv2d_2_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_ap_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_2_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_3_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE,
  conv2d, forward_conv2d,
  &AI_NET_OBJ_INSTANCE, &conv2d_4_layer, AI_STATIC,
  .tensors = &conv2d_3_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_3_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool,
  &AI_NET_OBJ_INSTANCE, &dense_1_layer, AI_STATIC,
  .tensors = &conv2d_4_chain, 
  .groups = 1, 
  .nl_func = nl_func_relu_array_f32, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(2, 2), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_ap_array_f32, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output0),
  AI_TENSOR_LIST_ENTRY(&dense_1_output),
  AI_TENSOR_LIST_ENTRY(&dense_1_weights, &dense_1_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_layer, 8,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &dense_1_nl_layer, AI_STATIC,
  .tensors = &dense_1_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_1_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_1_output),
  AI_TENSOR_LIST_ENTRY(&dense_1_nl_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_1_nl_layer, 8,
  NL_TYPE,
  nl, forward_relu,
  &AI_NET_OBJ_INSTANCE, &dense_2_layer, AI_STATIC,
  .tensors = &dense_1_nl_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_1_nl_output),
  AI_TENSOR_LIST_ENTRY(&dense_2_output),
  AI_TENSOR_LIST_ENTRY(&dense_2_weights, &dense_2_bias),
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_layer, 10,
  DENSE_TYPE,
  dense, forward_dense,
  &AI_NET_OBJ_INSTANCE, &dense_2_nl_layer, AI_STATIC,
  .tensors = &dense_2_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  dense_2_nl_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&dense_2_output),
  AI_TENSOR_LIST_ENTRY(&dense_2_nl_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  dense_2_nl_layer, 10,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &dense_2_nl_layer, AI_STATIC,
  .tensors = &dense_2_nl_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 172056, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 10416, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_HCR_FLOAT_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_HCR_FLOAT_OUT_NUM, &dense_2_nl_output),
  &conv2d_1_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool hcr_float_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2d_4_scratch0_array.data = AI_PTR(activations + 4800);
    conv2d_4_scratch0_array.data_start = AI_PTR(activations + 4800);
    conv2d_2_scratch0_array.data = AI_PTR(activations + 8112);
    conv2d_2_scratch0_array.data_start = AI_PTR(activations + 8112);
    input_0_output_array.data = AI_PTR(NULL);
    input_0_output_array.data_start = AI_PTR(NULL);
    conv2d_1_output_array.data = AI_PTR(activations + 0);
    conv2d_1_output_array.data_start = AI_PTR(activations + 0);
    conv2d_2_output_array.data = AI_PTR(activations + 8688);
    conv2d_2_output_array.data_start = AI_PTR(activations + 8688);
    conv2d_3_output_array.data = AI_PTR(activations + 0);
    conv2d_3_output_array.data_start = AI_PTR(activations + 0);
    conv2d_4_output_array.data = AI_PTR(activations + 5568);
    conv2d_4_output_array.data_start = AI_PTR(activations + 5568);
    dense_1_output_array.data = AI_PTR(activations + 0);
    dense_1_output_array.data_start = AI_PTR(activations + 0);
    dense_1_nl_output_array.data = AI_PTR(activations + 0);
    dense_1_nl_output_array.data_start = AI_PTR(activations + 0);
    dense_2_output_array.data = AI_PTR(activations + 720);
    dense_2_output_array.data_start = AI_PTR(activations + 720);
    dense_2_nl_output_array.data = AI_PTR(NULL);
    dense_2_nl_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool hcr_float_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    dense_2_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_2_bias_array.data = AI_PTR(weights + 171912);
    dense_2_bias_array.data_start = AI_PTR(weights + 171912);
    dense_2_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_2_weights_array.data = AI_PTR(weights + 145992);
    dense_2_weights_array.data_start = AI_PTR(weights + 145992);
    dense_1_bias_array.format |= AI_FMT_FLAG_CONST;
    dense_1_bias_array.data = AI_PTR(weights + 145272);
    dense_1_bias_array.data_start = AI_PTR(weights + 145272);
    dense_1_weights_array.format |= AI_FMT_FLAG_CONST;
    dense_1_weights_array.data = AI_PTR(weights + 7032);
    dense_1_weights_array.data_start = AI_PTR(weights + 7032);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(weights + 6984);
    conv2d_4_bias_array.data_start = AI_PTR(weights + 6984);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(weights + 1800);
    conv2d_4_weights_array.data_start = AI_PTR(weights + 1800);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(weights + 1752);
    conv2d_3_bias_array.data_start = AI_PTR(weights + 1752);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(weights + 456);
    conv2d_3_weights_array.data_start = AI_PTR(weights + 456);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(weights + 444);
    conv2d_2_bias_array.data_start = AI_PTR(weights + 444);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(weights + 120);
    conv2d_2_weights_array.data_start = AI_PTR(weights + 120);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(weights + 108);
    conv2d_1_bias_array.data_start = AI_PTR(weights + 108);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(weights + 0);
    conv2d_1_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_hcr_float_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_HCR_FLOAT_MODEL_NAME,
      .model_signature   = AI_HCR_FLOAT_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 230262,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_hcr_float_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_hcr_float_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_hcr_float_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_hcr_float_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= hcr_float_configure_weights(net_ctx, &params->params);
  ok &= hcr_float_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_hcr_float_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_hcr_float_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_HCR_FLOAT_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME


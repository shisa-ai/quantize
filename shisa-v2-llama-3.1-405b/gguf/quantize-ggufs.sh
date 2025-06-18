#!/bin/bash

LLAMA_BIN="/data/llama.cpp/build/bin"
FP16="shisa-v2-llama3.1-405b-fp16.gguf"
IMATRIX="imatrix.dat"
NAME="shisa-v2-llama3.1-405b"

# Q8_0
${LLAMA_BIN}/llama-quantize ${FP16} ${NAME}-Q8_0.gguf Q8_0

# Q4_K_M
${LLAMA_BIN}/llama-quantize --imatrix ${IMATRIX} ${FP16} ${NAME}-Q4_K_M.gguf Q4_K_M

# IQ4_XS
${LLAMA_BIN}/llama-quantize --imatrix ${IMATRIX} ${FP16} ${NAME}-IQ4_XS.gguf IQ4_XS

# IQ3_M
${LLAMA_BIN}/llama-quantize --imatrix ${IMATRIX} ${FP16} ${NAME}-IQ3_M.gguf IQ3_M

# IQ3_XS
${LLAMA_BIN}/llama-quantize --imatrix ${IMATRIX} ${FP16} ${NAME}-IQ3_XS.gguf IQ3_XS

# IQ2_XXS
${LLAMA_BIN}/llama-quantize --imatrix ${IMATRIX} ${FP16} ${NAME}-IQ2_XXS.gguf IQ2_XXS

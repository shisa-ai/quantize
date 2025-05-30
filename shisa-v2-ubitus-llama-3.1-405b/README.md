# FP8 Dynamic
This is the easiest compression and only takes ~1h to do.
```
python quantize-dynamic-fp8.py
```


# W8A8-INT8
This uses [shisa-ai/shisa-v2-sharegpt](https://huggingface.co/datasets/shisa-ai/shisa-v2-sharegpt) as calibration set. It took about 9h for the smoothquant, just a hair under 2 days to run the full quant using all 8 H100s - it hit a maximum of 125GB (>85%) memory usage/GPU.
```
python quantize-llmcompressor-w8a8-int8.py \
    --model-id shisa-ai/shisa-v2-ubitus-llama3.1-405b \
    --out-dir shisa-v2-ubitus-llama3.1-405b-W8A8-INT8 \
    --calib-samples 2048 \
    --group-size 32 \
    --save-smoothquant
```

TODO:

# GPTQ W4A16  - H200 uses 120GB
python quantize-gptqmodel-w4a16.py --model-id shisa-ai/shisa-v2-ubitus-llama3.1-405b --out-dir shisa-v2-ubitus-llama3.1-405b-W4A16 --calib-samples 2048 --group-size 3216 --batch-size 4

# GPTQ W8A16
python quantize-gptqmodel-w4a16.py --model-id shisa-ai/shisa-v2-ubitus-llama3.1-405b --out-dir shisa-v2-ubitus-llama3.1-405b-W4816 --calib-samples 2048 --group-size 32 --batch-size 4 --bits 8


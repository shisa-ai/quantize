#!/usr/bin/env python3
"""
Debug tokenizer loading issue
"""

from transformers import AutoTokenizer
import torch

def test_tokenizer():
    model_id = "shisa-ai/shisa-v2-ubitus-llama3.1-405b"
    
    print("=" * 60)
    print("DEBUGGING TOKENIZER LOADING")
    print("=" * 60)
    
    # Test 1: Basic loading
    print("\n1. Testing basic AutoTokenizer.from_pretrained...")
    try:
        result = AutoTokenizer.from_pretrained(model_id)
        print(f"   Result type: {type(result)}")
        print(f"   Result value: {result}")
        if hasattr(result, 'vocab_size'):
            print(f"   Vocab size: {result.vocab_size}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 2: With trust_remote_code
    print("\n2. Testing with trust_remote_code=True...")
    try:
        result = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"   Result type: {type(result)}")
        print(f"   Result value: {result}")
        if hasattr(result, 'vocab_size'):
            print(f"   Vocab size: {result.vocab_size}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 3: With use_fast=False
    print("\n3. Testing with use_fast=False...")
    try:
        result = AutoTokenizer.from_pretrained(model_id, use_fast=False)
        print(f"   Result type: {type(result)}")
        print(f"   Result value: {result}")
        if hasattr(result, 'vocab_size'):
            print(f"   Vocab size: {result.vocab_size}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 4: Try the base Llama 3.1 tokenizer
    print("\n4. Testing base Llama 3.1 405B tokenizer...")
    try:
        result = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-405B")
        print(f"   Result type: {type(result)}")
        print(f"   Result value: {result}")
        if hasattr(result, 'vocab_size'):
            print(f"   Vocab size: {result.vocab_size}")
    except Exception as e:
        print(f"   ERROR: {e}")
    
    # Test 5: Check what files are in the model repo
    print("\n5. Checking model repository files...")
    try:
        from huggingface_hub import list_repo_files
        files = list_repo_files(model_id)
        tokenizer_files = [f for f in files if 'token' in f.lower()]
        print(f"   Tokenizer-related files: {tokenizer_files}")
    except Exception as e:
        print(f"   ERROR: {e}")

if __name__ == "__main__":
    test_tokenizer()

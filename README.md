# Implementing GPT-2 124M with JAX, FLAX, and NNX

## Overview

This project implements the GPT-2 124M (small) model, a transformer-based language model developed by OpenAI, using JAX and FLAX with the NNX API. The goal is to recreate the architecture of GPT-2, load pre-trained weights from Hugging Face, optimize the model using JAX's JIT compilation, and demonstrate its usage by generating text from a sample input. This README provides a comprehensive guide to understanding, setting up, and using the code, along with explanations of the key components and their functionality.

## Purpose

The implementation serves as both a practical tool for generating text and an educational resource for learning about transformer architectures, JAX, FLAX, and NNX. It assumes some familiarity with Python and machine learning but includes detailed explanations of JAX, FLAX, and transformer concepts where necessary.

## Prerequisites

Before you begin, ensure you have the following:

- **Python 3.x**: Preferably the latest stable version.
- **JAX, FLAX, and Transformers Libraries**: These are installed via pip as detailed in the setup section.
- **Hardware**: The code is optimized for CPU, GPU, or TPU. Ensure your system supports JAX's hardware acceleration (instructions for TPU are included).
- **Basic Knowledge**: Familiarity with neural networks, transformers, and Python programming.

## Installation and Setup

### Install Required Libraries

You need to install the following libraries. Run the following commands in your terminal or a notebook cell:

```bash
!pip install jax jaxlib flax transformers
!pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

- **JAX**: Provides high-performance numerical computing and automatic differentiation.
- **FLAX**: A neural network library built on JAX, using the NNX API for modern model building.
- **Transformers**: Hugging Face's library for accessing pre-trained models and tokenizers.
- **Other Dependencies**: NumPy and PyTorch are used for weight conversions and compatibility.

Adjust the JAX installation based on your hardware (CPU, GPU, or TPU). The TPU installation link ensures compatibility with Google Cloud TPUs.

### Verify Installation

After installation, import the modules and check available devices:

```python
import jax
print(jax.devices())
```

This will list the available JAX devices (e.g., CPU, GPU, TPU), confirming that JAX is correctly set up.

## Project Structure

The code is structured in a Jupyter notebook (or Python script) and includes several key sections:

1. **Setup**: Installing libraries and importing modules.
2. **Model Architecture**: Defining the GPT-2 components using FLAX's NNX API.
3. **Loading Pre-trained Weights**: Fetching and mapping weights from Hugging Face's GPT-2 model.
4. **JIT Compilation**: Optimizing the forward pass with JAX's JIT.
5. **Text Generation**: Implementing a generate function to produce text from prompts.

### Key Components

#### Multi-Head Attention

- **Purpose**: Allows the model to focus on different parts of the input sequence using causal (masked) attention.
- **Features**:
  - Projects input to query (Q), key (K), and value (V) vectors.
  - Uses a causal mask to ensure tokens only attend to previous tokens.
  - Includes dropout for regularization, disabled during inference.

#### Feed-Forward Network (FFN)

- **Purpose**: Applies two linear transformations with a GELU activation.
- **Details**:
  - Expands the embedding dimension (typically 4x) and projects it back.
  - Uses dropout for regularization.

#### Transformer Block

- **Purpose**: Combines multi-head attention and FFN with residual connections and layer normalization.
- **Structure**:
  - Pre-layer normalization (pre-LN) design.
  - Residual connections ensure stable gradient flow during training.

#### GPT-2 Model

- **Overview**: A decoder-only transformer with token and positional embeddings, multiple transformer blocks, and final layer normalization.
- **Input/Output**:
  - Takes input IDs and outputs logits (probabilities over the vocabulary).
  - Matches Hugging Face's `GPT2Model` architecture.

## Loading Pre-trained Weights

### Steps

1. **Load Hugging Face Model**:
   - Use `transformers` to fetch the GPT-2 124M configuration and pre-trained model.
   - Convert PyTorch state dictionary to JAX arrays.

2. **Initialize NNX Model**:
   - Create an instance of the `GPT2Model` with random weights and a seed.

3. **Map Weights**:
   - Adjust for differences between PyTorch (Hugging Face) and JAX/FLAX conventions (e.g., transpose linear weights).
   - Ensure parameter names align with Hugging Face's state dict.

### Potential Issues

- **Transposition**: Linear weights must be transposed to match JAX/FLAX expectations.
- **Naming**: Ensure exact key matching between Hugging Face and NNX models.
- **Dropout**: No weights to load for dropout layers.

## JIT Compilation

### Why JIT?

- **Performance**: Compiles the forward pass into efficient machine code using XLA, leveraging hardware acceleration.
- **Optimization**: Fuses operations for faster execution, especially beneficial for repeated calls.

### Implementation

The `forward` function is JIT-compiled using `jax.jit`, treating the model as a static argument to ensure the computation graph is optimized based on its structure.

## Text Generation

### Generate Function

The `generate` function produces text based on a prompt, using techniques like:

- **Sampling Methods**: Top-k, top-p (nucleus), and temperature scaling for controlled randomness.
- **Repetition Penalty**: Prevents repetitive outputs.
- **Greedy Decoding**: Option for deterministic output.

### Parameters

- `prompt`: Initial text.
- `max_length`: Maximum output length.
- `temperature`: Controls randomness (lower = more deterministic).
- `top_k/top_p`: Filters for sampling the most probable tokens.
- `repetition_penalty`: Discourages repetition.
- `eos_token_id/pad_token_id`: Control sequence termination and padding.

### Output

Returns generated text as a string, decoded from token IDs.

## Usage Example

To generate text, use the following code snippet:

```python
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
generated_text = generate(model, tokenizer, "Hi my name is", max_length=35)
print(generated_text)
```

This will output a continuation of the prompt, such as "Hi my name is John and I like to..."

## Troubleshooting

- **Installation Issues**: Ensure all dependencies are correctly installed and compatible with your hardware.
- **Weight Loading Errors**: Check for transposition errors or naming mismatches in the state dict.
- **Performance Problems**: Verify JIT compilation and hardware acceleration setup.
- **Generation Quality**: Adjust parameters like temperature and top-k/top-p for better results.

## Acknowledgments

- OpenAI for developing the original GPT-2 model.
- Hugging Face for providing pre-trained models and tokenizers.
- JAX and FLAX teams for their powerful libraries.

This README provides a thorough guide to understanding and using the GPT-2 124M implementation with JAX, FLAX, and NNX. For further details, refer to the code comments and external documentation for JAX, FLAX, and Transformers.

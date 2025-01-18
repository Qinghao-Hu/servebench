import torch
import time
from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

model_path = "/nobackup/model/qwen2-vl/Qwen2-VL-7B-Instruct"

##############################
# 1. Load model and processor
##############################
model = Qwen2VLForConditionalGeneration.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
    device_map="auto",
)

# For Qwen2-VL, set min/max_pixels if necessary
min_pixels = 256 * 28 * 28  # Qinghao: It means 256 tokens
max_pixels = 12800 * 28 * 28
processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)

##############################
# 2. Prepare the input
##############################
messages = [
    {
        "role": "user",
        "content": [
            {"type": "image", "image": "vlm/demo.jpeg"},
            {"type": "text", "text": "Describe this image."},
        ],
    }
]

# For Qwen2-VL, you typically wrap text via its chat template
text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

image_inputs, video_inputs = process_vision_info(messages)

inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)

inputs = inputs.to("cuda")

input_token_count = inputs.input_ids.shape[1]
print(f"Input tokens: {input_token_count}")

# Inference: Generation of the output
start_time = time.time()
generated_ids = model.generate(**inputs, max_new_tokens=128)
end_time = time.time()
latency = end_time - start_time

generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

generated_token_count = len(generated_ids_trimmed[0])
print(f"Generated tokens: {generated_token_count}")
print(f"Latency: {latency:.2f} seconds")
print(f"Tokens per second: {generated_token_count/latency:.2f}")
print("\nGenerated text:")
print(output_text[0])

# # The number of tokens in the "prefill" pass = length of the input_ids
# prefill_tokens = inputs["input_ids"].shape[-1]

# ################################################################################
# # 3. Prefill (initial forward) timing
# ################################################################################
# torch.cuda.synchronize()
# start_time = time.time()  # overall start

# prefill_start = time.time()
# with torch.no_grad():
#     out = model(**inputs, use_cache=True)
# torch.cuda.synchronize()
# prefill_end = time.time()

# prefill_time = prefill_end - prefill_start
# print(f"[Prefill] tokens = {prefill_tokens}, time = {prefill_time:.4f} s")

# past_key_values = out.past_key_values

# last_token = torch.argmax(out.logits[:, -1, :], dim=-1, keepdim=True)

# ################################################################################
# # 4. Decoding: measure time to first token (TTFT) and total decoding time
# ################################################################################
# max_new_tokens = 128
# decoding_start = time.time()  # start of the decoding loop
# torch.cuda.synchronize()

# generated_tokens = []

# # last_token = inputs["input_ids"][:, -1:]  # the last token from the prefill

# time_to_first_token = None

# with torch.no_grad():
#     for step in range(max_new_tokens):
#         step_start = time.time()
#         generated_tokens.append(last_token)

#         output = model(input_ids=last_token, past_key_values=past_key_values, use_cache=True)
#         logits = output.logits
#         past_key_values = output.past_key_values

#         # Greedy decode:
#         next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)

#         last_token = next_token

#         # Measure TTFT right after the first token is produced
#         if step == 0 and time_to_first_token is None:
#             torch.cuda.synchronize()
#             step_end = time.time()
#             # TTFT is the time from overall start to the end of the first decoding step
#             time_to_first_token = step_end - start_time

# torch.cuda.synchronize()
# decoding_end = time.time()
# decoding_time = decoding_end - decoding_start

# decoding_tokens = len(generated_tokens)

# print(f"[Decoding] tokens = {decoding_tokens}, time = {decoding_time:.4f} s")

# ################################################################################
# # 5. Print TTFT and final text
# ################################################################################
# print(f"[Time to First Token] = {time_to_first_token:.4f} s")

# generated_tokens = torch.cat(generated_tokens, dim=1)
# output_text = processor.batch_decode(generated_tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)

# print("=" * 50)
# print("Output:\n", output_text[0])
# print("=" * 50)


# # srun -J bench -N 1 --gpus-per-node 1 python vlm/vlm_inference.py

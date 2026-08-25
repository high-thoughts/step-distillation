import argparse
import importlib
import json
import math
import os
import random
import re
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import torch
# import torch_npu
import torch.distributed as dist
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from transformers import AutoTokenizer

from peft import PeftModel

llada_module = importlib.import_module("LLaDA-Prometheus")
LLaDAModelLM = llada_module.LLaDAModelLM
LLaDAConfig = llada_module.LLaDAConfig
from gsm8k import GSM8KDataset


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Evaluate distilled LLaDA checkpoints on GSM8K")
	parser.add_argument(
		"--base_model_path",
		type=str,
		default="/mnt/users/cuihaitao-20251013/models/dLLM-Var",
		help="Path to the base LLaDA model checkpoint",
	)
	parser.add_argument(
		"--checkpoint_path",
		type=str,
		default=None,
		help="Path to the distilled LoRA checkpoint directory. If omitted, the latest checkpoint under --checkpoint_root is used",
	)
	parser.add_argument(
		"--checkpoint_root",
		type=str,
		default="./output",
		help="Root directory to search for checkpoints when --checkpoint_path is not provided",
	)
	parser.add_argument(
		"--output_file",
		type=str,
		default=None,
		help="Optional path to write per-sample evaluation results (JSONL)",
	)
	parser.add_argument(
		"--accuracy_file",
		type=str,
		default=None,
		help="Optional path to write accuracy metrics (JSON)",
	)
	parser.add_argument(
		"--batch_size",
		type=int,
		default=1,
		help="Batch size for inference",
	)
	parser.add_argument(
		"--max_gen_length",
		type=int,
		default=256,
		help="Maximum number of tokens generated for each sample",
	)
	parser.add_argument(
		"--block_length",
		type=int,
		default=64,
		help="Token block length used by the blockwise generator",
	)
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.9,
		help="Confidence threshold for the greedy unmasking function",
	)
	parser.add_argument(
		"--num_few_shot",
		type=int,
		default=0,
		help="Number of few-shot examples prepended in prompts",
	)
	parser.add_argument(
		"--subsample",
		type=int,
		default=-1,
		help="Optional number of test samples to evaluate (-1 uses full test set)",
	)
	parser.add_argument(
		"--dtype",
		type=str,
		choices=["bfloat16", "float16", "float32"],
		default="bfloat16",
		help="Computation dtype for model weights",
	)
	parser.add_argument(
		"--seed",
		type=int,
		default=42,
		help="Random seed for subsampling",
	)
	parser.add_argument(
		"--base_only",
		action="store_true",
		help="Evaluate only the base model without loading an adapter checkpoint",
	)
	return parser.parse_args()


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.npu.is_available():
		torch.npu.manual_seed_all(seed)



def init_distributed() -> Tuple[bool, int, int, int]:
	world_size_env = int(os.environ.get("WORLD_SIZE", "1"))
	if world_size_env <= 1 and "LOCAL_RANK" not in os.environ:
		return False, 0, 1, 0
	backend = "hccl" if torch.npu.is_available() else "gloo"
	local_rank = int(os.environ.get("LOCAL_RANK", 0))
	if torch.npu.is_available():
		torch.npu.set_device(local_rank)
	if not dist.is_initialized():
		dist.init_process_group(backend=backend)
	rank = dist.get_rank()
	world_size = dist.get_world_size()
	return True, rank, world_size, local_rank


def load_tokenizer(base_model_path: str):
	tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
	if tokenizer.pad_token is None:
		tokenizer.pad_token = tokenizer.eos_token
	tokenizer.padding_side = "left"
	return tokenizer


def build_rank_output_path(base_path: str, rank: int) -> str:
	root, ext = os.path.splitext(base_path)
	if ext:
		return f"{root}_rank{rank}{ext}"
	return f"{base_path}_rank{rank}.jsonl"


def _to_abs_path(path: str, base_dir: str) -> str:
	resolved = os.path.expanduser(path)
	if not os.path.isabs(resolved):
		resolved = os.path.join(base_dir, resolved)
	return os.path.abspath(resolved)


def find_latest_checkpoint(root_dir: str) -> str:
	candidates: List[Tuple[float, str]] = []
	if not os.path.isdir(root_dir):
		raise FileNotFoundError(f"Checkpoint root {root_dir} does not exist or is not a directory")
	for current_dir, dirnames, filenames in os.walk(root_dir):
		basename = os.path.basename(current_dir)
		if not basename.startswith("checkpoint-"):
			continue
		adapter_file = os.path.join(current_dir, "adapter_model.safetensors")
		if os.path.isfile(adapter_file):
			candidates.append((os.path.getmtime(current_dir), current_dir))
	if not candidates:
		raise FileNotFoundError(
			f"No checkpoint directories containing adapter_model.safetensors were found under {root_dir}"
		)
	candidates.sort(key=lambda item: item[0], reverse=True)
	return candidates[0][1]


def resolve_checkpoint_path(
	checkpoint_path: Optional[str],
	checkpoint_root: str,
	base_dir: str,
) -> str:
	if checkpoint_path:
		resolved = _to_abs_path(checkpoint_path, base_dir)
		if not os.path.isdir(resolved):
			raise FileNotFoundError(f"Provided checkpoint_path does not exist: {resolved}")
		return resolved
	root_abs = _to_abs_path(checkpoint_root, base_dir)
	return find_latest_checkpoint(root_abs)


def load_model(base_model_path: str, checkpoint_path: Optional[str], dtype: torch.dtype, device: torch.device) -> LLaDAModelLM:
	config = LLaDAConfig.from_pretrained(base_model_path, trust_remote_code=True)
	base_model = LLaDAModelLM.from_pretrained(
		base_model_path,
		config=config,
		trust_remote_code=True,
		torch_dtype=dtype,
	)
	if checkpoint_path:
		peft_model = PeftModel.from_pretrained(base_model, checkpoint_path)
		peft_model.to(device)
		merged_model = peft_model.merge_and_unload()
		merged_model.to(device)
		merged_model.eval()
		return merged_model
	base_model.to(device)
	base_model.eval()
	return base_model


# --- Custom generate with loop count stats ---
def generate_with_stats(
    self: LLaDAModelLM,
    input_ids: torch.LongTensor,
    attention_mask: Optional[torch.Tensor] = None,
    max_gen_length: int = 1024,
    block_length: int = 64,
    threshold: float = 0.9,
    eos_token_id: int = 126081,
):
    """
    Block-wise generation that records the number of decoding iterations per block.
    Returns (generated_ids, avg_loop_count_per_block).
    The average is computed over the blocks actually decoded until EOS or max length.
    """
    # Replicate core logic of LLaDAModelLM.generate while counting per-block while-loop iterations
    batchsize, prompt_length = input_ids.shape
    max_num_blocks = max_gen_length // block_length

    output_ids = input_ids
    block_x = torch.full((batchsize, block_length), self.mask_id, dtype=torch.long, device=self.device)
    output_ids = torch.cat([output_ids, block_x], dim=-1)

    # Prefill phase to initialize cache on current sequence including the first masked block
    prefill_outputs = self.prefill_phase(output_ids, block_length)
    past_key_values = prefill_outputs['past_key_values']
    logits = prefill_outputs['logits']
    output_ids[:, -block_length:] = self.unmask_function_greedy(
        logits=logits,
        x=output_ids[:, -block_length:],
        threshold=threshold,
    )

    # Free prefill tensors ASAP
    del prefill_outputs
    del logits

    block_loop_counts = []

    # Decoding loop over blocks
    for _ in range(max_num_blocks):
        iter_count = 0
        outputs = None

        # Iteratively fill the current block
        while (output_ids[:, -block_length:] == self.mask_id).sum() > 0:
            iter_count += 1
            outputs = self(
                input_ids=output_ids[:, -block_length:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            output_ids[:, -block_length:] = self.unmask_function_greedy(
                logits=outputs.logits,
                x=output_ids[:, -block_length:],
                threshold=threshold,
            )

        # Record how many decoding iterations this block required
        block_loop_counts.append(iter_count)

        # Ensure cache is updated even if the while loop didn't run
        if outputs is None:
            outputs = self(
                input_ids=output_ids[:, -block_length:],
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )

        past_key_values = outputs.past_key_values

        # Early stop when EOS token appears in current block
        if (output_ids[:, -block_length:] == eos_token_id).any():
            gen_tokens = output_ids[:, prompt_length:]
            avg_loops = float(sum(block_loop_counts)) / max(1, len(block_loop_counts))
            return gen_tokens, avg_loops

        # Advance to next block
        block_x = torch.full((batchsize, block_length), self.mask_id, dtype=torch.long, device=self.device)
        output_ids = torch.cat([output_ids, block_x], dim=-1)

    gen_tokens = output_ids[:, prompt_length:]
    avg_loops = float(sum(block_loop_counts)) / max(1, len(block_loop_counts))
    return gen_tokens, avg_loops


ANSWER_PATTERNS = [
	re.compile(r"\\boxed\{([^{}]+)\}"),
	re.compile(r"####\s*([^\n]+)"),
	re.compile(r"答案[:：]\s*([^\n]+)"),
]


def extract_answer(text: str) -> Optional[str]:
	for pattern in ANSWER_PATTERNS:
		matches = pattern.findall(text)
		if matches:
			candidate = matches[-1]
			cleaned = clean_answer(candidate)
			if cleaned:
				return cleaned
	# Fallback: take last number in string
	numeric_matches = re.findall(r"-?\d+(?:\.\d+)?", text)
	if numeric_matches:
		return clean_answer(numeric_matches[-1])
	return None


def clean_answer(text: str) -> str:
	cleaned = text.strip()
	cleaned = cleaned.replace(",", "")
	cleaned = cleaned.replace("$", "")
	cleaned = cleaned.strip()
	return cleaned


def answers_match(pred: Optional[str], gold: Optional[str]) -> bool:
	if pred is None or gold is None:
		return False
	# Try numeric comparison when possible
	try:
		pred_val = float(pred)
		gold_val = float(gold)
		return math.isclose(pred_val, gold_val, rel_tol=1e-5, abs_tol=1e-5)
	except ValueError:
		pass
	return pred.strip() == gold.strip()


def build_dataloader(
	tokenizer,
	num_few_shot: int,
	subsample: int,
	batch_size: int,
	distributed: bool,
	rank: int,
	world_size: int,
) -> DataLoader:
	dataset = GSM8KDataset(
		tokenizer=tokenizer,
		num_examples=num_few_shot,
		add_reasoning=True,
		subsample=subsample,
	)
	if distributed:
		indices = list(range(len(dataset)))[rank::world_size]
		shard = Subset(dataset, indices)
	else:
		shard = dataset
	return DataLoader(
		shard,
		batch_size=batch_size,
		shuffle=False,
		collate_fn=dataset.collate_fn,
	)


def evaluate(
	model: LLaDAModelLM,
	tokenizer,
	dataloader: DataLoader,
	device: torch.device,
	output_file: Optional[str],
	max_gen_length: int,
	block_length: int,
	threshold: float,
	distributed: bool,
	rank: int,
) -> Tuple[float, int, int, float]:
	total = 0
	correct = 0
	results: List[str] = []

	# Sum of per-sample average loop counts to compute dataset-level average
	loop_avg_sum = 0.0

	target_file = None
	if output_file:
		target_file = output_file if not distributed else build_rank_output_path(output_file, rank)
		output_dir = os.path.dirname(target_file)
		if output_dir:
			os.makedirs(output_dir, exist_ok=True)

	progress = tqdm(
		dataloader,
		desc="Evaluating",
		dynamic_ncols=True,
		disable=distributed and rank != 0,
	)

	for batch_idx, batch in enumerate(progress):
		input_ids = batch["input_ids"].to(device)
		attention_mask = (input_ids != tokenizer.pad_token_id).to(device)

		with torch.no_grad():
			generated, avg_block_loops = model.generate(
				input_ids=input_ids,
				attention_mask=attention_mask,
				max_gen_length=max_gen_length,
				block_length=block_length,
				threshold=threshold,
				eos_token_id=tokenizer.eos_token_id or tokenizer.pad_token_id,
			)

		# Accumulate per-sample average block loops for dataset-level metric
		batch_size_cur = len(batch["questions"])
		loop_avg_sum += float(avg_block_loops) * float(batch_size_cur)

		generated = generated.cpu()
		input_ids = input_ids.cpu()
		attention_mask = attention_mask.cpu()

		for i in range(len(batch["questions"])):
			prompt_len = int(attention_mask[i].sum().item())
			prompt_tokens = input_ids[i, -prompt_len:]
			gen_tokens = generated[i]

			full_tokens = torch.cat([prompt_tokens, gen_tokens], dim=0)
			full_text = tokenizer.decode(full_tokens, skip_special_tokens=True)
			pred_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

			gold_answer = batch["answers"][i]
			pred_answer = extract_answer(pred_text)
			gold_normalized = extract_answer(gold_answer)
			is_correct = answers_match(pred_answer, gold_normalized)

			record = {
				"index": total,
				"question": batch["questions"][i],
				"gold_answer_raw": gold_answer,
				"gold_answer": gold_normalized,
				"prediction_raw": pred_text,
				"prediction_answer": pred_answer,
				"full_completion": full_text,
				"correct": is_correct,
			}

			if target_file:
				results.append(json.dumps(record, ensure_ascii=False))

			if is_correct:
				correct += 1
			total += 1

	if target_file and results:
		with open(target_file, "w", encoding="utf-8") as f:
			for line in results:
				f.write(line + "\n")

	accuracy = correct / total if total > 0 else 0.0
	avg_loops_local = loop_avg_sum / total if total > 0 else 0.0
	return accuracy, correct, total, avg_loops_local


def main():
	args = parse_args()
	distributed, rank, world_size, local_rank = init_distributed()
	set_seed(args.seed)
	script_dir = os.path.dirname(os.path.abspath(__file__))

	if args.base_only and args.checkpoint_path:
		raise ValueError("--base_only cannot be used together with --checkpoint_path")

	if args.base_only:
		checkpoint_path = None
	else:
		if distributed:
			payload: List[object] = [True, ""]
			if rank == 0:
				try:
					resolved_ckpt = resolve_checkpoint_path(
						checkpoint_path=args.checkpoint_path,
						checkpoint_root=args.checkpoint_root,
						base_dir=script_dir,
					)
					discovery_error = ""
				except Exception as err:
					payload[0] = False
					discovery_error = str(err)
					resolved_ckpt = ""
				payload[1] = resolved_ckpt if payload[0] else discovery_error
			else:
				payload = [None, None]
			dist.broadcast_object_list(payload, src=0)
			success = bool(payload[0])
			if not success:
				raise RuntimeError(f"Failed to resolve checkpoint_path: {payload[1]}")
			checkpoint_path = str(payload[1])
		else:
			checkpoint_path = resolve_checkpoint_path(
				checkpoint_path=args.checkpoint_path,
				checkpoint_root=args.checkpoint_root,
				base_dir=script_dir,
			)

	device = (
		torch.device("npu", local_rank)
		if torch.npu.is_available()
		else torch.device("cpu")
	)
	dtype = args.dtype

	tokenizer = load_tokenizer(args.base_model_path)

	dataloader = build_dataloader(
		tokenizer=tokenizer,
		num_few_shot=args.num_few_shot,
		subsample=args.subsample,
		batch_size=args.batch_size,
		distributed=distributed,
		rank=rank,
		world_size=world_size,
	)
	model = load_model(
		base_model_path=args.base_model_path,
		checkpoint_path=checkpoint_path,
		dtype=dtype,
		device=device,
	)

	# Patch model.generate with stats-enhanced version
	import types as _types
	model.generate = _types.MethodType(generate_with_stats, model)

	if args.output_file is None:
		if not distributed or rank == 0:
			timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
			if checkpoint_path:
				default_root = checkpoint_path
				default_name = f"gsm8k_eval_{timestamp}.jsonl"
			else:
				default_root = os.path.join(script_dir, "base_eval_outputs")
				default_name = f"gsm8k_eval_base_{timestamp}.jsonl"
			default_path = os.path.join(default_root, default_name)
		else:
			default_path = None
		if distributed:
			path_container = [default_path]
			dist.broadcast_object_list(path_container, src=0)
			args.output_file = path_container[0]
		else:
			args.output_file = default_path

	local_accuracy, local_correct, local_total, local_avg_loops = evaluate(
		model=model,
		tokenizer=tokenizer,
		dataloader=dataloader,
		device=device,
		output_file=args.output_file,
		max_gen_length=args.max_gen_length,
		block_length=args.block_length,
		threshold=args.threshold,
		distributed=distributed,
		rank=rank,
	)

	if distributed:
		correct_tensor = torch.tensor([local_correct], device=device, dtype=torch.long)
		total_tensor = torch.tensor([local_total], device=device, dtype=torch.long)
		loops_sum_tensor = torch.tensor([local_avg_loops * local_total], device=device, dtype=torch.float32)
		dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
		dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
		dist.all_reduce(loops_sum_tensor, op=dist.ReduceOp.SUM)
		global_correct = int(correct_tensor.item())
		global_total = int(total_tensor.item())
		accuracy = global_correct / global_total if global_total > 0 else 0.0
		avg_loops_global = float(loops_sum_tensor.item()) / global_total if global_total > 0 else 0.0
	else:
		global_correct = local_correct
		global_total = local_total
		accuracy = local_accuracy
		avg_loops_global = local_avg_loops

	if not distributed or rank == 0:
		result_path_display = (
			args.output_file
			if not distributed
			else f"{args.output_file}_rank*"
		)
		model_identifier = checkpoint_path if checkpoint_path else args.base_model_path
		print(f"Model evaluated: {model_identifier}")
		print(f"Results saved to: {result_path_display}")
		print(f"Accuracy: {accuracy * 100:.2f}% ({global_correct}/{global_total})")
		print(f"Avg block decode iterations per sample: {avg_loops_global:.3f}")

		# Save accuracy metrics to file
		if args.accuracy_file:
			accuracy_data = {
				"model": model_identifier,
				"checkpoint_path": checkpoint_path if checkpoint_path else "base_model",
				"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
				"accuracy": accuracy,
				"accuracy_percent": f"{accuracy * 100:.2f}%",
				"correct": global_correct,
				"total": global_total,
				"avg_block_decode_iterations_per_sample": avg_loops_global,
				"hyperparameters": {
					"max_gen_length": args.max_gen_length,
					"block_length": args.block_length,
					"threshold": args.threshold,
					"batch_size": args.batch_size,
					"num_few_shot": args.num_few_shot,
					"dtype": args.dtype,
					"subsample": args.subsample if args.subsample > 0 else "full",
				}
			}
			accuracy_dir = os.path.dirname(args.accuracy_file)
			if accuracy_dir:
				os.makedirs(accuracy_dir, exist_ok=True)
			with open(args.accuracy_file, "w", encoding="utf-8") as f:
				json.dump(accuracy_data, f, ensure_ascii=False, indent=2)
			print(f"Accuracy metrics saved to: {args.accuracy_file}")

	if distributed:
		dist.destroy_process_group()


if __name__ == "__main__":
	main()


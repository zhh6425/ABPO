import argparse
import os
import datasets

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/vepfs-cnbj3fa964354bf4/zhuoxuh/DATA/vision-r1")

    args = parser.parse_args()

    data_source = "vision-r1"

    train_dataset = datasets.load_dataset('parquet', data_files='/vepfs-cnbj3fa964354bf4/zhuoxuh/huggingface/datasets/Vision-R1-rl/train.parquet')['train']
    test_dataset = datasets.load_dataset('parquet', data_files='/vepfs-cnbj3fa964354bf4/zhuoxuh/huggingface/datasets/Vision-R1-rl/test.parquet')['train']

    instruction_following = (
        r"You FIRST think about the reasoning process as an internal monologue and then provide the final answer. "
        r"The reasoning process MUST BE enclosed within <think> </think> tags. "
        r"The final answer MUST BE put in \boxed{}."
    )

    # add a row to each data item that represents a unique id
    def make_map_fn(split):
        def process_fn(example, idx):
            problem = example.pop("problem")
            prompt = problem + " " + instruction_following
            answer = example.pop("answer")
            images = example.pop("images", [])

            data = {
                "data_source": data_source,
                "prompt": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                "images": [img for img in images],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "split": split,
                    "index": idx,
                    "answer": answer,
                    "question": problem,
                },
                "solution": ""
            }
            return data

        return process_fn

    train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True, num_proc=8)
    test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True, num_proc=8)

    local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
    test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))


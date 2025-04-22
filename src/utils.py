
import re
from datasets import Dataset
def create_masked_sentence(text_chunk, keyword):
    """
    Create a sentence with a masked word.

    Args:
        text_chunk: The original text chunk
        keyword: The keyword to mask
        position: The position of the keyword

    Returns:
        Sentence with the keyword masked
    """
    # words = text_chunk.split()
    # print(f"words:\n{words}")
    # print(f"keyword:\n{keyword}")


    # Mask the matching keyword in the text chunk
    masked_sentence = re.sub(r'\b' + re.escape(keyword) + r'\b', '[MASK]', text_chunk, flags=re.IGNORECASE)
    return masked_sentence

def prepare_finetuning_dataset(chunks_for_fintuning):
    """
    Prepare a dataset for finetuning from the evaluation results,
    focusing on incorrectly predicted masked words.

    Args:
        chunks_for_finetuning: List of dictionaries containing evaluation results for each chunk
        [
            {
                "chunk_id": int,
                "text": str,
                "needs_finetuning": bool,
                "accuracy": float,
                "evaluation_results": [{
                        "keyword": str,
                        "masked_sentence": str,
                        "prediction": str,
                        "is_correct": bool
                    }]
            }
        ]

    Returns:
        finetuning_dataset: Dataset object ready for finetuning
    """
    # Format the examples for masked word prediction
    formatted_examples = []

    for chunk_detail in chunks_for_fintuning:
        # print(chunk_detail)
        if not chunk_detail["needs_finetuning"]:
            continue  # Skip chunks that don't need finetuning

        # Extract incorrectly predicted keywords
        for result in chunk_detail["evaluation_results"]:
            if not result["is_correct"]:
                example = {
                    "instruction": "Predict the masked word in the sentence",
                    "input": result["masked_sentence"],
                    "output": f"The masked word is {result['keyword']}"
                }
                formatted_examples.append(example)

    # Convert to HF Dataset
    dataset_dict = {
        "instruction": [ex["instruction"] for ex in formatted_examples],
        "input": [ex["input"] for ex in formatted_examples],
        "output": [ex["output"] for ex in formatted_examples]
    }

    finetuning_dataset = Dataset.from_dict(dataset_dict)
    return finetuning_dataset

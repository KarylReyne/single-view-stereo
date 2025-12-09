import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def setup_models(config):
    model = AutoModelForCausalLM.from_pretrained(
        config["model_path"],  
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        attn_implementation="sdpa",
        torch_dtype=torch.float16
    )
    model = model.to(config["device"])
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(config["model_path"])
    model.resize_token_embeddings(len(tokenizer))
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer

def interpret_prompt(prompt, config):
    model, tokenizer = setup_models(config)

    instruction = lambda p: f"""
        You are given a query that specifies a baseline distance B as a floating-point number. Please return B in the following format:
        <begin_baseline>B<end_baseline>
        Remember that your answer has to be in the exact format specified above. Don't output anything else.
        Here is the query:
        <begin_query>{p}<end_query>
    """
    apply_template = lambda p: tokenizer.apply_chat_template([
            {"role": "system", "content": "You are a helpful assistant"}, # from ReasonIR p.19 fig.9
            {"role": "user", "content": instruction(p)}
        ],
        tokenize=False,
        add_generation_prompt=True
    )

    prompt_embeddings = tokenizer(
        [apply_template(prompt)], 
        return_tensors="pt", 
        padding=True, 
        padding_side="left"
    ).to(config["device"])
    generated_encoded_tokens = model.generate(
        **prompt_embeddings,
        max_new_tokens=128,
        temperature=config["temperature"]
    )
    generated_response = tokenizer.batch_decode(
        generated_encoded_tokens, 
        skip_special_tokens=True
    )[0]

    B = None
    try:
        B = float(generated_response.split("<begin_baseline>")[-1].split("<end_baseline>")[0])
    except Exception as e:
        raise e
    assert B != None

    # TODO: maybe add batched self-consistency, just for fun?
    
    return B
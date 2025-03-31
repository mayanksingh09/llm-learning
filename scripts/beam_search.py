from transformers import AutoTokenizer, AutoModelForCausalLM

prefixes = ["The dog jumped over",
            "There's a snake in"]
model_name = "gpt2"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

def beam_search(input_text, beam_width=3, max_length=20):
    input_ids = tokenizer.encode(input_text, return_tensors='pt')
    beam_output = model.generate(
        input_ids,
        num_beams=beam_width,
        max_length=max_length,
        early_stopping=True
    )
    return tokenizer.decode(beam_output[0], skip_special_tokens=True)

if __name__ == "__main__":
    for prefix in prefixes:
        print(f"Prefix: {prefix}")
        generated_text = beam_search(prefix)
        print(f"Generated Text: {generated_text}\n")
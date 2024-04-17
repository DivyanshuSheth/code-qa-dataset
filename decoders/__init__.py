def build_decoder(decoder_name: str, decoder_backend: str, **decoder_kwargs):
    match decoder_backend:
        case 'vllm':
            from decoders.vllm_model import VLLMModel as ModelCls
        case 'openai':
            from decoders.openai_model import OpenAIModel as ModelCls
        case 'hf':
            from decoders.hf_model import HFModel as ModelCls
        case _:
            raise Exception(f"Unknown model backend: {decoder_backend}")
    if decoder_name == "bigcode/gpt_bigcode-santacoder":
        return ModelCls.from_pretrained(decoder_name, max_model_len=2048, **decoder_kwargs)    
    return ModelCls.from_pretrained(decoder_name, **decoder_kwargs)
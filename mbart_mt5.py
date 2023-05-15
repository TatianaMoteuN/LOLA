import torch

#pip3 install transformers

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartTokenizer, MBartModel, T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration


def translate_with_mbart_and_mt5(input_text, target_language):

  #load mBART model
    mbart_model_name = "facebook/mbart-large-cc25"
    mbart_tokenizer = MBartTokenizer.from_pretrained(mbart_model_name)
    mbart_encoder = MBartForConditionalGeneration.from_pretrained(mbart_model_name)
    
    #Load mT5 decoder
    mt5_model_name = "google/mt5-large"
    mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_model_name)
    mt5_decoder = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)

    #we freeze the decoder parameters
    #for param in mbart_model_name.model.decoder.parameters():
    # param.requires_grad = False

    # Set both models to evaluation mode
    mbart_encoder.eval()
    mt5_decoder.eval()

    # Tokenize the input text
    input_ids = mbart_tokenizer.encode(input_text, return_tensors="pt")

    # Encode the input text using mBART encoder
    encoded_input = mbart_encoder(input_ids=input_ids)
    # Generate translations using mT5 decoder
    translated_ids = mt5_decoder.generate(input_ids=encoded_input["last_hidden_state"],
                                          attention_mask=encoded_input["attention_mask"],
                                          decoder_start_token_id=mt5_tokenizer.pad_token_id,
                                          forced_bos_token_id=mt5_tokenizer.pad_token_id,
                                          decoder_attention_mask=torch.tensor([[1] * input_ids.shape[1]]))

    # Decode the translated IDs back into text
    translated_text = mt5_tokenizer.decode(translated_ids[0], skip_special_tokens=True)

    return translated_text


# Example usage
input_text = "Hello, how are you?"
target_language = "fr"  # Target language is French

translation = translate_with_mbart_and_mt5(input_text, target_language)
print(f"Translated text: {translation}")
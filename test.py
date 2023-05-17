import torch

#pip3 install transformers

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartTokenizer, MBartModel, T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration


article = "UN Chief Says There Is No Military Solution in Syria"


def get_translation_model_and_tokenizer(src_lang, dst_lang):
  """
  Given the source and destination languages, returns the appropriate model
  See the language codes here: https://developers.google.com/admin-sdk/directory/v1/languages
  For the 3-character language codes, you can google for the code!
  """
  #load mBART model
  mbart_model_name = "facebook/mbart-large-cc25"
  # initialize the tokenizer & model
  mbart_tokenizer = MBartTokenizer.from_pretrained(mbart_model_name)
  model = MBartForConditionalGeneration.from_pretrained(mbart_model_name)

  #Load mT5 decoder
  mt5_model_name = "google/mt5-large"
  mt5_tokenizer = T5Tokenizer.from_pretrained(mt5_model_name)
  model2 = MT5ForConditionalGeneration.from_pretrained(mt5_model_name)

  # return them for use
  return model, mbart_tokenizer, model2, mt5_tokenizer


# source & destination languages
src = "en"
dst = "zh"

model, mbart_tokenizer, model2, mt5_tokenizer = get_translation_model_and_tokenizer(src, dst)

# encode the text into tensor of integers using the appropriate tokenizer
inputs = mbart_tokenizer.encode(article, return_tensors="pt", max_new_tokens=512, truncation=True)
print(inputs)

# generate the translation output using beam search
beam_outputs = model2.generate(inputs, num_beams=3)
# decode the output and ignore special tokens
print(mt5_tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
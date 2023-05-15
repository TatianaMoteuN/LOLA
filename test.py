import torch

#pip3 install transformers

from transformers import MBartForConditionalGeneration, MBartTokenizer, MBart50TokenizerFast
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers import MBartTokenizer, MBartModel, T5Tokenizer, T5ForConditionalGeneration, MT5ForConditionalGeneration


article = """
Albert Einstein ( 14 March 1879 – 18 April 1955) was a German-born theoretical physicist, widely acknowledged to be one of the greatest physicists of all time. 
Einstein is best known for developing the theory of relativity, but he also made important contributions to the development of the theory of quantum mechanics. 
Relativity and quantum mechanics are together the two pillars of modern physics. 
His mass–energy equivalence formula E = mc2, which arises from relativity theory, has been dubbed "the world's most famous equation". 
His work is also known for its influence on the philosophy of science.
He received the 1921 Nobel Prize in Physics "for his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect", a pivotal step in the development of quantum theory. 
His intellectual achievements and originality resulted in "Einstein" becoming synonymous with "genius"
"""


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
inputs = mbart_tokenizer.encode(article, return_tensors="pt", max_length=512, truncation=True)
print(inputs)

# generate the translation output using beam search
beam_outputs = model2.generate(inputs, num_beams=3)
# decode the output and ignore special tokens
print(mt5_tokenizer.decode(beam_outputs[0], skip_special_tokens=True))
import torch
from transformers import BertConfig, PreTrainedModel, TFPreTrainedModel, TFBertForMaskedLM, \
    BertForMaskedLM, BertTokenizer
import logging


#config = BertConfig.from_json_file("config.json")
#model = BertForPreTraining.from_pretrained('model.ckpt-100000', from_tf=True, config=config)

logging.basicConfig(level=logging.INFO)

tokenizer = BertTokenizer.from_pretrained('bert_pytorch_model/vocab.txt')
'10385'
text = '[CLS] 24852 46667 41570 41787 27521 [MASK] 18656 [SEP]'
tokenized_text = tokenizer.tokenize(text)
indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)
masked_index = tokenized_text.index('[MASK]')

# Create the segments tensors.
segments_ids = [0] * len(tokenized_text)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

model = BertForMaskedLM.from_pretrained('bert_pytorch_model')
model.eval()

# Predict all tokens
with torch.no_grad():
    predictions = model(tokens_tensor, segments_tensors)

sorted_predictions = torch.sort(predictions[0][0][masked_index], descending=True)
num_suggested = 5
pred_tokens = []
for i in range(num_suggested):
    # predicted_index = torch.argmax(predictions[0][0][masked_index]).item()
    predicted_index = sorted_predictions[1][i].item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
    pred_tokens.append(predicted_token)

print(pred_tokens)




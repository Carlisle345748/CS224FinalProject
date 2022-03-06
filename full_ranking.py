import os

import torch
from transformers import AutoTokenizer, AutoModel
from datasets.search import FaissIndex

from cocondenser import CondenserLTR

if __name__ == '__main__':
	WEIGHTS_NAME = "pytorch_model.bin"
	model_dir = "/Users/carlisle/CS224N_Project/result/ABS-BM25-Condenser-100000-1000"
	device = 'cuda' if torch.cuda.is_available() else 'cpu'

	tokenizer = AutoTokenizer.from_pretrained("Luyu/co-condenser-marco")
	state_dict = torch.load(os.path.join(model_dir, WEIGHTS_NAME), map_location="cpu")
	q_enc = AutoModel.from_pretrained("Luyu/co-condenser-marco")
	p_enc = AutoModel.from_pretrained("Luyu/co-condenser-marco")
	model = CondenserLTR(q_enc=q_enc, p_enc=p_enc, psg_per_qry=8)
	model.load_state_dict(state_dict)
	model.to(device)

	query = ["123123", "321321312", "1231231"]
	passage = ["123123", "321321312", "1231231"]
	query = tokenizer(
		query,
		truncation=True,
		max_length=32,
		padding=True,
		return_tensors="pt",
	).to(device)

	passage = tokenizer(
		passage,
		truncation=True,
		max_length=128,
		padding=True,
		return_tensors="pt"
	).to(device)

	q_reps = model.encode_query(query)
	p_reps = model.encode_passage(passage)

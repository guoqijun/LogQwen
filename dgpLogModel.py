import peft
import torch
from transformers import BertTokenizerFast, BertModel, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from torch import nn


def stack_and_pad_left(tensors):
    # 找到第一维度的最大长度
    max_len = max(tensor.shape[0] for tensor in tensors)

    # 创建一个存放结果的列表
    padded_tensors = []
    padding_masks = []

    for tensor in tensors:
        # 计算需要填充的长度
        pad_len = max_len - tensor.shape[0]

        # 使用零填充
        padded_tensor = torch.nn.functional.pad(tensor, (0, 0, pad_len, 0))
        padded_tensors.append(padded_tensor)

        # 创建填充位置的掩码
        padding_mask = torch.cat([torch.zeros(pad_len, dtype=torch.long),
                                  torch.ones(tensor.shape[0], dtype=torch.long)])
        padding_masks.append(padding_mask)

    # 堆叠所有填充后的张量
    stacked_tensor = torch.stack(padded_tensors)
    padding_masks = torch.stack(padding_masks)

    return stacked_tensor, padding_masks


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # load the model into memory using 4-bit precision
    bnb_4bit_use_double_quant=False,  # use double quantition
    bnb_4bit_quant_type="nf4",  # use NormalFloat quantition
    bnb_4bit_compute_dtype=torch.bfloat16  # use hf for computing when we need
)


class DgpLogModel(nn.Module):
    def __init__(self, Bert_path, Llama_path, device=torch.device("cuda:0"), max_content_len=128):
        super().__init__()
        self.max_content_len = max_content_len  # max length of each log messages (contents)
        self.device = device
        self.Llama_tokenizer = AutoTokenizer.from_pretrained(Llama_path, padding_side="right")
        self.Llama_tokenizer.pad_token = self.Llama_tokenizer.eos_token
        self.Llama_model = AutoModelForCausalLM.from_pretrained(Llama_path, quantization_config=bnb_config,
                                                                low_cpu_mem_usage=True,
                                                                device_map=device)  # embedding dim = 4096

        self.Bert_tokenizer = BertTokenizerFast.from_pretrained(Bert_path, do_lower_case=True)
        self.Bert_model = BertModel.from_pretrained(Bert_path, quantization_config=bnb_config, low_cpu_mem_usage=True,
                                                    device_map=device)

        self.projector = nn.Linear(self.Bert_model.config.hidden_size, self.Llama_model.config.hidden_size,
                                   device=device)

        self.instruc_tokens = self.Llama_tokenizer(
            ['Below is a sequence of system log messages:', '. Is this sequence normal or anomalous? \\n'],
            return_tensors="pt", padding=True).to(self.device)

    def forward(self, sequences):
        '''
        :param sequences: list of list: [seq, seq, ...,seq]  , seq:[item, ..., item]
        :return: Generated answer (token id).
        '''
        batch_size = len(sequences)
        data = sequences

        inputs = self.Bert_tokenizer(data, return_tensors="pt", max_length=self.max_content_len, padding=True,
                                     truncation=True).to(self.device)

        outputs = self.Bert_model(**inputs).pooler_output  # dim = 768
        outputs = outputs.float()
        outputs = self.projector(outputs)
        outputs = outputs.half()

        # seq_embeddings = torch.tensor_split(outputs, seq_positions)

        prefix = "The sequence is"
        answer_prefix_tokens = self.Llama_tokenizer(prefix, padding=True, return_tensors="pt")['input_ids'][0, 1:].to(
            self.device)

        if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
            instruc_embeddings = self.Llama_model.model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.model.embed_tokens(answer_prefix_tokens)
        else:
            instruc_embeddings = self.Llama_model.model.embed_tokens(self.instruc_tokens['input_ids'])
            answer_prefix_tokens_embeddings = self.Llama_model.model.embed_tokens(answer_prefix_tokens)

        ins1 = instruc_embeddings[0][self.instruc_tokens['attention_mask'][0].bool()]
        ins2 = instruc_embeddings[1][self.instruc_tokens['attention_mask'][1].bool()][1:]

        promot_embeddings = []
        promot_embeddings = torch.cat([ins1, outputs, ins2, answer_prefix_tokens_embeddings])

        inputs_embeds, attention_mask = stack_and_pad_left(promot_embeddings)
        attention_mask = attention_mask.to(self.device)

        pad_token_id = self.Llama_tokenizer.pad_token_id
        eos_token_id = self.Llama_tokenizer.eos_token_id
        if isinstance(eos_token_id, int):
            eos_token_id = [eos_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(self.device) if eos_token_id is not None else None

        unfinished_sequences = torch.ones(batch_size, dtype=torch.long, device=self.device)

        this_peer_finished = False
        answer = []
        while not this_peer_finished:
            Llama_output = self.Llama_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask).logits
            next_token_logits = Llama_output[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)

            # print(next_tokens)
            answer.append(next_tokens)

            if type(self.Llama_model) == peft.peft_model.PeftModelForCausalLM:
                next_tokens_embeddings = self.Llama_model.model.model.embed_tokens(next_tokens)
            else:
                next_tokens_embeddings = self.Llama_model.model.embed_tokens(next_tokens)

            inputs_embeds = torch.cat([inputs_embeds, next_tokens_embeddings[:, None, :]], dim=1)
            attention_mask = torch.cat([attention_mask, unfinished_sequences[:, None]], dim=1)

            if eos_token_id_tensor is not None:
                unfinished_sequences = unfinished_sequences.mul(
                    next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
                )

                # stop when each sentence is finished
                if unfinished_sequences.max() == 0:
                    this_peer_finished = True

                # stop if we exceed the maximum answer length
            if 5 < len(answer):
                this_peer_finished = True

        return torch.stack(answer, dim=1)

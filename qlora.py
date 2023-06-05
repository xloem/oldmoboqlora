# -*- coding: utf-8 -*-
#!pip install -q -U bitsandbytes
#!pip install -q -U git+https://github.com/huggingface/transformers.git 
#!pip install -q -U git+https://github.com/huggingface/peft.git
#!pip install -q -U git+https://github.com/huggingface/accelerate.git
#!pip install -q datasets

#from huggingface_hub import notebook_login
#notebook_login()

"""First let's load the model we are going to use -"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftConfig

#model_id = 'camelids/llama-7b-fp16-safetensors'#"EleutherAI/gpt-neox-20b"
peft_model_id = 'baffo32/llama7bqlora-asciiart'
ONLY_ODD_P2P = True
#MAIN_DEVICE = 3
config = PeftConfig.from_pretrained(peft_model_id)
config.inference_mode = False
#config = LoraConfig(
#    r=8, 
#    lora_alpha=32, 
#    #target_modules=["query_key_value"], 
#    lora_dropout=0.05, 
#    bias="none", 
#    task_type="CAUSAL_LM"
#)
model_id = config.base_model_name_or_path
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map={"":0} # training args device getter hardcoded to assume data parallel starts on device 0
)

"""Then we have to apply some preprocessing to the model to prepare it for training. For that use the `prepare_model_for_kbit_training` method from PEFT."""

from peft import prepare_model_for_kbit_training

model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel

model = PeftModel.from_pretrained(model, peft_model_id, is_trainable=True)
#model = get_peft_model(model, config)
print_trainable_parameters(model)

"""Let's load a dataset, to fine tune our model."""

#from datasets import load_dataset

#data = load_dataset("Abirate/english_quotes")

from datasets import Dataset
import requests
def generator():
  lines = []
  for line in requests.get(
      'https://archive.org/download/usenet-alt/alt.ascii-art.mbox.zip/alt.ascii-art.mbox',
      stream=True,
  ).iter_lines():
      if (
          line.startswith(b'From ') and
          len(line) > 8 and 
          line[6:].isdigit() and
          len(lines) and 
          not lines[-1].strip()
      ):
          mail = b'\n'.join(lines)
          yield dict(text=mail.decode(),len=len(mail))
          lines = []
      lines.append(line)
  if lines:
    mail = b'\n'.join(lines)
    yield dict(text=mail.decode(),len=len(mail))
#next(generator()) # catches typos with stack trace
data = Dataset.from_generator(generator)
def crop(text, **_):
    head, body = text.split('\n\n',1)
    head = head.split('\n')
    head = filter(
        lambda line:
            line.startswith('From: ') or
            line.startswith('Subject: '),
        head
    )
    mail = '\n'.join(head) + '\n\n' + body.strip()
    return dict(text=mail, len=len(mail))
data = data.map(lambda row: crop(**row), desc='Cropping')
data = data.filter(
    lambda row:
      (lambda body:
        body.count('\n') > 1 and
        len([
            c for c in body
            if c.isalpha()
        ]) < len(body) // 3
      )(row['text'].split('\n\n',1)[1]),
    desc='Filtering',
)
tokenizer = AutoTokenizer.from_pretrained(model_id) # ensure initial state hash
data = data.map(
    lambda rows: tokenizer(
        rows['text'],
        max_length=tokenizer.model_max_length,
        truncation=True,
    ),
    batched=True,
    desc='Tokenizing',
)
#data = data.sort('len')
import time
data = data.shuffle(seed=int(time.time()))
print(data[1024]['text'])

"""This is an addition to handle my old system. It prevents using P2P on root hubs when filled with K80s."""
if ONLY_ODD_P2P:
    def safe_to(cls, src_idx_getter = lambda obj: obj.device.index, target_idx_getter = lambda x: x.index):
        to = cls.to
        def safe_to(obj, device):
            try:
                src_idx = src_idx_getter(obj)
            except StopIteration:
                src_idx = None
            if src_idx is None:
                return to(obj, device)
            target_idx = target_idx_getter(device)
            if src_idx != -1 and target_idx != -1 and src_idx & ~1 != target_idx & ~1:
                # if the smallest index bit is not the same, the data leaves the K80
                return to(to(obj, -1), device)
            else:
                # otherwise the on-K80 P2P can be used
                return to(obj, device)
        return safe_to
    torch.nn.Module.to = safe_to(torch.nn.Module, lambda module: next(module.parameters()).device.index)
    safe_tensor_to = safe_to(torch.Tensor)
    def safe_parallel(comm_parallel, params2valid, params2srcdevs, src2idx, src2dev, srcdevsparams2safe):
        def safe_parallel(*params, **kwparams):
            src, devices = params2srcdevs(*params, **kwparams)
            src_idx = src2idx(src)
            if src_idx != -1:
                src_group = src_idx >> 1
                num_groups = (max(devices) >> 1) + 1
                target_idx_by_group = [[] for g in range(num_groups)]
                out_by_group = [None] * num_groups
                out_group_idx_by_target_idx = []
                for target_idx in devices:
                    group_idx = target_idx >> 1
                    out_group_idx_by_target_idx.append((group_idx, len(target_idx_by_group[group_idx])))
                    target_idx_by_group[group_idx].append(target_idx)
                if any([len(target_idx_by_group[group]) > 0 for group in range(num_groups) if group != src_group]):
                    assert params2valid(*params, **kwparams)
                    for group in range(num_groups):
                        if group == src_group:
                            the_src = src
                        else:
                            the_src = src2dev(src, group << 1)
                        safe_params, safe_kwparams = srcdevsparams2safe(
                            the_src,
                            target_idx_by_group[group],
                            *params,
                            **kwparams)
                        out_by_group[group] = comm_parallel(*safe_params, **safe_kwparams)
                    return tuple([out_by_group[group][idx] for group, idx in out_group_idx_by_target_idx])
            return comm_parallel(*params, **kwparams)
        setattr(torch.nn.parallel.comm, comm_parallel.__name__, safe_parallel)
        return safe_parallel
    safe_parallel(
        torch.nn.parallel.comm.scatter,
        params2valid = lambda tensor, devices, chunk_sizes=None, dim=0, streams=None, *, out=None: chunk_sizes is None and dim == 0 and out is None and tensor.shape[dim] == len(devices),
        params2srcdevs = lambda tensor, devices, *_, **__: (tensor, devices),
        src2idx = lambda tensor: tensor.device.index,
        src2dev = lambda tensor, dev: tensor.cpu().to(dev),
        srcdevsparams2safe =
            lambda tensor, devices, tensor_, devices_, *params, **kwparams:
                ((tensor[devices], devices, *params), kwparams)
    )
    safe_parallel(
        torch.nn.parallel.comm.broadcast_coalesced,
        params2valid = lambda *_, **__: True,
        params2srcdevs = lambda tensors, devices, *_, **__: ([tensor for tensor in tensors], devices),
        src2idx = lambda tensors: tensors[0].device.index,
        src2dev = lambda tensors, dev: [tensor.cpu().to(dev) for tensor in tensors],
        srcdevsparams2safe =
            lambda tensors, devices, tensors_, devices_, buffer_size=10485760:
                ((tensors, devices, buffer_size),{})
    )
    #torch_nn_parallel_comm_broadcast_coalesced = torch.nn.parallel.broadcast_coalesced
    #def safe_broadcast_coalesced(tensors, devices, buffer_size=10485760):
    #    tensors = [tensor for tensor in tensors]
    #    src_idx = tensors[0].device.index
    #    if src_idx != -1:
    #        src_idx &= ~1
    #        DIRECT, INDIRECT = 0, 1
    #        target_idx_by_direct = [[],[]]
    #        out_by_direct = [None, None]
    #        out_direct_idx_by_target_idx = []
    #        for target_idx in devices:
    #            direct_idx = INDIRECT if src_idx != target_idx & ~1 else DIRECT
    #            out_direct_idx_by_target_idx.append((direct_idx, len(target_idx_by_direct[direct_idx])))
    #            target_idx_by_direct[direct_idx].append(target_idx)
    #        if len(target_idx_by_direct[INDIRECT]):
    #            # this is the only bit that is different
    #            out_by_direct[DIRECT] = torch_nn_parallel_comm_broadcast_coalesced(tensors, target_idx_by_direct[DIRECT], buffer_size)
    #            out_by_direct[INDIRECT] = torch_nn_parallel_comm_broadcast_coalesced([tensor.cpu() for tensor in tensors], target_idx_by_direct[INDIRECT], buffer_size)
    #            return tuple([out_by_direct[direct][idx] for direct, idx in out_direct_idx_by_target_idx])
    #    return torch_nn_parallel_comm_broadcast_coalesced(tensor, devices, chunk_sizes, dim, streams, out=out)
    #torch_nn_parallel_comm_scatter = torch.nn.parallel.comm.scatter
    #def safe_scatter(tensor, devices, chunk_sizes=None, dim=0, streams=None, *, out=None):
    #    src_idx = tensor.device.index
    #    if src_idx != -1:
    #        src_idx &= ~1
    #        DIRECT, INDIRECT = 0, 1
    #        target_idx_by_direct = [[],[]]
    #        out_by_direct = [None, None]
    #        out_direct_idx_by_target_idx = []
    #        for target_idx in devices:
    #            direct_idx = INDIRECT if src_idx != target_idx & ~1 else DIRECT
    #            out_direct_idx_by_target_idx.append((direct_idx, len(target_idx_by_direct[direct_idx])))
    #            target_idx_by_direct[direct_idx].append(target_idx)
    #        if len(target_idx_by_direct[INDIRECT]):
    #            assert chunk_sizes is None and dim == 0 and out is None and tensor.shape[dim] == len(devices)
    #            out_by_direct[DIRECT] = torch_nn_parallel_comm_scatter(tensor[target_idx_by_direct[DIRECT]], target_idx_by_direct[DIRECT], chunk_sizes, dim, streams, out=out)
    #            out_by_direct[INDIRECT] = torch_nn_parallel_comm_scatter(tensor[target_idx_by_direct[INDIRECT]].cpu(), target_idx_by_direct[INDIRECT], chunk_sizes, dim, streams, out=out)
    #            return tuple([out_by_direct[direct][idx] for direct, idx in out_direct_idx_by_target_idx])
    #    return torch_nn_parallel_comm_scatter(tensor, devices, chunk_sizes, dim, streams, out=out)
    #torch.nn.parallel.comm.scatter = safe_scatter

"""Run the cell below to run the training! For the sake of the demo, we just ran it for few steps just to showcase how to use this integration with existing tools on the HF ecosystem."""

import transformers

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

trainer = transformers.Trainer(
    model=model,
    train_dataset=data,#["train"],
    args=transformers.TrainingArguments(
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,#4,
        warmup_steps=32,
        max_steps=256,
        learning_rate=2e-4,
        fp16=True,
        logging_steps=1,#4,
        output_dir="outputs",
        optim="paged_adamw_8bit",
        #device=MAIN_DEVICE,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)
model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
trainer.train()
model.config.use_cache = True

model.push_to_hub(peft_model_id)

#!nvidia-smi

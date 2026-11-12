# /// script
# dependencies = [
#   "nngeometry",
#   "datasets",
#   "transformers",
# ]
# [tool.uv.sources]
# nngeometry = { path = "..", editable = true }
# ///

import argparse

import torch
from datasets import load_dataset
from torch import nn
from torch.utils.data import DataLoader, default_collate
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.pytorch_utils import Conv1D

from nngeometry.layercollection import LayerCollection
from nngeometry.metrics import FIM, FIM_MonteCarlo
from nngeometry.object.pspace import PMatEKFACBlockDiag

parser = argparse.ArgumentParser(description="gpt2 benchmark on imdb")
parser.add_argument(
    "--model",
    default="gpt2",
    help="""Any model from huggingface.""",
)
parser.add_argument(
    "--ovr",
    action="store_true",
    help="""Use a binary classification task (One versus Rest)\
        instead of multi-class classification one with Monte Carlo.""",
)
parser.add_argument(
    "--device",
    default="cuda" if torch.cuda.is_available() else "cpu",
)
parser.add_argument(
    "--untie",
    action="store_true",
    help="""By default, weights between the embedding layer and the LLM head are tied.\
        Use this option to untie them.""",
)
parser.add_argument(
    "--layers",
    choices=["all", "head", "linear"],
    default="linear",
    help="""By default, all linear layers are used but the embeddings layers\
        and the lm_head layer. You can also use all layers or only the head.""",
)
args = parser.parse_args()

torch.manual_seed(0)

dataset = load_dataset("imdb")["train"]
dataset = dataset.select(range(100))

tokenizer = AutoTokenizer.from_pretrained(args.model)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


def tokenize(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        padding=False,
        max_length=512,
        return_attention_mask=True,
    )


dataset = dataset.map(tokenize, batched=True, remove_columns=["label"])
dataset = dataset.with_format("torch", columns=["input_ids", "attention_mask"])


def collate_fn(examples):
    batch = default_collate(examples)
    return batch["input_ids"], batch["attention_mask"]


loader = DataLoader(
    dataset,
    batch_size=1,
    num_workers=4,
    collate_fn=collate_fn,
)

model = AutoModelForCausalLM.from_pretrained(args.model)
model = model.to(args.device)
model.eval()


# for gpt2
def replace_oai_conv1d_with_nn_linear(module):
    for name, child in module.named_children():
        if isinstance(child, Conv1D):
            in_features, out_features = child.weight.shape

            linear = nn.Linear(
                in_features,
                out_features,
                bias=(child.bias is not None),
                device=child.weight.device,
                dtype=child.weight.dtype,
            )

            with torch.no_grad():
                linear.weight.copy_(child.weight.t())
                if child.bias is not None:
                    linear.bias.copy_(child.bias)

            linear.train(child.training)
            linear.weight.requires_grad_(child.weight.requires_grad)
            if child.bias is not None:
                linear.bias.requires_grad_(child.bias.requires_grad)

            setattr(module, name, linear)
        else:
            replace_oai_conv1d_with_nn_linear(child)


replace_oai_conv1d_with_nn_linear(model)

head = model.get_output_embeddings()
backbone = model.base_model

if args.untie:
    model.config.tie_word_embeddings = False
    head.weight = torch.nn.Parameter(head.weight.detach().clone())

input_embeddings = model.get_input_embeddings()

print(
    "embedding and head weights are tied : ",
    head.weight is input_embeddings.weight,
)

if args.layers == "all":
    layer_collection = LayerCollection.from_model(model, ignore_unsupported_layers=True)
elif args.layers == "linear":
    layer_collection = LayerCollection()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and module is not head:
            layer_collection.add_layer_from_model(model, module)
elif args.layers == "head":
    layer_collection = LayerCollection()
    layer_collection.add_layer_from_model(model, head)


def next_token_logits(input_ids, attention_mask):
    input_ids = input_ids.to(args.device)
    attention_mask = attention_mask.to(args.device)

    hidden = backbone(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        return_dict=True,
    ).last_hidden_state

    last_positions = attention_mask.sum(dim=1) - 1
    rows = torch.arange(input_ids.size(0), device=args.device)
    return head(hidden[rows, last_positions])


def next_token_logit_ovr(input_ids, attention_mask):
    logits = next_token_logits(input_ids, attention_mask)

    target = logits.detach().argmax(dim=1, keepdim=True)
    predicted = torch.gather(logits, 1, target)

    others = torch.scatter(logits, 1, target, float("-inf"))
    return predicted - torch.logsumexp(others, dim=1, keepdim=True)


def eigendecomposition(x):
    U, s, Vt = torch.linalg.svd(x.to(dtype=torch.float32))
    return s, U


if args.ovr:
    F = FIM(
        model,
        loader,
        representation=PMatEKFACBlockDiag,
        variant="classif_binary_logits",
        function=next_token_logit_ovr,
        layer_collection=layer_collection,
        verbose=True,
        eigendecomposition=eigendecomposition,
    )
else:
    F = FIM_MonteCarlo(
        model,
        loader,
        representation=PMatEKFACBlockDiag,
        variant="classif_logits",
        function=next_token_logits,
        layer_collection=layer_collection,
        verbose=True,
        eigendecomposition=eigendecomposition,
    )
F.update_diag(loader)

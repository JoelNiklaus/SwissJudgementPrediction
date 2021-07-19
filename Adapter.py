import math

import torch
import copy
from torch import nn
from transformers import RobertaForSequenceClassification


class Activation_Function_Class(nn.Module):
    """
    Implementation of various activation function.
    """

    def __init__(self, hidden_act):

        if hidden_act.lower() == "relu":
            self.f = nn.functional.relu
        elif hidden_act.lower() == "tanh":
            self.f = torch.tanh
        elif hidden_act.lower() == "swish":

            def swish(x):
                return x * torch.sigmoid(x)

            self.f = swish
        elif hidden_act.lower() == "gelu":

            def gelu_new(x):
                """
                Implementation of the gelu activation function currently in Google Bert repo (identical to OpenAI GPT).
                Also see https://arxiv.org/abs/1606.08415
                """
                return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

            self.f = gelu_new
        elif hidden_act.lower() == "leakyrelu":
            self.f = nn.functional.leaky_relu

        super().__init__()

    def forward(self, x):
        return self.f(x)


# Single Adapter

class Adapter(nn.Module):
    """
    Implementation of a single Adapter block.
    """

    def __init__(
        self,
        input_size,
        bottleneck_size=None,
        non_linearity="relu",
        init_weights=True,
    ):
        super().__init__()

        self.input_size = input_size

        # list for all modules of the adapter, passed into nn.Sequential()
        seq_list = []

        # if a downsample size is not passed, we just half the size of the original input
        self.bottleneck_size = bottleneck_size
        if bottleneck_size is None:
            self.bottleneck_size = self.input_size // 2

        # Linear down projection of the input
        seq_list.append(nn.Linear(self.input_size, self.bottleneck_size))

        # select non-linearity
        self.non_linearity = Activation_Function_Class(non_linearity.lower())

        seq_list.append(self.non_linearity)

        # sequential adapter, first downproject, then non-linearity then upsample. In the forward pass we include the
        # residual connection
        self.adapter_down = nn.Sequential(*seq_list)

        # Up projection to input size
        self.adapter_up = nn.Linear(self.bottleneck_size, self.input_size)

        # if we want to initialize with the bert strategy then this function is called for all the linear layers
        if init_weights:
            self.adapter_down.apply(self.init_weights)
            self.adapter_up.apply(self.init_weights)

    def forward(self, x):  # , residual_input=None):
        down = self.adapter_down(x)

        up = self.adapter_up(down)

        output = up

        # apply residual connection
        output = output + x

        return output

    @staticmethod
    def init_weights(module):
        """Initialize the weights."""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # std defaults to 0.02, this might need to be changed
            module.weight.data.normal_(mean=0.0, std=1e-3)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class RobertaOutput(nn.Module):
    def __init__(self, pretrained_self_dense: nn.Linear,
                 pretrained_self_ln: nn.LayerNorm,
                 config,
                 bottleneck_size: int):
        super().__init__()
        self.dense = copy.deepcopy(pretrained_self_dense)
        self.LayerNorm = copy.deepcopy(pretrained_self_ln)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter = Adapter(config.hidden_size, non_linearity=config.hidden_act,
                               bottleneck_size=bottleneck_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class RobertaSelfOutput(nn.Module):
    def __init__(self, pretrained_self_dense: nn.Linear,
                 pretrained_self_ln: nn.LayerNorm,
                 config,
                 bottleneck_size: int):
        super().__init__()
        self.dense = copy.deepcopy(pretrained_self_dense)
        self.LayerNorm = copy.deepcopy(pretrained_self_ln)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.adapter = Adapter(config.hidden_size, non_linearity=config.hidden_act,
                               bottleneck_size=bottleneck_size)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.adapter(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


def add_adapters(model: RobertaForSequenceClassification, bottleneck_size=256):
    # Add Adapters
    for i in range(model.roberta.config.num_hidden_layers):
        model.roberta.encoder.layer[i].attention.output = RobertaSelfOutput(
            model.roberta.encoder.layer[i].attention.output.dense,
            model.roberta.encoder.layer[i].attention.output.LayerNorm,
            model.roberta.config,
            bottleneck_size)
        model.roberta.encoder.layer[i].attention.output = \
            model.roberta.encoder.layer[i].attention.output.to(model.device)
        model.roberta.encoder.layer[i].output = RobertaOutput(
            model.roberta.encoder.layer[i].output.dense,
            model.roberta.encoder.layer[i].output.LayerNorm,
            model.roberta.config,
            bottleneck_size)
        model.roberta.encoder.layer[i].output = \
            model.roberta.encoder.layer[i].output.to(model.device)

    return model


if __name__ == '__main__':
    from transformers import AutoModelForSequenceClassification
    model = AutoModelForSequenceClassification.from_pretrained('xlm-roberta-base')
    model = add_adapters(model)

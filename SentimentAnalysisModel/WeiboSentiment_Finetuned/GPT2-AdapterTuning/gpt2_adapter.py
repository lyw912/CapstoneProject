import torch
import torch.nn as nn
from transformers.models.gpt2.modeling_gpt2 import GPT2Block
from adapter import AdapterLayer

class GPT2BlockWithAdapter(nn.Module):
    """
    GPT2Block layer with Adapter
    Adds Adapter layer to original GPT2Block for parameter-efficient fine-tuning
    """
    def __init__(self, config):
        super(GPT2BlockWithAdapter, self).__init__()
        # Create standard GPT2Block
        self.original_block = GPT2Block(config)

        # Add Adapter layer
        adapter_size = 64  # Adapter hidden layer size
        self.adapter = AdapterLayer(config.hidden_size, adapter_size)
    
    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_cache=False,
        output_attentions=False,
        **kwargs  # Use **kwargs to receive all other parameters
    ):
        # First pass through original GPT2Block, only passing supported parameters
        outputs = self.original_block(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )

        # First element of original output is hidden state
        hidden_states = outputs[0]

        # Pass hidden state through Adapter layer
        hidden_states = self.adapter(hidden_states)

        # Update output hidden state
        outputs = (hidden_states,) + outputs[1:]

        return outputs

    def load_state_dict(self, state_dict, strict=True):
        """
        Custom parameter loading method for loading from original GPT2Block
        """
        # Pass all parameters to original block
        return self.original_block.load_state_dict(state_dict, strict=strict) 
from transformers.models.gpt2.modeling_gpt2 import GPT2Block

class GPT2BlockWithAdapter(GPT2Block):
    def __init__(self, config):
        super().__init__(config)
        # Assume Adapter size is 64
        adapter_size = 64
        self.adapter = AdapterLayer(config.n_embd, adapter_size)

    def forward(
        self,
        hidden_states,
        layer_past=None,
        attention_mask=None,
        head_mask=None,
        use_cache=False,
            output_attentions=False,
        ):
        # Call original forward propagation method
        attn_outputs = super().forward(
            hidden_states,
            layer_past=layer_past,
            attention_mask=attention_mask,
            head_mask=head_mask,
            use_cache=use_cache,
            output_attentions=output_attentions,
        )
        # Get Transformer layer output
        a = attn_outputs[0]  # First output is attention result
        # Pass output through Adapter layer
        a = self.adapter(a)
        # Return modified output (other outputs remain unchanged)
        outputs = (a,) + attn_outputs[1:]
        return outputs
"""
Each GPT2Block contains a series of Self-Attention and Feed-Forward layers,
which together form the foundation of the model architecture.

"""



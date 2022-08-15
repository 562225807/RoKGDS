import torch.nn as nn
import torch
import math
from transformers.modeling_bert import BertPreTrainedModel, BertSelfOutput, \
    BertIntermediate, BertPooler, BertOutput, \
    add_start_docstrings, add_start_docstrings_to_callable, prune_linear_layer


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.output_attentions = config.output_attentions

        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.know_query = nn.Linear(config.hidden_size, self.all_head_size)
        self.know_key = nn.Linear(config.hidden_size, self.all_head_size)
        self.know_value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        self.know_dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_save=False,
        exist_length=None,
        query_state=None,
    ):
        if query_state is not None:
            mixed_query_layer = self.query(query_state)
            mixed_know_query_layer = self.know_query(query_state)
        else:
            mixed_query_layer = self.query(hidden_states)
            mixed_know_query_layer = self.know_query(hidden_states)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if use_save:
            if isinstance(encoder_hidden_states, tuple):
                mixed_key_layer = hidden_states[0]
                mixed_value_layer = hidden_states[1]

                mixed_know_key_layer = encoder_hidden_states[0]
                mixed_know_value_layer = encoder_hidden_states[1]

                if mixed_key_layer.shape[1] < max(exist_length):
                    tmp_pad = torch.zeros([mixed_key_layer.shape[0], 1, mixed_key_layer.shape[-1]]).to('cuda')
                    mixed_key_layer = torch.cat([mixed_key_layer, tmp_pad], 1)
                    mixed_value_layer = torch.cat([mixed_value_layer, tmp_pad], 1)

                query_state_key = self.key(query_state)
                query_state_value = self.value(query_state)

                mixed_key_layer[list(range(mixed_key_layer.shape[0])), exist_length - 1, :] = query_state_key.squeeze(1)
                mixed_value_layer[list(range(mixed_value_layer.shape[0])), exist_length - 1, :] = query_state_value.squeeze(1)
                attention_mask = attention_mask[list(range(mixed_key_layer.shape[0])), :, exist_length - 1, :].unsqueeze(2)

            else:
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

                mixed_know_key_layer = self.know_key(encoder_hidden_states)
                mixed_know_value_layer = self.know_value(encoder_hidden_states)
        else:
            if encoder_hidden_states is not None:
                mixed_key_layer = self.key(hidden_states)
                mixed_value_layer = self.value(hidden_states)

                mixed_know_key_layer = self.know_key(encoder_hidden_states)
                mixed_know_value_layer = self.know_value(encoder_hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        know_query_layer = self.transpose_for_scores(mixed_know_query_layer)
        know_key_layer = self.transpose_for_scores(mixed_know_key_layer)
        know_value_layer = self.transpose_for_scores(mixed_know_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)

        know_attention_scores = torch.matmul(know_query_layer, know_key_layer.transpose(-1, -2))
        know_attention_scores = know_attention_scores / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        if encoder_attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            know_attention_scores = know_attention_scores + encoder_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        know_attention_probs = nn.Softmax(dim=-1)(know_attention_scores)

        import json
        with open("attention.txt", "a+", encoding='utf-8') as f:
            f.write(json.dumps(torch.mean(attention_probs[0], 0).tolist()) + '\n')
        with open("know_attention.txt", "a+", encoding='utf-8') as f:
            f.write(json.dumps(torch.mean(know_attention_probs[0], 0).tolist()) + '\n')

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)
        know_attention_probs = self.know_dropout(know_attention_probs)
         # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask
        if head_mask is not None:
            know_attention_probs = know_attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)
        know_context_layer = torch.matmul(know_attention_probs, know_value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        know_context_layer = know_context_layer.permute(0, 2, 1, 3).contiguous()
        new_know_context_layer_shape = know_context_layer.size()[:-2] + (self.all_head_size,)
        know_context_layer = know_context_layer.view(*new_know_context_layer_shape)

        outputs = ((context_layer + know_context_layer) / 2, attention_probs) if self.output_attentions else ((context_layer + know_context_layer) / 2,)
        if use_save:
            return outputs, mixed_key_layer, mixed_value_layer, mixed_know_key_layer, mixed_know_value_layer
        return outputs


class BertAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)
        self.pruned_heads = set()

    def prune_heads(self, heads):
        if len(heads) == 0:
            return
        mask = torch.ones(self.self.num_attention_heads, self.self.attention_head_size)
        heads = set(heads) - self.pruned_heads  # Convert to set and remove already pruned heads
        for head in heads:
            # Compute how many pruned heads are before the head and move the index accordingly
            head = head - sum(1 if h < head else 0 for h in self.pruned_heads)
            mask[head] = 0
        mask = mask.view(-1).contiguous().eq(1)
        index = torch.arange(len(mask))[mask].long()

        # Prune linear layers
        self.self.query = prune_linear_layer(self.self.query, index)
        self.self.key = prune_linear_layer(self.self.key, index)
        self.self.value = prune_linear_layer(self.self.value, index)
        self.output.dense = prune_linear_layer(self.output.dense, index, dim=1)

        # Update hyper params and store pruned heads
        self.self.num_attention_heads = self.self.num_attention_heads - len(heads)
        self.self.all_head_size = self.self.attention_head_size * self.self.num_attention_heads
        self.pruned_heads = self.pruned_heads.union(heads)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_save=False,
        exist_length=None,
        query_state=None,
    ):
        if query_state is not None:
            self_outputs = self.self(
                hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
                use_save=use_save, exist_length=exist_length, query_state=query_state,
            )
        else:
            self_outputs = self.self(
                hidden_states, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask,
                use_save=use_save, exist_length=exist_length,
            )
        if use_save:
            self_outputs, key_layer, value_layer, know_key_layer, know_value_layer = self_outputs

        if query_state is not None:
            attention_output = self.output(self_outputs[0], query_state)
        else:
            attention_output = self.output(self_outputs[0], hidden_states)
        outputs = (attention_output,) + self_outputs[1:]  # add attentions if we output them

        if use_save:
            return outputs, key_layer, value_layer, know_key_layer, know_value_layer
        return outputs


class BertLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = BertAttention(config)
        self.is_decoder = config.is_decoder
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_save=False,
        query_state=None,
        exist_length=None,
    ):
        if query_state is not None:
            self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                                    encoder_hidden_states, encoder_attention_mask,
                                                    use_save=use_save, exist_length=exist_length, query_state=query_state)
        else:
            self_attention_outputs = self.attention(hidden_states, attention_mask, head_mask,
                                                    encoder_hidden_states, encoder_attention_mask,
                                                    use_save=use_save, exist_length=exist_length)

        if use_save:
            self_attention_outputs, key_layer, value_layer, know_key_layer, know_value_layer = self_attention_outputs

        attention_output = self_attention_outputs[0]
        outputs = self_attention_outputs[1:]  # add self attentions if we output attention weights

        # if self.is_decoder and encoder_hidden_states is not None:
        #     if query_state is not None:
        #         cross_attention_outputs = self.crossattention(
        #             attention_output, None, head_mask, encoder_hidden_states, encoder_attention_mask, use_save=use_save, is_knowledge=True,
        #         )
        #         cross_attention_outputs, know_key_layer, know_value_layer = cross_attention_outputs
        #         attention_output = cross_attention_outputs[0]
        #         outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights
        #     else:
        #         cross_attention_outputs = self.crossattention(
        #             attention_output, attention_mask, head_mask, encoder_hidden_states, encoder_attention_mask, use_save=use_save, is_knowledge=True,
        #         )
        #         if use_save:
        #             cross_attention_outputs, know_key_layer, know_value_layer = cross_attention_outputs
        #         attention_output = cross_attention_outputs[0]
        #         outputs = outputs + cross_attention_outputs[1:]  # add cross attentions if we output attention weights

        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        outputs = (layer_output,) + outputs

        if use_save:
            return outputs, key_layer, value_layer, know_key_layer, know_value_layer
        return outputs


class BertEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.output_attentions = config.output_attentions
        self.output_hidden_states = config.output_hidden_states
        self.layer = nn.ModuleList([BertLayer(config) for _ in range(config.num_hidden_layers)])
        self.layer_key_values = [None for _ in range(config.num_hidden_layers)]
        self.know_layer_key_values = [None for _ in range(config.num_hidden_layers)]

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_save=False,
        exist_length=None,
    ):
        all_hidden_states = ()
        all_attentions = ()

        if use_save and self.layer_key_values[0] is not None:
            query_states = hidden_states[list(range(hidden_states.shape[0])), exist_length - 1, :].unsqueeze(1)
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (query_states,)

                layer_outputs = layer_module(
                    self.layer_key_values[i], attention_mask, head_mask[i], self.know_layer_key_values[i], encoder_attention_mask,
                    use_save, query_states, exist_length,
                )
                layer_outputs, key_layer, value_layer, know_key_layer, know_value_layer = layer_outputs
                self.layer_key_values[i] = (key_layer, value_layer)
                self.know_layer_key_values[i] = (know_key_layer, know_value_layer)
                query_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)
            hidden_states[list(range(hidden_states.shape[0])), exist_length - 1, :] = query_states.squeeze()
        else:
            for i, layer_module in enumerate(self.layer):
                if self.output_hidden_states:
                    all_hidden_states = all_hidden_states + (hidden_states,)

                layer_outputs = layer_module(
                    hidden_states, attention_mask, head_mask[i], encoder_hidden_states, encoder_attention_mask,
                    use_save,
                )
                if use_save:
                    layer_outputs, key_layer, value_layer, know_key_layer, know_value_layer = layer_outputs
                    self.layer_key_values[i] = (key_layer, value_layer)
                    self.know_layer_key_values[i] = (know_key_layer, know_value_layer)

                hidden_states = layer_outputs[0]

                if self.output_attentions:
                    all_attentions = all_attentions + (layer_outputs[1],)

        # Add last layer
        if self.output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        outputs = (hidden_states,)
        if self.output_hidden_states:
            outputs = outputs + (all_hidden_states,)
        if self.output_attentions:
            outputs = outputs + (all_attentions,)

        return outputs  # last-layer hidden state, (all hidden states), (all attentions)

    def clear_key_values(self):
        self.layer_key_values = [None for _ in range(len(self.layer_key_values))]
        self.know_layer_key_values = [None for _ in range(len(self.know_layer_key_values))]


BERT_START_DOCSTRING = r"""
    This model is a PyTorch `torch.nn.Module <https://pytorch.org/docs/stable/nn.html#torch.nn.Module>`_ sub-class.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general
    usage and behavior.

    Parameters:
        config (:class:`~transformers.BertConfig`): Model configuration class with all the parameters of the model.
            Initializing with a config file does not load the weights associated with the model, only the configuration.
            Check out the :meth:`~transformers.PreTrainedModel.from_pretrained` method to load the model weights.
"""

BERT_INPUTS_DOCSTRING = r"""
    Args:
        input_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`):
            Indices of input sequence tokens in the vocabulary.

            Indices can be obtained using :class:`transformers.BertTokenizer`.
            See :func:`transformers.PreTrainedTokenizer.encode` and
            :func:`transformers.PreTrainedTokenizer.encode_plus` for details.

            `What are input IDs? <../glossary.html#input-ids>`__
        attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on padding token indices.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.

            `What are attention masks? <../glossary.html#attention-mask>`__
        token_type_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Segment token indices to indicate first and second portions of the inputs.
            Indices are selected in ``[0, 1]``: ``0`` corresponds to a `sentence A` token, ``1``
            corresponds to a `sentence B` token

            `What are token type IDs? <../glossary.html#token-type-ids>`_
        position_ids (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Indices of positions of each input sequence tokens in the position embeddings.
            Selected in the range ``[0, config.max_position_embeddings - 1]``.

            `What are position IDs? <../glossary.html#position-ids>`_
        head_mask (:obj:`torch.FloatTensor` of shape :obj:`(num_heads,)` or :obj:`(num_layers, num_heads)`, `optional`, defaults to :obj:`None`):
            Mask to nullify selected heads of the self-attention modules.
            Mask values selected in ``[0, 1]``:
            :obj:`1` indicates the head is **not masked**, :obj:`0` indicates the head is **masked**.
        inputs_embeds (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Optionally, instead of passing :obj:`input_ids` you can choose to directly pass an embedded representation.
            This is useful if you want more control over how to convert `input_ids` indices into associated vectors
            than the model's internal embedding lookup matrix.
        encoder_hidden_states  (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`, `optional`, defaults to :obj:`None`):
            Sequence of hidden-states at the output of the last layer of the encoder. Used in the cross-attention
            if the model is configured as a decoder.
        encoder_attention_mask (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`, defaults to :obj:`None`):
            Mask to avoid performing attention on the padding token indices of the encoder input. This mask
            is used in the cross-attention if the model is configured as a decoder.
            Mask values selected in ``[0, 1]``:
            ``1`` for tokens that are NOT MASKED, ``0`` for MASKED tokens.
"""


@add_start_docstrings(
    "The bare Bert Model transformer outputting raw hidden-states without any specific head on top.",
    BERT_START_DOCSTRING,
)
class DecoderModel(BertPreTrainedModel):
    """

    The model can behave as an encoder (with only self-attention) as well
    as a decoder, in which case a layer of cross-attention is added between
    the self-attention layers, following the architecture described in `Attention is all you need`_ by Ashish Vaswani,
    Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser and Illia Polosukhin.

    To behave as an decoder the model needs to be initialized with the
    :obj:`is_decoder` argument of the configuration set to :obj:`True`; an
    :obj:`encoder_hidden_states` is expected as an input to the forward pass.

    .. _`Attention is all you need`:
        https://arxiv.org/abs/1706.03762

    """

    def __init__(self, config):
        super().__init__(config)
        self.config = config

        self.encoder = BertEncoder(config)
        self.pooler = BertPooler(config)

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        """ Prunes heads of the model.
            heads_to_prune: dict of {layer_num: list of heads to prune in this layer}
            See base class PreTrainedModel
        """
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    @add_start_docstrings_to_callable(BERT_INPUTS_DOCSTRING)
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        use_save=False,
        exist_length=None,
    ):
        r"""
    Return:
        :obj:`tuple(torch.FloatTensor)` comprising various elements depending on the configuration (:class:`~transformers.BertConfig`) and inputs:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor`: of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token)
            further processed by a Linear layer and a Tanh activation function. The Linear
            layer weights are trained from the next sentence prediction (classification)
            objective during pre-training.

            This output is usually *not* a good summary
            of the semantic content of the input, you're often better with averaging or pooling
            the sequence of hidden-states for the whole input sequence.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape
            :obj:`(batch_size, num_heads, sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.

    Examples::

        from transformers import BertModel, BertTokenizer
        import torch

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        model = BertModel.from_pretrained('bert-base-uncased')

        input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute", add_special_tokens=True)).unsqueeze(0)  # Batch size 1
        outputs = model(input_ids)

        last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

        """

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        device = input_ids.device if input_ids is not None else inputs_embeds.device

        if attention_mask is None:
            attention_mask = torch.ones(input_shape, device=device)
        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        if attention_mask.dim() == 3:
            extended_attention_mask = attention_mask[:, None, :, :]
        elif attention_mask.dim() == 2:
            # Provided a padding mask of dimensions [batch_size, seq_length]
            # - if the model is a decoder, apply a causal mask in addition to the padding mask
            # - if the model is an encoder, make the mask broadcastable to [batch_size, num_heads, seq_length, seq_length]
            if self.config.is_decoder:
                batch_size, seq_length = input_shape
                seq_ids = torch.arange(seq_length, device=device)
                causal_mask = seq_ids[None, None, :].repeat(batch_size, seq_length, 1) <= seq_ids[None, :, None]
                causal_mask = causal_mask.to(
                    attention_mask.dtype
                )  # causal and attention masks must have same type with pytorch version < 1.3
                extended_attention_mask = causal_mask[:, None, :, :] * attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, None, :]
        else:
            raise ValueError(
                "Wrong shape for input_ids (shape {}) or attention_mask (shape {})".format(
                    input_shape, attention_mask.shape
                )
            )

        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype)  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        # If a 2D ou 3D attention mask is provided for the cross-attention
        # we need to make broadcastabe to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)

            if encoder_attention_mask.dim() == 3:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, :, :]
            elif encoder_attention_mask.dim() == 2:
                encoder_extended_attention_mask = encoder_attention_mask[:, None, None, :]
            else:
                raise ValueError(
                    "Wrong shape for encoder_hidden_shape (shape {}) or encoder_attention_mask (shape {})".format(
                        encoder_hidden_shape, encoder_attention_mask.shape
                    )
                )

            encoder_extended_attention_mask = encoder_extended_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            encoder_extended_attention_mask = (1.0 - encoder_extended_attention_mask) * -10000.0
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        if head_mask is not None:
            if head_mask.dim() == 1:
                head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                head_mask = head_mask.expand(self.config.num_hidden_layers, -1, -1, -1, -1)
            elif head_mask.dim() == 2:
                head_mask = (
                    head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                )  # We can specify head_mask for each layer
            head_mask = head_mask.to(
                dtype=next(self.parameters()).dtype
            )  # switch to fload if need + fp16 compatibility
        else:
            head_mask = [None] * self.config.num_hidden_layers

        embedding_output = self.embeddings(
            input_ids=input_ids, position_ids=position_ids, token_type_ids=token_type_ids, inputs_embeds=inputs_embeds
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            use_save=use_save,
            exist_length=exist_length,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = None
        #pooled_output = self.pooler(sequence_output)

        outputs = (sequence_output, pooled_output,) + encoder_outputs[
            1:
        ]  # add hidden_states and attentions if they are here

        return outputs  # sequence_output, pooled_output, (hidden_states), (attentions)

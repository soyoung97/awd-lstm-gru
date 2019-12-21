import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import math
from embed_regularize import embedded_dropout
from locked_dropout import LockedDropout
from weight_drop import WeightDrop

class Attention(nn.Module):
    def __init__(self, feature_dim):
        super(Attention, self).__init__()

        self.attn_1 = nn.Linear(feature_dim, feature_dim)
        self.attn_2 = nn.Linear(feature_dim, 1)

        # inititalize
        nn.init.xavier_uniform_(self.attn_1.weight)
        nn.init.xavier_uniform_(self.attn_2.weight)
        self.attn_1.bias.data.fill_(0.0)
        self.attn_2.bias.data.fill_(0.0)

    def forward(self, x, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """
        sequence_length = x.shape[1]

        self_attention_scores = self.attn_2(torch.tanh(self.attn_1(x)))

        # Attend for each time step using the previous context
        context_vectors = []
        attention_vectors = []

        for t in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            weighted_attention_scores = F.softmax(
                self_attention_scores[:, :t + 1, :].clone(), dim=1)

            context_vectors.append(
                torch.sum(weighted_attention_scores * x[:, :t + 1, :].clone(), dim=1))

            if return_attention:
                attention_vectors.append(
                    weighted_attention_scores.cpu().detach().numpy())

        context_vectors = torch.stack(context_vectors).transpose(0, 1)

        return context_vectors, attention_vectors


class PositionalAttention(nn.Module):
    def __init__(self,
                 feature_dim,
                 positioning_embedding=20,
                 num_building_blocks=3):
        super(PositionalAttention, self).__init__()
        self.num_building_blocks = num_building_blocks

        self.positioning_generator = nn.LSTM(
            feature_dim, positioning_embedding, batch_first=True)

        self.sigma_generator = nn.Linear(positioning_embedding, 1)
        self.mu_generator = nn.Linear(
            positioning_embedding, num_building_blocks)

    def flatten_parameters(self):
        """
        Flatten parameters of all reccurrent components in the model.
        """
        self.positioning_generator.flatten_parameters()

    @staticmethod
    def normal_pdf(x, mu, sigma):
        """Return normalized Gaussian_pdf(x)."""
        x = torch.exp(-(x - mu)**2 / (2 * sigma**2 + 10e-4))
        # Normalize the Gaussian PDF result
        x = F.normalize(x, p=1)
        return x

    def forward(self, x, pad_lengths, return_attention=False):
        """
        Input x is encoder output
        return_attention decides whether to return
        attention scores over the encoder output
        """

        batch_size = x.shape[0]
        sequence_length = x.shape[1]

        # Need the lengths to normalize each sentence to respective length
        # for the building blocks - 1/N and j/N
        sentence_lengths = pad_lengths.expand(
            sequence_length, batch_size)

        # Running our linear and rnn layers (we run it all at once and parse through the sequence after)
        # positioning_weights, _ = self.positioning_generator(x)
        self.flatten_parameters()

        # pack for efficiency if more than one element (else unpadded)
        if not return_attention:
            packed_input = pack_padded_sequence(x, pad_lengths,
                                                batch_first=True)
            packed_output, _ = self.positioning_generator(packed_input)
            positioning_weights, _ = pad_packed_sequence(packed_output, batch_first=True,
                                                         total_length=sequence_length)
        else:
            positioning_weights, _ = self.positioning_generator(x)

        mu_weights = F.relu(self.mu_generator(positioning_weights))
        sigma_weights = torch.sigmoid(
            self.sigma_generator(positioning_weights))

        # Setting up Building Blocks
        prev_mu = torch.zeros(batch_size, device=device)
        building_blocks = torch.ones(
            (sequence_length, batch_size, self.num_building_blocks), device=device)
        building_blocks[:, :, 1] = 1/sentence_lengths
        building_blocks[:, :, 2] = (torch.arange(
            sequence_length, dtype=torch.float, device=device)+1).unsqueeze(1).expand(-1, batch_size) / sentence_lengths

        # Attend for each time step using the previous context
        position_vectors = []  # Which positions to attend to
        attention_vectors = []

        # we go over the whole sequence - even though it is padded so the max
        # length might be shorter.
        for j in range(sequence_length):
            # For each timestep the context that is attented grows
            # as there are more available previous hidden states
            bb = building_blocks[j].clone()
            bb[:, 0] = prev_mu

            mu = torch.bmm(mu_weights[:, j, :].clone(
            ).unsqueeze(1), bb.unsqueeze(2)).squeeze()

            # need to clamp to direct attention to previous segment of sequence
            # max dynamic and expands as we look further down sequence
            mu = torch.max(mu, j/pad_lengths)
            prev_mu = mu

            sigma = sigma_weights[:, j, :]

            # relative counter that represents 0-1 where to attend on sequence up till now
            rel_counter = torch.arange(
                j+1, dtype=torch.float, device=device)
            rel_counter = rel_counter.expand(
                batch_size, -1) / pad_lengths.view(batch_size, 1)

            gaussian_weighted_attention = self.normal_pdf(
                rel_counter, mu.unsqueeze(1), sigma).unsqueeze(2)

            # multiply the weights with the hidden encoded states found till this point
            applied_positional_attention = x[:, :j+1,
                                             :].clone() * gaussian_weighted_attention
            position_vectors.append(
                torch.sum(applied_positional_attention, dim=1))

            if return_attention:
                attention_vectors.append(
                    gaussian_weighted_attention.cpu().detach().numpy())

        context_vectors = torch.stack(position_vectors).transpose(0, 1)

        return context_vectors, attention_vectors


class AttentiveRNNLanguageModel(nn.Module):
    """
    Implements an Attentive Language Model according to http://www.aclweb.org/anthology/I17-1045
    """

    def __init__(self, vocab_size,
                 embedding_size=65,
                 hidden_size=65,
                 n_layers=1,
                 dropout_p_input=0.5,
                 dropout_p_encoder=0.0,
                 dropout_p_decoder=0.5,
                 attention=False,
                 positional_attention=True,
                 positioning_embedding=20,
                 tie_weights=True):

        super(AttentiveRNNLanguageModel, self).__init__()

        self.embedding_size = embedding_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.vocab_size = vocab_size
        self.attention = attention
        self.positional_attention = positional_attention

        self.input_dropout = nn.Dropout(dropout_p_input)
        self.embedding = nn.Embedding(vocab_size, embedding_size)
        #self.embedding equal to self.encoder in RNNModel
        self.encoder = nn.LSTM(embedding_size, hidden_size,
                               n_layers, batch_first=True,
                               dropout=dropout_p_encoder)
        if self.attention:
            self.attention_score_module = Attention(hidden_size)
        if self.positional_attention:
            self.position_score_module = PositionalAttention(
                hidden_size, positioning_embedding=positioning_embedding)

        # concatenation FF Layer to combine context and prev output
        if self.attention or self.positional_attention:
            self.concatenation_layer = nn.Linear(hidden_size * 2, hidden_size)

        if self.attention and self.positional_attention:
            raise NotImplementedError(
                "Attention and Positional Attention cannot be both activated")

        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.decoder_dropout = nn.Dropout(dropout_p_decoder)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if self.embedding_size != hidden_size:
                #raise ValueError(
                #    'When using the tied flag, encoder embedding_size must be equal to hidden_size')
                self.embedding_size = hidden_size
            self.decoder.weight = self.embedding.weight

        self.init_weights()

    def forward(self, input, pad_lengths, return_attention=False):
        batch_size = input.size(0)  # get the batch size
        total_length = input.size(1)  # get the max sequence length

        embedded = self.embedding(input)
        embedded = self.input_dropout(embedded)

        self.flatten_parameters()
        # pack for efficiency if more than one element (else unpadded)
        if not return_attention:
            packed_input = pack_padded_sequence(embedded, pad_lengths,
                                                batch_first=True)
            packed_output, _ = self.encoder(packed_input)
            encoder_output, _ = pad_packed_sequence(packed_output, batch_first=True,
                                                    total_length=total_length)
        else:
            encoder_output, _ = self.encoder(embedded)

        if self.attention:
            context_vectors, attention_score = self.attention_score_module(
                encoder_output, return_attention=return_attention)

        if self.positional_attention:
            context_vectors, attention_score = self.position_score_module(
                encoder_output, pad_lengths, return_attention=return_attention)

        if self.attention or self.positional_attention:
            combined_encoding = torch.cat(
                (context_vectors, encoder_output), dim=2)
            # concatenation layer
            encoder_output = torch.tanh(
                self.concatenation_layer(combined_encoding))

        output = self.decoder_dropout(encoder_output)
        decoded = self.decoder(output.contiguous())

        if return_attention:
            return decoded, attention_score

        return decoded

    def flatten_parameters(self):
        """
        Flatten parameters of all reccurrent components in the model.
        """
        self.encoder.flatten_parameters()

    def init_weights(self, init_range=0.1):
        """
        Standard weight initialization
        """
        self.embedding.weight.data.uniform_(
            -init_range, init_range)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-init_range, init_range)

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, dropout=0.5, dropouth=0.5, dropouti=0.5, dropoute=0.1, wdrop=0, tie_weights=False):
        super(RNNModel, self).__init__()
        self.lockdrop = LockedDropout()
        self.idrop = nn.Dropout(dropouti)
        self.hdrop = nn.Dropout(dropouth)
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        assert rnn_type in ['LSTM', 'QRNN', 'GRU', 'biLSTM'], 'RNN type is not supported'
        if rnn_type == 'LSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        if rnn_type == 'GRU':
            self.rnns = [torch.nn.GRU(ninp if l == 0 else nhid, nhid if l != nlayers - 1 else ninp, 1, dropout=0) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns]
        elif rnn_type == 'QRNN':
            from torchqrnn import QRNNLayer
            self.rnns = [QRNNLayer(input_size=ninp if l == 0 else nhid, hidden_size=nhid if l != nlayers - 1 else (ninp if tie_weights else nhid), save_prev_x=True, zoneout=0, window=2 if l == 0 else 1, output_gate=True) for l in range(nlayers)]
            for rnn in self.rnns:
                rnn.linear = WeightDrop(rnn.linear, ['weight'], dropout=wdrop)
        elif rnn_type == 'biLSTM':
            self.rnns = [torch.nn.LSTM(ninp if l == 0 else nhid, nhid//2 if l != nlayers - 1 else (ninp//2 if tie_weights else nhid), 1, dropout=0, bidirectional=True) for l in range(nlayers)]
            if wdrop:
                self.rnns = [WeightDrop(rnn, ['weight_hh_l0'], dropout=wdrop) for rnn in self.rnns] 
        print(self.rnns)
        #self.rnns = torch.nn.ModuleList(self.rnns)
        self.rnns = torch.nn.ModuleList([torch.nn.DataParallel(r, dim=1) for r in self.rnns])
        self.decoder = nn.Linear(nhid, ntoken)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            #if nhid != ninp:
            #    raise ValueError('When using the tied flag, nhid must be equal to emsize')
            self.decoder.weight = self.encoder.weight

        self.init_weights()

        self.rnn_type = rnn_type
        self.ninp = ninp
        self.nhid = nhid
        self.nlayers = nlayers
        self.dropout = dropout
        self.dropouti = dropouti
        self.dropouth = dropouth
        self.dropoute = dropoute
        self.tie_weights = tie_weights

    def reset(self):
        if self.rnn_type == 'QRNN': [r.reset() for r in self.rnns]

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.fill_(0)
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, return_h=False):
        emb = embedded_dropout(self.encoder, input, dropout=self.dropoute if self.training else 0)
        #emb = self.idrop(emb)

        emb = self.lockdrop(emb, self.dropouti)
        raw_output = emb
        new_hidden = []
        #raw_output, hidden = self.rnn(emb, hidden)
        raw_outputs = []
        outputs = []
        for l, rnn in enumerate(self.rnns):
            current_input = raw_output
            raw_output, new_h = rnn(raw_output, hidden[l])
            new_hidden.append(new_h)
            raw_outputs.append(raw_output)
            if l != self.nlayers - 1:
                #self.hdrop(raw_output)
                raw_output = self.lockdrop(raw_output, self.dropouth)
                outputs.append(raw_output)
        hidden = new_hidden

        output = self.lockdrop(raw_output, self.dropout)
        outputs.append(output)
###
        result = output.view(output.size(0)*output.size(1), output.size(2))
        if return_h:
            return result, hidden, raw_outputs, outputs
        return result, hidden

    def init_hidden(self, bsz):
        weight = next(self.parameters()).data
        if self.rnn_type == 'LSTM':
            return [(weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_(),
                    weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
        if self.rnn_type == 'biLSTM':
            a =[(weight.new(2, bsz, self.nhid//2 if l != self.nlayers - 1 else (self.ninp//2 if self.tie_weights else self.nhid)).zero_(),
                    weight.new(2, bsz, self.nhid//2 if l != self.nlayers - 1 else (self.ninp//2 if self.tie_weights else self.nhid)).zero_())
                    for l in range(self.nlayers)]
            return a
 
        elif self.rnn_type == 'QRNN' or self.rnn_type == 'GRU':
            return [weight.new(1, bsz, self.nhid if l != self.nlayers - 1 else (self.ninp if self.tie_weights else self.nhid)).zero_()
                   for l in range(self.nlayers)]

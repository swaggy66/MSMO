import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss
from transformers import BertPreTrainedModel, BertModel
from transformers import PreTrainedModel, RobertaModel, RobertaConfig


class BertABSATagger(BertPreTrainedModel):
    def __init__(self, bert_config):
        """
        bert_config: configuration for bert model
        """
        super(BertABSATagger, self).__init__(bert_config)
        self.num_labels = bert_config.num_labels

        # initialized with pre-trained BERT and perform finetuning
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size            
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        # print(tagger_input)
        tagger_input = self.bert_dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        # print(logits)
        # print(outputs[:-1])
        outputs = (logits,) + outputs[2:]
        # print((logits,))
        # print(outputs[2:])

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # print(active_loss)
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                # print(active_logits)
                active_labels = labels.view(-1)[active_loss]
                # print(active_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs


class RobertaPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple 
    interface for downloading and loading pretrained models.
    """
    config_class = RobertaConfig
    base_model_prefix = "roberta"

    # Copied from transformers.models.bert.modeling_bert.BertPreTrainedModel._init_weights
    def _init_weights(self, module):
        """ Initialize the weights """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class XLMRABSATagger(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        tagger_input = outputs[0]
        tagger_input = self.dropout(tagger_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        outputs = (logits,) + outputs[2:]

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs
    

import torch
from torch import autograd, nn
import torch.nn.functional as functional
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

# from layers import *
# from options import opt

class DANFeatureExtractor(BertPreTrainedModel):
    def __init__(self,bert_config):
        super(DANFeatureExtractor, self).__init__(bert_config)
        # super(BertABSATagger, self).__init__(bert_config)
        #self.word_emb = vocab.init_embed_layer() #stanford_emb

    
        self.num_labels = bert_config.num_labels     

        # initialized with pre-trained BERT and perform finetuning
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)
        
        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size 

          
        self.classifier = nn.Linear(penultimate_hidden_size, penultimate_hidden_size)

        # if sum_pooling:
        #     self.avg = SummingLayer(self.word_emb)
        # else:
        #     self.avg = AveragingLayer(self.word_emb)



    
    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
            position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)


        return outputs


class DANFeatureExtractorXLM(RobertaPreTrainedModel):
    def __init__(self, config):
        super(DANFeatureExtractorXLM,self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = config.hidden_size 
        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.classifier = nn.Linear(penultimate_hidden_size, penultimate_hidden_size)

        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
                            attention_mask=attention_mask, head_mask=head_mask)

       

        return outputs

class LSTMFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 dropout,
                 bdrnn,
                 attn_type):
        super(LSTMFeatureExtractor, self).__init__()
        self.num_layers = num_layers
        self.bdrnn = bdrnn
        self.attn_type = attn_type
        self.hidden_size = hidden_size//2 if bdrnn else hidden_size
        self.n_cells = self.num_layers*2 if bdrnn else self.num_layers
        
        self.word_emb = vocab.init_embed_layer()
        self.rnn = nn.LSTM(input_size=vocab.emb_size, hidden_size=self.hidden_size,
                num_layers=num_layers, dropout=dropout, bidirectional=bdrnn)
        if attn_type == 'dot':
            # self.attn = DotAttentionLayer(hidden_size)
            pass

    def forward(self, input):
        data, lengths = input
        lengths_list = lengths.tolist()
        batch_size = len(data)
        embeds = self.word_emb(data)
        packed = pack_padded_sequence(embeds, lengths_list, batch_first=True)
        state_shape = self.n_cells, batch_size, self.hidden_size
        h0 = c0 = embeds.data.new(*state_shape)
        output, (ht, ct) = self.rnn(packed, (h0, c0))

        if self.attn_type == 'last':
            return ht[-1] if not self.bdrnn \
                          else ht[-2:].transpose(0, 1).contiguous().view(batch_size, -1)
        elif self.attn_type == 'avg':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return torch.sum(unpacked_output, 1) / lengths.float().view(-1, 1)
        elif self.attn_type == 'dot':
            unpacked_output = pad_packed_sequence(output, batch_first=True)[0]
            return self.attn((unpacked_output, lengths))
        else:
            raise Exception('Please specify valid attention (pooling) mechanism')


class CNNFeatureExtractor(nn.Module):
    def __init__(self,
                 vocab,
                 num_layers,
                 hidden_size,
                 kernel_num,
                 kernel_sizes,
                 dropout):
        super(CNNFeatureExtractor, self).__init__()
        self.word_emb = vocab.init_embed_layer()
        self.kernel_num = kernel_num
        self.kernel_sizes = kernel_sizes

        self.convs = nn.ModuleList([nn.Conv2d(1, kernel_num, (K, vocab.emb_size)) for K in kernel_sizes])
        
        assert num_layers >= 0, 'Invalid layer numbers'
        self.fcnet = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.fcnet.add_module('f-dropout-{}'.format(i), nn.Dropout(p=dropout))
            if i == 0:
                self.fcnet.add_module('f-linear-{}'.format(i),
                        nn.Linear(len(kernel_sizes)*kernel_num, hidden_size))
            else:
                self.fcnet.add_module('f-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            self.fcnet.add_module('f-relu-{}'.format(i), nn.ReLU())

    def forward(self, input):
        data, lengths = input
        batch_size = len(data)
        embeds = self.word_emb(data)
        # conv
        embeds = embeds.unsqueeze(1) # batch_size, 1, seq_len, emb_size
        x = [functional.relu(conv(embeds)).squeeze(3) for conv in self.convs]
        x = [functional.max_pool1d(i, i.size(2)).squeeze(2) for i in x]
        x = torch.cat(x, 1)
        # fcnet
        return self.fcnet(x)


class SentimentClassifier(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 output_size,
                 dropout,
                 batch_norm=False):
        super(SentimentClassifier, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
        self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    def forward(self, input):
        return self.net(input)
    


class mBertABSASentimentClassifier(BertPreTrainedModel):
    def __init__(self, bert_config):
        super(mBertABSASentimentClassifier, self).__init__(bert_config)
    #     assert num_layers >= 0, 'Invalid layer numbers'
    #     self.net = nn.Sequential()
    #     for i in range(num_layers):
    #         if dropout > 0:
    #             self.net.add_module('p-dropout-{}'.format(i), nn.Dropout(p=dropout))
    #         self.net.add_module('p-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
    #         if batch_norm:
    #             self.net.add_module('p-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
    #         self.net.add_module('p-relu-{}'.format(i), nn.ReLU())

    #     self.net.add_module('p-linear-final', nn.Linear(hidden_size, output_size))
    #     self.net.add_module('p-logsoftmax', nn.LogSoftmax(dim=-1))

    # def forward(self, input):
    #     return self.net(input)
        self.num_labels = bert_config.num_labels

        # initialized with pre-trained BERT and perform finetuning
        #不需要嵌入层
        self.bert = BertModel(bert_config, add_pooling_layer=False)
        
        
        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size            
        self.classifier = nn.Linear(penultimate_hidden_size, bert_config.num_labels)

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
    #             position_ids=None, head_mask=None, teacher_probs=None):

    #     outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
    #                         attention_mask=attention_mask, head_mask=head_mask)
    def forward(self, feature_input,input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):
        # outputs = self.bert(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #             attention_mask=attention_mask, head_mask=head_mask)
        feature_input = feature_input[0]
        
        tagger_input = self.bert_dropout(feature_input)
        logits = self.classifier(tagger_input)
        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
      
        # print(outputs[:-1])
        outputs = (logits,) + tuple(feature_input[2:])
        # print((logits,))
        # print(outputs[2:])

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                # print(active_loss)
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                # print(active_logits)
                active_labels = labels.view(-1)[active_loss]
                # print(active_labels)
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs

class XLMABSASentimentClassifier(RobertaPreTrainedModel):
    def __init__(self, config):
        super(XLMABSASentimentClassifier,self).__init__(config)
        self.num_labels = config.num_labels

        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)

        self.init_weights()

    def forward(self, feature_input,input_ids=None, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, head_mask=None, teacher_probs=None):

        feature_input = feature_input[0]
        
        # outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, head_mask=head_mask)

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        #tagger_input = outputs[0]
        tagger_input = self.dropout(feature_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        outputs = (logits,) + tuple(feature_input[2:])

        if labels is not None:
            # print("We are using true labels!")
            loss_fct = CrossEntropyLoss()
            if attention_mask is not None:
                active_loss = attention_mask.view(-1) == 1
                active_logits = logits.view(-1, self.num_labels)[active_loss]
                active_labels = labels.view(-1)[active_loss]
                loss = loss_fct(active_logits, active_labels)
            else:
                loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
            outputs = (loss,) + outputs

        # use soft labels to train the model
        if teacher_probs is not None:
            # print("We are using soft labels!")
            loss_kd_func = MSELoss(reduction='none')
            active_loss = attention_mask.view(-1) == 1

            pred_probs = torch.nn.functional.softmax(logits, dim=-1)
            loss_kd = loss_kd_func(pred_probs, teacher_probs)  # batch_size x max_seq_len x num_labels

            loss_kd = torch.mean(loss_kd.view(-1, self.num_labels)[active_loss.view(-1)])

            outputs = (loss_kd,) + outputs

        return outputs


class LanguageDetector(nn.Module):
    def __init__(self,
                 num_layers,
                 hidden_size,
                 dropout,
                 batch_norm=False):
        super(LanguageDetector, self).__init__()
        assert num_layers >= 0, 'Invalid layer numbers'
        self.net = nn.Sequential()
        for i in range(num_layers):
            if dropout > 0:
                self.net.add_module('q-dropout-{}'.format(i), nn.Dropout(p=dropout))
            self.net.add_module('q-linear-{}'.format(i), nn.Linear(hidden_size, hidden_size))
            if batch_norm:
                self.net.add_module('q-bn-{}'.format(i), nn.BatchNorm1d(hidden_size))
            self.net.add_module('q-relu-{}'.format(i), nn.ReLU())

        self.net.add_module('q-linear-final', nn.Linear(hidden_size, 1))

    def forward(self, input):
        return self.net(input)
    

class mBERTABSALanguageDetector(BertPreTrainedModel):
    def __init__(self,bert_config):
        super(mBERTABSALanguageDetector, self).__init__(bert_config)
    #     assert num_layers >= 0, 'Invalid layer numbers'

        self.num_language = 1

        # initialized with pre-trained BERT and perform finetuning
        #不需要嵌入层
        #self.bert = BertModel(bert_config, add_pooling_layer=False)

        self.bert_dropout = nn.Dropout(bert_config.hidden_dropout_prob)

        # hidden size at the penultimate layer
        penultimate_hidden_size = bert_config.hidden_size            
        self.classifier = nn.Linear(penultimate_hidden_size, self.num_language)


   
    def forward(self, feature_input, labels=None):

        feature_input = feature_input[0]
        
        tagger_input = self.bert_dropout(feature_input)
        logits = self.classifier(tagger_input)
        #outputs = logits
        return logits

      

class XLMABSALanguageDetector(RobertaPreTrainedModel):
    def __init__(self, config):
        super(XLMABSALanguageDetector,self).__init__(config)
        # self.num_labels = config.num_labels
        self.num_language = 1

        #self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, self.num_language)

        self.init_weights()

    # def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
    #             position_ids=None, head_mask=None, teacher_probs=None):
    def forward(self, feature_input, labels=None):

        # outputs = self.roberta(input_ids, position_ids=position_ids, token_type_ids=token_type_ids,
        #                     attention_mask=attention_mask, head_mask=head_mask)
        feature_input = feature_input[0]

        # the hidden states of the last Bert Layer, shape: (bsz, seq_len, hsz)
        #tagger_input = outputs[0]
        tagger_input = self.dropout(feature_input)
        # print("tagger_input.shape:", tagger_input.shape)

        logits = self.classifier(tagger_input)
        #outputs = (logits,) + outputs[2:]

      

        return logits

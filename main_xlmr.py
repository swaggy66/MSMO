import argparse
import os
import logging
import random
import pickle
import numpy as np
from tqdm import tqdm, trange
import sys
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
import torch.optim as optim
from torchnet.meter import ConfusionMeter
from tensorboardX import SummaryWriter

from transformers import BertConfig, BertTokenizer, BertTokenizerFast
from transformers import XLMRobertaConfig, XLMRobertaTokenizer, XLMRobertaTokenizerFast
from transformers import AdamW, get_linear_schedule_with_warmup

from model import BertABSATagger, XLMRABSATagger
from seq_utils import compute_metrics_absa
from data_utils import XABSAKDDataset
from data_utils import build_or_load_dataset, get_tag_vocab, write_results_to_log ,build_or_load_dataset_languagedetect_source,build_or_load_dataset_languagedetect_target,build_or_load_dataset_languagedetect_source2target_codeswutch,build_or_load_dataset_languagedetect_target2source_codeswitch,build_or_load_dataset_languagedetect_source_label,build_or_load_dataset_languagedetect_target_label,build_or_load_dataset_alignid_source,build_or_load_dataset_alignid_target
import re
from model_adan import *
import utils

logger = logging.getLogger(__name__)
import debugpy
import torch.nn.functional as F
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9502))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass


# MODEL_CLASSES = {
#     'bert': (BertConfig, BertABSATagger, BertTokenizerFast),
#     'mbert': (BertConfig, BertABSATagger, BertTokenizerFast),
#     'xlmr': (XLMRobertaConfig, XLMRABSATagger, XLMRobertaTokenizerFast)
# }


MODEL_CLASSES = {
    'bert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'mbert': (BertConfig, BertABSATagger, BertTokenizerFast),
    'xlmr': (XLMRobertaConfig, XLMRABSATagger, XLMRobertaTokenizerFast),
    'adan_bert':(BertConfig, DANFeatureExtractor, BertTokenizerFast),
    'adan_language_detect':(BertConfig, mBERTABSALanguageDetector, BertTokenizerFast),
    'adan_sentiment':(BertConfig, mBertABSASentimentClassifier, BertTokenizerFast),
    'adan_xml':(XLMRobertaConfig, DANFeatureExtractorXLM, XLMRobertaTokenizerFast),
    'adan_language_detect_xlm':(XLMRobertaConfig, XLMABSALanguageDetector, XLMRobertaTokenizerFast),
    'adan_sentiment_xlm':(XLMRobertaConfig, XLMABSASentimentClassifier, XLMRobertaTokenizerFast)
 
}



def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


def init_args():
    parser = argparse.ArgumentParser()
    # basic settings
    parser.add_argument("--tfm_type", default='mbert', type=str, required=True,
                        help="The base transformer, selected from: [bert, mbert, xlmr]")
    parser.add_argument("--model_name_or_path", default='./mBERT', type=str, required=True,
                        help="Path to the pre-trained model or shortcut name")
    parser.add_argument("--exp_type", default='acs', type=str, required=True,
                        help="Experiment type, selected from: [supervised, zero_shot,acs ...]")

    # source/target data and languages
    parser.add_argument("--data_dir", default='./data/', type=str, required=True, help="Base data dir")
    parser.add_argument("--src_lang", default='en', type=str, required=True, help="source language")
    parser.add_argument("--tgt_lang", default='fr', type=str, required=True, help="target language")
    parser.add_argument("--data_select", default=1.0, type=float, help="ratio of the selected data to train, 1 is to use all data")
    parser.add_argument('--tagging_schema', type=str, default='BIEOS')
    parser.add_argument('--label_path', type=str, default='')
    parser.add_argument("--ignore_cached_data", action='store_true')
    parser.add_argument("--train_data_sampler", type=str, default='random')
    parser.add_argument("--do_lower_case", action='store_true', help="Set this flag if you are using an uncased model.")

    parser.add_argument('--task', type=str, default='absa')
    # Other parameters
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_seq_length", default=128, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--do_train", action='store_true', help="Whether to run training.")
    parser.add_argument("--do_distill", action='store_true', help="Whether to run knowledge distillation.")
    parser.add_argument("--trained_teacher_paths", type=str, help="path of the trained model")
    parser.add_argument("--do_eval", action='store_true', help="Whether to run eval on the dev/test set.")
    parser.add_argument("--eval_begin_end", default="15-19", type=str)

    # train configs
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Rul evaluation during training at each logging step.")
    parser.add_argument("--per_gpu_train_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=16, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--freeze_bottom_layer", default=-1, type=int)
    parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=20.0, type=float, 
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--train_begin_saving_step", type=int, default=10000, help="Starting point of evaluation.")
    parser.add_argument("--train_begin_saving_epoch", type=int, default=10, help="Starting point of evaluation.")
    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=100,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as model_name ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=44,
                        help="random seed for initialization")

    parser.add_argument('--max_len', type=int, default=110)


    parser.add_argument('--model', default='dan')  # dan or lstm or cnn

    #mbert
    parser.add_argument('--featuremodel', default='adan_bert')  # dan or lstm or cnn
    parser.add_argument('--languagedetectmodel', default='adan_language_detect')  # dan or lstm or cnn
    parser.add_argument('--sentimentmodel', default='adan_sentiment')  # dan or lstm or cnn

    #xlmr
    # parser.add_argument('--featuremodel', default='adan_xml')  # dan or lstm or cnn
    # parser.add_argument('--languagedetectmodel', default='adan_language_detect_xlm')  # dan or lstm or cnn
    # parser.add_argument('--sentimentmodel', default='adan_sentiment_xlm')  # dan or lstm or cnn
   

    parser.add_argument('--P_F_learning_rate', type=float, default=0.0005)
    parser.add_argument('--Q_learning_rate', type=float, default=0.0005)

    parser.add_argument('--n_critic', type=int, default=5)
    parser.add_argument('--clip_lower', type=float, default=-0.01)
    parser.add_argument('--clip_upper', type=float, default=0.01)
    parser.add_argument('--lambd', type=float, default=0.01)
    parser.add_argument('--fix_emb', action='store_true')
    parser.add_argument('--model_save_file', default='./save_model_adan')
    
    #一致性损失
    # parser.add_argument('--uda_weight', default=0.00025,type=float)
    # parser.add_argument('--uda_weight', default=0.0035,type=float)
    parser.add_argument('--uda_weight', default=0.0025,type=float)


    parser.add_argument('--uda_threshold', type=float, default=0.0)



    # distributed training
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()

    # set up output dir: './outputs/mbert-en-fr-zero_shot/'
    output_dir = f"./outputs/{args.tfm_type}-{args.src_lang}-{args.tgt_lang}-{args.exp_type}"
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    args.output_dir = output_dir

    return args


def get_optimizer_grouped_parameters(args, model, no_grad=None):
    no_decay = ["bias", "LayerNorm.weight"]
    if no_grad is not None:
        logger.info(" The frozen parameters are:")
        for n, p in model.named_parameters():
            p.requires_grad = False if any(nd in n for nd in no_grad) else True
            if not p.requires_grad:
                logger.info("   Freeze: %s", n)
        logger.info(" The parameters to be fine-tuned are:")
        for n, p in model.named_parameters():
            if p.requires_grad:
                logger.info("   Fine-tune: %s", n)
    else:
        for n, p in model.named_parameters():
            if not p.requires_grad:
                assert False, "parameters to update with requires_grad=False"

    outputs = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay) and p.requires_grad],
         "weight_decay": 0.0}
    ]
    return outputs

# Repeating dataloader for UDA
def repeat_dataloader(iterable):
    """ repeat dataloader """
    while True:
        for x in iterable:
            yield x    

#计算KL散度（单向）
def compute_kl_loss(p, q, pad_mask=None):
    
    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1), F.softmax(p, dim=-1), reduction='none')
    
    p_loss = p_loss.sum(dim=-1)
    q_loss = q_loss.sum(dim=-1)
    
    if pad_mask is not None:
        p_loss.masked_fill_(pad_mask, 0.)
        q_loss.masked_fill_(pad_mask, 0.)

    p_loss = p_loss.sum() / (~pad_mask).sum() #mean
    q_loss = q_loss.sum() / (~pad_mask).sum() #mean

    loss = (p_loss + q_loss) / 2
    return loss   
#方面词分数转换为方面词跨度概率（连乘的方式）
def tok_score_2_span_prob(args,batch_align_ids, batch_e_scores, max_entity, config, delta = 1e-20):
    
    list_entity_prob = [[torch.tensor([]) for j in range(max_entity)] for i in range(batch_align_ids.shape[0])] # list (batch) of list (max_entity) of empty tensor
    
    batch_without_entity = True
    for batch_id in range(batch_align_ids.shape[0]):
        
        sent_align_id = batch_align_ids[batch_id]
        sent_e_score = batch_e_scores[batch_id]

        _, tag2idx, _ = get_tag_vocab(task=args.task, tagging_schema=args.tagging_schema, 
                                  label_path=args.label_path)
        # Ensure the shapes match by slicing the longer tensor
        # if sent_align_id.shape[0] != sent_e_score.shape[0]:
        #     min_length = min(sent_align_id.shape[0], sent_e_score.shape[0])
        #     sent_align_id = sent_align_id[:min_length]
        #     sent_e_score = sent_e_score[:min_length]
        for i in range(1, max_entity+1):
            e_score = sent_e_score[sent_align_id][sent_align_id == i]
            #print(e_score)
            e_prob = torch.nn.functional.softmax(e_score, dim=-1)
            #print(e_prob)
            if e_prob.shape[0] == 0:
                continue
            elif e_prob.shape[0] == 1:
                per_prob = e_prob[0,tag2idx["S-POS"]] 
                org_prob = e_prob[0,tag2idx["S-NEG"]] 
                loc_prob = e_prob[0,tag2idx["S-NEU"]] 
                #misc_prob = e_prob[0,config.label2idx["S-MISC"]] 
                o_prob = e_prob[0,tag2idx["O"]] 
                
                # Numerical Stability                
                per_prob = per_prob + delta if per_prob < delta else per_prob
                org_prob = org_prob + delta if org_prob < delta else org_prob
                loc_prob = loc_prob + delta if loc_prob < delta else loc_prob
                #misc_prob = misc_prob + delta if misc_prob < delta else misc_prob
                o_prob = o_prob + delta if o_prob < delta else o_prob
                #print('probs:', per_prob, org_prob, loc_prob, o_prob)
                # rest_prob = 1 - (per_prob + org_prob + loc_prob + misc_prob + o_prob)
                rest_prob = 1 - (per_prob + org_prob + loc_prob + o_prob)
                if rest_prob <= 0:
                    # print('probs:', per_prob, org_prob, loc_prob, misc_prob, o_prob)
                    print('probs:', per_prob, org_prob, loc_prob, o_prob)
                assert rest_prob > 0 
                
                assert per_prob.requires_grad
                assert rest_prob.requires_grad
  
                # entity_prob = torch.stack([per_prob, org_prob, loc_prob, misc_prob, o_prob, rest_prob], dim=0)
                entity_prob = torch.stack([per_prob, org_prob, loc_prob, o_prob, rest_prob], dim=0)
                list_entity_prob[batch_id][i-1] = entity_prob
                #print('list_entity_prob',list_entity_prob)
                
                batch_without_entity = False
            else:
                per_prob = e_prob[0,tag2idx["B-POS"]]
                per_prob = per_prob * e_prob[-1,tag2idx["E-POS"]]
                for i_prob in e_prob[1:-1,tag2idx["I-POS"]]:
                    per_prob = per_prob * i_prob
                    
                org_prob = e_prob[0,tag2idx["B-NEG"]]
                org_prob = org_prob * e_prob[-1,tag2idx["E-NEG"]]
                for i_prob in e_prob[1:-1,tag2idx["I-NEG"]]:
                    org_prob = org_prob * i_prob

                loc_prob = e_prob[0,tag2idx["B-NEU"]]
                loc_prob = loc_prob * e_prob[-1,tag2idx["E-NEU"]]
                for i_prob in e_prob[1:-1,tag2idx["I-NEU"]]:
                    loc_prob = loc_prob * i_prob

                # misc_prob = e_prob[0,config.label2idx["B-MISC"]]
                # misc_prob = misc_prob * e_prob[-1,config.label2idx["E-MISC"]]
                # for i_prob in e_prob[1:-1,config.label2idx["I-MISC"]]:
                #     misc_prob = misc_prob * i_prob

                o_prob = 1.0
                for prob in e_prob[:,tag2idx["O"]]:
                    o_prob = o_prob * prob
                    
                # Numerical Stability                
                per_prob = per_prob + delta if per_prob < delta else per_prob
                org_prob = org_prob + delta if org_prob < delta else org_prob
                loc_prob = loc_prob + delta if loc_prob < delta else loc_prob
                #misc_prob = misc_prob + delta if misc_prob < delta else misc_prob
                o_prob = o_prob + delta if o_prob < delta else o_prob

                # rest_prob = 1 - (per_prob + org_prob + loc_prob + misc_prob + o_prob)
                rest_prob = 1 - (per_prob + org_prob + loc_prob + o_prob)
                if rest_prob <= 0:
                    # print('probs:', per_prob, org_prob, loc_prob, misc_prob, o_prob)
                    print('probs:', per_prob, org_prob, loc_prob, o_prob)
                assert rest_prob > 0 
                
                # entity_prob = torch.stack([per_prob, org_prob, loc_prob, misc_prob, o_prob, rest_prob], dim=0)
                entity_prob = torch.stack([per_prob, org_prob, loc_prob, o_prob, rest_prob], dim=0)                
                list_entity_prob[batch_id][i-1] = entity_prob
                #print('list_entity',list_entity_prob)
                
                batch_without_entity = False
                
    if not batch_without_entity:
        return list_entity_prob
    else:
        return None


def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step


def train_adan(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    #随机采样语言辨别器所需要的数据，总数和训练数据一样，acs为四个数据集
    if args.train_data_sampler == 'random':
        train_sampler_source = RandomSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_source = SequentialSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_target = RandomSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_target = SequentialSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_s2t = RandomSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_s2t = SequentialSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_t2s = RandomSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_t2s = SequentialSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #构建批量的语言辨别器所需要的数据
    train_dataloader_source = DataLoader(train_dataset_source_without_label, sampler=train_sampler_source, batch_size=args.train_batch_size)
    train_dataloader_target = DataLoader(train_dataset_target_without_label, sampler=train_sampler_target, batch_size=args.train_batch_size)
    train_dataloader_s2t = DataLoader(train_dataset_s2t_without_label, sampler=train_sampler_s2t, batch_size=args.train_batch_size)
    train_dataloader_t2s = DataLoader(train_dataset_t2s_without_label, sampler=train_sampler_t2s, batch_size=args.train_batch_size)



    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    #独立三个模型的参数
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform XABSA task with label list being {tag_list} (n_labels={num_tags})")

    #加载特征提取器
    # args.tfm_type = args.tfm_type.lower() 
    args.featuremodel = args.featuremodel.lower() 
    logger.info(f"Load pre-trained {args.featuremodel} model from `{args.model_name_or_path}`")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.featuremodel]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'

    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    # model.to(args.device)
    #新增语言辨别器和特征提取层
        # models
    if args.featuremodel.lower() == 'adan_bert':
        F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
        F.to(args.device)
    # if args.model.lower() == 'dan':
    #     F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
    #     F.to(args.device)
    # elif opt.model.lower() == 'lstm':
    #     F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
    #             opt.bdrnn, opt.attn)
    # elif opt.model.lower() == 'cnn':
    #     F = CNNFeatureExtractor(vocab, opt.F_layers,
    #             opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception('Unknown model')
    config_class1, model_class1, tokenizer_class1 = MODEL_CLASSES[args.sentimentmodel]
    config1 = config_class1.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer1 = tokenizer_class1.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    P = mBertABSASentimentClassifier.from_pretrained(args.model_name_or_path,config=config1)

    #加载语言辨别器 其中分辨源语言或者是目标语言（标签只有两种）
    num_tags_language = 1
    config_class2, model_class2, tokenizer_class2 = MODEL_CLASSES[args.languagedetectmodel]
    config2 = config_class2.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags_language, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer2 = tokenizer_class2.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    Q = mBERTABSALanguageDetector.from_pretrained(args.model_name_or_path,config=config2)
    F, P, Q = F.to(args.device), P.to(args.device), Q.to(args.device)
    optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()),
                           lr=args.learning_rate, eps=args.adam_epsilon)
    optimizerQ = optim.Adam(Q.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    schedulerQ = get_linear_schedule_with_warmup(optimizerQ, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
   
    
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, F, no_grad=no_grad)
    # optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, Q, no_grad=no_grad)
    # optimizer2 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler2 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, P, no_grad=no_grad)
    # optimizer3 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler3 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)




    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        #新增FPQ的训练
        F.train()
        P.train()
        Q.train()
        # training accuracy
        correct, total = 0, 0
        sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
        grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)


      

        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_source = tqdm(train_dataloader_source, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target = tqdm(train_dataloader_target, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_s2t = tqdm(train_dataloader_s2t, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_t2s = tqdm(train_dataloader_t2s, desc="Iteration", disable=args.local_rank not in [-1, 0])

        # inputs_ch = ({'input_ids':      batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
        # inputs_ch_next = next(inputs_ch)
        # print(inputs_ch_next)
        q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
        q_inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs_with_label = {'input_ids':     batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    }
            inputs_labels = {'labels': batch['labels'].to(args.device)}
     
            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_ch_next = next(inputs_ch_iter)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
                inputs_ch_next = next(chn_train_iter)

            # Q iterations
            n_critic = args.n_critic
            if n_critic>0 and ((n_epoch==0 and step<=25) or (step%500==0)):
                n_critic = 10
            utils.freeze_net(F)
            utils.freeze_net(P)
            utils.unfreeze_net(Q)
            for qiter in range(n_critic):
                # clip Q weights
                for p in Q.parameters():
                    p.data.clamp_(args.clip_lower, args.clip_upper)
                Q.zero_grad()
                # get a minibatch of data
                try:
                    # labels are not used
                    #q_inputs_en, _ = next(train_dataset_source_without_label)
                    # q_inputs_en = next(train_dataset_source_without_label)
                    # q_inputs_en = ({'input_ids':   batch['input_ids'].to(args.device),
                    # 'attention_mask': batch['attention_mask'].to(args.device),
                    # 'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    # } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
                    q_inputs_en = next(q_inputs_en_iter) 
                except StopIteration:
                    # check if dataloader is exhausted
                    # yelp_train_iter_Q = iter(train_dataset_source_without_label)
                    # q_inputs_en, _ = next(train_dataset_source_without_label)
                    q_train_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
                    q_inputs_en = next(q_train_en_iter) 
                try:
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    #q_inputs_ch= next(train_dataset_target_without_label)
                    q_inputs_ch= next(q_inputs_ch_iter)
                except StopIteration:
                    # chn_train_iter_Q = iter(train_dataset_target_without_label)
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    # q_inputs_ch = next(train_dataset_target_without_label)
                    q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))

                    q_inputs_ch= next(q_inputs_ch_iter)

                features_en = F(**q_inputs_en)
                #print(features_en)
                o_en_ad = Q(features_en)
                l_en_ad = torch.mean(o_en_ad)
                (-l_en_ad).backward() 
               
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_en_q = (sum_en_q[0] + 1, sum_en_q[1] + l_en_ad.item())

                features_ch = F(**q_inputs_ch)
                o_ch_ad = Q(features_ch)
                l_ch_ad = torch.mean(o_ch_ad)
                l_ch_ad.backward()
                # logging.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                #logging.info(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_ch_q = (sum_ch_q[0] + 1, sum_ch_q[1] + l_ch_ad.item())

                optimizerQ.step()
                schedulerQ.step()  # Update learning rate schedule

            # F&P iteration
            utils.unfreeze_net(F)
            utils.unfreeze_net(P)
            utils.freeze_net(Q)
            if args.fix_emb:
           
                utils.freeze_net(F.parameters)
            # clip Q weights
            for p in Q.parameters():
                p.data.clamp_(args.clip_lower, args.clip_upper)
            F.zero_grad()
            P.zero_grad()
          
            features_en = F(**inputs_with_label)
            o_en_sent = P(features_en,**inputs_with_label)
            #l_en_sent = functional.nll_loss(o_en_sent, inputs_labels)
            l_en_sent = o_en_sent[0]
            loss = l_en_sent
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # l_en_sent.backward(retain_graph=True)
            # loss.backward
            loss.backward(retain_graph=True)
            o_en_ad = Q(features_en)
            l_en_ad = torch.mean(o_en_ad)
            (args.lambd*l_en_ad).backward(retain_graph=True)

            # training accuracy
            # _, pred = torch.max(o_en_sent, 1)
            # total += inputs_labels.size(0)
            # correct += (pred == inputs_labels).sum().item()

            features_ch = F(**inputs_ch_next)
            o_ch_ad = Q(features_ch)
            l_ch_ad = torch.mean(o_ch_ad)
            (-args.lambd*l_ch_ad).backward()

          

            # loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(F.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(P.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(Q.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                F.zero_grad()
                P.zero_grad()
                # model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    output_dir_F = os.path.join(args.output_dir, 'checkpointF-{}'.format(global_step))
                    if not os.path.exists(output_dir_F):
                        os.makedirs(output_dir_F)
                    # model_to_save = model.module if hasattr(model, 'module') else model 
                    # model_to_save.save_pretrained(output_dir)
                    F_model_to_save = F.module if hasattr(F, 'module') else F 
                    F_model_to_save.save_pretrained(output_dir_F)
                    tokenizer.save_pretrained(output_dir_F)
                    torch.save(args, os.path.join(output_dir_F, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_F)
                    
                    output_dir_Q = os.path.join(args.output_dir, 'checkpointQ-{}'.format(global_step))
                    if not os.path.exists(output_dir_Q):
                        os.makedirs(output_dir_Q)
                    Q_model_to_save = Q.module if hasattr(Q, 'module') else Q 
                    Q_model_to_save.save_pretrained(output_dir_Q)
                    tokenizer.save_pretrained(output_dir_Q)
                    torch.save(args, os.path.join(output_dir_Q, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_Q)
                    
                    output_dir_P = os.path.join(args.output_dir, 'checkpointP-{}'.format(global_step))
                    if not os.path.exists(output_dir_P):
                        os.makedirs(output_dir_P)
                    P_model_to_save = P.module if hasattr(P, 'module') else P
                    P_model_to_save.save_pretrained(output_dir_P)
                    tokenizer.save_pretrained(output_dir_P)
                    torch.save(args, os.path.join(output_dir_P, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_P)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
        
        # end of epoch
        # logger.info('Ending epoch {}'.format(n_epoch+1))
        # logs
        if sum_en_q[0] > 0:
            logger.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
            logger.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
        # evaluate
        #logger.info('Training Accuracy: {}%'.format(100.0*correct/total))
        # logger.info('Evaluating English Validation set:')
        # evaluate_languagefetect(args, yelp_valid_loader, F, P)
        # logger.info('Evaluating Foreign validation set:')
        # acc = evaluate_languagefetect(args, chn_valid_loader, F, P)
        # if acc > best_acc:
        #     logger.info(f'New Best Foreign validation accuracy: {acc}')
        #     best_acc = acc
        #     torch.save(F.state_dict(),
        #             '{}/netF_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(P.state_dict(),
        #             '{}/netP_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(Q.state_dict(),
        #             '{}/netQ_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        # logger.info('Evaluating Foreign test set:')
        # evaluate_languagefetect(args, chn_test_loader, F, P)

        # logger.info(f'Best Foreign validation accuracy: {best_acc}')


        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step   

def train_adan_con(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label,train_dataset_source_with_label,train_dataset_target_with_label,train_dataset_alignid_source,train_dataset_alignid_target, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    #随机采样语言辨别器所需要的数据，总数和训练数据一样，acs为四个数据集
    if args.train_data_sampler == 'random':
        train_sampler_source = RandomSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_source = SequentialSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_target = RandomSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_target = SequentialSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_s2t = RandomSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_s2t = SequentialSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_t2s = RandomSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_t2s = SequentialSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # else:
    #     train_sampler_twl = SequentialSampler(train_dataset_target_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_swl = SequentialSampler(train_dataset_source_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_twl = SequentialSampler(train_dataset_target_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # else:
    #     train_sampler_twl_alignid = SequentialSampler(train_dataset_alignid_target) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_swl_alignid = SequentialSampler(train_dataset_alignid_source) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_twl_alignid = SequentialSampler(train_dataset_alignid_target) if args.local_rank == -1 else DistributedSampler(train_dataset)


    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    train_dataloader_source = DataLoader(train_dataset_source_without_label, sampler=train_sampler_source, batch_size=args.train_batch_size)
    train_dataloader_target = DataLoader(train_dataset_target_without_label, sampler=train_sampler_target, batch_size=args.train_batch_size)
    train_dataloader_s2t = DataLoader(train_dataset_s2t_without_label, sampler=train_sampler_s2t, batch_size=args.train_batch_size)
    train_dataloader_t2s = DataLoader(train_dataset_t2s_without_label, sampler=train_sampler_t2s, batch_size=args.train_batch_size)
    train_dataloader_swl = DataLoader(train_dataset_source_with_label, sampler=train_sampler_swl, batch_size=args.train_batch_size)
    train_dataloader_twl = DataLoader(train_dataset_target_with_label, sampler=train_sampler_twl, batch_size=args.train_batch_size)

    train_dataloader_swl_alignid = DataLoader(train_dataset_alignid_source, sampler=train_sampler_swl_alignid, batch_size=args.train_batch_size)
    train_dataloader_twl_alignid = DataLoader(train_dataset_alignid_target, sampler=train_sampler_twl_alignid, batch_size=args.train_batch_size)
   

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
   


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform XABSA task with label list being {tag_list} (n_labels={num_tags})")

    # args.tfm_type = args.tfm_type.lower() 
    args.featuremodel = args.featuremodel.lower() 
    logger.info(f"Load pre-trained {args.featuremodel} model from `{args.model_name_or_path}`")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.featuremodel]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'

    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    # model.to(args.device)
        # models
    if args.featuremodel.lower() == 'adan_bert':
        F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
        F.to(args.device)
    # if args.model.lower() == 'dan':
    #     F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
    #     F.to(args.device)
    # elif opt.model.lower() == 'lstm':
    #     F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
    #             opt.bdrnn, opt.attn)
    # elif opt.model.lower() == 'cnn':
    #     F = CNNFeatureExtractor(vocab, opt.F_layers,
    #             opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception('Unknown model')
    config_class1, model_class1, tokenizer_class1 = MODEL_CLASSES[args.sentimentmodel]
    config1 = config_class1.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer1 = tokenizer_class1.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    P = mBertABSASentimentClassifier.from_pretrained(args.model_name_or_path,config=config1)

    num_tags_language = 1
    config_class2, model_class2, tokenizer_class2 = MODEL_CLASSES[args.languagedetectmodel]
    config2 = config_class2.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags_language, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer2 = tokenizer_class2.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    Q = mBERTABSALanguageDetector.from_pretrained(args.model_name_or_path,config=config2)
    F, P, Q = F.to(args.device), P.to(args.device), Q.to(args.device)
    optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()),
                           lr=args.learning_rate, eps=args.adam_epsilon)
    optimizerQ = optim.Adam(Q.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    schedulerQ = get_linear_schedule_with_warmup(optimizerQ, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
   
    




    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        F.train()
        P.train()
        Q.train()
        # training accuracy
        correct, total = 0, 0
        sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
        grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)


        # train_dataset_source_without_label=iter(train_dataloader_source)
        # train_dataset_target_without_label=iter(train_dataloader_target)
        # train_dataset_s2t_without_label=iter(train_dataloader_s2t)
        # train_dataset_t2s_without_label=iter(train_dataloader_t2s)
        
        #一致性损失
        label_kl_epoch_loss = 0
        unlabel_kl_epoch_loss = 0
        uda_epoch_loss = 0
        uda_time = 0
        num_uda = 0
        total_uda = 1e-8
        skipped_uda_batch = 0
        unlabel_skip = 0
        trans_skip = 0
        missing_skip = 0




        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_source = tqdm(train_dataloader_source, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target = tqdm(train_dataloader_target, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_s2t = tqdm(train_dataloader_s2t, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_t2s = tqdm(train_dataloader_t2s, desc="Iteration", disable=args.local_rank not in [-1, 0])
        #一致性
        epoch_iterator_source_label = tqdm(train_dataloader_swl, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target_label = tqdm(train_dataloader_twl, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        epoch_iterator_source_alignid = tqdm(train_dataloader_swl_alignid, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target_alignid = tqdm(train_dataloader_twl_alignid, desc="Iteration", disable=args.local_rank not in [-1, 0])
       
        # inputs_ch = ({'input_ids':      batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
        # inputs_ch_next = next(inputs_ch)
        # print(inputs_ch_next)
        q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
        q_inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
        #带标签
        q_inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_label)))
        inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_label)))
        inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_label)))
        q_inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_label)))
       # 带alignid
        q_inputs_ch_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
        inputs_ch_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
        
        inputs_en_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
        q_inputs_en_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
       
        # q_inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_target)))
        # inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_target)))
        # inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source)))
        # q_inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source)))
       
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs_with_label = {'input_ids':     batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    }
            inputs_labels = {'labels': batch['labels'].to(args.device)}
            #目标语言迭代数据
            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_ch_next = next(inputs_ch_iter)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
                inputs_ch_next = next(chn_train_iter)

            # Q iterations
            n_critic = args.n_critic
            if n_critic>0 and ((n_epoch==0 and step<=25) or (step%500==0)):
                n_critic = 10
            utils.freeze_net(F)
            utils.freeze_net(P)
            utils.unfreeze_net(Q)
            for qiter in range(n_critic):
                # clip Q weights
                for p in Q.parameters():
                    p.data.clamp_(args.clip_lower, args.clip_upper)
                Q.zero_grad()
                # get a minibatch of data
                try:
                    # labels are not used
                    #q_inputs_en, _ = next(train_dataset_source_without_label)
                    # q_inputs_en = next(train_dataset_source_without_label)
                    # q_inputs_en = ({'input_ids':   batch['input_ids'].to(args.device),
                    # 'attention_mask': batch['attention_mask'].to(args.device),
                    # 'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    # } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
                    q_inputs_en = next(q_inputs_en_iter) 
                except StopIteration:
                    # check if dataloader is exhausted
                    # yelp_train_iter_Q = iter(train_dataset_source_without_label)
                    # q_inputs_en, _ = next(train_dataset_source_without_label)
                    q_train_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
                    q_inputs_en = next(q_train_en_iter) 
                try:
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    #q_inputs_ch= next(train_dataset_target_without_label)
                    q_inputs_ch= next(q_inputs_ch_iter)
                except StopIteration:
                    # chn_train_iter_Q = iter(train_dataset_target_without_label)
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    # q_inputs_ch = next(train_dataset_target_without_label)
                    q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))

                    q_inputs_ch= next(q_inputs_ch_iter)

                features_en = F(**q_inputs_en)
                #print(features_en)
                o_en_ad = Q(features_en)
                l_en_ad = torch.mean(o_en_ad)
                (-l_en_ad).backward() 
                #使得特征提取层向着最大化损失的效果更新参数，强制迫使F学习语言的不变性
                # logging.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                # logging.info(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_en_q = (sum_en_q[0] + 1, sum_en_q[1] + l_en_ad.item())

                features_ch = F(**q_inputs_ch)
                o_ch_ad = Q(features_ch)
                l_ch_ad = torch.mean(o_ch_ad)
                l_ch_ad.backward()
                # logging.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                #logging.info(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_ch_q = (sum_ch_q[0] + 1, sum_ch_q[1] + l_ch_ad.item())

                optimizerQ.step()
                schedulerQ.step()  # Update learning rate schedule

            # F&P iteration
            utils.unfreeze_net(F)
            utils.unfreeze_net(P)
            utils.freeze_net(Q)
            if args.fix_emb:
                #冻结初始化的embeding参数（需要斟酌）
                utils.freeze_net(F.parameters)
            # clip Q weights
            for p in Q.parameters():
                p.data.clamp_(args.clip_lower, args.clip_upper)
            F.zero_grad()
            P.zero_grad()
            #对四个混合数据进行特征提取和方面词情感极性预测
            features_en = F(**inputs_with_label)
            o_en_sent = P(features_en,**inputs_with_label)
            #l_en_sent = functional.nll_loss(o_en_sent, inputs_labels)
            l_en_sent = o_en_sent[0]
            loss = l_en_sent
            print('loss:',loss)
            #新的
            loss.backward(retain_graph=True)
            o_en_ad = Q(features_en)
            l_en_ad = torch.mean(o_en_ad)
            (args.lambd*l_en_ad).backward(retain_graph=True)

            # training accuracy
            # _, pred = torch.max(o_en_sent, 1)
            # total += inputs_labels.size(0)
            # correct += (pred == inputs_labels).sum().item()

            features_ch = F(**inputs_ch_next)
            o_ch_ad = Q(features_ch)
            l_ch_ad = torch.mean(o_ch_ad)
            (-args.lambd*l_ch_ad).backward()
            # Translation-based consistency training 
            # uda_dataloader_repeat_1 = repeat_dataloader(train_dataloader_swl)
            # uda_dataloader_repeat_2 = repeat_dataloader(train_dataloader_twl)

            # uda_dataloader_repeat_3 = repeat_dataloader(train_dataloader_swl_alignid)
            # uda_dataloader_repeat_4 = repeat_dataloader(train_dataloader_twl_alignid)
          
            # uda_batch_1 = [t.to(config.device) for t in next(uda_dataloader_repeat_1)]
            # uda_batch_2 = [t.to(config.device) for t in next(uda_dataloader_repeat_2)]

            # uda_batch_3 = [t.to(config.device) for t in next(uda_dataloader_repeat_3)]
            # uda_batch_4 = [t.to(config.device) for t in next(uda_dataloader_repeat_4)]


            # # input_ids_1, attention_mask_1, token_type_ids_1, orig_to_tok_index_1, word_seq_len_1, label_ids_1, align_ids_1 = uda_batch_1
            # # input_ids_2, attention_mask_2, token_type_ids_2, orig_to_tok_index_2, word_seq_len_2, label_ids_2, align_ids_2 = uda_batch_2
            # inputs_1 = {
            #     'input_ids': uda_batch_1['input_ids'].to(args.device),
            #     'attention_mask': uda_batch_1['attention_mask'].to(args.device),
            #     'token_type_ids': uda_batch_1['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
            #     'labels': uda_batch_1['labels'].to(args.device)
            # }
            # inputs_2 = {
            #     'input_ids': uda_batch_2['input_ids'].to(args.device),
            #     'attention_mask': uda_batch_2['attention_mask'].to(args.device),
            #     'token_type_ids': uda_batch_2['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
            #     'labels': uda_batch_2['labels'].to(args.device)
            # }

            # inputs_3 = {
            #     'labels': uda_batch_3['labels'].to(args.device)
            # }

            # inputs_4 = {
            #     'labels': uda_batch_4['labels'].to(args.device)
            # }
            
            #一致性对齐
            try:
                inputs_ch_next2 = next(inputs_ch_iter2)  
     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter2 = iter(({
                    'labels':   batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
                inputs_ch_next2 = next(chn_train_iter2)

            try:
                inputs_en_next2 = next(inputs_en_iter2)  
      
            except:
                # # check if Chinese data is exhausted
                en_train_iter2 = iter(({ 
                    'labels':   batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
                inputs_en_next2 = next(en_train_iter2)

            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_ch_next1 = next(q_inputs_ch_iter1)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':   batch['labels'].to(args.device)
                    } for step, batch in enumerate(epoch_iterator_target_label)))
                inputs_ch_next1 = next(chn_train_iter1)

            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_en_next1 = next(q_inputs_en_iter1)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                en_train_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':   batch['labels'].to(args.device)
                    } for step, batch in enumerate(epoch_iterator_source_label)))
                inputs_en_next1 = next(en_train_iter1)
           


            if args.uda_weight > 0.0:
                # _, unlabel_e_scores, _ = model(words=input_ids, word_seq_lens=word_seq_len, orig_to_tok_index=orig_to_tok_index,
                #                            input_mask=attention_mask, labels=label_ids, output_emission=True)
                # _, trans_e_scores, _ = model(words=input_ids2, word_seq_lens=word_seq_len2, orig_to_tok_index=orig_to_tok_index2,
                #                            input_mask=attention_mask2, labels=label_ids2, output_emission=True)
                #源语言的分数
                features_en = F(**inputs_en_next1)
                o_en_sent = P(features_en,**inputs_en_next1)
                unlabel_e_scores=o_en_sent[1]

                 #目标语言的分数
                features_foreign = F(**inputs_ch_next1)
                o_foreign_sent = P(features_foreign,**inputs_ch_next1)
                trans_e_scores=o_foreign_sent[1]
               

                max_entity = 10
                # unlabel_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids, batch_e_scores=unlabel_e_scores, 
                #                                              max_entity=max_entity, config=config)
                # trans_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids2, batch_e_scores=trans_e_scores, 
                #                                            max_entity=max_entity, config=config)
                unlabel_entity_probs = tok_score_2_span_prob(args,batch_align_ids=inputs_en_next2['labels'], batch_e_scores=unlabel_e_scores, 
                                                             max_entity=max_entity,config=config)
                trans_entity_probs = tok_score_2_span_prob(args,batch_align_ids=inputs_ch_next2['labels'], batch_e_scores=trans_e_scores, 
                                                           max_entity=max_entity,config=config)


                if (unlabel_entity_probs is not None) and (trans_entity_probs is not None):

                    unlabel_entity_probs_list = []
                    trans_entity_probs_list = []
                    print(len(unlabel_entity_probs))
                    for batch_id in range(len(unlabel_entity_probs)):                    
                        # Discard ENTIRE SENTENCE if entity is missing
                        unlabel_prob_sent = []
                        trans_prob_sent = []
                        no_missing = True
                        for entity_id in range(max_entity):
                            if unlabel_entity_probs[batch_id][entity_id].nelement() != 0 and trans_entity_probs[batch_id][entity_id].nelement() != 0:
                                print(1)
                                unlabel_prob_sent.append(unlabel_entity_probs[batch_id][entity_id])
                                trans_prob_sent.append(trans_entity_probs[batch_id][entity_id])
                            elif unlabel_entity_probs[batch_id][entity_id].nelement() != trans_entity_probs[batch_id][entity_id].nelement():
                                print(2)
                                missing_skip +=1
                                no_missing = False
                        if no_missing:
                            print(3)
                            print('unlabel_prob_sent',unlabel_prob_sent)
                            unlabel_entity_probs_list.extend(unlabel_prob_sent)
                            trans_entity_probs_list.extend(trans_prob_sent)
                    print('unlabel_entity_probs_list',unlabel_entity_probs_list)
                    unlabel_entity_probs = torch.stack(unlabel_entity_probs_list)
                    trans_entity_probs = torch.stack(trans_entity_probs_list)
                    assert unlabel_entity_probs.shape[0] == trans_entity_probs.shape[0]

                    trans_entity_log_probs = torch.log(trans_entity_probs)

                    # UDA loss masking
                    uda_loss_mask = torch.max(unlabel_entity_probs[:,:-1], dim=-1)[0] > args.uda_threshold
                    uda_loss_mask = uda_loss_mask.type(torch.float32)
                    uda_loss_mask = uda_loss_mask.to(args.device)

                    # KL
                    KL = torch.nn.KLDivLoss(reduction='none') # use batchmean instead of mean to align with math definition
                    # Use bidirectional KL
                    uda_loss = (torch.sum(KL(trans_entity_log_probs, unlabel_entity_probs), dim=-1) \
                              + torch.sum(KL(torch.log(unlabel_entity_probs), trans_entity_probs), dim=-1)) / 2
                    
                    uda_loss = torch.sum(uda_loss * uda_loss_mask, dim=-1) / torch.max(torch.sum(uda_loss_mask, dim=-1), torch.tensor(1.).to(args.device))
                    print('uda_loss:',uda_loss)
                    #uda_loss.backward
                    loss = loss + uda_loss * args.uda_weight

                    num_uda += torch.sum(uda_loss_mask, dim=-1).item()
                    total_uda += uda_loss_mask.shape[0]

                    uda_epoch_loss += uda_loss.item()
                else:
                    skipped_uda_batch += 1
                    if unlabel_entity_probs is None:
                        unlabel_skip += 1
                    if trans_entity_probs is None:
                        trans_skip += 1
           
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

     



            # loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(F.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(P.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(Q.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                F.zero_grad()
                P.zero_grad()
                # model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    output_dir_F = os.path.join(args.output_dir, 'checkpointF-{}'.format(global_step))
                    if not os.path.exists(output_dir_F):
                        os.makedirs(output_dir_F)
                    # model_to_save = model.module if hasattr(model, 'module') else model 
                    # model_to_save.save_pretrained(output_dir)
                    F_model_to_save = F.module if hasattr(F, 'module') else F 
                    F_model_to_save.save_pretrained(output_dir_F)
                    tokenizer.save_pretrained(output_dir_F)
                    torch.save(args, os.path.join(output_dir_F, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_F)
                    
                    output_dir_Q = os.path.join(args.output_dir, 'checkpointQ-{}'.format(global_step))
                    if not os.path.exists(output_dir_Q):
                        os.makedirs(output_dir_Q)
                    Q_model_to_save = Q.module if hasattr(Q, 'module') else Q 
                    Q_model_to_save.save_pretrained(output_dir_Q)
                    tokenizer.save_pretrained(output_dir_Q)
                    torch.save(args, os.path.join(output_dir_Q, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_Q)
                    
                    output_dir_P = os.path.join(args.output_dir, 'checkpointP-{}'.format(global_step))
                    if not os.path.exists(output_dir_P):
                        os.makedirs(output_dir_P)
                    P_model_to_save = P.module if hasattr(P, 'module') else P
                    P_model_to_save.save_pretrained(output_dir_P)
                    tokenizer.save_pretrained(output_dir_P)
                    torch.save(args, os.path.join(output_dir_P, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_P)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
        
        # end of epoch
        # logger.info('Ending epoch {}'.format(n_epoch+1))
        # logs
        if sum_en_q[0] > 0:
            logger.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
            logger.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
        # evaluate
        #logger.info('Training Accuracy: {}%'.format(100.0*correct/total))
        # logger.info('Evaluating English Validation set:')
        # evaluate_languagefetect(args, yelp_valid_loader, F, P)
        # logger.info('Evaluating Foreign validation set:')
        # acc = evaluate_languagefetect(args, chn_valid_loader, F, P)
        # if acc > best_acc:
        #     logger.info(f'New Best Foreign validation accuracy: {acc}')
        #     best_acc = acc
        #     torch.save(F.state_dict(),
        #             '{}/netF_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(P.state_dict(),
        #             '{}/netP_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(Q.state_dict(),
        #             '{}/netQ_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        # logger.info('Evaluating Foreign test set:')
        # evaluate_languagefetect(args, chn_test_loader, F, P)

        # logger.info(f'Best Foreign validation accuracy: {best_acc}')


        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step   


def train_adan_con_xlmr(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label,train_dataset_source_with_label,train_dataset_target_with_label,train_dataset_alignid_source,train_dataset_alignid_target, model, tokenizer):
    """ Train the model """

    if args.local_rank in [-1, 0]:
       tb_writer = SummaryWriter()

    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    args.train_batch_size_xlm = args.per_gpu_train_batch_size * max(1, args.n_gpu)*2
   
    if args.train_data_sampler == 'random':
        train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler = SequentialSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    
    #随机采样语言辨别器所需要的数据，总数和训练数据一样，acs为四个数据集
    if args.train_data_sampler == 'random':
        train_sampler_source = RandomSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_source = SequentialSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_target = RandomSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_target = SequentialSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_s2t = RandomSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_s2t = SequentialSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    if args.train_data_sampler == 'random':
        train_sampler_t2s = RandomSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    else:
        train_sampler_t2s = SequentialSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

    #按平行语料来训练语言辨别器
    # train_sampler_source = SequentialSampler(train_dataset_source_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
 
    # train_sampler_target = SequentialSampler(train_dataset_target_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
   
    # train_sampler_s2t = SequentialSampler(train_dataset_s2t_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

    # train_sampler_t2s = SequentialSampler(train_dataset_t2s_without_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
       
 #不带alignid
    # if args.train_data_sampler == 'random':
    #     train_sampler_swl = RandomSampler(train_dataset_source_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # else:
    #     train_sampler_swl = SequentialSampler(train_dataset_source_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # if args.train_data_sampler == 'random':
    #     train_sampler_twl = RandomSampler(train_dataset_target_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # else:
    #     train_sampler_twl = SequentialSampler(train_dataset_target_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_swl = SequentialSampler(train_dataset_source_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_twl = SequentialSampler(train_dataset_target_with_label) if args.local_rank == -1 else DistributedSampler(train_dataset)

#带alignid
    # if args.train_data_sampler == 'random':
    #     train_sampler_swl_alignid = RandomSampler(train_dataset_alignid_source) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # else:
    #     train_sampler_swl_alignid = SequentialSampler(train_dataset_alignid_source) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # if args.train_data_sampler == 'random':
    #     train_sampler_twl_alignid = RandomSampler(train_dataset_alignid_target) if args.local_rank == -1 else DistributedSampler(train_dataset)
    # else:
    #     train_sampler_twl_alignid = SequentialSampler(train_dataset_alignid_target) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_swl_alignid = SequentialSampler(train_dataset_alignid_source) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_sampler_twl_alignid = SequentialSampler(train_dataset_alignid_target) if args.local_rank == -1 else DistributedSampler(train_dataset)


    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)
    #构建批量的语言辨别器所需要的数据
    train_dataloader_source = DataLoader(train_dataset_source_without_label, sampler=train_sampler_source, batch_size=args.train_batch_size)
    train_dataloader_target = DataLoader(train_dataset_target_without_label, sampler=train_sampler_target, batch_size=args.train_batch_size)
    train_dataloader_s2t = DataLoader(train_dataset_s2t_without_label, sampler=train_sampler_s2t, batch_size=args.train_batch_size)
    train_dataloader_t2s = DataLoader(train_dataset_t2s_without_label, sampler=train_sampler_t2s, batch_size=args.train_batch_size)
    #不带alignid
    train_dataloader_swl = DataLoader(train_dataset_source_with_label, sampler=train_sampler_swl, batch_size=args.train_batch_size_xlm)
    train_dataloader_twl = DataLoader(train_dataset_target_with_label, sampler=train_sampler_twl, batch_size=args.train_batch_size_xlm)

    #带alignid
    train_dataloader_swl_alignid = DataLoader(train_dataset_alignid_source, sampler=train_sampler_swl_alignid, batch_size=args.train_batch_size_xlm)
    train_dataloader_twl_alignid = DataLoader(train_dataset_alignid_target, sampler=train_sampler_twl_alignid, batch_size=args.train_batch_size_xlm)
   

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    #独立三个模型的参数
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                   args.train_batch_size * args.gradient_accumulation_steps * (torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform XABSA task with label list being {tag_list} (n_labels={num_tags})")

    #加载特征提取器
    # args.tfm_type = args.tfm_type.lower() 
    args.featuremodel = args.featuremodel.lower() 
    logger.info(f"Load pre-trained {args.featuremodel} model from `{args.model_name_or_path}`")

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.featuremodel]
    config = config_class.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer = tokenizer_class.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'

    # model = model_class.from_pretrained(args.model_name_or_path, config=config)
    # model.to(args.device)
    #新增语言辨别器和特征提取层
        # models
    if args.featuremodel.lower() == 'adan_xml':
        # F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
        F = DANFeatureExtractorXLM.from_pretrained(args.model_name_or_path,config=config)        
        F.to(args.device)
    # if args.model.lower() == 'dan':
    #     F = DANFeatureExtractor.from_pretrained(args.model_name_or_path,config=config)
    #     F.to(args.device)
    # elif opt.model.lower() == 'lstm':
    #     F = LSTMFeatureExtractor(vocab, opt.F_layers, opt.hidden_size, opt.dropout,
    #             opt.bdrnn, opt.attn)
    # elif opt.model.lower() == 'cnn':
    #     F = CNNFeatureExtractor(vocab, opt.F_layers,
    #             opt.hidden_size, opt.kernel_num, opt.kernel_sizes, opt.dropout)
    else:
        raise Exception('Unknown model')
    config_class1, model_class1, tokenizer_class1 = MODEL_CLASSES[args.sentimentmodel]
    config1 = config_class1.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer1 = tokenizer_class1.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    P = XLMABSASentimentClassifier.from_pretrained(args.model_name_or_path,config=config1)

    #加载语言辨别器 其中分辨源语言或者是目标语言（标签只有两种）
    num_tags_language = 1
    config_class2, model_class2, tokenizer_class2 = MODEL_CLASSES[args.languagedetectmodel]
    config2 = config_class2.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=num_tags_language, id2label=idx2tag, label2id=tag2idx
    )
    # logger.info(f"config info: \n {config}")
    tokenizer2 = tokenizer_class2.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
        do_lower_case=args.do_lower_case
    )
    Q = XLMABSALanguageDetector.from_pretrained(args.model_name_or_path,config=config2)
    F, P, Q = F.to(args.device), P.to(args.device), Q.to(args.device)
    optimizer = optim.Adam(list(F.parameters()) + list(P.parameters()),
                           lr=args.learning_rate, eps=args.adam_epsilon)
    optimizerQ = optim.Adam(Q.parameters(), lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    schedulerQ = get_linear_schedule_with_warmup(optimizerQ, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
   
    
    # optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, F, no_grad=no_grad)
    # optimizer1 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler1 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, Q, no_grad=no_grad)
    # optimizer2 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler2 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # optimizer_grouped_parameters1 = get_optimizer_grouped_parameters(args, P, no_grad=no_grad)
    # optimizer3 = AdamW(optimizer_grouped_parameters1, lr=args.learning_rate, eps=args.adam_epsilon)
    # # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    # scheduler3 = get_linear_schedule_with_warmup(optimizer1, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)




    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        #新增FPQ的训练
        F.train()
        P.train()
        Q.train()
        # training accuracy
        correct, total = 0, 0
        sum_en_q, sum_ch_q = (0, 0.0), (0, 0.0)
        grad_norm_p, grad_norm_q = (0, 0.0), (0, 0.0)



        # train_dataset_source_without_label=iter(train_dataloader_source)
        # train_dataset_target_without_label=iter(train_dataloader_target)
        # train_dataset_s2t_without_label=iter(train_dataloader_s2t)
        # train_dataset_t2s_without_label=iter(train_dataloader_t2s)
        
    
        label_kl_epoch_loss = 0
        unlabel_kl_epoch_loss = 0
        uda_epoch_loss = 0
        uda_time = 0
        num_uda = 0
        total_uda = 1e-8
        skipped_uda_batch = 0
        unlabel_skip = 0
        trans_skip = 0
        missing_skip = 0




        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_source = tqdm(train_dataloader_source, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target = tqdm(train_dataloader_target, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_s2t = tqdm(train_dataloader_s2t, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_t2s = tqdm(train_dataloader_t2s, desc="Iteration", disable=args.local_rank not in [-1, 0])
       
        epoch_iterator_source_label = tqdm(train_dataloader_swl, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target_label = tqdm(train_dataloader_twl, desc="Iteration", disable=args.local_rank not in [-1, 0])
        
        epoch_iterator_source_alignid = tqdm(train_dataloader_swl_alignid, desc="Iteration", disable=args.local_rank not in [-1, 0])
        epoch_iterator_target_alignid = tqdm(train_dataloader_twl_alignid, desc="Iteration", disable=args.local_rank not in [-1, 0])
       
        # inputs_ch = ({'input_ids':      batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
        # inputs_ch_next = next(inputs_ch)
        # print(inputs_ch_next)
        q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
        inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
        q_inputs_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))

        q_inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_label)))
        inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_label)))
        inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_label)))
        q_inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_label)))
       # 带alignid
        q_inputs_ch_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
        inputs_ch_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
        
        inputs_en_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
        q_inputs_en_iter2 = iter(({'labels':batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
       
        # q_inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_target)))
        # inputs_ch_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_target)))
        # inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source)))
        # q_inputs_en_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
        #             'attention_mask': batch['attention_mask'].to(args.device),
        #             'token_type_ids': batch['token_type_ids'].to(args.device),'labels':batch['labels'].to(args.device) if args.tfm_type != 'xlmr' else None
        #             } for step, batch in enumerate(epoch_iterator_source)))
       
        
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs_with_label = {'input_ids':     batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    }
            inputs_labels = {'labels': batch['labels'].to(args.device)}
     
            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_ch_next = next(inputs_ch_iter)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))
                inputs_ch_next = next(chn_train_iter)

            # Q iterations
            n_critic = args.n_critic
            if n_critic>0 and ((n_epoch==0 and step<=25) or (step%500==0)):
                n_critic = 10
            utils.freeze_net(F)
            utils.freeze_net(P)
            utils.unfreeze_net(Q)
            for qiter in range(n_critic):
                # clip Q weights
                for p in Q.parameters():
                    p.data.clamp_(args.clip_lower, args.clip_upper)
                Q.zero_grad()
                # get a minibatch of data
                try:
                    # labels are not used
                    #q_inputs_en, _ = next(train_dataset_source_without_label)
                    # q_inputs_en = next(train_dataset_source_without_label)
                    # q_inputs_en = ({'input_ids':   batch['input_ids'].to(args.device),
                    # 'attention_mask': batch['attention_mask'].to(args.device),
                    # 'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    # } for step, batch in enumerate(epoch_iterator_source))  # Chinese labels are not used
                    q_inputs_en = next(q_inputs_en_iter) 
                except StopIteration:
                    # check if dataloader is exhausted
                    # yelp_train_iter_Q = iter(train_dataset_source_without_label)
                    # q_inputs_en, _ = next(train_dataset_source_without_label)
                    q_train_en_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_source)))
                    q_inputs_en = next(q_train_en_iter) 
                try:
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    #q_inputs_ch= next(train_dataset_target_without_label)
                    q_inputs_ch= next(q_inputs_ch_iter)
                except StopIteration:
                    # chn_train_iter_Q = iter(train_dataset_target_without_label)
                    # q_inputs_ch, _ = next(train_dataset_target_without_label)
                    # q_inputs_ch = next(train_dataset_target_without_label)
                    q_inputs_ch_iter = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                    } for step, batch in enumerate(epoch_iterator_target)))

                    q_inputs_ch= next(q_inputs_ch_iter)

                features_en = F(**q_inputs_en)
                #print(features_en)
                o_en_ad = Q(features_en)
                l_en_ad = torch.mean(o_en_ad)
                (-l_en_ad).backward() 
         
                # logging.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                # logging.info(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_en_q = (sum_en_q[0] + 1, sum_en_q[1] + l_en_ad.item())

                features_ch = F(**q_inputs_ch)
                o_ch_ad = Q(features_ch)
                l_ch_ad = torch.mean(o_ch_ad)
                l_ch_ad.backward()
                # logging.debug(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                #logging.info(f'Q grad norm: {Q.net[1].weight.grad.data.norm()}')
                logging.info(f'Q grad norm: {Q.classifier.weight.grad.data.norm()}')
                sum_ch_q = (sum_ch_q[0] + 1, sum_ch_q[1] + l_ch_ad.item())

                optimizerQ.step()
                schedulerQ.step()  # Update learning rate schedule

            # F&P iteration
            utils.unfreeze_net(F)
            utils.unfreeze_net(P)
            utils.freeze_net(Q)
            if args.fix_emb:
  
                utils.freeze_net(F.parameters)
            # clip Q weights
            for p in Q.parameters():
                p.data.clamp_(args.clip_lower, args.clip_upper)
            F.zero_grad()
            P.zero_grad()

            features_en = F(**inputs_with_label)
            o_en_sent = P(features_en,**inputs_with_label)
            #l_en_sent = functional.nll_loss(o_en_sent, inputs_labels)
            l_en_sent = o_en_sent[0]
            loss = l_en_sent
            print('loss:',loss)
            #新的
            loss.backward(retain_graph=True)
            o_en_ad = Q(features_en)
            l_en_ad = torch.mean(o_en_ad)
            (args.lambd*l_en_ad).backward(retain_graph=True)

            # training accuracy
            # _, pred = torch.max(o_en_sent, 1)
            # total += inputs_labels.size(0)
            # correct += (pred == inputs_labels).sum().item()

            features_ch = F(**inputs_ch_next)
            o_ch_ad = Q(features_ch)
            l_ch_ad = torch.mean(o_ch_ad)
            (-args.lambd*l_ch_ad).backward()
            # Translation-based consistency training 
            # uda_dataloader_repeat_1 = repeat_dataloader(train_dataloader_swl)
  
            try:
                inputs_ch_next2 = next(inputs_ch_iter2)  
     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter2 = iter(({
                    'labels':   batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_target_alignid)))
                inputs_ch_next2 = next(chn_train_iter2)

            try:
                inputs_en_next2 = next(inputs_en_iter2)  
      
            except:
                # # check if Chinese data is exhausted
                en_train_iter2 = iter(({ 
                    'labels':   batch['labels'].to(args.device) 
                    } for step, batch in enumerate(epoch_iterator_source_alignid)))
                inputs_en_next2 = next(en_train_iter2)

            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_ch_next1 = next(q_inputs_ch_iter1)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                chn_train_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':   batch['labels'].to(args.device)
                    } for step, batch in enumerate(epoch_iterator_target_label)))
                inputs_ch_next1 = next(chn_train_iter1)

            try:
                # inputs_ch = ({'input_ids':   batch['input_ids'].to(args.device),
                #     'attention_mask': batch['attention_mask'].to(args.device),
                #     'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                #     } for step, batch in enumerate(epoch_iterator_target))  # Chinese labels are not used
                inputs_en_next1 = next(q_inputs_en_iter1)  
                #inputs_ch_next2 = next(inputs_ch)     
            except:
                # # check if Chinese data is exhausted
                en_train_iter1 = iter(({'input_ids':   batch['input_ids'].to(args.device),
                    'attention_mask': batch['attention_mask'].to(args.device),
                    'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                    'labels':   batch['labels'].to(args.device)
                    } for step, batch in enumerate(epoch_iterator_source_label)))
                inputs_en_next1 = next(en_train_iter1)
           


            if args.uda_weight > 0.0:
                # _, unlabel_e_scores, _ = model(words=input_ids, word_seq_lens=word_seq_len, orig_to_tok_index=orig_to_tok_index,
                #                            input_mask=attention_mask, labels=label_ids, output_emission=True)
                # _, trans_e_scores, _ = model(words=input_ids2, word_seq_lens=word_seq_len2, orig_to_tok_index=orig_to_tok_index2,
                #                            input_mask=attention_mask2, labels=label_ids2, output_emission=True)
                features_en = F(**inputs_en_next1)
                o_en_sent = P(features_en,**inputs_en_next1)
                unlabel_e_scores=o_en_sent[1]

                features_foreign = F(**inputs_ch_next1)
                o_foreign_sent = P(features_foreign,**inputs_ch_next1)
                trans_e_scores=o_foreign_sent[1]
               

                max_entity = 10
                # unlabel_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids, batch_e_scores=unlabel_e_scores, 
                #                                              max_entity=max_entity, config=config)
                # trans_entity_probs = tok_score_2_span_prob(batch_align_ids=align_ids2, batch_e_scores=trans_e_scores, 
                #                                            max_entity=max_entity, config=config)
                #print('input',inputs_en_next2['labels'][0])
                unlabel_entity_probs = tok_score_2_span_prob(args,batch_align_ids=inputs_en_next2['labels'], batch_e_scores=unlabel_e_scores, 
                                                             max_entity=max_entity,config=config)
                trans_entity_probs = tok_score_2_span_prob(args,batch_align_ids=inputs_ch_next2['labels'], batch_e_scores=trans_e_scores, 
                                                           max_entity=max_entity,config=config)


                if (unlabel_entity_probs is not None) and (trans_entity_probs is not None):

                    unlabel_entity_probs_list = []
                    trans_entity_probs_list = []
                    #print(len(unlabel_entity_probs)) 8*num_gpu 
                    for batch_id in range(len(unlabel_entity_probs)):                    
                        # Discard ENTIRE SENTENCE if entity is missing
                        unlabel_prob_sent = []
                        trans_prob_sent = []
                        no_missing = True
                        for entity_id in range(max_entity):
                            if unlabel_entity_probs[batch_id][entity_id].nelement() != 0 and trans_entity_probs[batch_id][entity_id].nelement() != 0:
                                #print(1)
                                unlabel_prob_sent.append(unlabel_entity_probs[batch_id][entity_id])
                                trans_prob_sent.append(trans_entity_probs[batch_id][entity_id])
                            elif unlabel_entity_probs[batch_id][entity_id].nelement() != trans_entity_probs[batch_id][entity_id].nelement():
                                #print(2)
                                missing_skip +=1
                                no_missing = False
                        if no_missing:
                            #print(3)
                            #print(unlabel_prob_sent)
                            unlabel_entity_probs_list.extend(unlabel_prob_sent)
                            trans_entity_probs_list.extend(trans_prob_sent)
                    #print(unlabel_entity_probs_list)
                    #print('unlabel_entity_probs_list',unlabel_entity_probs_list)
                    unlabel_entity_probs = torch.stack(unlabel_entity_probs_list)
                    trans_entity_probs = torch.stack(trans_entity_probs_list)
                    assert unlabel_entity_probs.shape[0] == trans_entity_probs.shape[0]

                    trans_entity_log_probs = torch.log(trans_entity_probs)

                    # UDA loss masking
                    uda_loss_mask = torch.max(unlabel_entity_probs[:,:-1], dim=-1)[0] > args.uda_threshold
                    uda_loss_mask = uda_loss_mask.type(torch.float32)
                    uda_loss_mask = uda_loss_mask.to(args.device)

                    # KL
                    KL = torch.nn.KLDivLoss(reduction='none') # use batchmean instead of mean to align with math definition
                    # Use bidirectional KL
                    uda_loss = (torch.sum(KL(trans_entity_log_probs, unlabel_entity_probs), dim=-1) \
                              + torch.sum(KL(torch.log(unlabel_entity_probs), trans_entity_probs), dim=-1)) / 2
                    
                    uda_loss = torch.sum(uda_loss * uda_loss_mask, dim=-1) / torch.max(torch.sum(uda_loss_mask, dim=-1), torch.tensor(1.).to(args.device))
                    print('uda_loss:',uda_loss)
                    #uda_loss.backward
                    loss = loss + uda_loss * args.uda_weight

                    num_uda += torch.sum(uda_loss_mask, dim=-1).item()
                    total_uda += uda_loss_mask.shape[0]

                    uda_epoch_loss += uda_loss.item()
                else:
                    skipped_uda_batch += 1
                    if unlabel_entity_probs is None:
                        unlabel_skip += 1
                    if trans_entity_probs is None:
                        trans_skip += 1
           
            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            # l_en_sent.backward(retain_graph=True)
            # loss.backward
            loss.backward(retain_graph=True)
           


            # loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                # torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(F.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(P.parameters(), args.max_grad_norm)
                torch.nn.utils.clip_grad_norm_(Q.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                F.zero_grad()
                P.zero_grad()
                # model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after X steps 
                    output_dir_F = os.path.join(args.output_dir, 'checkpointF-{}'.format(global_step))
                    if not os.path.exists(output_dir_F):
                        os.makedirs(output_dir_F)
                    # model_to_save = model.module if hasattr(model, 'module') else model 
                    # model_to_save.save_pretrained(output_dir)
                    F_model_to_save = F.module if hasattr(F, 'module') else F 
                    F_model_to_save.save_pretrained(output_dir_F)
                    tokenizer.save_pretrained(output_dir_F)
                    torch.save(args, os.path.join(output_dir_F, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_F)
                    
                    output_dir_Q = os.path.join(args.output_dir, 'checkpointQ-{}'.format(global_step))
                    if not os.path.exists(output_dir_Q):
                        os.makedirs(output_dir_Q)
                    Q_model_to_save = Q.module if hasattr(Q, 'module') else Q 
                    Q_model_to_save.save_pretrained(output_dir_Q)
                    tokenizer.save_pretrained(output_dir_Q)
                    torch.save(args, os.path.join(output_dir_Q, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_Q)
                    
                    output_dir_P = os.path.join(args.output_dir, 'checkpointP-{}'.format(global_step))
                    if not os.path.exists(output_dir_P):
                        os.makedirs(output_dir_P)
                    P_model_to_save = P.module if hasattr(P, 'module') else P
                    P_model_to_save.save_pretrained(output_dir_P)
                    tokenizer.save_pretrained(output_dir_P)
                    torch.save(args, os.path.join(output_dir_P, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir_P)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break
        
        # end of epoch
        # logger.info('Ending epoch {}'.format(n_epoch+1))
        # logs
        if sum_en_q[0] > 0:
            logger.info(f'Average English Q output: {sum_en_q[1]/sum_en_q[0]}')
            logger.info(f'Average Foreign Q output: {sum_ch_q[1]/sum_ch_q[0]}')
        # evaluate
        #logger.info('Training Accuracy: {}%'.format(100.0*correct/total))
        # logger.info('Evaluating English Validation set:')
        # evaluate_languagefetect(args, yelp_valid_loader, F, P)
        # logger.info('Evaluating Foreign validation set:')
        # acc = evaluate_languagefetect(args, chn_valid_loader, F, P)
        # if acc > best_acc:
        #     logger.info(f'New Best Foreign validation accuracy: {acc}')
        #     best_acc = acc
        #     torch.save(F.state_dict(),
        #             '{}/netF_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(P.state_dict(),
        #             '{}/netP_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        #     torch.save(Q.state_dict(),
        #             '{}/netQ_epoch_{}.pth'.format(args.model_save_file, n_epoch))
        # logger.info('Evaluating Foreign test set:')
        # evaluate_languagefetect(args, chn_test_loader, F, P)

        # logger.info(f'Best Foreign validation accuracy: {best_acc}')


        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

        # save a checkpoint when each epoch ends after a specific epoch
        '''
        n_epoch_name = n_epoch
        if n_epoch_name >= args.train_begin_saving_epoch: 
            output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(n_epoch_name))
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model 
            model_to_save.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            torch.save(args, os.path.join(output_dir, 'training_args.bin'))
            logger.info("Saving model checkpoint to %s", output_dir)
        '''

    if args.local_rank in [-1, 0]:
        tb_writer.close()

    return global_step, train_loss / global_step   


def train_kd(args, train_dataset, model, tokenizer):
    """ Train the model with the soft labels """
    # prepare dataloader
    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size)

    # compute total update steps
    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps) + 1
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    if args.freeze_bottom_layer >= 0:
        no_grad = ["embeddings"] + ["layer." + str(layer_i) + "." for layer_i in range(12) if layer_i < args.freeze_bottom_layer]
    else:
        no_grad = None
    optimizer_grouped_parameters = get_optimizer_grouped_parameters(args, model, no_grad=no_grad)
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    # scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)

    # Train!
    logger.info("***** Running training with soft labels *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total optimization steps = %d", t_total)
    logger.info("  Learning rate = %f", args.learning_rate)
    logger.info("  Model saved path = %s", args.output_dir)

    global_step = 0
    train_loss, logging_loss = 0.0, 0.0
    model.zero_grad()

    train_iterator = trange(int(args.num_train_epochs), desc="Epoch", disable=args.local_rank not in [-1, 0])
    set_seed(args)  # Added here for reproductibility (even between python 2 and 3)
    for n_epoch, _ in enumerate(train_iterator):
        epoch_train_loss = 0.0
        epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=args.local_rank not in [-1, 0])
        for step, batch in enumerate(epoch_iterator):
            model.train()
            # batch = tuple(t.to(args.device) for t in batch)
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'teacher_probs':  batch['teacher_probs'].to(args.device)}
            ouputs = model(**inputs)
            loss = ouputs[0]  # model outputs are always tuple in pytorch-transformers (see doc)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            train_loss += loss.item()
            epoch_train_loss += loss.item()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                # update trainable parameters every gradient_accumulation_steps
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

                optimizer.step()
                scheduler.step()  # Update learning rate schedule
                model.zero_grad()
                global_step += 1

                if args.local_rank in [-1, 0] and global_step % args.save_steps == 0 and global_step >= args.train_begin_saving_step:
                    # Save model checkpoint per each N steps after 1000 steps 
                    output_dir = os.path.join(args.output_dir, 'checkpoint-{}'.format(global_step))
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model 
                    model_to_save.save_pretrained(output_dir)
                    tokenizer.save_pretrained(output_dir)
                    torch.save(args, os.path.join(output_dir, 'training_args.bin'))
                    logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < args.max_steps < global_step:
                epoch_iterator.close()
                break
        if 0 < args.max_steps < global_step:
            train_iterator.close()
            break

        logger.info(f"Current epoch train loss: {epoch_train_loss:.5f}")

    return global_step, train_loss / global_step


def evaluate_languagedetect(args, loader, F, P):
    F.eval()
    P.eval()
    it = iter(loader)
    correct = 0
    total = 0
    confusion = ConfusionMeter(args.num_labels)
    with torch.no_grad():
        for inputs, targets in tqdm(it):
            outputs = P(F(inputs))
            _, pred = torch.max(outputs, 1)
            confusion.add(pred.data, targets.data)
            total += targets.size(0)
            correct += (pred == targets).sum().item()
    accuracy = correct / total
    logger.info('Accuracy on {} samples: {}%'.format(total, 100.0*accuracy))
    logger.debug(confusion.conf)
    return accuracy


def evaluate(args, eval_dataset, model, idx2tag, mode, step=None):
    """
    Perform evaluation on a given `eval_datset` 
    """
    eval_output_dir = args.output_dir
    # eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            outputs = model(**inputs)
            # logits: (bsz, seq_len, label_size)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        num_eval_steps += 1

        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / num_eval_steps
    # argmax operation over the last dimension
    pred_labels = np.argmax(preds, axis=-1)

    result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write= {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")
    """
    output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
    with open(output_eval_file, "w") as writer:
        #logger.info("***** %s results *****" % mode)
        for key in sorted(result.keys()):
            if 'eval_loss' in key:
                logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        #logger.info("***** %s results *****" % mode)
    """
    return results

def evaluate_adan(args, eval_dataset, model_F,model_P, idx2tag, mode, step=None):
    """
    Perform evaluation on a given `eval_datset` 
    """
    eval_output_dir = args.output_dir
    # eval_dataset, eval_evaluate_label_ids = load_and_cache_examples(args, args.task_name, tokenizer, mode=mode)
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    eval_loss, num_eval_steps = 0.0, 0
    preds, pred_labels, gold_labels = None, None, None
    results = {}

    for batch in tqdm(eval_dataloader, desc="Evaluating"):
        model_F.eval()
        model_P.eval()
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         batch['labels'].to(args.device)}
            inputs_no_labels = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None
                      }
            outputs_F = model_F(**inputs_no_labels)
            # outputs = model(**inputs)
            outputs = model_P(outputs_F,**inputs)
            # logits: (bsz, seq_len, label_size)
            tmp_eval_loss, logits = outputs[:2]
            eval_loss += tmp_eval_loss.mean().item()

        num_eval_steps += 1

        if preds is None:
            preds = logits.cpu().numpy()
            gold_labels = inputs['labels'].cpu().numpy()
        else:
            preds = np.append(preds, logits.cpu().numpy(), axis=0)
            gold_labels = np.append(gold_labels, inputs['labels'].cpu().numpy(), axis=0)
    
    eval_loss = eval_loss / num_eval_steps
    # argmax operation over the last dimension
    pred_labels = np.argmax(preds, axis=-1)

    result, ground_truth, predictions = compute_metrics_absa(pred_labels, gold_labels, idx2tag, args.tagging_schema)
    
    result['eval_loss'] = eval_loss
    results.update(result)

    if mode == 'test':
        file_to_write= {'results': results, 'labels': ground_truth, 'preds': predictions}
        file_name_to_write = f'{args.saved_model_dir}/{args.src_lang}-{args.tgt_lang}-preds-{step}.pickle'
        pickle.dump(file_to_write, open(file_name_to_write, 'wb'))
        logger.info(f"Write predictions to {file_name_to_write}")
    """
    output_eval_file = os.path.join(eval_output_dir, "%s_results.txt" % mode)
    with open(output_eval_file, "w") as writer:
        #logger.info("***** %s results *****" % mode)
        for key in sorted(result.keys()):
            if 'eval_loss' in key:
                logger.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        #logger.info("***** %s results *****" % mode)
    """
    return results


def get_teacher_probs(args, dataset, model_class, teacher_model_path):
    teacher_model = model_class.from_pretrained(teacher_model_path)
    teacher_model.to(args.device)
    
    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # compute logits for the dataset using the model!
    logger.info(f"***** Compute logits for [{args.tgt_lang}] using the model {teacher_model_path} *****")
    logger.info("  Num examples = %d", len(dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    preds = None
    teacher_model.eval()
    for batch in eval_dataloader:
        with torch.no_grad():
            inputs = {'input_ids':      batch['input_ids'].to(args.device),
                      'attention_mask': batch['attention_mask'].to(args.device),
                      'token_type_ids': batch['token_type_ids'].to(args.device) if args.tfm_type != 'xlmr' else None,
                      'labels':         None}
            outputs = teacher_model(**inputs)
            logits = outputs[0]

        # nb_eval_steps += 1
        preds = logits.detach() if preds is None else torch.cat((preds, logits.detach()), dim=0) # dataset_len x max_seq_len x label_len

    preds = torch.nn.functional.softmax(preds, dim=-1)
    
    return preds


def get_multi_teacher_probs(args, dataset, model_class):
    teacher_paths = args.trained_teacher_paths 
    
    # obtain all preds
    all_preds = []
    for teacher_path in teacher_paths:
        preds = get_teacher_probs(args, dataset, model_class, teacher_path)
        all_preds.append(preds)
    
    logger.info("Fuse the soft labels from three pre-trained models")
    combined_preds = 1/3 * all_preds[0] + 1/3 * all_preds[1] + 1/3 * all_preds[2]

    return combined_preds
    # 提取步数的函数
def extract_step(checkpoint_name):
    match = re.search(r'checkpoint.*-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else None
    
def main():
    # --------------------------------
    # Prepare the tags, env etc.
    args = init_args()
    print("\n", "="*30, f"NEW EXP ({args.src_lang} -> {args.tgt_lang} for {args.exp_type}", "="*30, "\n")

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(f"Output directory ({args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome.")

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device_id = 1
        # device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        os.environ['MASTER_ADDR'] = args.MASTER_ADDR
        os.environ['MASTER_PORT'] = args.MASTER_PORT
        torch.distributed.init_process_group(backend='nccl', rank=args.local_rank, world_size=1)
        args.n_gpu = 1
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    # not using 16-bits training
    logger.info(f"Process rank: {args.local_rank}, device: {device}, n_gpu: {args.n_gpu}") 
    logger.info(f"Distributed training: {bool(args.local_rank != -1)}, 16-bits training: False")

    # Set seed
    set_seed(args)

    # Set up task and the label
    tag_list, tag2idx, idx2tag = get_tag_vocab('absa', args.tagging_schema, args.label_path)
    num_tags = len(tag_list)
    args.num_labels = num_tags
    logger.info(f"Perform XABSA task with label list being {tag_list} (n_labels={num_tags})")

    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()

    # hard-code lr and bs based on parameter search
    if args.tfm_type == 'mbert':
        lr, batch_size = 5e-5, 16  
    elif args.tfm_type == 'xlmr':
        lr, batch_size = 4e-5, 25
    # args.learning_rate = lr
    # args.per_gpu_train_batch_size = batch_size
    logger.info(f"We hard-coded set lr={args.learning_rate} and bs={args.per_gpu_train_batch_size}")
    

    # -----------------------------------------------------------------
    # Training process (train a model using the data and save the model)
    if args.do_train:
        logger.info("\n\n***** Prepare to conduct training  *****\n")

        # Set up model (from pre-trained tfms)
        args.tfm_type = args.tfm_type.lower() 
        logger.info(f"Load pre-trained {args.tfm_type} model from `{args.model_name_or_path}`")

        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        config = config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
        )
        # logger.info(f"config info: \n {config}")
        tokenizer = tokenizer_class.from_pretrained(
            args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
            do_lower_case=args.do_lower_case
        )
        # config.absa_type = args.absa_type  # type of task-specific layer, 'linear'

        model = model_class.from_pretrained(args.model_name_or_path, config=config)
        model.to(args.device)

        # Distributed and parallel training
        if args.local_rank != -1:
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                              output_device=args.local_rank,
                                                              find_unused_parameters=True)
        elif args.n_gpu > 1:
            model = torch.nn.DataParallel(model)

        # load the training dataset
        print()
        logger.info("Prepare training examples...")
        #print(args)
        train_dataset = build_or_load_dataset(args, tokenizer, mode='train')
        
        #不使用标签和训练
        train_dataset_source = build_or_load_dataset_languagedetect_source(args, tokenizer, mode='train')
        train_dataset_target = build_or_load_dataset_languagedetect_target(args, tokenizer, mode='train')
        train_dataset_source2target = build_or_load_dataset_languagedetect_source2target_codeswutch(args, tokenizer, mode='train')
        train_dataset_target2source = build_or_load_dataset_languagedetect_target2source_codeswitch(args, tokenizer, mode='train')
        train_dataset_source_with_label=build_or_load_dataset_languagedetect_source_label(args, tokenizer, mode='train')
        train_dataset_target_with_label=build_or_load_dataset_languagedetect_target_label(args, tokenizer, mode='train')
        
        train_dataset_alignid_source = build_or_load_dataset_alignid_source(args, tokenizer, mode='train')
        train_dataset_alignid_target = build_or_load_dataset_alignid_target(args, tokenizer, mode='train')
        # train_dataset_source_without_label = train_dataset_source.encodings
        # train_dataset_target_without_label = train_dataset_target.encodings
        # train_dataset_s2t_without_label = train_dataset_source2target.encodings
        # train_dataset_t2s_without_label = train_dataset_target2source.encodings
        train_dataset_source_without_label = train_dataset_source
        train_dataset_target_without_label = train_dataset_target
        train_dataset_s2t_without_label = train_dataset_source2target
        train_dataset_t2s_without_label = train_dataset_target2source

        #加入对抗平均网络的训练
        # _, _ = train_adan(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label, model, tokenizer)


        #加入对抗平均网络和一致性训练
        _, _ = train_adan_con(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label,train_dataset_source_with_label,train_dataset_target_with_label,train_dataset_alignid_source,train_dataset_alignid_target, model, tokenizer)
        # _, _ = train_adan(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label, model, tokenizer)
        # _, _ = train_adan_con_xlmr(args, train_dataset,train_dataset_source_without_label,train_dataset_target_without_label,train_dataset_s2t_without_label,train_dataset_t2s_without_label,train_dataset_source_with_label,train_dataset_target_with_label,train_dataset_alignid_source,train_dataset_alignid_target, model, tokenizer)


        # begin training!
        # _, _ = train(args, train_dataset, model, tokenizer)
        
        # Create output directory if needed
        if not os.path.exists(args.output_dir) and args.local_rank in [-1, 0]:
            os.mkdir(args.output_dir)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(args.output_dir)
            tokenizer.save_pretrained(args.output_dir)
            torch.save(args, os.path.join(args.output_dir, 'training_args.bin')) 


    if args.do_distill:
        print()
        logger.info("********** Training with KD **********")
        config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        tokenizer = tokenizer_class.from_pretrained(args.model_name_or_path)
        if args.exp_type == 'macs_kd':
            dataset = build_or_load_dataset(args, tokenizer, mode='unlabeled_mtl')
        else:
            dataset = build_or_load_dataset(args, tokenizer, mode='unlabeled')
        
        # teacher_model_path = get_teacher_model_path(args)
        if args.exp_type == 'acs_kd_s':
            teacher_model_path = args.trained_teacher_paths
            teacher_probs = get_teacher_probs(args, dataset, model_class, teacher_model_path)
        # use the three combinations as multi-teacher distill
        elif args.exp_type == 'acs_kd_m':
            teacher_probs = get_multi_teacher_probs(args, dataset, model_class)
        # MTL version of distillation
        elif args.exp_type == 'macs_kd':
            teacher_model_path = args.trained_teacher_paths
            teacher_probs = get_teacher_probs(args, dataset, model_class, teacher_model_path)
        
        train_dataset = XABSAKDDataset(dataset.encodings, teacher_probs)
        logger.info(f"Obtained the soft labels")

        # initialize the model with the translated data
        # student_model_path = f"outputs/{args.tfm_type}-{args.src_lang}-{args.tgt_lang}-smt/checkpoint"
        student_model_path = args.student_model_path

        s_config_class, s_model_class, _ = MODEL_CLASSES[args.tfm_type]
        config = s_config_class.from_pretrained(
            args.config_name if args.config_name else args.model_name_or_path,
            num_labels=num_tags, id2label=idx2tag, label2id=tag2idx
        )
        #student_model_path = args.model_name_or_path   # not good
        
        student_model = s_model_class.from_pretrained(student_model_path)
        student_model.to(args.device)
        logger.info(f"We initialize the student model with {student_model_path}")
        
        _, _ = train_kd(args, train_dataset, student_model, tokenizer)


    # -----------------------------------------------------------------
    # Evaluation process (whether it is supervised setting or zero-shot)
    if args.do_eval:
        exp_type = args.exp_type
        logger.info("\n\n***** Prepare to conduct evaluation *****\n")
        logger.info(f"We are evaluating for *{args.tgt_lang}* under *{args.exp_type}* setting...")
        
        dev_results, test_results = {}, {}
        best_f1, best_checkpoint, best_global_step = -999999.0, None, None
        all_checkpoints, global_steps = [], []
        # 假设有两个列表分别存储不同类型的检查点
        all_checkpointsF, all_checkpointsP= [],[]


        # find the dir containing trained model, different dirs under different settings
        # if the model is multilingual, we will only use one target language for the output dir
        if 'mtl' in exp_type:
            one_tgt_lang = 'fr'
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{one_tgt_lang}-{exp_type}"
        
        elif exp_type == 'zero_shot':
            saved_model_dir = f"outputs/{args.tfm_type}-{args.src_lang}-{args.src_lang}-supervised"
            if not os.path.exists(saved_model_dir):
                raise Exception("No trained models can be found!")
        
        else:
            saved_model_dir = args.output_dir
        
        args.saved_model_dir = saved_model_dir
        # print(args.saved_model_dir)

        # retrieve all the saved checkpoints for model selection
        # for f in os.listdir(saved_model_dir):
        #     sub_dir = os.path.join(saved_model_dir, f)
        #     if os.path.isdir(sub_dir):
        #         all_checkpoints.append(sub_dir)
        # logger.info(f"We will perform validation on the following checkpoints: {all_checkpoints}")

        #分别加载模型
        # 遍历保存模型的主目录
        for f in os.listdir(args.output_dir):
            sub_dir = os.path.join(args.output_dir, f)
            if os.path.isdir(sub_dir):
                # 根据子目录名称分类存储到不同的列表
                if f.startswith('checkpointF-'):
                    all_checkpointsF.append(sub_dir)
                elif f.startswith('checkpointP-'):
                    all_checkpointsP.append(sub_dir)

        logger.info(f"We will perform validation on the following checkpoints for F: {all_checkpointsF}")
        logger.info(f"We will perform validation on the following checkpoints for P: {all_checkpointsP}")
        
        # load the dev and test dataset
        # config_class, model_class, tokenizer_class = MODEL_CLASSES[args.tfm_type]
        # config = config_class.from_pretrained(all_checkpoints[0])
        # tokenizer = tokenizer_class.from_pretrained(all_checkpoints[0])
        #加载F和P
        config_class1, model_class1, tokenizer_class1 = MODEL_CLASSES[args.featuremodel]
        config1 = config_class1.from_pretrained(all_checkpointsF[0])
        tokenizer1 = tokenizer_class1.from_pretrained(all_checkpointsF[0])

        config_class2, model_class2, tokenizer_class2 = MODEL_CLASSES[args.sentimentmodel]
        config2 = config_class2.from_pretrained(all_checkpointsP[0])
        tokenizer2 = tokenizer_class2.from_pretrained(all_checkpointsP[0])
        logger.info("Load DEV dataset...")
        dev_dataset = build_or_load_dataset(args, tokenizer1, mode='dev')
        logger.info("Load TEST dataset...")
        test_dataset = build_or_load_dataset(args, tokenizer1, mode='test')

        # # 提取步数的函数
        # def extract_step(checkpoint_name):
        #     match = re.search(r'checkpoint.*-(\d+)', checkpoint_name)
        #     return int(match.group(1)) if match else None

        # 将检查点按步数排序并同步
        checkpointsF_sorted = sorted(all_checkpointsF, key=extract_step)
        checkpointsP_sorted = sorted(all_checkpointsP, key=extract_step)

        # 同步步数的检查点
        synchronized_checkpoints = [(f, p) for f, p in zip(checkpointsF_sorted, checkpointsP_sorted) if extract_step(f) == extract_step(p)]

        # global_steps_checkpoint = []

        # for checkpointF,checkpointP in zip(all_checkpointsF,all_checkpointsP):
        #     global_step = checkpointF.split('-')[-1] if len(checkpointF) > 1 else ""
            # only perform evaluation at the specific epochs
        for checkpointF, checkpointP in synchronized_checkpoints:
            # global_step = extract_step(checkpointF)
            # if global_step is None:
            #     continue
            global_step = checkpointF.split('-')[-1] if len(checkpointF) > 1 else ""

            eval_begin, eval_end = args.eval_begin_end.split('-')
            if int(eval_begin) <= int(global_step) < int(eval_end):
                global_steps.append(global_step)

                # reload the model and conduct inference
                logger.info(f"\nLoad the trained model from {checkpointF}...")
                logger.info(f"\nLoad the trained model from {checkpointP}...")
                # model = model_class.from_pretrained(checkpoint, config=config)
                # model.to(args.device)
                #加载F和P
                model1 = model_class1.from_pretrained(checkpointF, config=config1)
                model1.to(args.device)
                model2 = model_class2.from_pretrained(checkpointP, config=config2)
                model2.to(args.device)

                #dev_result = evaluate(args, dev_dataset, model, idx2tag, mode='dev')
                #评估
                dev_result = evaluate_adan(args, dev_dataset, model1,model2, idx2tag, mode='dev')
                # regard the micro-f1 as the criteria of model selection
                metrics = 'micro_f1'
                if dev_result[metrics] > best_f1:
                    best_f1 = dev_result[metrics]
                    best_checkpointF = checkpointF
                    best_checkpointP = checkpointP
                    best_global_step = global_step

                # add the global step to the name of these metrics for recording
                # 'micro_f1' --> 'micro_f1_1000'
                dev_result = dict((k + '_{}'.format(global_step), v) for k, v in dev_result.items())
                dev_results.update(dev_result)

                # test_result = evaluate(args, test_dataset, model, idx2tag, mode='test', step=global_step)
                # test_result = evaluate(args, test_dataset, model1,model2, idx2tag, mode='test', step=global_step)
                test_result = evaluate_adan(args, test_dataset, model1,model2, idx2tag, mode='test', step=global_step)
                test_result = dict((k + '_{}'.format(global_step), v) for k, v in test_result.items())
                test_results.update(test_result)
    
        # print test results over last few steps
        logger.info(f"\n\nThe best checkpoint is {best_checkpointF}and{best_checkpointP}")
        best_step_metric = f"{metrics}_{best_global_step}"
        print(f"F1 scores on test set: {test_results[best_step_metric]:.4f}")

        print("\n* Results *:  Dev  /  Test  \n")
        metric_names = ['micro_f1', 'precision', 'recall', 'eval_loss']
        for gstep in global_steps:
            print(f"Step-{gstep}:")
            for name in metric_names:
                name_step = f'{name}_{gstep}'
                print(f"{name:<10}: {dev_results[name_step]:.4f} / {test_results[name_step]:.4f}", sep='  ')
            print()

        results_log_dir = './results_log'
        if not os.path.exists(results_log_dir):
            os.mkdir(results_log_dir)
        log_file_path = f"{results_log_dir}/{args.tfm_type}-{args.exp_type}-{args.tgt_lang}.txt"
        write_results_to_log(log_file_path, test_results[best_step_metric], args, dev_results, test_results, global_steps)


if __name__ == '__main__':
    main()

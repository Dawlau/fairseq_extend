import math
from argparse import Namespace
from typing import Iterator

import torch
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.data import encoders

from fairseq.logging import metrics
from comet import download_model, load_from_checkpoint
from torch.nn.parameter import Parameter

@register_criterion("comet")
class CometCriterion(FairseqCriterion):
    def __init__(self, task):
        super().__init__(task)
        self.tokenizer = encoders.build_tokenizer(Namespace(
            tokenizer='moses'
        ))
        self.tgt_dict = task.target_dictionary
        self.comet_path = download_model("Unbabel/wmt22-comet-da")
        self.comet = load_from_checkpoint(self.comet_path)


    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return iter([])


    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T
        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens, prev_output_tokens = sample["target"], sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_output_tokens, tgt_tokens)
        #get loss only on tokens, not on lengths
        outs = outputs["word_ins"].get("out", None)
        masks = outputs["word_ins"].get("mask", None)

        loss, reward = self._compute_loss(src_tokens, outs, tgt_tokens, masks)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.detach(),
            "nll_loss": loss.detach(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
            "reward": reward.detach()
        }

        return loss, sample_size, logging_output

    def decode(self, toks, escape_unk=False):
        with torch.no_grad():
            s = self.tgt_dict.string(
                toks.int().cpu(),
                "@@ ",
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=(
                    "UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"
                )
            )
            s = self.tokenizer.decode(s)
        return s

    def _compute_loss(self, src, outputs, targets, masks=None):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len
        """
        batch_size = outputs.size(0)
        seq_len = outputs.size(1)
        vocab_size = outputs.size(2)

        with torch.no_grad():
            probs = torch.nn.functional.softmax(outputs, dim=-1).view(-1, vocab_size)
            sample_idx = torch.multinomial(probs, 1, replacement=True).view(batch_size, seq_len)

            data = []

            for source_sentence, generated_sentence, target_sentence in zip(src, sample_idx, targets):
                data.append({
                    "src": self.decode(source_sentence),
                    "mt": self.decode(generated_sentence),
                    "ref": self.decode(target_sentence),
                })

            reward = torch.tensor(self.comet.predict(data, batch_size=8, gpus=1)[0]).unsqueeze(-1).expand(-1, seq_len).to(outputs.device)

        if masks is not None:
           outputs, targets = outputs[masks], targets[masks]
           reward, sample_idx = reward[masks], sample_idx[masks]

        log_probs = torch.log_softmax(outputs, dim=-1)
        log_probs = log_probs.gather(1, sample_idx.unsqueeze(-1)).squeeze()
        loss = -log_probs*reward
        loss = loss.mean()

        return loss, reward.mean()

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        reward_sum= sum(log.get("reward", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)

        # we divide by log(2) to convert the loss from base e to base 2
        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar("reward", reward_sum / sample_size, sample_size, round=3)
import random
import torch
import numpy as np
from torch import nn, optim
from torch.nn import functional as F
from transformers import AutoTokenizer, AutoModel
from transformers.modeling_outputs import QuestionAnsweringModelOutput


class LSTMModel(nn.Module):
    def __init__(
        self,
        model_name,
        model_config,
        sep_token_id,
        mask_token_id,
        masking_prob,
        masking_ratio,
    ):
        super().__init__()
        self.sep_token_id = sep_token_id
        self.mask_token_id = mask_token_id
        self.masking_prob = masking_prob
        self.masking_ratio = masking_ratio

        self.model_name = model_name
        self.backbone_model = AutoModel.from_pretrained(model_name, config=model_config)
        self.lstm = nn.LSTM(
            input_size=model_config.hidden_size,
            hidden_size=1024,
            num_layers=3,
            dropout=0.3,
            bidirectional=True,
            batch_first=True,
        )
        self.fc = nn.Linear(1024 * 2, 2, bias=True)

        for name, param in self.lstm.named_parameters():
            if "weight" in name:
                nn.init.kaiming_normal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        nn.init.kaiming_normal_(self.fc.weight)
        if self.fc.bias is not None:
            nn.init.zeros_(self.fc.bias)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        start_positions=None,
        end_positions=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):

        outputs = self.backbone_model(
            input_ids=self._random_masking(input_ids)
            if (self.training and self.masking_prob != 0.0)
            else input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]
        sequence_output, (_, _) = self.lstm(sequence_output)
        logits = self.fc(sequence_output)

        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutput(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def _random_masking(self, input_ids):
        masked_input_ids = input_ids.clone()
        for input_id in masked_input_ids:
            if random.random() < self.masking_prob:
                sep_idx = np.where(input_id.cpu().numpy() == self.sep_token_id)[0][0]
                mask_cnt = 0
                while mask_cnt < int(sep_idx * self.masking_ratio):
                    mask_idx = random.randint(1, sep_idx - 1)
                    if input_id[mask_idx] != self.mask_token_id:
                        input_id[mask_idx] = self.mask_token_id
                        mask_cnt += 1
                        if (
                            input_id[mask_idx + 1] != self.mask_token_id
                            and input_id[mask_idx + 1] != self.sep_token_id
                        ):
                            input_id[mask_idx + 1] = self.mask_token_id
                            mask_cnt += 1

        return masked_input_ids

import os
import sys

import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import SequentialSampler, DataLoader
from tqdm import tqdm
import numpy as np



sys.path.append("../")
sys.path.append("../../")
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# from utilities.metrics import uncertainty_metrics
from src.metrics import uncertainty_metrics

"""
Code from https://github.com/gpleiss/temperature_scaling/blob/master/temperature_scaling.py
"""


class ModelWithTemperature(nn.Module):
    """
    A thin decorator, which wraps a model with temperature scaling
    model (nn.Module):
        A classification neural network
        NB: Output of the neural network should be the classification logits,
            NOT the softmax (or log softmax)!
    """

    def __init__(self, model):
        super(ModelWithTemperature, self).__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, input):
        logits = self.model(input)
        return self.temperature_scale(logits)

    def temperature_scale(self, logits):
        """
        Perform temperature scaling on logits
        """
        # Expand temperature to match the size of logits
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def eval_nll(self):
        loss = self.nll_criterion(self.temperature_scale(self.logits), self.labels)
        loss.backward()
        return loss

    # This function probably should live outside of this class, but whatever
    def set_temperature(self, valid_loader, device, model_type):
        """
        Tune the tempearature of the model (using the validation set).
        We're going to set it to optimize NLL.
        valid_loader (DataLoader): validation set loader
        """
        # self.cuda()
        self.nll_criterion = nn.CrossEntropyLoss().cuda()
        self.ece_criterion = _ECELoss().cuda()
        self.device = device
        self.model_type = model_type

        # First: collect all the logits and labels for the validation set
        logits_list = []
        labels_list = []
        # with torch.no_grad():
        #     for input, label in valid_loader:
        #         input = input.cuda()
        #         logits = self.model(input)
        #         logits_list.append(logits)
        #         labels_list.append(label)
        #     logits = torch.cat(logits_list).cuda()
        #     labels = torch.cat(labels_list).cuda()

        eval_loss = 0.0
        nb_eval_steps = 0
        preds = None
        out_label_ids = None

        for batch in tqdm(valid_loader, desc="Evaluating"):
            self.model.eval()
            batch = tuple(t.to(self.device) for t in batch)

            with torch.no_grad():
                inputs = {"input_ids": batch[0], "attention_mask": batch[1], "labels": batch[3]}
                if self.model_type != "distilbert":
                    inputs["token_type_ids"] = (
                        batch[2] if self.model_type in ["bert", "xlnet", "albert"] else None
                    )  # XLM, DistilBERT, RoBERTa, and XLM-RoBERTa don't use segment_ids
                outputs = self.model(**inputs)
                tmp_eval_loss, logits = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if preds is None:
                preds = logits.detach().cpu().numpy()
                out_label_ids = inputs["labels"].detach().cpu().numpy()
            else:
                preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                out_label_ids = np.append(out_label_ids, inputs["labels"].detach().cpu().numpy(), axis=0)

        # Calculate NLL and ECE before temperature scaling
        self.logits = torch.tensor(preds)
        self.labels = torch.tensor(out_label_ids)
        before_temperature_nll = self.nll_criterion(self.logits, self.labels).item()
        before_temperature_ece = self.ece_criterion(self.logits, self.labels).item()
        print('Before temperature - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

        # Next: optimize the temperature w.r.t. NLL
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        optimizer.step(self.eval_nll)

        # Calculate NLL and ECE after temperature scaling
        after_temperature_nll = self.nll_criterion(self.temperature_scale(self.logits), self.labels).item()
        after_temperature_ece = self.ece_criterion(self.temperature_scale(self.logits), self.labels).item()
        print('Optimal temperature: %.3f' % self.temperature.item())
        print('After temperature - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return self


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).
    The input to this loss is the logits of a model, NOT the softmax scores.
    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:
    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |
    We then return a weighted average of the gaps, based on the number
    of samples in each bin
    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """

    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece


def test_temp_scaling(eval_dataset, args, model):
    # Now we're going to wrap the model with a decorator that adds temperature scaling
    tmp_model = ModelWithTemperature(model)

    args.eval_batch_size = args.per_gpu_eval_batch_size * max(1, args.n_gpu)
    eval_sampler = SequentialSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # Tune the model temperature, and save the results
    tmp_model.set_temperature(eval_dataloader, args.device, args.model_type)
    model_filename = os.path.join(args.output_dir, 'model_with_temperature.pth')
    torch.save(model.state_dict(), model_filename)
    print('Temperature scaled model saved to %s' % model_filename)
    print('Done!')

    temperature = tmp_model.temperature
    logits = tmp_model.logits
    labels = tmp_model.labels
    calibration_scores = uncertainty_metrics(logits, labels.detach().cpu().numpy())
    calibration_scores.update({"temperature": float(temperature)})
    return calibration_scores, logits

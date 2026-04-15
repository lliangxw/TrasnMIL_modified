import sys
import numpy as np
import inspect
import importlib
import random
import pandas as pd

#---->
from MyOptimizer import create_optimizer
from MyLoss import create_loss
from MyLoss.attention_loss import attention_loss
from utils.utils import cross_entropy_torch

#---->
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics

#---->
import pytorch_lightning as pl


class  ModelInterface(pl.LightningModule):

    #---->init
    def __init__(self, model, loss, optimizer, **kargs):
        super(ModelInterface, self).__init__()

        self.val_step_outputs = []
        self.test_step_outputs = []

        self.save_hyperparameters()
        self.load_model()
        self.loss = create_loss(loss)
        self.optimizer = optimizer
        self.n_classes = model.n_classes
        self.log_path = kargs['log']

        #---->acc
        self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        self.data_test = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
        #---->Metrics
        if self.n_classes > 2: 
            self.AUROC = torchmetrics.AUROC(task="multiclass", num_classes = self.n_classes, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task='multiclass', num_classes = self.n_classes,
                                                                           average='micro'),
                                                     torchmetrics.CohenKappa(task='multiclass', num_classes = self.n_classes),
                                                     torchmetrics.F1Score(task='multiclass', num_classes = self.n_classes,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task='multiclass', average = 'macro',
                                                                         num_classes = self.n_classes),
                                                     torchmetrics.Precision(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes),
                                                     torchmetrics.Specificity(task='multiclass', average = 'macro',
                                                                            num_classes = self.n_classes)])
        else : 
            self.AUROC = torchmetrics.AUROC(task="binary", num_classes=2, average = 'macro')
            metrics = torchmetrics.MetricCollection([torchmetrics.Accuracy(task="binary", num_classes = 2,
                                                                           average = 'micro'),
                                                     torchmetrics.CohenKappa(task="binary",num_classes = 2),
                                                     torchmetrics.F1Score(task="binary", num_classes = 2,
                                                                     average = 'macro'),
                                                     torchmetrics.Recall(task="binary", average = 'macro',
                                                                         num_classes = 2),
                                                     torchmetrics.Precision(task="binary",average = 'macro',
                                                                            num_classes = 2)])
        self.valid_metrics = metrics.clone(prefix = 'val_')
        self.test_metrics = metrics.clone(prefix = 'test_')

        #--->random
        self.shuffle = kargs['data'].data_shuffle
        self.count = 0


    #---->remove v_num
    def get_progress_bar_dict(self):
        # don't show the version number
        items = super().get_progress_bar_dict()
        items.pop("v_num", None)
        return items

    def get_attention_loss(self, attns, heatmap, has_heatmap, header_attention, head_fusion='attn'):
        return attention_loss(attns, heatmap, has_heatmap, header_attention, head_fusion)

    # def training_step(self, batch, batch_idx):
    #     #---->inference
    #     data, label, slide_id, heatmap, has_heatmap = batch
    #     results_dict, attns, h = self.model(data=data, label=label)
    #     logits = results_dict['logits']
    #     Y_prob = results_dict['Y_prob']
    #     Y_hat = results_dict['Y_hat']

    #     #---->loss
    #     # loss = self.loss(logits, label) + self.get_attention_loss(attns, heatmap, has_heatmap)

    #     cls_loss = self.loss(logits, label)
    #     attn_loss = self.get_attention_loss(attns, heatmap, has_heatmap)
    #     loss = cls_loss + attn_loss

    #     print(f"cls_loss: {cls_loss.item():.6f}")
    #     print(f"attn_loss: {attn_loss.item():.6f}")
    #     print(f"total_loss: {loss.item():.6f}")

    #     #---->acc log
    #     Y_hat = int(Y_hat)
    #     Y = int(label)
    #     self.data[Y]["count"] += 1
    #     self.data[Y]["correct"] += (Y_hat == Y)

    #     return {'loss': loss} 


    # def training_epoch_end(self, training_step_outputs):
    #     for c in range(self.n_classes):
    #         count = self.data[c]["count"]
    #         correct = self.data[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('training : class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]

    def on_train_epoch_end(self): ### Replacing the deleted training_epoch_end(self, training_step_outputs)
        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = None if count == 0 else float(correct) / count
            print(f"training : class {c}: acc {acc}, correct {correct}/{count}")
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

    
    # def training_step(self, batch, batch_idx):
    #     loss = ...
    #     # if you need to aggregate stuff:
    #     self.train_step_outputs.append(loss.detach())
    #     self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=False)
    #     return loss

    def training_step(self, batch, batch_idx):  ### Replacing the deleted training_step(self, batch, batch_idx)

        lambda_attn = 0.0
        data, label, slide_id, heatmap, has_heatmap = batch
        results_dict, attns, h = self.model(data=data, label=label)
        logits = results_dict["logits"]
        Y_hat = results_dict["Y_hat"]

        # loss = self.loss(logits, label) + self.get_attention_loss(attns, heatmap, has_heatmap)

        cls_loss = self.loss(logits, label)
        if has_heatmap == 1:
            _fc2_p = self.model._fc2.weight.detach()
            header_attention = (h * _fc2_p).reshape((self.n_classes, 8, 64)).sum(dim=-1)[0,:]

            if not torch.isfinite(header_attention).all():
                print("header_attention invalid", header_attention[~torch.isfinite(header_attention)])
            if not torch.isfinite(h).all():
                print("h invalid", torch.isnan(h).sum(), torch.isinf(h).sum())

            attn_loss = self.get_attention_loss(attns, heatmap, has_heatmap, header_attention)
            attn_loss = attn_loss.to(dtype=torch.float32)
            loss = cls_loss + lambda_attn * attn_loss
        else:
            attn_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            loss = cls_loss

        print(f"cls_loss: {cls_loss.item():.6f}")
        if has_heatmap == 1:
            print(f"attn_loss: {attn_loss.item():.6f}")
        print(f"total_loss: {loss.item():.6f}")

        # per-class acc counter
        y_hat_int = int(Y_hat)
        y_int = int(label)
        self.data[y_int]["count"] += 1
        self.data[y_int]["correct"] += (y_hat_int == y_int)

        self.log("train_cls_loss", cls_loss, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        self.log("train_attn_loss", lambda_attn * attn_loss, prog_bar=False, on_step=True, on_epoch=False, logger=True)
        self.log("train_total_loss", loss, prog_bar=True, on_step=True, on_epoch=False, logger=True)
        return loss

    # def validation_step(self, batch, batch_idx):
    #     data, label, slide_id = batch
    #     results_dict = self.model(data=data, label=label)
    #     logits = results_dict['logits']
    #     Y_prob = results_dict['Y_prob']
    #     Y_hat = results_dict['Y_hat']

    #     incorrect_mask = Y_hat != label
    #     incorrect_ids = [slide_id[i] for i in range(len(slide_id)) if incorrect_mask[i]]
    #     incorrect_labels = [label[i] for i in range(len(slide_id)) if incorrect_mask[i]]

    #     # Print them (or save)
    #     for sid, slabel in zip(incorrect_ids, incorrect_labels):
    #         print(f"Incorrect prediction: {sid}, {slabel}")

    #     # # Optional: return for validation_epoch_end
    #     # return {"incorrect_ids": incorrect_ids}


    #     #---->acc log
    #     Y = int(label)
    #     self.data_test[Y]["count"] += 1
    #     self.data_test[Y]["correct"] += (Y_hat.item() == Y)

    #     return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def validation_step(self, batch, batch_idx): ### Replacing the deleted validation_step(self, batch, batch_idx)
        lambda_attn = 0.0
        data, label, slide_id, heatmap, has_heatmap = batch
        results_dict, attns, h = self.model(data=data, label=label)
        logits = results_dict["logits"]
        Y_prob = results_dict["Y_prob"]
        Y_hat = results_dict["Y_hat"]

        # logits = logits.detach().cpu()
        # Y_prob = Y_prob.detach().cpu()
        # Y_hat = Y_hat.detach().cpu()
        # label = label.detach().cpu()
        heatmap = heatmap.detach().cpu()
        attns = [attn.detach().cpu() for attn in attns]
        h = h.detach().cpu()

        incorrect_mask = (Y_hat != label)
        incorrect_ids = [slide_id[i] for i in range(len(slide_id)) if incorrect_mask[i]]
        incorrect_labels = [label[i] for i in range(len(slide_id)) if incorrect_mask[i]]

        for sid, slabel in zip(incorrect_ids, incorrect_labels):
            print(f"Incorrect prediction: {sid}, {slabel}")

        # per-class acc counter (note: you used data_test here)
        y_int = int(label)
        self.data_test[y_int]["count"] += 1
        self.data_test[y_int]["correct"] += (Y_hat.item() == y_int)

        out = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat, "label": label, "heatmap": heatmap, "has_heatmap": has_heatmap, "attns": attns, "h": h}
        self.val_step_outputs.append(out)
        return out


    # def validation_epoch_end(self, val_step_outputs):
    #     logits = torch.cat([x['logits'] for x in val_step_outputs], dim = 0)
    #     probs = torch.cat([x['Y_prob'] for x in val_step_outputs], dim = 0)
    #     max_probs = torch.stack([x['Y_hat'] for x in val_step_outputs])
    #     target = torch.stack([x['label'] for x in val_step_outputs], dim = 0)
        
    #     #---->
    #     self.log('val_loss', cross_entropy_torch(logits, target), prog_bar=True, on_epoch=True, logger=True) #### Commented

    #     # ### Added
    #     # print("validation process test")
    #     # print("validation loss = " + str(cross_entropy_torch(logits, target)))


    #     # self.log('auc', self.AUROC(probs, target.squeeze()), prog_bar=True, on_epoch=True, logger=True)
    #     self.log_dict(self.valid_metrics(max_probs.squeeze() , target.squeeze()),
    #                       on_epoch = True, logger = True)

    #     #---->acc log
    #     for c in range(self.n_classes):
    #         count = self.data_test[c]["count"]
    #         correct = self.data_test[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('validation : class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data_test = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
        
    #     #---->random, if shuffle data, change seed
    #     if self.shuffle == True:
    #         self.count = self.count+1
    #         random.seed(self.count*50)
    

    def on_validation_epoch_end(self): ### Replacing the deleted validation_epoch_end(self, val_step_outputs)
        if len(self.val_step_outputs) == 0:
            return

        lambda_attn = 0.0
        logits = torch.cat([x["logits"] for x in self.val_step_outputs], dim=0)
        probs = torch.cat([x["Y_prob"] for x in self.val_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.val_step_outputs])
        target = torch.stack([x["label"] for x in self.val_step_outputs], dim=0)

        cls_val_loss = cross_entropy_torch(logits, target).detach().cpu()
        attn_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
        attn_count = 0
        _fc2_p = self.model._fc2.weight.detach()

        for i in range(len(self.val_step_outputs)):
            output = self.val_step_outputs[i]
            has_heatmap = int(output["has_heatmap"])
            if has_heatmap == 1:
                h = output["h"]
                header_attention = (h * _fc2_p).reshape(self.n_classes, 8, 64).sum(dim=-1)[0, :]

                if not torch.isfinite(header_attention).all():
                    print("header_attention invalid", header_attention[~torch.isfinite(header_attention)])
                if not torch.isfinite(output["attns"][0]).all():
                    print("attn tensor invalid", output["attns"][0].shape,
                        output["attns"][0].min(), output["attns"][0].max(),
                        torch.isnan(output["attns"][0]).sum(), torch.isinf(output["attns"][0]).sum())

                sample_attn_loss = self.get_attention_loss(
                    output["attns"],
                    output["heatmap"],
                    has_heatmap,
                    header_attention,
                )
                sample_attn_loss = sample_attn_loss.to(dtype=torch.float32)
                if not torch.isfinite(sample_attn_loss):
                    print("Skipping non-finite validation attention loss for this sample")
                    continue
                attn_loss = attn_loss + sample_attn_loss
                attn_count += 1

        if attn_count > 0:
            attn_loss = attn_loss / attn_count

        print(f"validation cls_loss: {cls_val_loss.item():.6f}")
        if attn_count > 0:
            print(f"validation attn_loss: {attn_loss.item():.6f}")
        print(f"validation total_loss: {(cls_val_loss + lambda_attn * attn_loss).item():.6f}")
        self.log("val_cls_loss", cls_val_loss, prog_bar=False, on_epoch=True, logger=True)
        self.log("val_attn_loss", lambda_attn * attn_loss, prog_bar=False, on_epoch=True, logger=True)
        self.log("val_loss", cls_val_loss + lambda_attn * attn_loss, prog_bar=True, on_epoch=True, logger=True)
        self.log_dict(self.valid_metrics(max_probs.squeeze(), target.squeeze()),
                    on_epoch=True, logger=True)

        for c in range(self.n_classes):
            count = self.data_test[c]["count"]
            correct = self.data_test[c]["correct"]
            acc = None if count == 0 else float(correct) / count
            print(f"validation : class {c}: acc {acc}, correct {correct}/{count}")
        self.data_test = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        if self.shuffle:
            self.count += 1
            random.seed(self.count * 50)

        self.val_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = create_optimizer(self.optimizer, self.model)
        return [optimizer]

    # def test_step(self, batch, batch_idx):
    #     data, label, slide_id = batch
    #     results_dict = self.model(data=data, label=label)
    #     logits = results_dict['logits']
    #     Y_prob = results_dict['Y_prob']
    #     Y_hat = results_dict['Y_hat']

    #     #---->acc log
    #     Y = int(label)
    #     self.data[Y]["count"] += 1
    #     self.data[Y]["correct"] += (Y_hat.item() == Y)

    #     return {'logits' : logits, 'Y_prob' : Y_prob, 'Y_hat' : Y_hat, 'label' : label}

    def test_step(self, batch, batch_idx):  ### Replacing the deleted test_step(self, batch, batch_idx)
        data, label, slide_id, heatmap, has_heatmap = batch
        results_dict = self.model(data=data, label=label)
        logits = results_dict["logits"]
        Y_prob = results_dict["Y_prob"]
        Y_hat = results_dict["Y_hat"]

        y_int = int(label)
        self.data[y_int]["count"] += 1
        self.data[y_int]["correct"] += (Y_hat.item() == y_int)

        out = {"logits": logits, "Y_prob": Y_prob, "Y_hat": Y_hat, "label": label}
        self.test_step_outputs.append(out)
        return out

    # def test_epoch_end(self, output_results):
    #     probs = torch.cat([x['Y_prob'] for x in output_results], dim = 0)
    #     max_probs = torch.stack([x['Y_hat'] for x in output_results])
    #     target = torch.stack([x['label'] for x in output_results], dim = 0)
        
    #     #---->
    #     auc = self.AUROC(probs, target.squeeze())
    #     metrics = self.test_metrics(max_probs.squeeze() , target.squeeze())
    #     metrics['auc'] = auc
    #     for keys, values in metrics.items():
    #         print(f'{keys} = {values}')
    #         metrics[keys] = values.cpu().numpy()
    #     print()
    #     #---->acc log
    #     for c in range(self.n_classes):
    #         count = self.data[c]["count"]
    #         correct = self.data[c]["correct"]
    #         if count == 0: 
    #             acc = None
    #         else:
    #             acc = float(correct) / count
    #         print('class {}: acc {}, correct {}/{}'.format(c, acc, correct, count))
    #     self.data = [{"count": 0, "correct": 0} for i in range(self.n_classes)]
    #     #---->
    #     result = pd.DataFrame([metrics])
    #     result.to_csv(self.log_path / 'result.csv')

    def on_test_epoch_end(self): ### Replacing the deleted test_epoch_end(self, output_results)
        if len(self.test_step_outputs) == 0:
            return

        probs = torch.cat([x["Y_prob"] for x in self.test_step_outputs], dim=0)
        max_probs = torch.stack([x["Y_hat"] for x in self.test_step_outputs])
        target = torch.stack([x["label"] for x in self.test_step_outputs], dim=0)

        auc = self.AUROC(probs, target.squeeze())
        metrics = self.test_metrics(max_probs.squeeze(), target.squeeze())
        metrics["auc"] = auc

        for k, v in metrics.items():
            print(f"{k} = {v}")
            metrics[k] = v.detach().cpu().numpy() if hasattr(v, "detach") else v

        print()

        for c in range(self.n_classes):
            count = self.data[c]["count"]
            correct = self.data[c]["correct"]
            acc = None if count == 0 else float(correct) / count
            print(f"class {c}: acc {acc}, correct {correct}/{count}")
        self.data = [{"count": 0, "correct": 0} for _ in range(self.n_classes)]

        result = pd.DataFrame([metrics])
        result.to_csv(self.log_path / "result.csv", index=False)

        self.test_step_outputs.clear()


    def load_model(self):
        name = self.hparams.model.name
        # Change the `trans_unet.py` file name to `TransUnet` class name.
        # Please always name your model file name as `trans_unet.py` and
        # class name or funciton name corresponding `TransUnet`.
        if '_' in name:
            camel_name = ''.join([i.capitalize() for i in name.split('_')])
        else:
            camel_name = name
        try:
            Model = getattr(importlib.import_module(
                f'models.{name}'), camel_name)
        except:
            raise ValueError('Invalid Module File Name or Invalid Class Name!')
        self.model = self.instancialize(Model)
        pass

    def instancialize(self, Model, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.hparams.
        """
        class_args = inspect.getargspec(Model.__init__).args[1:]
        inkeys = self.hparams.model.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = getattr(self.hparams.model, arg)
        args1.update(other_args)
        return Model(**args1)

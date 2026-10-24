import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)

from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from utils.metrics import metric
import numpy as np
import torch
import torch.nn as nn
from torch import optim
import numpy as np
import os
import time
import warnings
import matplotlib.pyplot as plt
import numpy as np
import copy
from models.PLFM import FreModel
import layers.SolarLLMProcess_utils as LLMutils
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM
from release_config import llm_paths

warnings.filterwarnings('ignore')


class exp_TSLFL_LLM_Solar(Exp_Basic):
    def __init__(self, args, ft=False, path=False):
        self.ft = ft
        self.avg_speed = None
        self.path = path
        self.decay = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
        self.pred_len = args.pred_len
        self.fremodel = FreModel(args).to('cuda:0')
        self.fremodel.load_state_dict(torch.load(self.path, map_location='cuda:0'), strict=False)

        # 原始基础模型路径
        base_model_name, peft_model_path = llm_paths("solar")
        # 使用QLoRA微调后保存的路径
        # Adapter path is supplied by the release profile.
        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        # 加载QLoRA模型
        self.qwen_model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        super(exp_TSLFL_LLM_Solar, self).__init__(args)

    def _build_model(self):

        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model


    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        self.feature_names = data_set.cols
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _predict(self, batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_x):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, spec_x, None, dec_inp, None)

            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        
        low_freq_pred, high_freq_pred = outputs
        pred = low_freq_pred + high_freq_pred

        # LLM 处理过程
        x_combin = torch.cat((batch_x, batch_x_semantic), dim=2)
        y_low_combin = torch.cat((low_freq_pred, batch_y_semantic), dim=2)
        y_high_combin = torch.cat((high_freq_pred, batch_y_semantic), dim=2)
        y_combin = torch.cat((pred, batch_y_semantic), dim=2)

        field_x = LLMutils.compute_derived_features_tensor(x_combin)
        field_y = LLMutils.compute_derived_features_tensor(y_combin, self.args.pred_len)
        field_low_y = LLMutils.compute_derived_features_tensor(y_low_combin, self.args.pred_len)
        field_high_y = LLMutils.compute_derived_features_tensor(y_high_combin, self.args.pred_len)

        prompts = LLMutils.make_prompts_from_fields(field_x, field_y, batch_x_time, batch_y_time, self.args.pred_len,
                                        field_high_y, field_low_y)
        print(f"Generated {len(prompts)} prompts for LLM processing.")
        print(prompts[0])

        power_refined = LLMutils.Qwen(self.tokenizer, self.qwen_model, prompts, self.args.pred_len,
                          field_y['POWER'], trues=batch_y)
        outputs_refined = torch.tensor(power_refined).unsqueeze(-1).float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
        # outputs = outputs[:, -self.args.pred_len:, :]

        return outputs_refined, batch_y, pred


    # def vali(self, vali_loader, criterion):
    #     total_mse_loss = []
    #     total_mae_loss = []
    #     self.model.eval()
    #     self.fremodel.eval()

    #     with torch.no_grad():
    #         for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(vali_loader):
    #             batch_x = batch_x.float().to(self.device)
    #             batch_x_semantic = batch_x_semantic.float().to(self.device)
    #             batch_x_time = batch_x_time.int().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_y_semantic = batch_y_semantic.float().to(self.device)
    #             batch_y_time = batch_y_time.int().to(self.device)
    #             temp_batch_x = copy.deepcopy(batch_x)

    #             spec_x = self.fremodel(temp_batch_x, batch_y, None, None)

    #             outputs, batch_y = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_x)

    #             pred = outputs
    #             true = batch_y
            
    #             mse = criterion(pred, true)
    #             total_mse_loss.append(mse.item())
    #     mse_loss = np.average(total_mse_loss)
    #     mae_loss = np.average(total_mae_loss)
    #     self.model.train()
    #     return mse_loss

    # def train(self, setting):
    #     _, train_loader = self._get_data(flag='train')
    #     _, vali_loader = self._get_data(flag='val')
    #     _, test_loader = self._get_data(flag='test')

    #     path = os.path.join(self.args.checkpoints, setting)
    #     if not os.path.exists(path):
    #         os.makedirs(path)

    #     time_now = time.time()

    #     train_steps = len(train_loader)
    #     early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

    #     model_optim = self._select_optimizer()
    #     criterion = self._select_criterion()
    #     speed_list = []
    #     test_mse_list = []
    #     test_mae_list = []
    #     vali_list = []

    #     if self.args.use_amp:
    #         scaler = torch.cuda.amp.GradScaler()
    #     for epoch in range(self.args.train_epochs):
    #         iter_count = 0
    #         train_loss = []
    #         iter_tol = 0

    #         self.model.train()
    #         self.fremodel.eval()

    #         epoch_time = time.time()
    #         for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(train_loader):
    #             iter_count += 1
    #             iter_tol += 1
    #             model_optim.zero_grad()
    #             batch_x = batch_x.float().to(self.device)
    #             batch_x_semantic = batch_x_semantic.float().to(self.device)
    #             batch_x_time = batch_x_time.int().to(self.device)
    #             batch_y = batch_y.float().to(self.device)
    #             batch_y_semantic = batch_y_semantic.float().to(self.device)
    #             batch_y_time = batch_y_time.int().to(self.device)

    #             temp_batch_x = copy.deepcopy(batch_x)

    #             spec_x = self.fremodel(temp_batch_x, batch_y, None, None)
  
    #             outputs, batch_y = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_x)

    #             '''
    #                 FALoss to replace MSE
    #             '''
    #             fft_o = torch.fft.rfft(outputs, dim=1)  # B x L x C
    #             fft_t = torch.fft.rfft(batch_y, dim=1)  # B x L x C
    #             loss = self.args.lamb * torch.sum(torch.sum(torch.sum(torch.abs(fft_o - fft_t)))) / (fft_o.shape[2] * fft_o.shape[1] * fft_o.shape[0])

    #             train_loss.append(loss.item())

    #             if (i + 1) % 100 == 0:
    #                 print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
    #                 speed = (time.time() - time_now) / iter_count
    #                 left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
    #                 print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
    #                 iter_count = 0
    #                 time_now = time.time()

    #             if self.args.use_amp:
    #                 scaler.scale(loss).backward()
    #                 scaler.step(model_optim)
    #                 scaler.update()
    #             else:
    #                 loss.backward()
    #                 model_optim.step()
            
    #         total_time = time.time() - epoch_time
    #         print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    #         print("Average Speed: {}".format(total_time / iter_tol))
    #         speed_list.append(total_time / iter_tol)
    #         train_loss = np.average(train_loss)
    #         vali_loss = self.vali(vali_loader, criterion)
    #         test_loss = self.vali(test_loader, criterion)
    #         vali_list.append(vali_loss)

    #         test_mse_list.append(test_loss)

    #         print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
    #             epoch + 1, train_steps, train_loss, vali_loss, test_loss))
    #         early_stopping(vali_loss, self.model, path)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break

    #         adjust_learning_rate(model_optim, epoch + 1, self.args)

    #     best_model_path = path + '/' + 'checkpoint.pth'
    #     self.model.load_state_dict(torch.load(best_model_path, map_location='cuda:0'))
    #     # self.avg_speed = round(np.mean(speed_list) * 1000, 2)

    #     # mse = test_mse_list[np.argmin(vali_list)]
    #     # mae = test_mae_list[np.argmin(vali_list)]

    #     # f = open("result.txt", 'a') 
    #     # f.write(setting + "  \n")
    #     # f.write('mse:{}, mae:{} bs:{} lr:{} dm:{} main_fre:{}'.format(mse, mae, self.args.batch_size, self.args.learning_rate, self.args.d_model, self.args.d_ff, self.args.main_fre))
    #     # f.write('\n')
    #     # f.write('\n')
    #     # f.close()

    #     return best_model_path

    def test(self, setting, test=1):
        _, test_loader = self._get_data(flag='test')
        if test:
            setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_dt{}_{}_{}'.format(
                "TSLFL_Forcast",
                self.args.model_id,
                self.args.model,
                self.args.features,
                self.args.seq_len,
                self.args.label_len,
                self.args.pred_len,
                self.args.d_model,
                self.args.n_heads,
                self.args.e_layers,
                self.args.d_layers,
                self.args.d_ff,
                self.args.factor,
                self.args.embed,
                self.args.distil,
                self.args.des, 0)
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join(self.args.checkpoints, setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []
        orgs = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        self.fremodel.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_semantic = batch_x_semantic.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_semantic = batch_y_semantic.float().to(self.device)
                temp_batch_x = copy.deepcopy(batch_x)
                
                '''
                    Note that during the inference process, PLFM directly outputs the forecasting spectra, without test data leakage
                '''
                spec_in = self.fremodel(temp_batch_x, batch_y, None, None)

                outputs, batch_y, org = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time, spec_in)

                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs.detach().cpu().numpy()
                org = org.detach().cpu().numpy()
                # spec_temp = spec_temp.detach().cpu().numpy()
                batch_x_cpu = batch_x.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                trues.append(true)
                orgs.append(org)

                preds_now = np.concatenate(preds, axis=0)
                trues_now = np.concatenate(trues, axis=0)
                org_now = np.concatenate(orgs, axis=0)
                
                mae, mse, rmse, mape, mspe, smape = metric(preds_now, trues_now)
                print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))
                mae, mse, rmse, mape, mspe, smape = metric(org_now, trues_now)
                print('org mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))

        
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe, smape = metric(preds, trues)

        print('mse:{}, mae:{} bs:{} lr:{} dm:{} df:{} main_fre:{}'.format(mse, mae, self.args.batch_size, self.args.learning_rate, self.args.d_model, self.args.d_ff, self.args.main_fre))
        # print('mse:{}, mae:{}'.format(mse, mae))

        f = open("LLM_results.txt", 'a') 
        f.write(setting + "  \n")
        f.write(f'seed: {self.args.seed}, pred_len: {self.pred_len}\n')
        f.write('mae, mse, mape:\n')
        f.write(f'{mae}\t{mse}\t{mape}')
        # f.write('mse:{}, mae:{} bs:{} lr:{} dm:{} df:{}  main_fre:{}'.format(mse, mae, self.args.batch_size, self.args.learning_rate, self.args.d_model, self.args.d_ff, self.args.main_fre))
        f.write('\n')
        f.write('\n')
        f.close()

        # np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path + 'pred.npy', preds)
        # np.save(folder_path + 'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            logging.info(best_model_path)
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                spec_x = self.fremodel(batch_x, batch_y, batch_x_mark, batch_y_mark)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark, spec_x)

                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return

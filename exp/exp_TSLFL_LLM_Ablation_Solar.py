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

warnings.filterwarnings('ignore')


class Exp_TSLFL_LLM_Ablation_Solar(Exp_Basic):
    def __init__(self, args, ft=False, path=False):
        self.ft = ft
        self.avg_speed = None
        self.decay = [0, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 1.5, 2]
        self.pred_len = args.pred_len

        # 原始基础模型路径
        base_model_name = "Qwen3/Qwen3-8B"
        # 使用QLoRA微调后保存的路径
        peft_model_path = "Qwen3/Solar_Finetune_test/v4-20250720-082152/checkpoint-219"
        # 加载Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name, padding_side='left')
        # 加载QLoRA模型
        self.qwen_model = AutoPeftModelForCausalLM.from_pretrained(
            peft_model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        super(Exp_TSLFL_LLM_Ablation_Solar, self).__init__(args)

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

    def _predict(self, batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, None, dec_inp, None, None)

            if self.args.output_attention:
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()
        
        pred = outputs
        # pred = torch.rand(batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]).float().to(self.device)  # 模拟预测结果
        low_freq_pred = high_freq_pred = torch.zeros_like(pred).float().to(self.device)

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

        power_refined = LLMutils.Qwen(self.tokenizer, self.qwen_model, prompts, self.args.pred_len,
                          field_y['POWER'], trues=batch_y)
        outputs_refined = torch.tensor(power_refined).unsqueeze(-1).float().to(self.device)
        batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)

        return outputs_refined, batch_y


    def test(self, setting, test=1):

        # setting = '{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_eb{}_dt{}_{}_{}'.format(
        #     "long_term_forecast",
        #     self.args.model_id,
        #     'DLinear_custom',
        #     self.args.features,
        #     self.args.seq_len,
        #     self.args.label_len,
        #     self.args.pred_len,
        #     self.args.d_model,
        #     self.args.n_heads,
        #     self.args.e_layers,
        #     self.args.d_layers,
        #     self.args.d_ff,
        #     self.args.embed,
        #     self.args.distil,
        #     self.args.des, 0)
        setting = "long_term_forecast_Solar_72_2_DLinear_custom_ftM_sl72_ll0_pl2_dm256_nh8_el2_dl1_df512_expand2_dc4_fc3_ebtimeF_dtTrue_Exp_0"

        _, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))
            # self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth'), map_location='cuda:0'))

        preds = []
        trues = []
        coarse_preds = []
        coarse_trues = []

        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_x_semantic = batch_x_semantic.float().to(self.device)
                # batch_x_time = batch_x_time.int().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_y_semantic = batch_y_semantic.float().to(self.device)
                # batch_y_time = batch_y_time.int().to(self.device)
                # temp_batch_x = copy.deepcopy(batch_x)
                
                '''
                    Note that during the inference process, PLFM directly outputs the forecasting spectra, without test data leakage
                '''

                outputs, batch_y = self._predict(batch_x, batch_x_semantic, batch_x_time, batch_y, batch_y_semantic, batch_y_time)

                batch_y = batch_y.detach().cpu().numpy()

                outputs = outputs.detach().cpu().numpy()
                # spec_temp = spec_temp.detach().cpu().numpy()
                batch_x_cpu = batch_x.detach().cpu().numpy()
                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)
                trues.append(true)

                preds_now = np.concatenate(preds, axis=0)
                trues_now = np.concatenate(trues, axis=0)
                mae, mse, rmse, mape, mspe, smape = metric(preds_now, trues_now)
                print('mse:{}, mae:{}, mape:{}'.format(mse, mae, mape))

        
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

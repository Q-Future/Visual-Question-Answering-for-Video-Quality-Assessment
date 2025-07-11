from pytorchvideo.models.hub import slowfast_r50
from torch import nn
import torch


class slowfast(torch.nn.Module):
    def __init__(self):
        super(slowfast, self).__init__()
        # slowfast=
        # slowfast_pretrained_features = nn.Sequential(*list(slowfast_r50(pretrained=True).children())[0])
        slowfast_pretrained_features = torch.load('./slowfast.pth')
        # for p in feature_extraction.parameters():
        #     if torch.isnan(p).any():
        #         print(p)
        #     p[torch.isnan(p)] = 0
       

        self.feature_extraction = torch.nn.Sequential()
        self.slow_avg_pool = torch.nn.Sequential()
        self.fast_avg_pool = torch.nn.Sequential()
        self.adp_avg_pool = torch.nn.Sequential()

        for x in range(0, 5):
            self.feature_extraction.add_module(str(x), slowfast_pretrained_features[x])
        # for p in self.feature_extraction.parameters():
        #
        #         print(p)


               
        # for p in self.feature_extraction.parameters():
        #             if torch.isnan(p).any():
        #                 print(1)
        #                 q = torch.where(torch.isnan(p), torch.zeros_like(p), p)
        #
        #                 p = q
                # p=q
        self.slow_avg_pool.add_module('slow_avg_pool', slowfast_pretrained_features[5].pool[0])
        self.fast_avg_pool.add_module('fast_avg_pool', nn.AdaptiveAvgPool2d(output_size=1))
        self.adp_avg_pool.add_module('adp_avg_pool', slowfast_pretrained_features[6].output_pool)

        self.num_positions = 2000
        self.position_embedding = nn.Embedding(self.num_positions, 256)
        self.register_buffer("position_ids", torch.arange(self.num_positions).expand((1, -1)), persistent=False)


    def forward(self, x):
        # for p in self.feature_extraction.parameters():
        #     print(p.numel())
        #     if torch.isnan(p).any():
        #         print(p.dtype)
        #     if p.numel()==0:
        #         print(p.dtype)
                # q = torch.where(torch.isnan(p), torch.zeros_like(p), p)
                #
                # p = q
        x=x.to(self.position_ids.device)
        # print(x.dtype)
        self.feature_extraction.to(self.position_ids.device)
        # print(x.device)

        xx = x.unsqueeze(0).transpose(1, 2)[:, :, :round(x.shape[0] / 4)]
        a = []
        a.append([xx][0])
        a.append([x.unsqueeze(0).transpose(1, 2)][0])
        # print(a[1].device)
        x1 = self.feature_extraction(a)
        # for p in self.feature_extraction.parameters():
        #     if p.numel() == 0:
        #         print(p)
        #     if torch.isnan(p).any():
        #         print(p)

            # print(p)
        # print(self.position_ids.dtype)
        # AdaptiveAvgPool3d(output_size=1)
        # slow_feature = self.slow_avg_pool(x[0])
        fast_feature = x1[1]
        # print(fast_feature)

        # slow_feature = self.adp_avg_pool(slow_feature)
        fast_feature = self.fast_avg_pool(fast_feature).squeeze(0).squeeze(-1).squeeze(-1).transpose(0, 1)

        embeddings = fast_feature
        embeddings = embeddings + self.position_embedding(self.position_ids)[0,:x.shape[0]]
        return embeddings

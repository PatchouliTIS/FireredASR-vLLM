import torch
import torch.nn as nn


class Adapter(nn.Module):
    def __init__(self, encoder_dim, llm_dim, downsample_rate=2):
        super().__init__()
        self.ds = downsample_rate
        self.linear1 = nn.Linear(encoder_dim * downsample_rate, llm_dim)
        # 使用inplace=True减少内存分配
        self.relu = nn.ReLU(inplace=True)
        self.linear2 = nn.Linear(llm_dim, llm_dim)

    def forward(self, x, x_lens):
        batch_size, seq_len, feat_dim = x.size()
        num_frames_to_discard = seq_len % self.ds
        if num_frames_to_discard > 0:
            x = x[:, :-num_frames_to_discard, :]
        seq_len = x.size(1)

        # 优化: 使用reshape替代contiguous+view，自动处理内存布局
        # reshape会在需要时自动调用contiguous，避免不必要的拷贝
        x = x.reshape(
            batch_size, seq_len // self.ds, feat_dim * self.ds
        )

        x = self.linear1(x)
        x = self.relu(x)  # inplace操作，节省内存
        x = self.linear2(x)

        # 优化: 直接整除，避免中间变量
        new_x_lens = torch.clamp(x_lens, max=seq_len) // self.ds
        return x, new_x_lens

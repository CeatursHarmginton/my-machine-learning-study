import torch
import torch.nn as nn

class LinearChainCRF(nn.Module):
    """Linear-Chain CRF cô đọng trong 34 dòng: Forward Log-Sum-Exp + Giải mã quy hoạch động Viterbi."""
    def __init__(self, num_tags: int):
        super().__init__()
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.trans = nn.Parameter(torch.zeros(num_tags, num_tags))

    def forward(self, emit: torch.Tensor, tags: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        b_sz, seq_len, _ = emit.shape
        b_idx, safe_t = torch.arange(b_sz, device=emit.device), tags.clamp(min=0)
        alpha = self.start + emit[:, 0]
        gold = self.start[safe_t[:, 0]] + emit[b_idx, 0, safe_t[:, 0]]
        for t in range(1, seq_len):
            m = mask[:, t]
            nxt = torch.logsumexp(alpha.unsqueeze(2) + self.trans + emit[:, t].unsqueeze(1), dim=1)
            alpha = torch.where(m.unsqueeze(1), nxt, alpha)
            gold = gold + (self.trans[safe_t[:, t - 1], safe_t[:, t]] + emit[b_idx, t, safe_t[:, t]]) * m.float()
        log_z = torch.logsumexp(alpha + self.end, dim=1)
        last_tag = safe_t[b_idx, mask.long().sum(1) - 1]
        return (log_z - (gold + self.end[last_tag])).mean()

    @torch.no_grad()
    def decode(self, emit: torch.Tensor, mask: torch.Tensor) -> list[list[int]]:
        v, ptrs = self.start + emit[:, 0], []
        for t in range(1, emit.size(1)):
            best_v, best_id = (v.unsqueeze(2) + self.trans).max(dim=1)
            v = torch.where(mask[:, t].unsqueeze(1), best_v + emit[:, t], v)
            ptrs.append(best_id)
        last = (v + self.end).argmax(dim=1).tolist()
        lens, out = mask.long().sum(1).tolist(), []
        for b, cur in enumerate(last):
            seq = [cur]
            for t in range(lens[b] - 2, -1, -1):
                cur = int(ptrs[t][b, cur])
                seq.append(cur)
            out.append(seq[::-1])
        return out

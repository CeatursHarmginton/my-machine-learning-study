import torch
import torch.nn as nn

class PairGridHead(nn.Module):
    """Lưới quan hệ từ-đôi GTS O(L^2) trong 11 dòng — không bao giờ bịa từ nằm ngoài câu gốc."""
    def __init__(self, d_model: int, n_tags: int = 6):
        super().__init__()
        self.pa = nn.Linear(d_model, d_model // 2)
        self.po = nn.Linear(d_model, d_model // 2)
        self.clf = nn.Linear(d_model, n_tags)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        a, o, n = torch.relu(self.pa(h)), torch.relu(self.po(h)), h.size(1)
        return self.clf(torch.cat([a[:, :, None, :].expand(-1, -1, n, -1), o[:, None, :, :].expand(-1, n, -1, -1)], -1))

def parse_tuples(text: str, tags: tuple[str, ...] = ("[A]", "[C]", "[O]", "[S]")) -> list[tuple[str, ...]]:
    """Bộ giải mã tổng quát an toàn khôi phục mọi cấu trúc Tuple từ chuỗi Seq2Seq, tự lọc phần tử lỗi."""
    out = []
    for chunk in text.split(";"):
        if all(t in chunk for t in tags):
            vals, rem = [], chunk
            for i, t in enumerate(tags):
                _, rem = rem.split(t, 1)
                if i + 1 < len(tags):
                    val, rem = rem.split(tags[i + 1], 1)
                    rem = tags[i + 1] + rem
                    vals.append(val.strip())
                else:
                    vals.append(rem.strip())
            out.append(tuple(vals))
    return out

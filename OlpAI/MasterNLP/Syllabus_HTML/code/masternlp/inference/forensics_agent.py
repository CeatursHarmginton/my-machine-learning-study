import ast
from collections import Counter
import torch
import torch.nn.functional as F

@torch.no_grad()
def llm_forensic_stats(logits: torch.Tensor, target_ids: torch.Tensor) -> dict[str, float]:
    """Đặc trưng pháp y Zero-Shot (Fast-DetectGPT): đo độ mượt bất thường & hạng xác suất token của văn bản AI."""
    lg, tg = logits[0, :-1], target_ids[0, 1:]
    lp = F.log_softmax(lg, dim=-1)
    p = lp.exp()
    tok_lp = lp.gather(-1, tg.unsqueeze(-1)).squeeze(-1)
    exp_lp = (p * lp).sum(-1)
    var_lp = (p * (lp - exp_lp.unsqueeze(-1)).pow(2)).sum(-1)
    ranks = (lg > lg.gather(-1, tg.unsqueeze(-1))).sum(-1) + 1
    z_curv = float((tok_lp - exp_lp).mean() / var_lp.sqrt().mean().clamp(min=1e-5))
    return {"ppl": float((-tok_lp.mean()).exp().clamp(max=1e4)), "log_rank": float(ranks.float().log().mean()), "z_curv": z_curv}

def safe_math_eval(expr: str) -> float | None:
    """Tính biểu thức toán học qua cây AST an toàn trong 12 dòng cho Tool-Integrated Reasoning Agent."""
    ops = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.Div: lambda a, b: a / b, ast.FloorDiv: lambda a, b: a // b, ast.Mod: lambda a, b: a % b, ast.Pow: lambda a, b: a ** b}
    def _run(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.USub):
            return -_run(n.operand)
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](_run(n.left), _run(n.right))
        raise ValueError("Blocked")
    try:
        return _run(ast.parse(expr.strip(), mode="eval").body)
    except Exception:
        return None

def majority_vote(answers: list) -> object:
    return Counter(a for a in answers if a is not None).most_common(1)[0][0]

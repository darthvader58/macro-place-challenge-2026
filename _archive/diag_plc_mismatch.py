"""Diagnose the placer-internal vs evaluator-plc proxy mismatch on ibm03."""
import importlib.util
import random
import numpy as np
import torch

spec = importlib.util.spec_from_file_location("p", "submissions/partcl_2026/placer.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

from macro_place.loader import load_benchmark_from_dir
from macro_place.objective import compute_proxy_cost

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

bench, plc_eval = load_benchmark_from_dir("external/MacroPlacement/Testcases/ICCAD04/ibm03")
n_hard = bench.num_hard_macros
plc_placer = mod._load_plc("ibm03")

sizes = bench.macro_sizes[:n_hard].numpy().astype(np.float64)
movable = bench.get_movable_mask()[:n_hard].numpy()
cw = float(bench.canvas_width)
ch = float(bench.canvas_height)
init_pos = bench.macro_positions[:n_hard].numpy().astype(np.float64).copy()
legal_pos = mod._legalize(init_pos, sizes, movable, 0.05, cw, ch, n_hard)

proxy_placer_internal = mod._eval_true_proxy(legal_pos, bench, plc_placer)
print(f"placer_plc proxy on legal (via _eval_true_proxy): {proxy_placer_internal:.4f}")

soft_init = bench.macro_positions[n_hard:].numpy().astype(np.float64).copy()
full_np = np.concatenate([legal_pos, soft_init], axis=0).astype(np.float32)
full_t = torch.from_numpy(full_np)

c_eval = compute_proxy_cost(full_t, bench, plc_eval)
print(
    f"evaluator_plc proxy on full_t:   {c_eval['proxy_cost']:.4f}  "
    f"wl={c_eval['wirelength_cost']:.3f}  den={c_eval['density_cost']:.3f}  cong={c_eval['congestion_cost']:.3f}"
)
c_placer = compute_proxy_cost(full_t, bench, plc_placer)
print(
    f"placer_plc proxy on full_t:      {c_placer['proxy_cost']:.4f}  "
    f"wl={c_placer['wirelength_cost']:.3f}  den={c_placer['density_cost']:.3f}  cong={c_placer['congestion_cost']:.3f}"
)

# Also check id(plc_eval) and id(plc_placer)
print(f"plc_eval id={id(plc_eval)}  plc_placer id={id(plc_placer)}  same={plc_eval is plc_placer}")
print(f"plc_eval grid_row,col = {plc_eval.grid_row},{plc_eval.grid_col}")
print(f"plc_placer grid_row,col = {plc_placer.grid_row},{plc_placer.grid_col}")
print(f"plc_eval canvas = {plc_eval.width} x {plc_eval.height}")
print(f"plc_placer canvas = {plc_placer.width} x {plc_placer.height}")

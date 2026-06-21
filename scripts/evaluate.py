import _path
import json
from pathlib import Path

import numpy as np
import torch
from tqdm.auto import tqdm

from config import get_settings
from data.dataset import CaptchaDataset
from data.labels import targets_to_strings
from eval.metrics import confusion_matrix_aggregate, exact_match_rate
from schemas import EvalResult, TestPredictions
from train.factory import build_model
from utils.checkpoint import load_checkpoint
from utils.device import get_device
from utils.seed import set_seed


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    dataset: CaptchaDataset,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    model.eval()
    all_true = []
    all_pred = []
    all_true_str = []
    all_pred_str = []
    for start in tqdm(range(0, len(dataset), batch_size), leave=False):
        end = min(start + batch_size, len(dataset))
        images = []
        targets = []
        labels = []
        for idx in range(start, end):
            image, target = dataset[idx]
            images.append(image)
            targets.append(target)
            labels.append("".join(str(d.item()) for d in target))
        batch_x = torch.stack(images).to(device)
        batch_y = torch.stack(targets)
        logits = model(batch_x)
        preds = logits.argmax(dim=-1).cpu()
        all_true.append(batch_y.numpy())
        all_pred.append(preds.numpy())
        all_true_str.extend(labels)
        all_pred_str.extend(targets_to_strings(preds))
    y_true = np.concatenate(all_true, axis=0)
    y_pred = np.concatenate(all_pred, axis=0)
    return y_true, y_pred, all_true_str, all_pred_str


def evaluate_checkpoint(
    model_name: str,
    stage: str,
    split: str,
    data_root: Path,
) -> EvalResult:
    settings = get_settings()
    device = get_device()
    model = build_model(model_name, settings).to(device)
    ckpt = settings.checkpoint_dir / model_name / stage / "final.pt"
    load_checkpoint(ckpt, model)
    dataset = CaptchaDataset(data_root)
    y_true, y_pred, labels, predictions = collect_predictions(
        model, dataset, device, settings.batch_size
    )
    result = EvalResult(
        model_name=model_name,
        checkpoint_stage=stage,
        split=split,
        exact_match=exact_match_rate(y_true, y_pred),
    )
    pred_path = (
        settings.output_dir
        / "predictions"
        / f"{model_name}_{stage}_{split}.json"
    )
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    pred_path.write_text(
        TestPredictions(
            model_name=model_name,
            checkpoint_stage=stage,
            split=split,
            labels=labels,
            predictions=predictions,
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    cm_path = (
        settings.output_dir
        / "confusion"
        / f"{model_name}_{stage}_{split}.npz"
    )
    cm_path.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix_aggregate(y_true, y_pred)
    np.savez(cm_path, confusion=cm)
    return result


def run_evaluation() -> list[EvalResult]:
    settings = get_settings()
    set_seed(settings.seed)
    clean_dir = settings.data_dir / "clean"
    results: list[EvalResult] = []
    configs = [
        ("vit", "clean"),
        ("vit", "finetune"),
        ("cnn", "clean"),
        ("cnn", "finetune"),
    ]
    for model_name, stage in configs:
        clean_res = evaluate_checkpoint(
            model_name, stage, "clean_test", clean_dir / "test"
        )
        adv_res = evaluate_checkpoint(
            model_name,
            stage,
            "adv_test",
            settings.data_dir / "adv" / model_name / "test",
        )
        adv_res.robustness_gap = clean_res.exact_match - adv_res.exact_match
        adv_res.attack_success_rate = 1.0 - adv_res.exact_match
        results.extend([clean_res, adv_res])
    out_path = settings.output_dir / "metrics" / "test_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [r.model_dump() for r in results]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return results


if __name__ == "__main__":
    run_evaluation()

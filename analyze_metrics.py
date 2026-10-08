import json
from pathlib import Path

def analyze_run(path, name):
    print(f"=== {name} ({path}) ===")
    p = Path(path)
    if not p.exists():
        print(f"File not found: {path}")
        return
    records = []
    with open(p, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    records.append(json.loads(line))
                except Exception:
                    pass
    
    val_records = [r for r in records if r.get("split") in ("val", "eval", "test")]
    train_records = [r for r in records if r.get("split") == "train"]
    epoch_records = [r for r in records if "epoch" in r.get("split", "")]

    print(f"Total train log steps: {len(train_records)}")
    print(f"Total val/test log steps: {len(val_records)}")
    
    print("\nVal/Test Evaluation Steps:")
    for v in val_records:
        step = v.get("step")
        loss = v.get("loss", 0.0)
        acc = v.get("accuracy", 0.0)
        auc = v.get("auc", 0.0)
        split = v.get("split")
        print(f"  step={step} split={split} loss={loss:.4f} acc={acc*100:.2f}% auc={auc:.4f}")

    print("\nTrain vs Val progression:")
    for v in val_records:
        v_step = v.get("step")
        nearby_train = [t for t in train_records if abs(t.get("step", 0) - v_step) <= 15]
        if nearby_train:
            avg_train_loss = sum(t.get("loss", 0.0) for t in nearby_train) / len(nearby_train)
            avg_train_acc = sum(t.get("accuracy", 0.0) for t in nearby_train) / len(nearby_train)
            v_loss = v.get("loss", 0.0)
            v_acc = v.get("accuracy", 0.0)
            gap = v_loss - avg_train_loss
            print(f"  step={v_step:4d} | Train Loss={avg_train_loss:.4f}, Val Loss={v_loss:.4f} (gap: {gap:+.4f}) | Train Acc={avg_train_acc*100:.2f}%, Val Acc={v_acc*100:.2f}%")

    if epoch_records:
        print("\nEpoch / Full Split summary:")
        for e in epoch_records:
            print(f"  split={e.get('split')} epoch={e.get('epoch')} loss={e.get('loss', 0.0):.4f} acc={e.get('accuracy', 0.0)*100:.2f}%")

if __name__ == "__main__":
    analyze_run("kaggle_out/runs/kaggle-jev-20261008-055555/metrics.jsonl", "Run 1 (20k questions, 1 Epoch)")
    print("\n" + "="*50 + "\n")
    analyze_run("kaggle_out_v3/runs/kaggle-jev-20261008-063713/metrics.jsonl", "Run 3 (50k questions, 1 Epoch)")

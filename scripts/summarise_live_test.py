"""Table 8 of the manuscript from the live per-sign test (scripts/live_test.py).

Reads every results/live_test/trials_*.csv and counts the FIRST attempt of every trial only (decided by the user,
5 Nov 2026, config_log section 34a): the first row of each trial_no in file order (rows are appended when a trial
finishes), whatever its repeated flag; repeated attempts (key R) are ignored, because R was mostly pressed after a
miss and would raise the success rate. For transparency the summary also states how many trials were repeated and
the success count when the last attempt is used instead (not for the table). Writes
  results/live_test/table8.csv / table8.md   (all participants together; columns exactly
      Gesture, Hands, Success, Miss, Success rate (%), Label shown on miss; 15 rows + Total)
  results/live_test/table8_<ID>.csv          (per participant, when there is more than one)
plus a text summary (in table8.md and on screen): successes out of the trials, the three signs missed most often,
and the share of misses with a wrong label vs no detection ("none").
"Label shown on miss" = the label_on_miss values of the misses with their counts, e.g. "Cari (2), none (1)".

Usage: python scripts/summarise_live_test.py [--dir results/live_test]
"""
import argparse
import csv
from collections import Counter
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
COLS = ["Gesture", "Hands", "Success", "Miss", "Success rate (%)", "Label shown on miss"]
CLASSES = yaml.safe_load(open(ROOT / "configs" / "classes.yaml", encoding="utf-8"))
NAMES = {int(k): v for k, v in CLASSES["names"].items()}
HANDS = {int(k): v for k, v in CLASSES["groups"].items()}


def labels_text(c):
    return ", ".join(f"{k} ({v})" for k, v in sorted(c.items(), key=lambda kv: (-kv[1], kv[0] == "none", kv[0]))) or "–"


def table(rows):
    out, tot_s, tot_m, miss_all = [], 0, 0, Counter()
    for gid in sorted(NAMES):
        r = [x for x in rows if int(x["gloss_id"]) == gid]
        s = sum(x["result"] == "success" for x in r)
        m = sum(x["result"] == "miss" for x in r)
        c = Counter(x["label_on_miss"] for x in r if x["result"] == "miss")
        miss_all += c
        tot_s, tot_m = tot_s + s, tot_m + m
        out.append({"Gesture": NAMES[gid], "Hands": HANDS[gid], "Success": s, "Miss": m,
                    "Success rate (%)": f"{100 * s / (s + m):.1f}" if s + m else "–", "Label shown on miss": labels_text(c)})
    wrong, none = sum(v for k, v in miss_all.items() if k != "none"), miss_all.get("none", 0)
    out.append({"Gesture": "Total", "Hands": "", "Success": tot_s, "Miss": tot_m,
                "Success rate (%)": f"{100 * tot_s / (tot_s + tot_m):.1f}" if tot_s + tot_m else "–",
                "Label shown on miss": f"wrong label ({wrong}), none ({none})"})
    return out, miss_all


def summary(rows, tab, miss_all, who):
    n = len(rows)
    s = sum(x["result"] == "success" for x in rows)
    worst = sorted((r for r in tab[:-1] if r["Miss"]), key=lambda r: (-r["Miss"], r["Gesture"]))[:3]
    wrong, none = sum(v for k, v in miss_all.items() if k != "none"), miss_all.get("none", 0)
    m = wrong + none
    expect = 150 * len({x["participant"] for x in rows})
    lines = [f"{who}: {s} of {n} trials successful ({100 * s / n:.1f} %)" if n else f"{who}: no trials",
             "Most missed: " + (", ".join(f"{r['Gesture']} ({r['Miss']} miss)" for r in worst) or "-"),
             f"Misses: wrong label {wrong} ({100 * wrong / m:.1f} %), not detected {none} ({100 * none / m:.1f} %)"
             if m else "Misses: none"]
    if n != expect:
        lines.append(f"NOTE: {n} valid trials, expected {expect} (150 per participant) - incomplete")
    return lines


def write(path, tab):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(tab)


def md(tab):
    out = ["| " + " | ".join(COLS) + " |", "|" + "|".join(["---"] * len(COLS)) + "|"]
    for r in tab:
        cells = [str(r[c]) for c in COLS]
        if r["Gesture"] == "Total":
            cells = [f"**{c}**" if c else "" for c in cells]
        out.append("| " + " | ".join(cells) + " |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=ROOT / "results" / "live_test")
    args = ap.parse_args()
    d = args.dir if args.dir.is_absolute() else ROOT / args.dir
    files = sorted(d.glob("trials_*.csv"))
    if not files:
        raise SystemExit(f"no trials_*.csv in {d}")
    rows, per, notes = [], {}, {}
    for f in files:
        allr = list(csv.DictReader(open(f, encoding="utf-8")))
        first = {}
        for x in allr:                      # file order = chronological order of the attempts
            first.setdefault(x["trial_no"], x)
        r = list(first.values())
        last = [x for x in allr if x["repeated"] == "0"]
        n_rep = sum(1 for x in allr if x["repeated"] == "1")
        pid = r[0]["participant"] if r else f.stem[7:]
        per[pid] = r
        rows += r
        thr = {x["threshold"] for x in allr}
        ck = {x["checkpoint_path"] for x in allr}
        notes[pid] = (f"{pid}: first attempts {sum(x['result'] == 'success' for x in r)}/{len(r)} successful (used); "
                      f"{len({x['trial_no'] for x in allr if x['repeated'] == '1'})} trials were repeated ({n_rep} repeated attempts); "
                      f"with the last attempt instead: {sum(x['result'] == 'success' for x in last)}/{len(last)} (not used)")
        print(f"{f.name}: {len(r)} trials, first attempt each; {n_rep} repeated attempts ignored; "
              f"threshold {thr}; checkpoint {ck}")
    if len({x["threshold"] for x in rows}) > 1 or len({x["checkpoint_path"] for x in rows}) > 1:
        raise SystemExit("participants were tested with different thresholds/checkpoints")

    text = ["# Table 8: live per-sign test, RT-DETR-L (P1 seed 0)", "",
            "First attempt of every trial only; repeated attempts (key R) are not counted.", ""]
    if len(per) > 1:
        for pid, r in per.items():
            tab, miss = table(r)
            write(d / f"table8_{pid}.csv", tab)
            meta = r[0] if r else {}
            text += [f"## Participant {pid} ({meta.get('signer_status', '')}, {meta.get('distance_m', '')} m, "
                     f"lighting: {meta.get('lighting', '')})", "", *md(tab), "", *summary(r, tab, miss, pid), notes[pid], ""]
        text += ["## All participants", ""]
    tab, miss = table(rows)
    write(d / "table8.csv", tab)
    lines = summary(rows, tab, miss, "All participants" if len(per) > 1 else next(iter(per)))
    lines += list(notes.values())
    text += [*md(tab), "", *lines, ""]
    (d / "table8.md").write_text("\n".join(text), encoding="utf-8")
    print("\n".join(md(tab)))
    print("\n".join(lines))
    print(f"-> {d / 'table8.csv'}, {d / 'table8.md'}")


if __name__ == "__main__":
    main()

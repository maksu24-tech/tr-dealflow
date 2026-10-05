"""Score model outputs against the hand-labelled test set.

    python -m dealflow score --labels data/etiketleme_dolu.xlsx

Ground truth = the human labels. For every article in eval_set and every model in
model_runs we compare: deal detection, company, stage, amount (+currency), lead
investors and Türkiye link. Name matching is forgiving on spelling (typos, "VC"
suffixes, fund vs. firm names) but strict on meaning.
"""
import csv
import json
import re
from difflib import SequenceMatcher

from openpyxl import load_workbook

from .clean import INVESTOR_ALIASES, _fold

STAGE_TR = {"ön tohum": "pre_seed", "tohum": "seed", "seri a": "series_a", "seri b": "series_b",
            "seri c": "series_c", "seri d+": "series_d_plus", "köprü": "bridge", "seri a öncesi": "bridge",
            "büyüme": "growth", "satın alma": "acquisition", "belirtilmemiş": "unknown"}
LINK_TR = {"türkiye merkezli": "hq_turkey", "yurtdışında türk kurucu": "turkish_founders",
           "sadece türk yatırımcı": "turkish_investor", "yok": "none"}
FIELDS = ["deal_detected", "company", "stage", "amount", "lead_investors", "turkey_link"]
NOISE = re.compile(r"\b(vc|ventures?|capital|partners|gsyf|fund|holding|a\.ş\.|inc|ltd|group|management)\b")


def _name(s: str) -> str:
    s = INVESTOR_ALIASES.get(s.strip(), s.strip())
    return " ".join(NOISE.sub(" ", _fold(s)).replace(".", " ").split())


def same_name(a: str, b: str) -> bool:
    x, y = _name(a), _name(b)
    if not x or not y:
        return False
    return x == y or x in y or y in x or SequenceMatcher(None, x, y).ratio() >= 0.85


def same_set(truth: list[str], pred: list[str]) -> bool:
    return all(any(same_name(t, p) for p in pred) for t in truth) and \
           all(any(same_name(p, t) for t in truth) for p in pred)


def _amount(v):
    if v is None or str(v).strip() == "":
        return None
    return float(re.sub(r"[^\d.]", "", str(v)))


def load_labels(path):
    ws = load_workbook(path, data_only=True)["Etiketler"]
    head = [c.value for c in ws[1]]
    col = {h: i for i, h in enumerate(head)}
    out = {}
    for row in ws.iter_rows(min_row=2, values_only=True):
        aid = row[col["ID (dokunma)"]]
        if not aid:
            continue
        is_deal = str(row[col["Yatırım/satın alma haberi mi?"]] or "").strip().lower() == "evet"
        lead = str(row[col["Lider yatırımcı(lar)"]] or "").strip()
        lead = [] if lead.lower() in ("", "belirtilmemiş", "none") else [x.strip() for x in lead.split(",") if x.strip()]
        stage = str(row[col["Tur aşaması"]] or "").strip().lower()
        link = str(row[col["Türkiye bağlantısı"]] or "").strip().lower()
        out[aid] = {
            "no": row[col["No"]], "title": row[col["Başlık"]], "is_deal": is_deal,
            "company": str(row[col["Şirket"]] or "").strip(),
            "stage": STAGE_TR.get(stage) if stage else None,
            "amount": _amount(row[col["Tutar"]]),
            "currency": (str(row[col["Para birimi"]] or "").strip().upper() or None),
            "lead": lead, "turkey_link": LINK_TR.get(link) if link else None,
            "note": row[col["Not"]],
        }
    return out


def model_answers(conn, model):
    ans = {}
    for aid, js in conn.execute("SELECT article_id, deal_json FROM model_runs WHERE model = ?", (model,)):
        ans.setdefault(aid, [])
        if js and not js.startswith('{"error'):
            ans[aid].append(json.loads(js))
    return ans


def pick(deals, company):
    for d in deals:
        if company and same_name(company, d.get("company", "")):
            return d
    return deals[0] if deals else None


def compare(truth, deals):
    """Return {field: True/False/None}. None = not scorable (missing label)."""
    r = {"deal_detected": bool(deals) == truth["is_deal"]}
    if not truth["is_deal"]:
        return r
    d = pick(deals, truth["company"])
    if d is None:  # model said "no deal": every labelled field counts as wrong
        labelled = {"company": truth["company"], "stage": truth["stage"], "amount": True,
                    "lead_investors": True, "turkey_link": truth["turkey_link"]}
        return {**r, **{f: (False if v else None) for f, v in labelled.items()}}
    r["company"] = same_name(truth["company"], d.get("company", "")) if truth["company"] else None
    r["stage"] = (d.get("stage") or "unknown") == truth["stage"] if truth["stage"] else None
    pa = d.get("amount")
    if truth["amount"] is None:
        r["amount"] = pa in (None, 0)
    else:
        r["amount"] = pa is not None and abs(pa - truth["amount"]) <= 0.02 * truth["amount"] \
            and (d.get("currency") or "").upper() == (truth["currency"] or "").upper()
    r["lead_investors"] = same_set(truth["lead"], d.get("lead_investors") or [])
    r["turkey_link"] = d.get("turkey_link") == truth["turkey_link"] if truth["turkey_link"] else None
    return r


def score(conn, labels, models, overrides=None):
    """overrides: {article_id: {field: value}} applied to the labels (adjudicated corrections)."""
    labels = {k: {**v, **(overrides or {}).get(k, {})} for k, v in labels.items()}
    table, rows = {}, []
    for m in models:
        ans = model_answers(conn, m)
        tally = {f: [0, 0] for f in FIELDS}
        for aid, t in labels.items():
            if t.get("_exclude"):  # genuinely ambiguous article: scored for nobody
                continue
            res = compare(t, ans.get(aid, []))
            for f, ok in res.items():
                if ok is None:
                    continue
                tally[f][0] += int(ok)
                tally[f][1] += 1
            rows.append({"model": m, "no": t["no"], "article_id": aid, **{f: res.get(f) for f in FIELDS},
                         "pred": json.dumps(pick(ans.get(aid, []), t["company"]), ensure_ascii=False)})
        table[m] = tally
    return table, rows


def write_rows(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

from __future__ import annotations

import calendar
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path

DATA_PATH = Path("liberedebride/reviews.json")
RESULTS_PATH = Path("current-review-output/results.json")
CURRENT_CID = "8053340920072279118"
MONTHS = (
    "janvier", "février", "mars", "avril", "mai", "juin",
    "juillet", "août", "septembre", "octobre", "novembre", "décembre",
)

def clean(value):
    return re.sub(r"\s+", " ", str(value or "")).strip()

def parse_iso(value):
    value = clean(value)
    if not value or value.startswith("1970-"):
        return None
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    dt = dt.astimezone(timezone.utc)
    if dt.year < 2007 or dt > datetime.now(timezone.utc) + timedelta(days=2):
        return None
    return dt

def parse_micros(value):
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    return parse_iso(datetime.fromtimestamp(number / 1_000_000, tz=timezone.utc).isoformat())

def day_label(dt):
    return f"{dt.day} {MONTHS[dt.month - 1]} {dt.year}"

def shift_months(dt, count):
    total = dt.year * 12 + dt.month - 1 - count
    year, month0 = divmod(total, 12)
    month = month0 + 1
    day = min(dt.day, calendar.monthrange(year, month)[1])
    return dt.replace(year=year, month=month, day=day)

def date_info(item, collected, order):
    exact = parse_iso(item.get("published_at")) or parse_micros(item.get("posted_at_unix_micros"))
    if exact:
        iso = exact.isoformat().replace("+00:00", "Z")
        return {
            "publishedAt": iso,
            "dateLabel": day_label(exact),
            "datePrecision": "day",
            "sortAt": iso,
            "dateSource": "google-timestamp",
        }

    text = clean(item.get("When") or item.get("when")).lower()
    text = re.sub(r"^(?:modifié|edited)\s+", "", text)
    patterns = (
        (r"(?:il y a\s+)?(?:(\d+)|un|une)\s+(minute|minutes|heure|heures|jour|jours|semaine|semaines|mois|an|ans|année|années)", "fr"),
        (r"(?:(\d+)|a|an|one)\s+(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago", "en"),
    )
    for pattern, language in patterns:
        match = re.search(pattern, text)
        if not match:
            continue
        amount = int(match.group(1) or 1)
        unit = match.group(2)
        if unit.startswith(("minute", "heure", "hour")):
            delta = timedelta(minutes=amount) if unit.startswith("minute") else timedelta(hours=amount)
            dt = collected - delta
            return {
                "publishedAt": dt.date().isoformat(),
                "dateLabel": day_label(dt),
                "datePrecision": "day",
                "sortAt": dt.isoformat().replace("+00:00", "Z"),
                "dateSource": "google-relative",
            }
        if unit.startswith(("jour", "day")):
            dt = collected - timedelta(days=amount)
            return {
                "publishedAt": dt.date().isoformat(),
                "dateLabel": day_label(dt),
                "datePrecision": "day",
                "sortAt": dt.isoformat().replace("+00:00", "Z"),
                "dateSource": "google-relative",
            }
        if unit.startswith(("semaine", "week")):
            dt = collected - timedelta(weeks=amount)
            return {
                "publishedAt": f"{dt.year:04d}-{dt.month:02d}",
                "dateLabel": f"{MONTHS[dt.month - 1]} {dt.year}",
                "datePrecision": "month",
                "sortAt": dt.isoformat().replace("+00:00", "Z"),
                "dateSource": "google-relative",
            }
        if unit.startswith("month") or unit == "mois":
            dt = shift_months(collected, amount)
            return {
                "publishedAt": f"{dt.year:04d}-{dt.month:02d}",
                "dateLabel": f"{MONTHS[dt.month - 1]} {dt.year}",
                "datePrecision": "month",
                "sortAt": dt.isoformat().replace("+00:00", "Z"),
                "dateSource": "google-relative",
            }
        dt = shift_months(collected, amount * 12)
        return {
            "publishedAt": str(dt.year),
            "dateLabel": str(dt.year),
            "datePrecision": "year",
            "sortAt": dt.isoformat().replace("+00:00", "Z"),
            "dateSource": "google-relative",
        }

    sort_at = (collected - timedelta(seconds=order)).isoformat().replace("+00:00", "Z")
    return {
        "publishedAt": "",
        "dateLabel": "date non communiquée",
        "datePrecision": "unknown",
        "sortAt": sort_at,
        "dateSource": "collection-order-only",
    }

def load_rows(path):
    if not path.exists():
        return []
    raw = path.read_text(errors="replace").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            return value.get("results") or value.get("data") or [value]
    except Exception:
        pass
    rows = []
    for line in raw.splitlines():
        try:
            value = json.loads(line)
            if isinstance(value, dict):
                rows.append(value)
        except Exception:
            pass
    return rows

data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
before = deepcopy(data)
original_ids = {review["id"] for review in data["reviews"]}
original_legacy_ids = {review["id"] for review in data["reviews"] if review.get("profile") == "legacy"}
existing = {review["id"]: review for review in data["reviews"]}

candidates = {}
for row in load_rows(RESULTS_PATH):
    if not isinstance(row, dict):
        continue
    cid = clean(row.get("cid"))
    if cid and cid != CURRENT_CID:
        continue
    groups = (row.get("user_reviews") or [], row.get("user_reviews_extended") or [])
    for group in groups:
        for item in group if isinstance(group, list) else []:
            if not isinstance(item, dict):
                continue
            review_id = clean(item.get("review_id"))
            if not review_id:
                continue
            old = candidates.get(review_id)
            comment = clean(item.get("text_original") or item.get("Description") or item.get("description"))
            old_comment = clean((old or {}).get("text_original") or (old or {}).get("Description"))
            if old is None or len(comment) > len(old_comment) or parse_iso(item.get("published_at")):
                candidates[review_id] = item

collected = datetime.now(timezone.utc)
added = updated = 0
for order, (review_id, item) in enumerate(candidates.items()):
    author = clean(item.get("Name") or item.get("author"))
    try:
        rating = int(round(float(item.get("Rating") or item.get("rating_float") or item.get("rating") or 0)))
    except (TypeError, ValueError):
        rating = 0
    if not author or not 1 <= rating <= 5:
        continue
    comment = clean(item.get("text_original") or item.get("Description") or item.get("description"))
    incoming_date = date_info(item, collected, order)
    review = existing.get(review_id)

    if review is None:
        review = {
            "id": review_id,
            "author": author,
            "rating": rating,
            "comment": comment,
            "profile": "current",
            "commentVerifiedEmpty": not bool(comment),
            **incoming_date,
        }
        data["reviews"].append(review)
        existing[review_id] = review
        added += 1
        continue

    changed = False
    if author != review.get("author"):
        review["author"] = author
        changed = True
    if rating != review.get("rating"):
        review["rating"] = rating
        changed = True
    if len(comment) > len(clean(review.get("comment"))):
        review["comment"] = comment
        review["commentVerifiedEmpty"] = False
        changed = True

    rank = {"unknown": 0, "year": 1, "month": 2, "day": 3}
    old_rank = rank.get(review.get("datePrecision"), 0)
    new_rank = rank.get(incoming_date.get("datePrecision"), 0)
    if new_rank > old_rank or incoming_date.get("dateSource") == "google-timestamp" and review.get("dateSource") != "google-timestamp":
        review.update(incoming_date)
        changed = True
    if changed:
        updated += 1

ids = [review["id"] for review in data["reviews"]]
assert len(ids) == len(set(ids)), "Un identifiant Google est dupliqué"
assert original_ids.issubset(set(ids)), "Un avis validé a disparu"
assert original_legacy_ids == {review["id"] for review in data["reviews"] if review.get("profile") == "legacy"}, "Les avis historiques ont été modifiés"

data["reviews"].sort(key=lambda review: review.get("sortAt") or "", reverse=True)
data["counts"] = {
    "total": len(data["reviews"]),
    "current": sum(review.get("profile") == "current" for review in data["reviews"]),
    "legacyFrozen": sum(review.get("profile") == "legacy" for review in data["reviews"]),
}
data["rating"] = round(sum(int(review["rating"]) for review in data["reviews"]) / len(data["reviews"]), 2)
data["legacyAutoUpdate"] = False

comparable_before = deepcopy(before)
comparable_after = deepcopy(data)
comparable_before.pop("updatedAt", None)
comparable_after.pop("updatedAt", None)
if comparable_after != comparable_before:
    data["updatedAt"] = collected.isoformat().replace("+00:00", "Z")
    DATA_PATH.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Ajoutés: {added}; enrichis: {updated}; total conservé: {len(data['reviews'])}")
else:
    print(f"Aucun changement fiable; les {len(data['reviews'])} avis sont conservés.")

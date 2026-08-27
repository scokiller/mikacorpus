from __future__ import annotations

import calendar
import json
import re
from copy import deepcopy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

DATA_PATH = Path("liberedebride/reviews.json")
RESULTS_PATH = Path("current-review-output/results.json")
CURRENT_CID = "8053340920072279118"
MONTHS_FR = (
    "janvier",
    "février",
    "mars",
    "avril",
    "mai",
    "juin",
    "juillet",
    "août",
    "septembre",
    "octobre",
    "novembre",
    "décembre",
)
PRECISION_RANK = {"unknown": 0, "year": 1, "month": 2, "day": 3}


def clean(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip()


def parse_iso(value: Any) -> datetime | None:
    text = clean(value)
    if not text or text.startswith("1970-"):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    parsed = parsed.astimezone(timezone.utc)
    now = datetime.now(timezone.utc)
    if parsed.year < 2007 or parsed > now + timedelta(days=2):
        return None
    return parsed


def parse_micros(value: Any) -> datetime | None:
    try:
        number = int(value or 0)
    except (TypeError, ValueError):
        return None
    if number <= 0:
        return None
    try:
        parsed = datetime.fromtimestamp(number / 1_000_000, tz=timezone.utc)
    except (OverflowError, OSError, ValueError):
        return None
    return parse_iso(parsed.isoformat())


def day_label(value: datetime) -> str:
    return f"{value.day} {MONTHS_FR[value.month - 1]} {value.year}"


def shift_months(value: datetime, count: int) -> datetime:
    total = value.year * 12 + value.month - 1 - count
    year, month_zero = divmod(total, 12)
    month = month_zero + 1
    day = min(value.day, calendar.monthrange(year, month)[1])
    return value.replace(year=year, month=month, day=day)


def exact_date_info(item: dict[str, Any]) -> dict[str, str] | None:
    exact = parse_iso(item.get("published_at")) or parse_micros(
        item.get("posted_at_unix_micros")
    )
    if exact is None:
        return None
    iso = exact.isoformat().replace("+00:00", "Z")
    return {
        "publishedAt": iso,
        "dateLabel": day_label(exact),
        "datePrecision": "day",
        "sortAt": iso,
        "dateSource": "google-timestamp",
    }


def relative_date_info(
    item: dict[str, Any], collected_at: datetime
) -> dict[str, str] | None:
    text = clean(item.get("When") or item.get("when") or item.get("dateLabel")).lower()
    text = re.sub(r"^(?:modifié|modifie|edited)\s+", "", text)
    patterns = (
        re.compile(
            r"(?:il y a\s+)?(?:(\d+)|un|une)\s+"
            r"(minute|minutes|heure|heures|jour|jours|semaine|semaines|mois|an|ans|année|années)"
        ),
        re.compile(
            r"(?:(\d+)|a|an|one)\s+"
            r"(minute|minutes|hour|hours|day|days|week|weeks|month|months|year|years)\s+ago"
        ),
    )
    match = None
    for pattern in patterns:
        match = pattern.search(text)
        if match:
            break
    if match is None:
        return None

    amount = int(match.group(1) or 1)
    unit = match.group(2)
    if unit.startswith("minute"):
        value = collected_at - timedelta(minutes=amount)
        precision = "day"
    elif unit.startswith(("heure", "hour")):
        value = collected_at - timedelta(hours=amount)
        precision = "day"
    elif unit.startswith(("jour", "day")):
        value = collected_at - timedelta(days=amount)
        precision = "day"
    elif unit.startswith(("semaine", "week")):
        value = collected_at - timedelta(weeks=amount)
        precision = "month"
    elif unit == "mois" or unit.startswith("month"):
        value = shift_months(collected_at, amount)
        precision = "month"
    else:
        value = shift_months(collected_at, amount * 12)
        precision = "year"

    sort_at = value.isoformat().replace("+00:00", "Z")
    if precision == "day":
        published_at = value.date().isoformat()
        label = day_label(value)
    elif precision == "month":
        published_at = f"{value.year:04d}-{value.month:02d}"
        label = f"{MONTHS_FR[value.month - 1]} {value.year}"
    else:
        published_at = str(value.year)
        label = str(value.year)

    return {
        "publishedAt": published_at,
        "dateLabel": label,
        "datePrecision": precision,
        "sortAt": sort_at,
        "dateSource": "google-relative",
    }


def date_info(
    item: dict[str, Any], collected_at: datetime
) -> dict[str, str] | None:
    return exact_date_info(item) or relative_date_info(item, collected_at)


def load_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8", errors="replace").strip()
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = None
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        nested = value.get("results") or value.get("data")
        if isinstance(nested, list):
            return [row for row in nested if isinstance(row, dict)]
        return [value]

    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def item_comment(item: dict[str, Any]) -> str:
    return clean(
        item.get("text_original")
        or item.get("Description")
        or item.get("description")
        or item.get("comment")
    )


def item_author(item: dict[str, Any]) -> str:
    return clean(item.get("Name") or item.get("author") or item.get("author_name"))


def item_rating(item: dict[str, Any]) -> int:
    try:
        value = float(
            item.get("Rating")
            or item.get("rating_float")
            or item.get("rating")
            or 0
        )
    except (TypeError, ValueError):
        return 0
    return int(round(value))


def candidate_score(item: dict[str, Any]) -> tuple[int, int, int]:
    exact = 1 if exact_date_info(item) else 0
    relative = 1 if clean(item.get("When") or item.get("when") or item.get("dateLabel")) else 0
    return exact, len(item_comment(item)), relative


def collect_candidates(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    candidates: dict[str, dict[str, Any]] = {}
    for row in rows:
        cid = clean(row.get("cid"))
        if cid and cid != CURRENT_CID:
            continue
        for group in (row.get("user_reviews") or [], row.get("user_reviews_extended") or []):
            if not isinstance(group, list):
                continue
            for item in group:
                if not isinstance(item, dict):
                    continue
                review_id = clean(item.get("review_id") or item.get("reviewId"))
                if not review_id:
                    continue
                previous = candidates.get(review_id)
                if previous is None or candidate_score(item) > candidate_score(previous):
                    candidates[review_id] = item
    return candidates


def main() -> None:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    reviews = data.get("reviews")
    if not isinstance(reviews, list) or not reviews:
        raise RuntimeError("La base des avis validés est absente ou vide.")

    before = deepcopy(data)
    original_ids = {clean(review.get("id")) for review in reviews}
    if "" in original_ids or len(original_ids) != len(reviews):
        raise RuntimeError("La base validée contient un identifiant vide ou dupliqué.")
    original_legacy_ids = {
        clean(review.get("id"))
        for review in reviews
        if review.get("profile") == "legacy"
    }
    existing = {clean(review.get("id")): review for review in reviews}
    candidates = collect_candidates(load_rows(RESULTS_PATH))
    collected_at = datetime.now(timezone.utc)
    added = 0
    enriched = 0
    skipped_without_date = 0

    for review_id, item in candidates.items():
        author = item_author(item)
        rating = item_rating(item)
        if not author or not 1 <= rating <= 5:
            continue
        comment = item_comment(item)
        incoming_date = date_info(item, collected_at)
        review = existing.get(review_id)

        if review is None:
            # Ne jamais classer une nouvelle carte avec la date de collecte.
            # Sans date ou période Google exploitable, elle attend le prochain passage.
            if incoming_date is None:
                skipped_without_date += 1
                continue
            review = {
                "id": review_id,
                "author": author,
                "rating": rating,
                "comment": comment,
                "profile": "current",
                "commentVerifiedEmpty": not bool(comment),
                **incoming_date,
            }
            reviews.append(review)
            existing[review_id] = review
            added += 1
            continue

        if review.get("profile") != "current":
            continue

        changed = False
        if author != clean(review.get("author")):
            review["author"] = author
            changed = True
        if rating != int(review.get("rating") or 0):
            review["rating"] = rating
            changed = True
        if len(comment) > len(clean(review.get("comment"))):
            review["comment"] = comment
            review["commentVerifiedEmpty"] = False
            changed = True

        if incoming_date is not None:
            old_rank = PRECISION_RANK.get(clean(review.get("datePrecision")), 0)
            new_rank = PRECISION_RANK.get(incoming_date["datePrecision"], 0)
            exact_upgrade = (
                incoming_date["dateSource"] == "google-timestamp"
                and review.get("dateSource") != "google-timestamp"
            )
            if new_rank > old_rank or exact_upgrade:
                review.update(incoming_date)
                changed = True

        if changed:
            enriched += 1

    final_ids = [clean(review.get("id")) for review in reviews]
    if len(final_ids) != len(set(final_ids)):
        raise RuntimeError("La fusion a créé un identifiant Google dupliqué.")
    if not original_ids.issubset(set(final_ids)):
        raise RuntimeError("La fusion a supprimé au moins un avis validé.")
    final_legacy_ids = {
        clean(review.get("id"))
        for review in reviews
        if review.get("profile") == "legacy"
    }
    if final_legacy_ids != original_legacy_ids:
        raise RuntimeError("Les avis historiques MR Performance ont été modifiés.")

    reviews.sort(key=lambda review: clean(review.get("sortAt")), reverse=True)
    data["counts"] = {
        "total": len(reviews),
        "current": sum(review.get("profile") == "current" for review in reviews),
        "legacyFrozen": sum(review.get("profile") == "legacy" for review in reviews),
    }
    data["rating"] = round(
        sum(int(review.get("rating") or 0) for review in reviews) / len(reviews), 2
    )
    data["legacyAutoUpdate"] = False

    comparable_before = deepcopy(before)
    comparable_after = deepcopy(data)
    comparable_before.pop("updatedAt", None)
    comparable_after.pop("updatedAt", None)
    if comparable_after != comparable_before:
        data["updatedAt"] = collected_at.isoformat().replace("+00:00", "Z")
        DATA_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"Ajoutés: {added}; enrichis: {enriched}; "
            f"ignorés sans date fiable: {skipped_without_date}; total: {len(reviews)}"
        )
    else:
        print(
            f"Aucun changement fiable; les {len(reviews)} avis restent intacts "
            f"({skipped_without_date} nouveau(x) sans date exploitable ignoré(s))."
        )


if __name__ == "__main__":
    main()

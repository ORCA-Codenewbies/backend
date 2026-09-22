from fastapi import APIRouter, Depends, Query, Request
import time
import uuid
import logging
from datetime import datetime, timezone

from backend.api.dependencies.rate_limit import check_rate_limit_unauthenticated
from data_sources.cyclone import get_active_cyclones
from data_sources.hazards import detect_catastrophic_hazards
from data_sources.incois import get_ocean_state_forecast
from agents.marine_safety.bsi import evaluate_conditions, SeaState, BoatProfile

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/v1/marine-alerts",
    tags=["Alerts"],
)

_alert_cache = {}
CACHE_TTL_SECONDS = 300

# BSI advisory → normalized alert severity and title
_BSI_ADVISORY_MAP = {
    "DANGER":  ("CRITICAL", "Do Not Fish – Dangerous Conditions"),
    "WARNING": ("HIGH",     "Marine Safety Warning"),
    "CAUTION": ("WARNING",  "Exercise Caution"),
}

@router.get("")
def get_marine_alerts(
    request: Request,
    lat: float = Query(..., description="Latitude"),
    lon: float = Query(..., description="Longitude"),
    _: str = Depends(check_rate_limit_unauthenticated)
):
    """
    Get live marine safety status.  Combines:
      1. Catastrophic hazard detection (cyclone, earthquake, tsunami)
      2. Operational BSI safety evaluation (wave height, wind speed)
    Results are cached by rounded coordinates for 5 minutes.
    """
    now = time.time()
    retrieved_at = datetime.now(timezone.utc).isoformat()

    # 1 decimal place = ~11.1km resolution
    cache_key = f"{round(lat, 1)},{round(lon, 1)}"

    if cache_key in _alert_cache:
        cached_entry = _alert_cache[cache_key]
        if now - cached_entry["timestamp"] < CACHE_TTL_SECONDS:
            logger.info(f"Returning cached alerts for {cache_key}")
            return cached_entry["data"]

    logger.info(f"Fetching fresh alerts for {cache_key} (lat={lat}, lon={lon})")

    # ── 1. Catastrophic hazard detection ──────────────────────────────────
    cyclone_data = get_active_cyclones(lat, lon)
    hazard_data = detect_catastrophic_hazards(lat, lon, "general_safety", cyclone_data)

    retrieved_at = datetime.now(timezone.utc).isoformat()
    normalized_alerts = []

    hazards = hazard_data.get("hazards", [])
    catastrophic_status = hazard_data.get("catastrophic_status", "UNKNOWN")

    if catastrophic_status == "UNKNOWN" or cyclone_data.get("source_status") == "JTWC_FETCH_FAILED":
        global_status = "UNKNOWN"
    elif catastrophic_status == "ACTIVE":
        global_status = "ALERT"
    else:
        global_status = "CLEAR"

    for hz in hazards:
        hz_type = hz.get("type", "MARINE_SAFETY")
        source = "JTWC" if hz_type == "CYCLONE" else "USGS" if hz_type in ["EARTHQUAKE", "TSUNAMI_WARNING"] else "MARINE_DATA"
        normalized_alerts.append({
            "id": str(uuid.uuid4()),
            "type": hz_type,
            "severity": hz.get("severity", "HIGH"),
            "title": f"{hz_type.replace('_', ' ').title()} Alert",
            "message": hz.get("description", "A marine hazard was detected."),
            "source": source,
            "distance_km": None,
            "updated_at": retrieved_at
        })

    # ── 2. BSI operational safety evaluation (reuses marine_safety logic) ─
    # Only run when there are no catastrophic hazards — catastrophic alerts
    # already subsume operational caution.
    if global_status == "CLEAR":
        try:
            osf = get_ocean_state_forecast(lat, lon, datetime.now(timezone.utc))

            def _val(obs, default):
                v = obs.value if obs and obs.value is not None else None
                try:
                    return float(v) if v is not None and str(v).upper() != "UNKNOWN" else default
                except (TypeError, ValueError):
                    return default

            wave_height = _val(osf.get("wave_height"), 1.2)
            wave_period = _val(osf.get("wave_period"), 6.5)
            wind_speed_ms = _val(osf.get("wind_speed"), 4.5)

            sea_state = SeaState(
                hs_m=wave_height,
                tz_s=wave_period,
                wave_dir_deg=180.0,
                swell_dir_deg=None,
                wind_speed_ms=wind_speed_ms,
            )
            boat = BoatProfile(max_hs_m=2.0)  # Small fishing vessel assumption
            bsi_report = evaluate_conditions(sea_state, boat)

            advisory = bsi_report.advisory  # SAFE / CAUTION / WARNING / DANGER
            if advisory in _BSI_ADVISORY_MAP:
                severity, title = _BSI_ADVISORY_MAP[advisory]
                hazard_notes = "; ".join(bsi_report.hazards) if bsi_report.hazards else ""
                message_parts = [
                    f"Wind: {wind_speed_ms:.1f} m/s.",
                    f"Wave height: {wave_height:.2f} m.",
                ]
                if hazard_notes:
                    message_parts.append(f"Conditions: {hazard_notes}.")
                message_parts.append("Exercise appropriate caution before going to sea.")

                normalized_alerts.append({
                    "id": str(uuid.uuid4()),
                    "type": "MARINE_SAFETY",
                    "severity": severity,
                    "title": title,
                    "message": " ".join(message_parts),
                    "source": "ORCA Marine Safety (BSI)",
                    "distance_km": None,
                    "updated_at": retrieved_at,
                })
                global_status = "ALERT"

        except Exception as e:
            logger.warning(f"BSI evaluation failed for {cache_key}: {e}")
            # Do not degrade to UNKNOWN; catastrophic check already returned CLEAR.

    response_data = {
        "status": global_status,
        "alerts": normalized_alerts,
        "retrieved_at": retrieved_at,
        "cache_ttl_seconds": CACHE_TTL_SECONDS
    }

    _alert_cache[cache_key] = {
        "timestamp": now,
        "data": response_data
    }

    return response_data

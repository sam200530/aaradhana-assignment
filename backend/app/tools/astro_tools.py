"""
AstroAgent tools — real ephemeris, geocoding, transits, and knowledge RAG.
All tools return structured dicts for easy eval assertion.
"""
from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

# ── Ephemeris ──────────────────────────────────────────────────────────────────
try:
    import swisseph as swe
    SWE_AVAILABLE = True
except ImportError:
    SWE_AVAILABLE = False

# ── Geo ────────────────────────────────────────────────────────────────────────
try:
    from geopy.geocoders import Nominatim
    from timezonefinder import TimezoneFinder
    GEO_AVAILABLE = True
except ImportError:
    GEO_AVAILABLE = False

import pytz

# ── Planet constants ───────────────────────────────────────────────────────────
PLANETS = {
    "Sun":     swe.SUN     if SWE_AVAILABLE else 0,
    "Moon":    swe.MOON    if SWE_AVAILABLE else 1,
    "Mercury": swe.MERCURY if SWE_AVAILABLE else 2,
    "Venus":   swe.VENUS   if SWE_AVAILABLE else 3,
    "Mars":    swe.MARS    if SWE_AVAILABLE else 4,
    "Jupiter": swe.JUPITER if SWE_AVAILABLE else 5,
    "Saturn":  swe.SATURN  if SWE_AVAILABLE else 6,
    "Uranus":  swe.URANUS  if SWE_AVAILABLE else 7,
    "Neptune": swe.NEPTUNE if SWE_AVAILABLE else 8,
    "Pluto":   swe.PLUTO   if SWE_AVAILABLE else 9,
}

ZODIAC_SIGNS = [
    "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
    "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces",
]

HOUSE_SYSTEMS = b"P"  # Placidus


def _deg_to_sign(deg: float) -> tuple[str, float]:
    """Convert ecliptic longitude to zodiac sign + degrees within sign."""
    idx = int(deg / 30) % 12
    within = deg % 30
    return ZODIAC_SIGNS[idx], within


def _jd_from_datetime(dt: datetime) -> float:
    """Convert datetime to Julian Day Number."""
    if not SWE_AVAILABLE:
        raise RuntimeError("pyswisseph not installed")
    return swe.julday(dt.year, dt.month, dt.day,
                      dt.hour + dt.minute / 60.0 + dt.second / 3600.0)


# ── Tool 1: geocode_place ──────────────────────────────────────────────────────

def geocode_place(place: str) -> dict:
    """
    Resolve a place name to lat/lon/timezone.
    Returns: {"place", "latitude", "longitude", "timezone", "error"?}
    """
    if not GEO_AVAILABLE:
        return {"error": "geopy/timezonefinder not installed", "place": place}

    try:
        geolocator = Nominatim(user_agent="astroagent/1.0")
        location = geolocator.geocode(place, timeout=10)
        if location is None:
            return {"error": f"Could not geocode '{place}'", "place": place}

        tf = TimezoneFinder()
        tz_name = tf.timezone_at(lat=location.latitude, lng=location.longitude)

        return {
            "place": place,
            "resolved_name": location.address,
            "latitude": round(location.latitude, 6),
            "longitude": round(location.longitude, 6),
            "timezone": tz_name or "UTC",
        }
    except Exception as e:
        return {"error": str(e), "place": place}


# ── Tool 2: compute_birth_chart ────────────────────────────────────────────────

def compute_birth_chart(
    date_of_birth: str,
    time_of_birth: str,
    latitude: float,
    longitude: float,
    timezone: str,
) -> dict:
    """
    Compute a natal chart using Swiss Ephemeris.
    date_of_birth: YYYY-MM-DD
    time_of_birth: HH:MM (24h)
    Returns: {"planets": {...}, "houses": [...], "ascendant": ..., "midheaven": ...}
    """
    if not SWE_AVAILABLE:
        return {"error": "pyswisseph not installed — cannot compute real chart"}

    try:
        # Parse date/time
        dt_str = f"{date_of_birth} {time_of_birth}"
        local_dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M")

        # Convert to UTC
        tz = pytz.timezone(timezone)
        local_aware = tz.localize(local_dt)
        utc_dt = local_aware.astimezone(pytz.utc).replace(tzinfo=None)

        jd = _jd_from_datetime(utc_dt)

        # Compute planets
        planets_data = {}
        for name, planet_id in PLANETS.items():
            result, _ = swe.calc_ut(jd, planet_id)
            lon = result[0]
            sign, deg_in_sign = _deg_to_sign(lon)
            planets_data[name] = {
                "longitude": round(lon, 4),
                "sign": sign,
                "degree": round(deg_in_sign, 2),
                "retrograde": result[3] < 0,
            }

        # Compute houses (Placidus)
        cusps, ascmc = swe.houses(jd, latitude, longitude, HOUSE_SYSTEMS)
        houses = []
        for i, cusp in enumerate(cusps, 1):
            sign, deg = _deg_to_sign(cusp)
            houses.append({
                "house": i,
                "cusp_longitude": round(cusp, 4),
                "sign": sign,
                "degree": round(deg, 2),
            })

        asc_sign, asc_deg = _deg_to_sign(ascmc[0])
        mc_sign, mc_deg = _deg_to_sign(ascmc[1])

        return {
            "birth_utc": utc_dt.isoformat(),
            "julian_day": round(jd, 6),
            "planets": planets_data,
            "houses": houses,
            "ascendant": {
                "longitude": round(ascmc[0], 4),
                "sign": asc_sign,
                "degree": round(asc_deg, 2),
            },
            "midheaven": {
                "longitude": round(ascmc[1], 4),
                "sign": mc_sign,
                "degree": round(mc_deg, 2),
            },
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 3: get_daily_transits ─────────────────────────────────────────────────

def get_daily_transits(
    date: str | None = None,
    natal_chart: dict | None = None,
) -> dict:
    """
    Get current planetary transits and, if a natal chart is supplied,
    compute major aspects to natal planets.
    date: YYYY-MM-DD (defaults to today UTC)
    """
    if not SWE_AVAILABLE:
        return {"error": "pyswisseph not installed"}

    try:
        if date is None:
            date = datetime.utcnow().strftime("%Y-%m-%d")

        dt = datetime.strptime(date, "%Y-%m-%d").replace(
            hour=12, minute=0, tzinfo=timezone.utc
        )
        jd = _jd_from_datetime(dt.replace(tzinfo=None))

        transiting = {}
        for name, planet_id in PLANETS.items():
            result, _ = swe.calc_ut(jd, planet_id)
            lon = result[0]
            sign, deg = _deg_to_sign(lon)
            transiting[name] = {
                "longitude": round(lon, 4),
                "sign": sign,
                "degree": round(deg, 2),
                "retrograde": result[3] < 0,
            }

        # Compute aspects to natal chart if provided
        aspects = []
        if natal_chart and "planets" in natal_chart:
            ASPECT_ORBS = {0: 8, 60: 6, 90: 7, 120: 8, 180: 8, 150: 3, 30: 3}
            ASPECT_NAMES = {0: "Conjunction", 60: "Sextile", 90: "Square",
                            120: "Trine", 180: "Opposition", 150: "Quincunx", 30: "Semi-sextile"}
            for t_name, t_data in transiting.items():
                for n_name, n_data in natal_chart["planets"].items():
                    diff = abs(t_data["longitude"] - n_data["longitude"]) % 360
                    if diff > 180:
                        diff = 360 - diff
                    for angle, orb in ASPECT_ORBS.items():
                        if abs(diff - angle) <= orb:
                            aspects.append({
                                "transit_planet": t_name,
                                "natal_planet": n_name,
                                "aspect": ASPECT_NAMES[angle],
                                "orb": round(abs(diff - angle), 2),
                                "exact_at": angle,
                            })
                            break

        return {
            "date": date,
            "transiting_planets": transiting,
            "aspects_to_natal": aspects,
        }
    except Exception as e:
        return {"error": str(e)}


# ── Tool 4: knowledge_lookup ───────────────────────────────────────────────────

# Curated astrology reference data (embedded to avoid external deps in minimal setup)
ASTROLOGY_KNOWLEDGE = [
    {
        "id": "sun_signs",
        "topic": "Sun sign meanings",
        "content": (
            "The Sun sign represents core identity and ego. "
            "Aries: pioneering, bold, competitive. Taurus: steadfast, sensual, determined. "
            "Gemini: curious, adaptable, communicative. Cancer: nurturing, intuitive, protective. "
            "Leo: creative, generous, dramatic. Virgo: analytical, practical, service-oriented. "
            "Libra: diplomatic, aesthetic, relationship-focused. Scorpio: intense, transformative, perceptive. "
            "Sagittarius: philosophical, adventurous, freedom-loving. Capricorn: ambitious, disciplined, traditional. "
            "Aquarius: innovative, humanitarian, unconventional. Pisces: compassionate, dreamy, spiritual."
        ),
    },
    {
        "id": "moon_signs",
        "topic": "Moon sign meanings",
        "content": (
            "The Moon represents emotions, instincts, and subconscious patterns. "
            "Moon in Aries: emotionally impulsive, needs independence. Moon in Taurus: seeks stability and comfort. "
            "Moon in Gemini: emotionally curious, needs variety. Moon in Cancer: deeply empathetic, home-loving. "
            "Moon in Leo: needs recognition and warmth. Moon in Virgo: emotionally analytical, worry-prone. "
            "Moon in Libra: seeks harmony, hates conflict. Moon in Scorpio: intensely emotional, transformative. "
            "Moon in Sagittarius: emotionally adventurous. Moon in Capricorn: emotionally reserved, ambitious. "
            "Moon in Aquarius: detached but humanitarian. Moon in Pisces: highly empathetic, dreamy."
        ),
    },
    {
        "id": "houses",
        "topic": "Astrological houses",
        "content": (
            "The 12 houses represent life domains. "
            "1st House: self, appearance, beginnings. 2nd: money, values, possessions. "
            "3rd: communication, siblings, local travel. 4th: home, family, roots. "
            "5th: creativity, romance, children, joy. 6th: health, daily routine, service. "
            "7th: partnerships, marriage, open enemies. 8th: transformation, shared resources, death/rebirth. "
            "9th: philosophy, higher learning, long travel. 10th: career, public image, authority. "
            "11th: friendships, groups, hopes, technology. 12th: hidden matters, spirituality, isolation, undoing."
        ),
    },
    {
        "id": "aspects",
        "topic": "Planetary aspects",
        "content": (
            "Aspects are angular relationships between planets. "
            "Conjunction (0°): planets merge energy, powerful focus. "
            "Sextile (60°): harmonious opportunity, gentle flow. "
            "Square (90°): tension, challenge, growth through friction. "
            "Trine (120°): natural ease, innate talent, flow. "
            "Opposition (180°): polarity, awareness through others, balance needed. "
            "Quincunx/Inconjunct (150°): adjustment required, awkward energy. "
            "Hard aspects (square, opposition) build character; soft aspects (trine, sextile) offer gifts."
        ),
    },
    {
        "id": "ascendant",
        "topic": "Ascendant (Rising Sign)",
        "content": (
            "The Ascendant (Rising Sign) is the zodiac sign on the eastern horizon at birth. "
            "It represents your outer personality, first impressions, and physical appearance. "
            "It is the mask you wear and how the world sees you. "
            "Aries Rising: bold, energetic, direct. Taurus Rising: calm, sensual, reliable. "
            "Gemini Rising: chatty, versatile, youthful. Cancer Rising: warm, protective, moon-ruled demeanor. "
            "Leo Rising: magnetic, regal, dramatic. Virgo Rising: precise, service-oriented, health-conscious. "
            "Libra Rising: charming, aesthetic, people-pleasing. Scorpio Rising: intense, mysterious, magnetic. "
            "Sagittarius Rising: optimistic, philosophical, adventurous. Capricorn Rising: serious, structured, ambitious. "
            "Aquarius Rising: quirky, progressive, humanitarian. Pisces Rising: dreamy, empathetic, spiritual."
        ),
    },
    {
        "id": "saturn_return",
        "topic": "Saturn Return",
        "content": (
            "Saturn Return occurs approximately every 29.5 years when Saturn returns to its natal position. "
            "The first return (ages 27-30) is a major life restructuring: career, relationships, and identity are tested. "
            "The second return (ages 57-60) brings wisdom and legacy-building. "
            "Saturn demands accountability, discipline, and letting go of what no longer serves. "
            "It is often challenging but ultimately builds lasting foundations."
        ),
    },
    {
        "id": "retrograde",
        "topic": "Retrograde planets",
        "content": (
            "Retrograde planets appear to move backward from Earth's perspective. "
            "Mercury Retrograde (3x/year, ~3 weeks): communication mishaps, technology glitches, revisiting past decisions. "
            "Venus Retrograde (~18 months apart, 6 weeks): reassessing relationships and values. "
            "Mars Retrograde (~2 years apart, 10 weeks): redirecting energy and drive. "
            "Outer planet retrogrades (Jupiter through Pluto) last months and mark inner reflection periods. "
            "Natal retrograde planets indicate internalized, reflective expression of that planet's energy."
        ),
    },
    {
        "id": "north_south_node",
        "topic": "Lunar Nodes",
        "content": (
            "The North Node (Rahu) and South Node (Ketu) are mathematical points, not planets. "
            "The South Node represents past-life karma, innate skills, and comfort zones. "
            "The North Node represents the soul's evolutionary direction and growth edge this lifetime. "
            "Nodal axis transits (every 18.5 years) mark major life chapters and karmic turning points."
        ),
    },
    {
        "id": "disclaimer",
        "topic": "Astrology disclaimer",
        "content": (
            "Astrology is a symbolic language for self-reflection and guidance. "
            "It is not a predictive science and cannot determine medical diagnoses, legal outcomes, or financial certainty. "
            "Always consult qualified professionals for health, legal, and financial decisions. "
            "Readings are for inspiration and personal insight, not deterministic fate."
        ),
    },
]


def knowledge_lookup(query: str, top_k: int = 3) -> dict:
    """
    Simple keyword-based knowledge retrieval over curated astrology notes.
    Returns top matching entries for the query.
    """
    query_lower = query.lower()
    scored = []
    for entry in ASTROLOGY_KNOWLEDGE:
        score = 0
        text = (entry["topic"] + " " + entry["content"]).lower()
        for word in query_lower.split():
            if len(word) > 3 and word in text:
                score += text.count(word)
        scored.append((score, entry))

    scored.sort(key=lambda x: -x[0])
    results = [e for _, e in scored[:top_k] if _ > 0 or len(scored) <= top_k]

    if not results:
        results = scored[:1][0][1:] if scored else []

    return {
        "query": query,
        "results": [
            {"topic": r["topic"], "content": r["content"], "id": r["id"]}
            for r in results[:top_k]
        ],
    }

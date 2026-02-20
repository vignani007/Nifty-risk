from dataclasses import dataclass


@dataclass(frozen=True)
class TrafficLightResult:
    exceptions: int
    zone: str
    plus_factor: float
    multiplier_m: float  # m = 3 + plus_factor


def basel_traffic_light(exceptions: int) -> TrafficLightResult:
    """
    Basel traffic light approach for 1-day 99% VaR backtesting over 250 obs.

    Zones:
      Green:  0-4 exceptions
      Yellow: 5-9 exceptions
      Red:    >=10 exceptions

    Plus factor mapping (Basel Table 2):
      0-4:  0.00
      5:    0.40
      6:    0.50
      7:    0.65
      8:    0.75
      9:    0.85
      >=10: 1.00

    Multiplier: m = 3.0 + plus_factor
    """
    x = int(exceptions)

    if x <= 4:
        zone, plus = "green", 0.00
    elif x == 5:
        zone, plus = "yellow", 0.40
    elif x == 6:
        zone, plus = "yellow", 0.50
    elif x == 7:
        zone, plus = "yellow", 0.65
    elif x == 8:
        zone, plus = "yellow", 0.75
    elif x == 9:
        zone, plus = "yellow", 0.85
    else:
        zone, plus = "red", 1.00

    m = 3.0 + plus
    return TrafficLightResult(exceptions=x, zone=zone, plus_factor=plus, multiplier_m=m)

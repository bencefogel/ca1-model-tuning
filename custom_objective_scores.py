import numpy as np


def trunk_score(icar: np.ndarray, taxis: np.ndarray, delay: float, dur: float, threshold_ratio=0.5) -> float:
    """
    Returns a combined score based on the magnitude and duration of a CaR spike in the trunk compartment.

    - Magnitude: how large the CaR current deviation is from baseline
    - Duration: how long the current stays below a threshold
    """

    # Define baseline period (before stimulus)
    t_baseline = taxis < delay

    # Define stimulus window
    t_stim = (taxis >= delay) & (taxis <= delay + dur)

    # Calculate baseline and peak CaR current
    icar_baseline = np.mean(icar[t_baseline])
    icar_peak = np.min(icar[t_stim])  # Most negative value = peak inward Ca current

    # Magnitude = difference from baseline to peak
    magnitude = abs(icar_peak - icar_baseline)

    # Define threshold as a fraction between baseline and peak
    threshold = icar_baseline + threshold_ratio * (icar_peak - icar_baseline)

    # Re-extract stim window current and time
    t_stim = taxis[(taxis >= delay) & (taxis <= delay + dur)]
    icar_stim = icar[(taxis >= delay) & (taxis <= delay + dur)]

    dt = np.mean(np.diff(t_stim))  # average timestep

    # Boolean mask: where current is more negative than the threshold
    below_thresh = icar_stim < threshold

    # Identify all "start" and "end" times of threshold crossings
    starts = np.where(np.diff(below_thresh.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(below_thresh.astype(int)) == -1)[0] + 1

    # Edge cases: signal starts or ends below threshold
    if below_thresh[0]:
        starts = np.insert(starts, 0, 0)
    if below_thresh[-1]:
        ends = np.append(ends, len(below_thresh))

    # Compute durations (in ms) of each spike
    durations = (ends - starts) * dt

    # Total and longest duration of all Ca²⁺ events
    total_duration = np.sum(durations)
    longest_duration = np.max(durations) if len(durations) > 0 else 0

    print(
        f"Detected {len(durations)} Ca spike(s) with total duration {total_duration:.2f} ms and longest duration {longest_duration:.2f} ms"
    )

    # Final score: multiply spike magnitude with its persistence
    return magnitude * longest_duration


def oblique_penalty(icar: np.ndarray, taxis: np.ndarray, delay: float, dur: float, threshold_ratio=0.5) -> float:
    """
    Returns a penalty score for unwanted CaR influx in non-trunk apical branches (tuft and oblique)

    Penalizes:
    - Total duration below threshold
    - Longest duration of a single event
    - Peak inward CaR current (magnitude)
    """

    # Stimulus window
    t_stim = (taxis >= delay) & (taxis <= delay + dur)
    icar_stim = icar[t_stim]
    t_stim_window = taxis[t_stim]

    # Calculate baseline and peak
    icar_baseline = np.mean(icar[taxis < delay])
    icar_peak = np.min(icar_stim)
    threshold = icar_baseline + threshold_ratio * (icar_peak - icar_baseline)

    # Boolean mask for below-threshold Ca²⁺ influx
    below_thresh = icar_stim < threshold
    dt = np.mean(np.diff(t_stim_window))

    # Find all threshold crossings
    starts = np.where(np.diff(below_thresh.astype(int)) == 1)[0] + 1
    ends = np.where(np.diff(below_thresh.astype(int)) == -1)[0] + 1

    if below_thresh[0]:
        starts = np.insert(starts, 0, 0)
    if below_thresh[-1]:
        ends = np.append(ends, len(below_thresh))

    # Compute duration of each spike-like event
    durations = (ends - starts) * dt
    total_duration = np.sum(durations)
    longest_duration = np.max(durations) if len(durations) > 0 else 0

    # Combine all penalty components (equal weights used here)
    penalty = (
        1 * abs(icar_peak - icar_baseline) +  # spike magnitude
        1 * longest_duration +                # duration of longest event
        1 * total_duration                    # total duration across all events
    )

    return penalty

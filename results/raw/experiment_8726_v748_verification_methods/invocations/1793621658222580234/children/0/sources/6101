import dataclasses

@dataclasses.dataclass
class MonitorTrajectoryFeatures:
    constraint_satisfaction_trend: list[float]
    evidence_presence_trend: list[float]
    unsupported_commitment_trend: list[float]
    trajectory_score: float

def evaluate_constraint_satisfaction(state_text: str, constraints: list[str]) -> float:
    if not constraints:
        return 1.0
    satisfied = 0
    state_lower = state_text.lower()
    for c in constraints:
        if c.lower() in state_lower:
            satisfied += 1
    return float(satisfied) / len(constraints)

def evaluate_evidence_presence(state_text: str) -> float:
    markers = ["because", "according to", "implies", "therefore", "given that", "since", "proof"]
    state_lower = state_text.lower()
    found = sum(1 for m in markers if m in state_lower)
    return min(1.0, float(found) / 2.0)

def evaluate_unsupported_commitment(state_text: str) -> float:
    risk_markers = ["definitely", "obviously", "the answer is", "must be", "clearly"]
    state_lower = state_text.lower()
    found = sum(1 for m in risk_markers if m in state_lower)
    return min(1.0, float(found) / 2.0)

def score_trajectory(states: list[str], constraints: list[str]) -> MonitorTrajectoryFeatures:
    cs_trend = [evaluate_constraint_satisfaction(s, constraints) for s in states]
    ev_trend = [evaluate_evidence_presence(s) for s in states]
    uc_trend = [evaluate_unsupported_commitment(s) for s in states]
    
    score = 0.0
    if len(states) > 0:
        score += cs_trend[-1]
        score += ev_trend[-1]
        score -= sum(uc_trend) / len(states)
    
    return MonitorTrajectoryFeatures(
        constraint_satisfaction_trend=cs_trend,
        evidence_presence_trend=ev_trend,
        unsupported_commitment_trend=uc_trend,
        trajectory_score=score
    )
package org.pcbqc.edge.core

import kotlin.math.max

object SecondLook {
    enum class Outcome { CONFIRMED, CLEARED, UNRESOLVED }

    fun target(ins: Inspection): Finding? =
        if (ins.verdict != Verdict.REVIEW) null
        else ins.findings.filter { it.severity != Severity.MINOR && !it.certain }.maxWithOrNull(compareBy<Finding> { it.severity }.thenBy { it.confidence })

    fun apply(ins: Inspection, target: Finding, seen: List<Finding>, quality: Quality, placeOk: Boolean): Pair<Inspection, Outcome> {
        val same = seen.filter { it.type == target.type }.maxOfOrNull { it.confidence } ?: 0f
        val outcome = when {
            same >= Rules.SURE -> Outcome.CONFIRMED
            !placeOk || !quality.ok || seen.isNotEmpty() || target.severity == Severity.CRITICAL -> Outcome.UNRESOLVED
            else -> Outcome.CLEARED
        }
        val list = when (outcome) {
            Outcome.CONFIRMED -> ins.findings.map { if (it === target) it.copy(confidence = max(it.confidence, same), cut = false, reason = it.reason + ", confirmed by close-up") else it }
            Outcome.CLEARED -> ins.findings.filter { it !== target }
            Outcome.UNRESOLVED -> ins.findings.map { if (it === target) it.copy(reason = it.reason + ", close-up could not settle it") else it }
        }
        val (v, why) = Rules.verdict(list, ins.quality)
        val trace = "${target.type} at ${target.zone}, first certainty ${(target.confidence * 100).toInt()}%, close-up ${(same * 100).toInt()}%, place ${if (placeOk) "recognised" else "not recognised"}, " +
            "${outcome.name.lowercase()}, verdict ${ins.verdict} to $v"
        return ins.copy(findings = list, verdict = v, verdictReason = why, secondLook = trace) to outcome
    }
}

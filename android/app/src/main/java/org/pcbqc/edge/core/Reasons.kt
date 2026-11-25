package org.pcbqc.edge.core

class Reason(val code: String, val short: String, val details: List<String>)

object Reasons {
    private val abbreviations = linkedMapOf(
        "short" to "SHT", "open_circuit" to "OPN", "missing_hole" to "MHL", "missing_pad" to "MPD", "mouse_bite" to "MBT",
        "spur" to "SPR", "spurious_copper" to "SCU", "pin_hole" to "PIN", "anomaly" to "DIF",
    )
    private val imageCodes = mapOf(
        "image is blurred" to "BLUR", "image is too dark" to "DARK", "image is too bright" to "BRIGHT",
        "too much glare" to "GLARE", "image is too small" to "SMALL",
    )

    val legend: List<Pair<String, String>> =
        listOf("REJ" to "reject", "RWK" to "rework", "REV-LOW" to "review, serious but not certain", "REV-MIN" to "review, minor only", "IMG" to "retake the picture") +
            abbreviations.map { (type, code) -> code to if (type == "anomaly") "differs from reference" else type.replace('_', ' ') }

    fun abbreviation(type: String) = abbreviations[type] ?: "OTH"

    private fun pct(v: Float) = "${(v * 100).toInt()}%"

    private fun codes(list: List<Finding>): String {
        val order = abbreviations.keys.toList()
        return list.map { it.type }.distinct().sortedBy { order.indexOf(it).let { i -> if (i < 0) order.size else i } }.joinToString("+") { abbreviation(it) }
    }

    private fun counted(list: List<Finding>) = list.groupBy { it.type }.values.joinToString(", ") { g ->
        "${if (g[0].type == "anomaly") "difference from reference" else g[0].label} x${g.size} (${pct(g.maxOf { it.confidence })})"
    }

    fun of(ins: Inspection): Reason {
        val need = pct(Rules.SURE)
        val critical = ins.findings.filter { it.severity == Severity.CRITICAL }
        val major = ins.findings.filter { it.severity == Severity.MAJOR }
        val sureMajor = major.filter { it.certain }
        fun doubt(list: List<Finding>) = if (list.any { it.confidence >= Rules.SURE }) "cut off by the picture edge" else "below $need"
        val (code, short, rule) = when (ins.verdict) {
            Verdict.PASS -> Triple("PASS", "No defects, image usable", "Rule: no findings in a usable image gives PASS")
            Verdict.RETAKE -> if (ins.notBoard) Triple(
                "IMG-NOBOARD", "This does not look like a bare board",
                if (ins.findings.size > Rules.MAX_MARKS) "Rule: ${ins.findings.size} marks is far more than a real board shows, so nothing is judged"
                else "Rule: scene score ${"%.1f".format(java.util.Locale.US, ins.times.scene)} is above the limit, so nothing is judged",
            ) else Triple(
                "IMG-" + ins.quality.problems.joinToString("+") { imageCodes[it] ?: "BAD" }, "Picture not usable: ${ins.quality.problems.joinToString(", ")}",
                if (ins.quality.unusable) "Rule: the camera picture is too dark to judge" else "Rule: nothing found, but the picture is too poor to call it good",
            )
            Verdict.REJECT -> {
                val sure = critical.filter { it.certain }
                Triple("REJ-" + codes(sure), "Critical: ${counted(sure)}", "Rule: a critical defect at $need certainty or more gives REJECT")
            }
            Verdict.REWORK -> Triple("RWK-" + codes(sureMajor), "Major: ${counted(sureMajor)}", "Rule: a major defect at $need certainty or more, with no critical defect, gives REWORK")
            Verdict.REVIEW -> when {
                critical.isNotEmpty() -> Triple(
                    "REV-LOW-" + codes(critical),
                    "Possible critical ${counted(critical)}, ${doubt(critical)}" + if (sureMajor.isEmpty()) "" else "; certain major ${counted(sureMajor)}",
                    "Rule: a critical defect below $need certainty, or cut off by the picture edge, goes to a person",
                )
                major.isNotEmpty() -> Triple("REV-LOW-" + codes(major), "Possible major ${counted(major)}, ${doubt(major)}", "Rule: a major defect below $need certainty, or cut off by the picture edge, goes to a person")
                else -> Triple("REV-MIN-" + codes(ins.findings), "Minor only: ${counted(ins.findings)}", "Rule: minor defects alone go to a person")
            }
        }
        val details = buildList {
            add(rule)
            (if (ins.notBoard) emptyList() else ins.findings).take(8).forEach { add("${abbreviation(it.type)} ${it.severity.name.lowercase()} ${pct(it.confidence)}, ${it.zone}: ${it.reason}") }
            if (ins.findings.size > 8 && !ins.notBoard) add("and ${ins.findings.size - 8} more")
            if (ins.findings.isNotEmpty() && !ins.quality.ok) add("Picture: ${ins.quality.problems.joinToString(", ")}")
            add("Reference: ${ins.goldenNote}")
            if (ins.sceneOverridden) add("Operator chose Inspect anyway")
            if (ins.secondLook.isNotEmpty()) add("Second look: ${ins.secondLook}")
        }
        return Reason(code, short, details)
    }
}

package org.pcbqc.edge.llm

import org.json.JSONObject
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Verdict

object Guard {
    private val denial = Regex(
        "(\\bno\\b[^.\\n]{0,30}\\b(defects?|issues?|problems?|errors?|faults?|changes?)\\b|\\bnot?\\b[^.\\n]{0,12}\\bdefective\\b|overlooked|not thorough|" +
            "\\bpass(ed|es)\\b|\\bpass the board|accept the board|safe to (use|ship)|ship the board|send the board to the next stage|free of defects)",
        RegexOption.IGNORE_CASE,
    )
    private val labels = listOf("short", "open circuit", "mouse bite", "spurious copper", "spur", "pin hole", "missing hole", "missing pad")
    private val words = mapOf("one" to 1, "two" to 2, "three" to 3, "four" to 4, "five" to 5, "six" to 6, "seven" to 7, "eight" to 8, "nine" to 9, "ten" to 10)

    fun tidy(text: String) = text.replace("**", "").replace(Regex("(?m)^#+\\s*"), "").replace(Regex("\\n{2,}"), "\n").replace(Regex(" {2,}"), " ").trim()

    private fun pattern(label: String) = Regex("\\b" + label.replace(" ", "[ -]?") + "(?:e?s)?\\b", RegexOption.IGNORE_CASE)

    private fun stems(s: String) = Regex("[a-z]{4,}").findAll(s.lowercase()).map { it.value.take(5) }.toSet() -
        setOf("board", "this", "that", "with", "from", "usual", "usually".take(5), "check", "there", "where", "likel")

    private fun section(text: String, name: String, vararg next: String): String? {
        val stop = if (next.isEmpty()) "\\z" else next.joinToString("|", "(?=", "|\\z)") { "^\\s*$it\\s*:" }
        return Regex("(?ims)^\\s*$name\\s*:(.*?)$stop").find(text)?.groupValues?.get(1)?.trim()
    }

    private fun mismatch(text: String, ins: Inspection): String? {
        val counts = ins.findings.groupBy({ it.label }, { it.certain }).mapValues { (_, v) -> setOf(v.size, v.count { it }) - 0 }
        for (l in labels) {
            if (!pattern(l).containsMatchIn(text)) continue
            if (l !in counts) return "reply mentioned a defect that was not found"
            val n = Regex("\\b(\\d+|${words.keys.joinToString("|")})\\s+(?:\\w+\\s+)?" + pattern(l).pattern, RegexOption.IGNORE_CASE).findAll(text)
            for (m in n) {
                val v = m.groupValues[1].lowercase().let { it.toIntOrNull() ?: words[it] } ?: continue
                if (v !in counts.getValue(l)) return "reply gave a wrong count"
            }
        }
        val totals = Regex("\\b(\\d+|${words.keys.joinToString("|")})\\s+(critical |major |minor )?(?:defects|issues|faults)\\b", RegexOption.IGNORE_CASE).findAll(text)
        for (m in totals) {
            val v = m.groupValues[1].lowercase().let { it.toIntOrNull() ?: words[it] } ?: continue
            val sev = m.groupValues[2].trim()
            val real = if (sev.isEmpty()) ins.findings.size else ins.findings.count { it.severity.name.equals(sev, true) }
            if (v != real) return "reply gave a wrong total"
        }
        val placesSaid = Regex("\\b(\\d+|${words.keys.joinToString("|")})\\s+places?\\b", RegexOption.IGNORE_CASE).findAll(text)
        for (m in placesSaid) {
            val v = m.groupValues[1].lowercase().let { it.toIntOrNull() ?: words[it] } ?: continue
            if (counts.values.none { v in it }) return "reply gave a wrong count"
        }
        val other = Regex("\\b(PASS|REJECT|REWORK|REVIEW|RETAKE)\\b").findAll(text).map { it.value }.firstOrNull { it != ins.verdict.name }
        if (other != null) return "reply named another verdict"
        if (ins.verdict != Verdict.PASS && denial.containsMatchIn(text)) return "reply contradicted the findings"
        return null
    }

    fun problem(text: String, ins: Inspection): String? {
        val summary = section(text, "SUMMARY", "LIKELY CAUSE", "ACTION") ?: return "reply had no summary"
        val cause = section(text, "LIKELY CAUSE", "ACTION") ?: return "reply had no cause"
        val action = section(text, "ACTION") ?: return "reply had no actions"
        val steps = Regex("(?m)^\\s*\\d[.)]\\s*(.+)$").findAll(action).map { it.groupValues[1] }.toList()
        if (steps.isEmpty()) return "reply had no action steps"
        mismatch(text, ins)?.let { return it }
        if (!summary.contains(ins.verdict.name)) return "reply left out the verdict"
        val worst = ins.findings.maxOfOrNull { it.severity }
        val top = ins.findings.filter { it.severity == worst }.map { it.label }.distinct()
        if (top.none { pattern(it).containsMatchIn(summary) || (it == "anomaly" && summary.contains("differ", true)) }) return "reply left out the main defect"
        val (causes, approved) = Playbook.allowed(ins)
        val known = stems(causes)
        if (stems(cause).count { it in known } < 2) return "reply gave an unapproved cause"
        val ok = approved.map { stems(it) }
        for (s in steps) {
            val w = stems(s)
            if (ok.none { a -> a.count { it in w } >= minOf(2, a.size) }) return "reply gave an unapproved step"
        }
        return null
    }

    fun answerProblem(text: String, ins: Inspection): String? = if (text.isBlank()) "answer was empty" else mismatch(text, ins)

    fun shiftProblem(text: String, facts: JSONObject): String? {
        val source = Playbook.shiftFacts(facts)
        val allowed = Regex("\\d+").findAll("$source 12").map { it.value }.toSet()
        val low = text.lowercase()
        val used = Regex("\\d+").findAll(text).map { it.value }.toList() + words.filter { Regex("\\b${it.key}\\b").containsMatchIn(low) && it.key != "one" }.map { it.value.toString() }
        val common = Playbook.commonest(facts)
        val said = stems(text)
        val known = stems(source) + stems("last hours inspected most common defect times first check repeating place points same several boards were verdict process hint none also found")
        return when {
            text.isBlank() -> "note was empty"
            used.any { it !in allowed } -> "note used a number that is not in the log"
            labels.any { l -> pattern(l).containsMatchIn(text) && !pattern(l).containsMatchIn(source) } -> "note mentioned a defect that is not in the log"
            common != null && labels.contains(common) && !pattern(common).containsMatchIn(text) -> "note left out the most common defect"
            said.isNotEmpty() && said.count { it in known } < 0.75 * said.size -> "note added things that are not in the log"
            else -> null
        }
    }
}

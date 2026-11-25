package org.pcbqc.edge.llm

import org.json.JSONObject
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Verdict

object Playbook {
    private class Entry(val cause: String, val steps: List<String>)

    private val entries = mapOf(
        "short" to Entry("Copper joins two tracks, usually from under-etching or a masking fault.", listOf("Quarantine the board and mark the spot", "Check etching time and the mask at that area", "Inspect the neighbouring boards of this batch")),
        "open_circuit" to Entry("A track is broken, usually from over-etching, a scratch or a masking fault.", listOf("Quarantine the board and mark the break", "Check the mask and etching strength", "Inspect the previous boards of this batch")),
        "mouse_bite" to Entry("Copper is missing along a track edge, usually from etching or mask damage.", listOf("Measure the remaining track width", "Send to rework if the track is too thin", "Check the mask edge and etching bath")),
        "spur" to Entry("A copper spike sticks out of a track, usually from incomplete etching.", listOf("Measure the gap to the next track", "Trim the spike if the gap is too small", "Check etching time")),
        "spurious_copper" to Entry("Copper was left where there should be none, usually from incomplete etching.", listOf("Check whether it touches a track or pad", "Remove the copper or send to rework", "Check etching time and board cleaning")),
        "pin_hole" to Entry("A tiny hole in the copper, usually from dust or plating trouble.", listOf("Check whether it lies on a pad or narrow track", "Send to rework if it does", "Check dust control and plating")),
        "missing_hole" to Entry("A drilled hole is absent, usually from a broken drill or a drilling program error.", listOf("Hold the board for drilling rework", "Check the drill bit and program", "Count the holes on the next board")),
        "missing_pad" to Entry("A solder pad is absent or badly formed, usually from a mask or etching fault.", listOf("Hold the board for rework or scrap", "Check the mask at that pad", "Inspect the same pad on the next board")),
        "anomaly" to Entry("This area differs from the reference board, and the type is unknown.", listOf("Look at the marked area under good light", "Clean the area and scan again", "Ask a supervisor if the difference remains")),
    )

    private fun entry(type: String) = entries[type] ?: entries.getValue("anomaly")

    private fun order(f: List<Finding>) = f.sortedWith(compareByDescending<Finding> { it.severity }.thenByDescending { it.confidence }).map { it.type }.distinct()

    fun allowed(ins: Inspection): Pair<String, List<String>> {
        val e = order(ins.findings).map { entry(it) }
        return e.joinToString(" ") { it.cause } to e.flatMap { it.steps }
    }

    private fun places(n: Int) = if (n == 1) "1 place" else "$n places"

    fun facts(ins: Inspection): String {
        val types = order(ins.findings)
        val main = ins.findings.filter { it.type == types.first() }.let { l -> l.filter { it.certain }.ifEmpty { l } }
        val also = types.drop(1).take(3).joinToString(", ") { it.replace('_', ' ') }
        val e = entry(types.first())
        return "Board: ${ins.boardId}\nVerdict: ${ins.verdict}\nMain defect: ${types.first().replace('_', ' ')}\nSeverity: ${main.maxOf { it.severity }.name.lowercase()}\n" +
            "Where: ${places(main.size)}, ${main.map { it.zone }.distinct().take(2).joinToString(" and ")}\n" + (if (also.isEmpty()) "" else "Also found: $also\n") +
            "Known cause: ${e.cause}\nApproved steps: ${e.steps.joinToString("; ")}"
    }

    private fun pairs(o: JSONObject?): List<Pair<String, Int>> =
        o?.keys()?.asSequence()?.map { it to o.getInt(it) }?.sortedByDescending { it.second }?.toList().orEmpty()

    fun commonest(f: JSONObject): String? = pairs(f.optJSONObject("defects")).firstOrNull()?.first

    fun shiftFacts(f: JSONObject): String {
        val top = pairs(f.optJSONObject("defects")).firstOrNull()
        val step = top?.let { entry(it.first.replace(' ', '_')).steps.getOrNull(1) } ?: "Nothing to check"
        return "Boards: ${f.optInt("boards")}\nVerdicts: ${pairs(f.optJSONObject("verdicts")).joinToString(", ") { "${it.second} ${it.first}" }}\n" +
            "Most common defect: ${top?.let { "${it.first}, ${it.second} times" } ?: "none"}\nRepeating place: ${f.optString("repeating_places").ifEmpty { "none" }}\n" +
            "Process hint: ${f.optString("process_hint").ifEmpty { "none" }}\nCheck first: $step"
    }

    fun shift(f: JSONObject): String = shiftFacts(f).replace("\n", ". ") + "."

    fun template(ins: Inspection): String {
        if (ins.verdict == Verdict.PASS) return "SUMMARY: No defects were found in a usable image.\nLIKELY CAUSE: none\nACTION:\n1. Send the board to the next stage"
        if (ins.notBoard) return "SUMMARY: This does not look like a bare circuit board, so nothing was judged.\nLIKELY CAUSE: the camera is not pointing at a board, or the picture is unusual\nACTION:\n1. Fill the frame with one bare board\n2. Use even light and avoid reflections\n3. Scan again, or use Inspect anyway if it is a board"
        if (ins.verdict == Verdict.RETAKE) return "SUMMARY: The picture is not good enough to judge this board (${ins.verdictReason}).\nLIKELY CAUSE: camera or lighting\nACTION:\n1. Clean the lens and hold the phone steady\n2. Fill the frame with the board and avoid glare\n3. Scan again"
        val types = order(ins.findings)
        val e = entry(types.first())
        val kinds = types.joinToString(", ") { it.replace('_', ' ') }
        return "SUMMARY: ${ins.findings.size} issue(s) found: $kinds. Verdict ${ins.verdict}.\nLIKELY CAUSE: ${e.cause}\nACTION:\n" +
            e.steps.mapIndexed { i, s -> "${i + 1}. $s" }.joinToString("\n")
    }
}

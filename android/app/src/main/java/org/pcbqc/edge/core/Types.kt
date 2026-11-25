package org.pcbqc.edge.core

import org.json.JSONArray
import org.json.JSONObject
import kotlin.math.max
import kotlin.math.min

data class Box(val x1: Float, val y1: Float, val x2: Float, val y2: Float) {
    val w get() = x2 - x1
    val h get() = y2 - y1
    val area get() = max(0f, w) * max(0f, h)
    val cx get() = (x1 + x2) / 2
    val cy get() = (y1 + y2) / 2

    fun union(o: Box) = Box(min(x1, o.x1), min(y1, o.y1), max(x2, o.x2), max(y2, o.y2))

    fun intersection(o: Box): Float =
        max(0f, min(x2, o.x2) - max(x1, o.x1)) * max(0f, min(y2, o.y2) - max(y1, o.y1))

    fun iou(o: Box): Float {
        val i = intersection(o)
        val u = area + o.area - i
        return if (u > 0) i / u else 0f
    }

    fun scaled(s: Float) = Box(x1 * s, y1 * s, x2 * s, y2 * s)
    fun shifted(dx: Float, dy: Float) = Box(x1 + dx, y1 + dy, x2 + dx, y2 + dy)
}

enum class Severity { MINOR, MAJOR, CRITICAL }

enum class Verdict { PASS, REVIEW, REWORK, REJECT, RETAKE }

data class Finding(
    val type: String,
    val confidence: Float,
    val box: Box,
    val source: String,
    val severity: Severity = Severity.MINOR,
    val areaPct: Float = 0f,
    val zone: String = "",
    val reason: String = "",
    val refX: Float? = null,
    val refY: Float? = null,
    val cut: Boolean = false,
) {
    val certain get() = confidence >= Rules.SURE && !cut
    val label get() = type.replace('_', ' ')
}

data class Quality(val sharpness: Float, val brightness: Float, val glarePct: Float, val problems: List<String>, val unusable: Boolean = false) {
    val ok get() = problems.isEmpty()
}

data class StageTimes(
    val prepareMs: Float = 0f,
    val inferMs: Float = 0f,
    val postMs: Float = 0f,
    val goldenMs: Float = 0f,
    val totalMs: Float = 0f,
    val tiles: Int = 0,
    val scene: Float = 0f,
)

data class Inspection(
    val boardId: String,
    val width: Int,
    val height: Int,
    val findings: List<Finding>,
    val verdict: Verdict,
    val verdictReason: String,
    val quality: Quality,
    val times: StageTimes,
    val goldenNote: String,
    val refId: Long = 0,
    val secondLook: String = "",
    val heat: String = "",
    val notBoard: Boolean = false,
    val sceneOverridden: Boolean = false,
) {
    val why: Reason get() = Reasons.of(this)

    fun record(): JSONObject = JSONObject().apply {
        val r = why
        put("board", boardId)
        put("verdict", verdict.name)
        put("reason_code", r.code)
        put("reason", r.short)
        put("why", JSONArray(r.details))
        put("reference_check", goldenNote)
        if (refId != 0L) put("ref_id", refId)
        if (secondLook.isNotEmpty()) put("second_look", secondLook)
        if (heat.isNotEmpty()) put("phone_heat", heat)
        put("scene_score", Math.round(times.scene * 100) / 100.0)
        if (notBoard) put("not_board", true)
        if (sceneOverridden) put("inspected_anyway", true)
        put("defects", JSONArray().apply {
            findings.forEach { f ->
                put(JSONObject().apply {
                    put("type", f.type)
                    put("severity", f.severity.name.lowercase())
                    put("confidence", Math.round(f.confidence * 100) / 100.0)
                    put("bbox", JSONArray(listOf(f.box.x1.toInt(), f.box.y1.toInt(), f.box.x2.toInt(), f.box.y2.toInt())))
                    put("zone", f.zone)
                    put("area_pct", Math.round(f.areaPct * 1000) / 1000.0)
                    put("found_by", f.source)
                    if (f.cut) put("cut_by_picture_edge", true)
                    if (f.refX != null && f.refY != null) put("ref_pos", JSONArray(listOf(f.refX.toDouble(), f.refY.toDouble())))
                })
            }
        })
    }
}

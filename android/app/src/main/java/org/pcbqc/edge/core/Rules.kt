package org.pcbqc.edge.core

import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

object Rules {
    private val electrical = setOf("short", "open_circuit")
    private val mounting = setOf("missing_hole", "missing_pad")
    private const val LARGE_PCT = 0.20f
    private const val LARGE_ANOMALY_PCT = 0.10f
    private const val REPEAT = 3
    const val SURE = 0.55f
    private const val FUSE = 0.5f
    private const val OVERLAP = 0.7f
    private const val EDGE = 0.004f
    const val KNOWN_RADIUS = 0.012f

    fun zone(b: Box, w: Int, h: Int): String {
        val col = listOf("left", "centre", "right")[(b.cx / w * 3).toInt().coerceIn(0, 2)]
        val row = listOf("top", "middle", "bottom")[(b.cy / h * 3).toInt().coerceIn(0, 2)]
        return if (row == "middle" && col == "centre") "centre" else "$row $col"
    }

    fun grade(raw: List<Finding>, w: Int, h: Int): List<Finding> {
        val counts = raw.groupingBy { it.type }.eachCount()
        return raw.map { f ->
            val pct = 100f * f.box.area / (w.toFloat() * h)
            val near = raw.any { o ->
                o !== f && o.box.iou(f.box) < 0.3f &&
                    hypot(o.box.cx - f.box.cx, o.box.cy - f.box.cy) < 1.5f * max(max(f.box.w, f.box.h), max(o.box.w, o.box.h))
            }
            val (sev, why) = when {
                f.type in electrical -> Severity.CRITICAL to "breaks the circuit electrically"
                f.type in mounting -> Severity.MAJOR to "a part cannot be mounted or soldered here"
                f.type == "anomaly" && pct > LARGE_ANOMALY_PCT -> Severity.MAJOR to "large difference from the reference board"
                f.type == "anomaly" -> Severity.MINOR to "small difference from the reference board"
                pct > LARGE_PCT -> Severity.MAJOR to "large area"
                (counts[f.type] ?: 0) >= REPEAT -> Severity.MAJOR to "repeats ${counts[f.type]} times, so the process may be drifting"
                near -> Severity.MAJOR to "close to another defect"
                else -> Severity.MINOR to "small and isolated"
            }
            val cut = touchesEdge(f.box, w, h)
            f.copy(severity = sev, areaPct = pct, zone = zone(f.box, w, h), reason = if (cut) "$why, cut off by the picture edge" else why, cut = cut)
        }.sortedWith(compareByDescending<Finding> { it.severity }.thenByDescending { it.confidence })
    }

    const val MAX_MARKS = 30
    const val DARK = 35f
    const val NO_BOARD = "this does not look like a bare board"

    fun verdict(graded: List<Finding>, quality: Quality, notBoard: Boolean = false): Pair<Verdict, String> {
        if (notBoard) return Verdict.RETAKE to NO_BOARD
        if (quality.unusable) return Verdict.RETAKE to quality.problems.joinToString(", ")
        if (graded.isEmpty()) {
            return if (quality.ok) Verdict.PASS to "No defects found in a usable image"
            else Verdict.RETAKE to quality.problems.joinToString(", ")
        }
        val worst = graded.maxOf { it.severity }
        val sure = graded.any { it.severity == worst && it.certain }
        val names = graded.filter { it.severity == worst }.map { it.label }.distinct().joinToString(", ")
        return when {
            worst == Severity.CRITICAL && sure -> Verdict.REJECT to "Critical: $names"
            worst == Severity.CRITICAL -> Verdict.REVIEW to "Possible critical ($names), low certainty: check by eye"
            worst == Severity.MAJOR && sure -> Verdict.REWORK to "Major: $names"
            worst == Severity.MAJOR -> Verdict.REVIEW to "Possible major ($names), low certainty: check by eye"
            else -> Verdict.REVIEW to "Minor: $names"
        }
    }

    private fun touchesEdge(b: Box, w: Int, h: Int): Boolean {
        if (max(w, h) <= 900) return false
        val m = EDGE * max(w, h)
        return b.x1 <= m || b.y1 <= m || b.x2 >= w - m || b.y2 >= h - m
    }

    fun fuse(dets: List<Finding>): List<Finding> {
        val keep = ArrayList<Finding>()
        for (d in dets.sortedByDescending { it.confidence }) {
            fun shared(k: Finding) = k.box.intersection(d.box) / min(k.box.area, d.box.area).coerceAtLeast(1e-6f)
            val same = keep.indexOfFirst { it.type == d.type && shared(it) > FUSE }
            if (same >= 0) {
                keep[same] = keep[same].copy(box = keep[same].box.union(d.box))
                continue
            }
            if (keep.none { shared(it) > OVERLAP && (d.type !in electrical || it.type in electrical) }) keep.add(d)
        }
        return keep
    }

    fun merge(model: List<Finding>, golden: List<Finding>): List<Finding> =
        model + golden.filter { g ->
            model.none { m ->
                val i = m.box.intersection(g.box)
                i > 0.3f * g.box.area || i > 0.3f * m.box.area
            }
        }

    class Known(val type: String, val x: Float, val y: Float)

    fun dropKnown(found: List<Finding>, known: List<Known>): Pair<List<Finding>, Int> {
        val kept = found.filter { f ->
            val x = f.refX
            val y = f.refY
            x == null || y == null || known.none { it.type == f.type && hypot(it.x - x, it.y - y) <= KNOWN_RADIUS }
        }
        return kept to found.size - kept.size
    }

    fun nms(dets: List<Finding>, thr: Float = 0.45f): List<Finding> {
        val keep = ArrayList<Finding>()
        for (d in dets.sortedByDescending { it.confidence }) {
            if (keep.none { it.box.iou(d.box) >= thr }) keep.add(d)
        }
        return keep
    }
}

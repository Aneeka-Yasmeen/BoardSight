package org.pcbqc.edge.core

import kotlin.math.hypot

object Drift {
    const val NEAR = 0.03f
    const val BOARDS_FOR_HOTSPOT = 3
    const val HISTORY = 50

    class Spot(val board: String, val type: String, val x: Float, val y: Float)

    class Hotspot(val type: String, val x: Float, val y: Float, val boards: Int) {
        val zone get() = Rules.zone(Box(x, y, x, y), 1, 1)
    }

    class Report(val boards: Int, val spots: List<Spot>, val hotspots: List<Hotspot>, val scattered: Int) {
        val hint: String
            get() = when {
                boards < BOARDS_FOR_HOTSPOT -> "Needs at least $BOARDS_FOR_HOTSPOT boards scanned against the same reference ($boards so far)."
                hotspots.isNotEmpty() -> "The same place fails on several boards, which points to a fixed cause such as a damaged mask, film or drill position."
                scattered > 0 -> "Defects are scattered, which points to a random cause such as dust, handling or uneven etching."
                else -> "No defects on the recent boards."
            }
    }

    fun analyse(boards: Int, spots: List<Spot>): Report {
        val left = spots.toMutableList()
        val hot = ArrayList<Hotspot>()
        while (left.isNotEmpty()) {
            val seed = left.maxByOrNull { s -> left.filter { near(it, s) }.map { it.board }.distinct().size } ?: break
            val group = left.filter { near(it, seed) }
            val n = group.map { it.board }.distinct().size
            if (n < BOARDS_FOR_HOTSPOT) break
            val type = group.groupingBy { it.type }.eachCount().maxByOrNull { it.value }!!.key
            hot += Hotspot(type, group.map { it.x }.average().toFloat(), group.map { it.y }.average().toFloat(), n)
            left.removeAll(group.toSet())
        }
        return Report(boards, spots, hot, left.size)
    }

    private fun near(a: Spot, b: Spot) = hypot(a.x - b.x, a.y - b.y) <= NEAR
}

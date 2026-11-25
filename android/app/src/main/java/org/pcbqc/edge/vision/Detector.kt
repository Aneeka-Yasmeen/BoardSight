package org.pcbqc.edge.vision

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Rect
import android.graphics.RectF
import org.json.JSONArray
import org.pcbqc.edge.core.Box
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Rules
import org.pcbqc.edge.core.StageTimes
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer
import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

class Detector(ctx: Context) : AutoCloseable {
    companion object {
        const val SIZE = 640
        const val STRIDE = 560
        const val LONG_SIDE = 2300
        const val NOT_BOARD = 1.2f

        fun median(v: List<Float>) = if (v.isEmpty()) 0f else v.sorted()[v.size / 2]
    }

    private val env = OrtEnvironment.getEnvironment()
    private val session: OrtSession
    val classes: List<String>
    private val pixels = IntArray(SIZE * SIZE)
    private val planes = FloatArray(3 * SIZE * SIZE)
    private val unit = FloatArray(256) { it / 255f }
    private val buf: FloatBuffer = ByteBuffer.allocateDirect(4 * 3 * SIZE * SIZE).order(ByteOrder.nativeOrder()).asFloatBuffer()
    private val paint = Paint(Paint.FILTER_BITMAP_FLAG)
    private val tile = Bitmap.createBitmap(SIZE, SIZE, Bitmap.Config.ARGB_8888)
    private val canvas = Canvas(tile)
    private val inputName: String

    init {
        val model = ctx.assets.open("pcb_yolo.onnx").use { it.readBytes() }
        val threads = min(4, max(1, Runtime.getRuntime().availableProcessors() - 2))
        val opts = OrtSession.SessionOptions().apply {
            setIntraOpNumThreads(threads)
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
        }
        session = env.createSession(model, opts)
        inputName = session.inputNames.first()
        val names = JSONArray(ctx.assets.open("classes.json").bufferedReader().use { it.readText() })
        classes = (0 until names.length()).map { names.getString(it) }
    }

    private class Clock {
        var pre = 0L
        var inf = 0L
        var post = 0L
        val scene = ArrayList<Float>()
    }

    @Synchronized
    private fun runTile(src: Bitmap, sx: Float, sy: Float, span: Float, conf: Float, clock: Clock): List<Finding> {
        val t0 = System.nanoTime()
        canvas.drawColor(Color.rgb(114, 114, 114))
        val right = min(src.width.toFloat(), sx + span)
        val bottom = min(src.height.toFloat(), sy + span)
        val k = SIZE / span
        canvas.drawBitmap(
            src,
            Rect(sx.toInt(), sy.toInt(), right.toInt(), bottom.toInt()),
            RectF(0f, 0f, (right - sx) * k, (bottom - sy) * k),
            paint,
        )
        tile.getPixels(pixels, 0, SIZE, 0, 0, SIZE, SIZE)
        val n = SIZE * SIZE
        for (i in 0 until n) {
            val p = pixels[i]
            planes[i] = unit[(p shr 16) and 0xFF]
            planes[n + i] = unit[(p shr 8) and 0xFF]
            planes[2 * n + i] = unit[p and 0xFF]
        }
        buf.rewind()
        buf.put(planes)
        buf.rewind()
        val t1 = System.nanoTime()
        val out = ArrayList<Finding>()
        OnnxTensor.createTensor(env, buf, longArrayOf(1, 3, SIZE.toLong(), SIZE.toLong())).use { input ->
            session.run(mapOf(inputName to input)).use { res ->
                val t2 = System.nanoTime()
                res.get("scene").ifPresent { clock.scene += (it as OnnxTensor).floatBuffer.get(0) }
                val tensor = res[0] as OnnxTensor
                val shape = tensor.info.shape
                val channels = shape[1].toInt()
                val count = shape[2].toInt()
                val data = FloatArray(channels * count)
                tensor.floatBuffer.get(data)
                val nc = channels - 4
                for (i in 0 until count) {
                    var best = 0f
                    var cls = -1
                    for (c in 0 until nc) {
                        val v = data[(4 + c) * count + i]
                        if (v > best) {
                            best = v
                            cls = c
                        }
                    }
                    if (best < conf || cls < 0) continue
                    val cx = data[i]
                    val cy = data[count + i]
                    val w = data[2 * count + i]
                    val h = data[3 * count + i]
                    val b = Box(cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2).scaled(1 / k).shifted(sx, sy)
                    out.add(Finding(classes[cls], best, b, "yolo"))
                }
                clock.inf += t2 - t1
                clock.pre += t1 - t0
                val kept = Rules.nms(out, 0.5f)
                clock.post += System.nanoTime() - t2
                return kept
            }
        }
    }

    private fun positions(n: Int): List<Int> {
        if (n <= SIZE) return listOf(0)
        val steps = ceil((n - SIZE) / STRIDE.toDouble()).toInt() + 1
        return (0 until steps).map { ((n - SIZE) * it / (steps - 1).toDouble()).roundToInt() }
    }

    fun inspect(img: Bitmap, conf: Float, longSide: Int = LONG_SIDE): Pair<List<Finding>, StageTimes> {
        val t0 = System.nanoTime()
        val clock = Clock()
        val big = max(img.width, img.height)
        if (big <= 900) {
            val d = runTile(img, 0f, 0f, big.toFloat(), conf, clock)
            return clip(Rules.fuse(d), img) to times(clock, t0, 1)
        }
        val s = longSide.toFloat() / big
        val work = if (kotlin.math.abs(s - 1f) < 0.02f) img
        else Bitmap.createScaledBitmap(img, (img.width * s).roundToInt(), (img.height * s).roundToInt(), true)
        val scale = img.width.toFloat() / work.width
        clock.pre += System.nanoTime() - t0
        val all = ArrayList<Finding>()
        var tiles = 0
        for (y in positions(work.height)) for (x in positions(work.width)) {
            all += runTile(work, x.toFloat(), y.toFloat(), SIZE.toFloat(), conf, clock)
            tiles++
        }
        val t1 = System.nanoTime()
        val merged = Rules.fuse(Rules.nms(all)).map { it.copy(box = it.box.scaled(scale)) }
        clock.post += System.nanoTime() - t1
        if (work !== img) work.recycle()
        return clip(merged, img) to times(clock, t0, tiles)
    }

    class LiveTile(val findings: List<Finding>, val index: Int, val count: Int, val ms: Float, val scene: Float)

    fun liveTile(frame: Bitmap, step: Int, conf: Float): LiveTile {
        val t0 = System.nanoTime()
        val spots = positions(frame.height).flatMap { y -> positions(frame.width).map { x -> x to y } }
        val i = step % spots.size
        val (x, y) = spots[i]
        val clock = Clock()
        val d = runTile(frame, x.toFloat(), y.toFloat(), SIZE.toFloat(), conf, clock)
        return LiveTile(d, i, spots.size, (System.nanoTime() - t0) / 1e6f, median(clock.scene))
    }

    private fun clip(d: List<Finding>, img: Bitmap) = d.map {
        it.copy(
            box = Box(
                it.box.x1.coerceIn(0f, img.width.toFloat()), it.box.y1.coerceIn(0f, img.height.toFloat()),
                it.box.x2.coerceIn(0f, img.width.toFloat()), it.box.y2.coerceIn(0f, img.height.toFloat()),
            ),
        )
    }

    private fun times(c: Clock, t0: Long, tiles: Int) =
        StageTimes(c.pre / 1e6f, c.inf / 1e6f, c.post / 1e6f, 0f, (System.nanoTime() - t0) / 1e6f, tiles, median(c.scene))

    override fun close() = session.close()
}

package org.pcbqc.edge

import android.app.ActivityManager
import android.app.Application
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Matrix
import android.graphics.Paint
import android.net.Uri
import android.os.BatteryManager
import android.os.Build
import android.os.Debug
import android.os.Process
import androidx.exifinterface.media.ExifInterface
import org.json.JSONArray
import org.json.JSONObject
import org.opencv.android.OpenCVLoader
import org.pcbqc.edge.core.Box
import org.pcbqc.edge.core.Drift
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Rules
import org.pcbqc.edge.core.SecondLook
import org.pcbqc.edge.core.Severity
import org.pcbqc.edge.data.LogDb
import org.pcbqc.edge.llm.Advisor
import org.pcbqc.edge.vision.Detector
import org.pcbqc.edge.vision.Reference
import java.io.File
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.roundToInt

class App : Application() {
    val engine by lazy { Engine(this) }

    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_UI_HIDDEN) engine.releaseLlama()
    }
}

class Engine(private val ctx: Context) {
    val detector by lazy { Detector(ctx) }
    val advisor = Advisor(ctx)
    val db = LogDb(ctx)
    val cvReady = OpenCVLoader.initLocal()
    private val prefs = ctx.getSharedPreferences("settings", Context.MODE_PRIVATE)
    private val images = File(ctx.filesDir, "images").apply { mkdirs() }
    private val refFile = File(ctx.filesDir, "reference.jpg")
    private val knownFile = File(ctx.filesDir, "reference_known.json")

    var confidence: Float
        get() = prefs.getFloat("conf2", 0.45f)
        set(v) = prefs.edit().putFloat("conf2", v).apply()

    var useLlama: Boolean
        get() = prefs.getBoolean("llama", true)
        set(v) = prefs.edit().putBoolean("llama", v).apply()

    var checkBoard: Boolean
        get() = prefs.getBoolean("board", true)
        set(v) = prefs.edit().putBoolean("board", v).apply()

    fun inspectAnyway(ins: Inspection): Inspection {
        val (v, why) = Rules.verdict(ins.findings, ins.quality)
        return ins.copy(verdict = v, verdictReason = why, notBoard = false, sceneOverridden = true)
    }

    val hasReference get() = refFile.exists()

    fun releaseLlama() {
        advisor.stop()
        Thread { advisor.unload() }.start()
    }

    fun setReference(b: Bitmap) {
        refFile.outputStream().use { b.compress(Bitmap.CompressFormat.JPEG, 94, it) }
        val known = JSONArray()
        detector.inspect(b, minOf(confidence, 0.15f)).first.forEach {
            known.put(JSONArray(listOf(it.type, (it.box.cx / b.width).toDouble(), (it.box.cy / b.height).toDouble())))
        }
        knownFile.writeText(known.toString())
    }

    fun clearReference() {
        refFile.delete()
        knownFile.delete()
    }

    private fun known(): List<Rules.Known> = runCatching {
        val a = JSONArray(knownFile.readText())
        (0 until a.length()).map { i -> a.getJSONArray(i).let { Rules.Known(it.getString(0), it.getDouble(1).toFloat(), it.getDouble(2).toFloat()) } }
    }.getOrDefault(emptyList())

    fun sureHitsOnReference(b: Bitmap): Int = detector.inspect(b, 0.5f).first.size

    private fun reference(): Bitmap? {
        if (!refFile.exists()) return null
        return BitmapFactory.decodeFile(refFile.path)
    }

    fun inspect(img: Bitmap, boardId: String, fromCamera: Boolean, useReference: Boolean = true): Inspection {
        val t0 = System.nanoTime()
        val quality = if (cvReady) Reference.quality(img, fromCamera) else org.pcbqc.edge.core.Quality(0f, 0f, 0f, emptyList())
        val (found, times) = detector.inspect(img, confidence)
        var goldenMs = 0f
        var note = "no reference board set"
        var extra = emptyList<Finding>()
        var toRef: ((Float, Float) -> Pair<Float, Float>)? = null
        val ref = if (useReference) reference() else null
        if (ref != null && cvReady) {
            val g0 = System.nanoTime()
            val r = runCatching { Reference.compare(ref, img) }.getOrElse { Reference.Result(emptyList(), "reference check failed", false) }
            goldenMs = (System.nanoTime() - g0) / 1e6f
            note = r.note
            extra = r.findings
            toRef = r.toReference
            ref.recycle()
        }
        fun place(list: List<Finding>) = list.map { f ->
            val at = toRef?.invoke(f.box.cx, f.box.cy)
            if (at == null) f else f.copy(refX = at.first, refY = at.second)
        }
        val (real, same) = if (toRef != null) Rules.dropKnown(place(found), known()) else found to 0
        if (same > 0) note += ", $same hit(s) also on the reference ignored"
        val graded = Rules.grade(Rules.merge(real, place(extra)), img.width, img.height)
        val notBoard = checkBoard && (times.scene > Detector.NOT_BOARD || found.size > Rules.MAX_MARKS)
        val (v, why) = Rules.verdict(graded, quality, notBoard)
        return Inspection(
            boardId, img.width, img.height, graded, v, why, quality,
            times.copy(goldenMs = goldenMs, totalMs = (System.nanoTime() - t0) / 1e6f), note,
            if (toRef != null) refFile.lastModified() else 0L,
            heat = heat(), notBoard = notBoard,
        )
    }

    fun secondLook(ins: Inspection, target: Finding, picture: Bitmap?, closeUp: Bitmap): Pair<Inspection, SecondLook.Outcome> {
        val side = minOf(closeUp.width, closeUp.height) / 2
        val part = Bitmap.createBitmap(closeUp, (closeUp.width - side) / 2, (closeUp.height - side) / 2, side, side)
        val small = Bitmap.createScaledBitmap(part, Detector.SIZE, Detector.SIZE, true)
        val q = if (cvReady) Reference.quality(small, true) else org.pcbqc.edge.core.Quality(0f, 0f, 0f, emptyList())
        val (seen, _) = detector.inspect(small, minOf(confidence, 0.25f))
        val placeOk = picture != null && cvReady && runCatching {
            val s = picture.width.toFloat() / ins.width
            val b = target.box.scaled(s)
            val w = maxOf(b.w, b.h) * 6f
            val span = maxOf(w, picture.width * 0.25f).coerceAtMost(minOf(picture.width, picture.height).toFloat()).toInt()
            val x = (b.cx - span / 2f).toInt().coerceIn(0, picture.width - span)
            val y = (b.cy - span / 2f).toInt().coerceIn(0, picture.height - span)
            Reference.samePlace(Bitmap.createBitmap(picture, x, y, span, span), small)
        }.getOrDefault(false)
        return SecondLook.apply(ins, target, seen, q, placeOk)
    }

    fun heat(): String {
        if (Build.VERSION.SDK_INT < 29) return ""
        val s = (ctx.getSystemService(Context.POWER_SERVICE) as android.os.PowerManager).currentThermalStatus
        return listOf("normal", "slightly warm", "warm, may slow down", "hot, slowed down", "very hot", "emergency", "shutdown").getOrElse(s) { "unknown" }
    }

    fun drift(): Drift.Report = if (!hasReference) Drift.analyse(0, emptyList()) else db.spots(refFile.lastModified()).let { Drift.analyse(it.first, it.second) }

    fun systemMemory(): Triple<Long, Long, Boolean> {
        val mi = ActivityManager.MemoryInfo()
        (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        return Triple(mi.availMem / 1_000_000, mi.totalMem / 1_000_000, mi.lowMemory)
    }

    fun annotate(img: Bitmap, ins: Inspection, side: Int = 1400): Bitmap {
        val s = minOf(1f, side.toFloat() / max(img.width, img.height))
        val out = Bitmap.createScaledBitmap(img, (img.width * s).roundToInt(), (img.height * s).roundToInt(), true)
            .copy(Bitmap.Config.ARGB_8888, true)
        val c = Canvas(out)
        val unit = max(out.width, out.height) / 400f
        val line = Paint(Paint.ANTI_ALIAS_FLAG).apply { style = Paint.Style.STROKE; strokeWidth = unit }
        val text = Paint(Paint.ANTI_ALIAS_FLAG).apply { textSize = unit * 8; color = Color.WHITE; isFakeBoldText = true }
        val back = Paint()
        (if (ins.notBoard) emptyList() else ins.findings).asReversed().forEach { f ->
            val col = severityColor(f.severity)
            line.color = col
            back.color = col
            val b = f.box.scaled(s)
            val p = unit * 2
            c.drawRect(b.x1 - p, b.y1 - p, b.x2 + p, b.y2 + p, line)
            val label = f.label
            val tw = text.measureText(label)
            val ty = if (b.y1 - p - unit * 10 > 0) b.y1 - p else b.y2 + p + unit * 10
            val tx = b.x1.coerceIn(0f, max(0f, out.width - tw - unit * 4))
            c.drawRect(tx, ty - unit * 10, tx + tw + unit * 4, ty, back)
            c.drawText(label, tx + unit * 2, ty - unit * 2.4f, text)
        }
        return out
    }

    fun store(ins: Inspection, annotated: Bitmap): Long {
        val f = File(images, "${ins.boardId}_${System.currentTimeMillis()}.jpg")
        f.outputStream().use { annotated.compress(Bitmap.CompressFormat.JPEG, 82, it) }
        images.listFiles()?.sortedByDescending { it.lastModified() }?.drop(300)?.forEach { it.delete() }
        return db.insert(ins, f.path)
    }

    fun clearLog() {
        db.clear()
        images.listFiles()?.forEach { it.delete() }
    }

    fun decode(uri: Uri): Bitmap? {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, o) }
        val big = max(o.outWidth, o.outHeight)
        if (big <= 0) return null
        val opt = BitmapFactory.Options().apply { inSampleSize = max(1, Integer.highestOneBit(max(1, big / 2300))) }
        val b = ctx.contentResolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, opt) } ?: return null
        val rot = ctx.contentResolver.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
        return upright(b, rot)
    }

    fun upright(b: Bitmap, deg: Int, maxSide: Int = 2600): Bitmap {
        val s = minOf(1f, maxSide.toFloat() / max(b.width, b.height))
        if (deg == 0 && s == 1f) return b
        val out = Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postScale(s, s); postRotate(deg.toFloat()) }, true)
        if (out !== b) b.recycle()
        return out
    }

    fun rotate(b: Bitmap, deg: Int): Bitmap =
        if (deg == 0) b else Bitmap.createBitmap(b, 0, 0, b.width, b.height, Matrix().apply { postRotate(deg.toFloat()) }, true)

    class Sample(val name: String, val set: String, val boxes: List<Pair<String, Box>>)

    fun samples(): List<Sample> {
        val txt = runCatching { ctx.assets.open("bench/bench.json").bufferedReader().use { it.readText() } }.getOrNull() ?: return emptyList()
        val arr = JSONArray(txt)
        return (0 until arr.length()).map { i ->
            val o = arr.getJSONObject(i)
            val bx = o.getJSONArray("boxes")
            Sample(
                o.getString("file"), o.getString("set"),
                (0 until bx.length()).map { k ->
                    val v = bx.getJSONArray(k)
                    v.getString(0) to Box(v.getDouble(1).toFloat(), v.getDouble(2).toFloat(), v.getDouble(3).toFloat(), v.getDouble(4).toFloat())
                },
            )
        }
    }

    fun sampleBitmap(s: Sample): Bitmap = ctx.assets.open("bench/${s.name}").use { BitmapFactory.decodeStream(it) }

    fun memoryMb(): Pair<Int, Int> {
        val am = ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager
        val pss = am.getProcessMemoryInfo(intArrayOf(Process.myPid())).firstOrNull()?.totalPss ?: (Debug.getPss().toInt())
        val peak = runCatching {
            File("/proc/self/status").readLines().first { it.startsWith("VmHWM") }.filter { it.isDigit() }.toInt() / 1024
        }.getOrDefault(0)
        return pss / 1024 to peak
    }

    fun powerWatts(): Float? {
        val i = ctx.registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED)) ?: return null
        if (i.getIntExtra(BatteryManager.EXTRA_PLUGGED, 0) != 0) return null
        val mv = i.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0)
        val bm = ctx.getSystemService(Context.BATTERY_SERVICE) as BatteryManager
        val raw = abs(bm.getLongProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW))
        if (raw == 0L || mv <= 0) return null
        val amps = if (raw < 20_000) raw / 1e3 else raw / 1e6
        return (amps * mv / 1000.0).toFloat()
    }

    fun device(): String = "${Build.MANUFACTURER} ${Build.MODEL}, Android ${Build.VERSION.RELEASE}, ${Build.SUPPORTED_ABIS.first()}"

    fun installModel(uris: List<Uri>, onProgress: (String) -> Unit): String {
        val dir = File(ctx.filesDir, "llm").apply { mkdirs() }
        var done = 0
        for (u in uris) {
            val name = ctx.contentResolver.query(u, arrayOf(android.provider.OpenableColumns.DISPLAY_NAME), null, null, null)
                ?.use { if (it.moveToFirst()) it.getString(0) else null } ?: ""
            val target = when {
                name.endsWith(".pte", true) -> Advisor.MODEL_FILE
                name.startsWith("tokenizer", true) -> Advisor.TOKENIZER_FILE
                else -> continue
            }
            val tmp = File(dir, "$target.part")
            ctx.contentResolver.openInputStream(u)?.use { input ->
                tmp.outputStream().use { out ->
                    val b = ByteArray(1 shl 20)
                    var total = 0L
                    while (true) {
                        val n = input.read(b)
                        if (n < 0) break
                        out.write(b, 0, n)
                        total += n
                        onProgress("Copying $name: ${total / 1_000_000} MB")
                    }
                }
            } ?: continue
            advisor.unload()
            File(dir, target).delete()
            if (tmp.renameTo(File(dir, target))) done++
        }
        return if (advisor.installed) "Llama files installed" else if (done > 0) "One file installed, the other is still needed" else "Choose the .pte file and tokenizer.model"
    }

    fun removeModel() {
        advisor.unload()
        File(ctx.filesDir, "llm").deleteRecursively()
        ctx.getExternalFilesDir("llm")?.listFiles()?.forEach { it.delete() }
    }

    fun benchmark(limit: Int, withLlama: Boolean, stopped: () -> Boolean, onProgress: (String) -> Unit): JSONObject {
        val set = samples().take(limit)
        val out = JSONObject().put("device", device()).put("confidence", confidence.toDouble()).put("iou", 0.33)
        detector.inspect(Bitmap.createBitmap(640, 640, Bitmap.Config.ARGB_8888), 0.5f)
        val groups = LinkedHashMap<String, IntArray>()
        val clean = IntArray(2)
        val ms = HashMap<String, ArrayList<Float>>()
        fun add(k: String, v: Float) = ms.getOrPut(k) { ArrayList() }.add(v)
        val watts = ArrayList<Float>()
        var worst: Inspection? = null
        for ((i, s) in set.withIndex()) {
            if (stopped()) break
            onProgress("Image ${i + 1} of ${set.size}")
            val b = sampleBitmap(s)
            val ins = inspect(b, "BENCH", false, false)
            add(if (ins.times.tiles > 1) "board" else "tile", ins.times.totalMs)
            add("prepare_per_tile", ins.times.prepareMs / ins.times.tiles)
            add("infer_per_tile", ins.times.inferMs / ins.times.tiles)
            add("post_per_tile", ins.times.postMs / ins.times.tiles)
            powerWatts()?.let { watts += it }
            if (s.boxes.isEmpty()) {
                clean[0]++
                if (ins.findings.isNotEmpty()) clean[1]++
            } else {
                val g = groups.getOrPut(s.set) { IntArray(3) }
                val used = HashSet<Int>()
                for (f in ins.findings.sortedByDescending { it.confidence }) {
                    val hit = s.boxes.indices.filter { it !in used && s.boxes[it].first == f.type }.maxByOrNull { s.boxes[it].second.iou(f.box) }
                    if (hit != null && s.boxes[hit].second.iou(f.box) >= 0.33f) {
                        used += hit
                        g[0]++
                    } else g[1]++
                }
                g[2] += s.boxes.size - used.size
                if (worst == null && ins.findings.isNotEmpty()) worst = ins
            }
            b.recycle()
        }
        fun r(v: Float) = Math.round(v * 1000) / 1000.0
        val acc = JSONObject()
        groups.forEach { (k, g) ->
            val p = g[0].toFloat() / max(1, g[0] + g[1])
            val rc = g[0].toFloat() / max(1, g[0] + g[2])
            acc.put(k, JSONObject().put("precision", r(p)).put("recall", r(rc)).put("f1", r(2 * p * rc / max(1e-6f, p + rc))).put("defects", g[0] + g[2]))
        }
        out.put("accuracy", acc)
        out.put("clean_images", clean[0]).put("clean_images_with_false_alarm", clean[1])
        val t = JSONObject()
        ms.forEach { (k, v) -> t.put("${k}_ms_median", r(v.sorted()[v.size / 2])) }
        ms["tile"]?.let { v -> t.put("fps_single_tile", r(1000f / v.sorted()[v.size / 2])) }
        ms["board"]?.let { v -> t.put("fps_full_board", r(1000f / v.sorted()[v.size / 2])) }
        out.put("timing", t)
        val w = worst
        if (withLlama && w != null && !stopped()) {
            onProgress("Loading Llama")
            val runs = JSONArray()
            for (k in 1..3) {
                if (stopped()) break
                val a = advisor.advise(w) { onProgress("Llama run $k of 3") }
                powerWatts()?.let { watts += it }
                a.stats?.let {
                    runs.put(
                        JSONObject().put("ttft_ms", it.ttftMs).put("total_ms", it.totalMs).put("prompt_tokens", it.promptTokens)
                            .put("generated_tokens", it.generatedTokens).put("prefill_tokens_per_s", r(it.prefillTps.toFloat()))
                            .put("decode_tokens_per_s", r(it.decodeTps.toFloat())).put("accepted", a.fromLlama),
                    )
                }
                if (a.stats == null) {
                    out.put("llama_note", a.note)
                    break
                }
            }
            out.put("llama_load_ms", advisor.loadMs).put("llama", runs)
        }
        val (pss, peak) = memoryMb()
        out.put("memory_pss_mb", pss).put("memory_peak_rss_mb", peak)
        out.put("power_w_mean", if (watts.isEmpty()) "unplug the phone to measure" else r(watts.average().toFloat()))
        return out
    }

    companion object {
        fun severityColor(s: Severity): Int = when (s) {
            Severity.CRITICAL -> Color.rgb(229, 57, 53)
            Severity.MAJOR -> Color.rgb(239, 124, 0)
            Severity.MINOR -> Color.rgb(190, 148, 0)
        }

        fun ms(v: Float) = String.format(Locale.US, if (v < 100) "%.1f ms" else "%.0f ms", v)
    }
}

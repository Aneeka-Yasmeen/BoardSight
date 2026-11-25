package org.pcbqc.edge.ui

import android.app.Application
import android.graphics.Bitmap
import android.net.Uri
import android.os.SystemClock
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.isActive
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import org.pcbqc.edge.App
import org.pcbqc.edge.core.Drift
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Quality
import org.pcbqc.edge.core.Rules
import org.pcbqc.edge.core.SecondLook
import org.pcbqc.edge.core.Verdict
import org.pcbqc.edge.data.LogRow
import org.pcbqc.edge.llm.Advice
import org.pcbqc.edge.vision.Detector
import java.util.concurrent.Executors
import kotlin.math.abs
import kotlin.math.max

class EdgeModel(app: Application) : AndroidViewModel(app) {
    val engine = (app as App).engine
    private val visionThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()
    private val llmThread = Executors.newSingleThreadExecutor().asCoroutineDispatcher()

    var tab by mutableStateOf(0)
    var live by mutableStateOf<List<Finding>>(emptyList())
    var liveFrame by mutableStateOf(1 to 1)
    var liveProgress by mutableStateOf(0 to 1)
    var liveVerdict by mutableStateOf<Verdict?>(null)
    var liveStrange by mutableStateOf(false)
    private var strangeImage: Bitmap? = null
    var liveFps by mutableFloatStateOf(0f)
    var auto by mutableStateOf(false)
    var busy by mutableStateOf<String?>(null)
    var result by mutableStateOf<Inspection?>(null)
    var picture by mutableStateOf<Bitmap?>(null)
    var adviceText by mutableStateOf("")
    var advice by mutableStateOf<Advice?>(null)
    var advising by mutableStateOf(false)
    var answer by mutableStateOf("")
    var asking by mutableStateOf(false)
    var rows by mutableStateOf<List<LogRow>>(emptyList())
    var shiftNote by mutableStateOf("")
    var message by mutableStateOf<String?>(null)
    var hasReference by mutableStateOf(engine.hasReference)
    var conveyor by mutableStateOf(false)
    var benchText by mutableStateOf("")
    var benching by mutableStateOf(false)
    var llamaReady by mutableStateOf(engine.advisor.installed)
    var setupNote by mutableStateOf("")
    var drift by mutableStateOf<Drift.Report?>(null)
    var reasonCounts by mutableStateOf<List<Pair<String, Int>>>(emptyList())
    var looking by mutableStateOf<Finding?>(null)
    var lookNote by mutableStateOf("")
    var memory by mutableStateOf("")
    var rowId = 0L
        private set

    @Volatile
    private var stopBench = false
    private var conveyorJob: Job? = null
    private var lastFrame = 0L

    private var lastSig: IntArray? = null
    private var firedSig: IntArray? = null
    private var still = 0
    val capture = mutableStateOf(0)

    private var liveStep = 0
    private var liveTiles = arrayOfNulls<List<Finding>>(0)
    private var liveSig: IntArray? = null
    private var liveScene = arrayOfNulls<Float>(0)

    private fun signature(frame: Bitmap): IntArray {
        val s = Bitmap.createScaledBitmap(frame, 24, 24, true)
        val px = IntArray(576)
        s.getPixels(px, 0, 24, 0, 0, 24, 24)
        s.recycle()
        return IntArray(576) { i -> (px[i] shr 8) and 0xFF }
    }

    private fun gap(a: IntArray?, b: IntArray) = if (a == null) 255.0 else a.indices.sumOf { abs(a[it] - b[it]) } / 576.0

    fun onFrame(frame: Bitmap) {
        val sig = signature(frame)
        if (gap(liveSig, sig) > 9.0) {
            liveTiles = arrayOfNulls(0)
            liveScene = arrayOfNulls(0)
            liveStep = 0
        }
        liveSig = sig
        val t = engine.detector.liveTile(frame, liveStep, engine.confidence)
        if (liveTiles.size != t.count) {
            liveTiles = arrayOfNulls(t.count)
            liveScene = arrayOfNulls(t.count)
        }
        liveScene[t.index] = t.scene
        liveTiles[t.index] = t.findings
        liveStep = (liveStep + 1) % t.count
        val done = liveTiles.count { it != null }
        val graded = Rules.grade(Rules.fuse(liveTiles.filterNotNull().flatten()), frame.width, frame.height)
        val verdict = if (done == t.count) Rules.verdict(graded, Quality(0f, 0f, 0f, emptyList())).first else null
        val now = SystemClock.elapsedRealtime()
        val fps = if (lastFrame > 0) 1000f / max(1L, now - lastFrame) else 0f
        lastFrame = now
        val strange = engine.checkBoard && (graded.size > Rules.MAX_MARKS || (done == t.count && Detector.median(liveScene.filterNotNull()) > Detector.NOT_BOARD))
        val fire = auto && !strange && done == t.count && busy == null && result == null && trigger(sig)
        val size = frame.width to frame.height
        viewModelScope.launch(Dispatchers.Main) {
            live = if (strange) emptyList() else graded
            liveStrange = strange
            liveFrame = size
            liveFps = if (liveFps == 0f) fps else liveFps * 0.8f + fps * 0.2f
            liveProgress = done to t.count
            liveVerdict = verdict
            if (fire) capture.value++
        }
    }

    private fun trigger(sig: IntArray): Boolean {
        still = if (gap(lastSig, sig) < 3.0) still + 1 else 0
        lastSig = sig
        if (still >= 6 && gap(firedSig, sig) > 8.0) {
            firedSig = sig
            still = 0
            return true
        }
        return false
    }

    fun inspect(img: Bitmap, fromCamera: Boolean, wanted: () -> Boolean = { true }, onDone: suspend () -> Unit = {}) {
        if (busy != null) return
        busy = "Inspecting"
        engine.advisor.stop()
        viewModelScope.launch {
            try {
                val id = withContext(Dispatchers.IO) { engine.db.nextBoard() }
                val ins = withContext(visionThread) { engine.inspect(img, id, fromCamera) }
                val pic = withContext(Dispatchers.Default) { engine.annotate(img, ins) }
                if (!wanted()) return@launch
                val row = withContext(Dispatchers.IO) { engine.store(ins, pic) }
                rowId = row
                strangeImage = if (ins.notBoard) img else null
                result = ins
                picture = pic
                answer = ""
                lookNote = ""
                advising = true
                viewModelScope.launch {
                    advise(ins, row)
                    onDone()
                    if (fromCamera && auto) {
                        delay(5000)
                        if (auto && rowId == row && result != null && looking == null) closeResult()
                    }
                }
            } catch (e: Throwable) {
                message = "Inspection failed: ${e.message}"
            } finally {
                busy = null
            }
        }
    }

    private suspend fun advise(ins: Inspection, id: Long) {
        advising = true
        advice = null
        adviceText = ""
        val a = withContext(llmThread) {
            if (engine.useLlama) engine.advisor.advise(ins) { t -> viewModelScope.launch(Dispatchers.Main) { if (rowId == id && advice == null) adviceText = t } }
            else Advice(org.pcbqc.edge.llm.Playbook.template(ins), false, "Playbook text: Llama is switched off", null)
        }
        withContext(Dispatchers.IO) { engine.db.setAdvice(id, a) }
        if (rowId == id) {
            advice = a
            adviceText = a.text
            advising = false
        }
    }

    fun inspectUri(uri: Uri) {
        viewModelScope.launch {
            val b = withContext(Dispatchers.IO) { runCatching { engine.decode(uri) }.getOrNull() }
            if (b == null) message = "That picture could not be opened" else inspect(b, false)
        }
    }

    fun setReference(b: Bitmap) {
        viewModelScope.launch {
            busy = "Saving reference"
            try {
                val n = withContext(visionThread) {
                    engine.setReference(b)
                    engine.sureHitsOnReference(b)
                }
                hasReference = true
                message = if (n == 0) "Reference board saved" else "Reference saved. The detector marked $n place(s) on it: make sure this board is really good"
            } catch (e: Throwable) {
                message = "Reference could not be saved: ${e.message}"
            } finally {
                busy = null
            }
        }
    }

    fun setReferenceUri(uri: Uri) {
        viewModelScope.launch {
            val b = withContext(Dispatchers.IO) { runCatching { engine.decode(uri) }.getOrNull() }
            if (b == null) message = "That picture could not be opened" else setReference(b)
        }
    }

    fun clearReference() {
        engine.clearReference()
        hasReference = false
    }

    fun closeResult() {
        engine.advisor.stop()
        result = null
        picture = null
        strangeImage = null
        looking = null
        advising = false
    }

    fun ask(q: String) {
        val ins = result ?: return
        if (q.isBlank() || asking || advising) return
        asking = true
        answer = ""
        viewModelScope.launch {
            val a = withContext(llmThread) { engine.advisor.ask(ins, q.trim()) { t -> viewModelScope.launch(Dispatchers.Main) { if (asking) answer = t } } }
            asking = false
            answer = a.text
        }
    }

    fun hold() {
        if (conveyor) {
            toggleConveyor()
            message = "Conveyor paused so this board stays open"
        }
        if (auto) {
            auto = false
            message = "Auto trigger paused so this board stays open"
        }
    }

    fun operator(decision: String, note: String, finalVerdict: String) {
        hold()
        val id = rowId
        viewModelScope.launch {
            withContext(Dispatchers.IO) { engine.db.setOperator(id, decision, note.trim(), finalVerdict) }
            message = "Saved: operator $decision, final verdict $finalVerdict"
        }
    }

    fun inspectAnyway() {
        val ins = result ?: return
        val img = strangeImage ?: return
        if (busy != null) return
        hold()
        busy = "Inspecting"
        viewModelScope.launch {
            try {
                val next = engine.inspectAnyway(ins)
                val pic = withContext(Dispatchers.Default) { engine.annotate(img, next) }
                val id = rowId
                withContext(Dispatchers.IO) { engine.db.replace(id, next) }
                strangeImage = null
                picture = pic
                result = next
                advising = true
                viewModelScope.launch { advise(next, id) }
            } finally {
                busy = null
            }
        }
    }

    fun startLook() {
        looking = result?.let { SecondLook.target(it) }
        if (looking != null) {
            auto = false
            if (conveyor) toggleConveyor()
        }
    }

    fun cancelLook() {
        looking = null
    }

    fun finishLook(closeUp: Bitmap) {
        val ins = result ?: return
        val t = looking ?: return
        if (busy != null) return
        busy = "Checking the close-up"
        engine.advisor.stop()
        viewModelScope.launch {
            try {
                val (next, outcome) = withContext(visionThread) { engine.secondLook(ins, t, picture, closeUp) }
                val id = rowId
                withContext(Dispatchers.IO) { engine.db.replace(id, next) }
                result = next
                looking = null
                lookNote = when (outcome) {
                    SecondLook.Outcome.CONFIRMED -> "Second look: ${t.label} confirmed by the close-up."
                    SecondLook.Outcome.CLEARED -> "Second look: ${t.label} not seen in a sharp close-up, so it was removed."
                    SecondLook.Outcome.UNRESOLVED -> "Second look: the close-up could not settle it. Check ${t.label} by eye."
                }
                advising = true
                viewModelScope.launch { advise(next, id) }
            } catch (e: Throwable) {
                message = "Second look failed: ${e.message}"
            } finally {
                busy = null
            }
        }
    }

    fun refreshMemory() {
        viewModelScope.launch {
            memory = withContext(Dispatchers.IO) {
                val (pss, peak) = engine.memoryMb()
                val (free, total, low) = engine.systemMemory()
                "This app: $pss MB now, $peak MB at its highest\nPhone: $free MB free of $total MB" +
                    (if (low) "\nAndroid reports low memory" else "") + "\nLlama is " + (if (engine.advisor.loaded) "loaded" else "not loaded")
            }
        }
    }

    fun freeLlama() {
        viewModelScope.launch {
            withContext(llmThread) { engine.advisor.unload() }
            refreshMemory()
        }
    }

    fun refreshLog() {
        viewModelScope.launch {
            rows = withContext(Dispatchers.IO) { engine.db.recent() }
            drift = withContext(Dispatchers.IO) { engine.drift() }
            reasonCounts = withContext(Dispatchers.IO) { engine.db.reasonCounts() }
        }
    }

    fun clearLog() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { engine.clearLog() }
            refreshLog()
        }
    }

    fun shift() {
        if (asking || advising) return
        asking = true
        shiftNote = "Llama is writing the shift note"
        viewModelScope.launch {
            val f = withContext(Dispatchers.IO) {
                engine.db.shiftFacts().apply {
                    val d = engine.drift()
                    if (d.boards >= Drift.BOARDS_FOR_HOTSPOT) {
                        if (d.hotspots.isNotEmpty()) put("repeating_places", d.hotspots.take(3).joinToString("; ") { "${it.type.replace('_', ' ')} at ${it.zone} on ${it.boards} boards" })
                        put("process_hint", d.hint)
                    }
                }
            }
            if (f.optInt("boards") == 0) {
                asking = false
                shiftNote = "No boards were inspected in the last 12 hours."
                return@launch
            }
            val a = withContext(llmThread) { engine.advisor.shift(f) { t -> viewModelScope.launch(Dispatchers.Main) { if (asking) shiftNote = t } } }
            asking = false
            shiftNote = a.text
        }
    }

    fun toggleConveyor() {
        if (conveyor) {
            conveyor = false
            conveyorJob?.cancel()
            return
        }
        val items = engine.samples().filter { it.set == "pcbdefect" || it.boxes.isEmpty() }.shuffled()
        if (items.isEmpty()) {
            message = "No demo images are bundled"
            return
        }
        conveyor = true
        conveyorJob = viewModelScope.launch {
            for (s in items) {
                if (!isActive || !conveyor) break
                val b = withContext(Dispatchers.IO) { engine.sampleBitmap(s) }
                while (busy != null) delay(100)
                var done = false
                if (!conveyor) break
                inspect(b, false, { conveyor }) { done = true }
                while (!done && (busy != null || advising)) delay(150)
                delay(2500)
            }
            conveyor = false
        }
    }

    fun benchmark(withLlama: Boolean) {
        if (benching) {
            stopBench = true
            engine.advisor.stop()
            return
        }
        benching = true
        stopBench = false
        benchText = "Starting"
        viewModelScope.launch {
            val r = withContext(llmThread) {
                runCatching { engine.benchmark(1000, withLlama, { stopBench }) { p -> viewModelScope.launch(Dispatchers.Main) { if (benching) benchText = p } } }
            }
            benching = false
            benchText = r.fold({ it.toString(2) }, { "Benchmark failed: ${it.message}" })
        }
    }

    fun installModel(uris: List<Uri>) {
        if (uris.isEmpty()) return
        setupNote = "Copying"
        viewModelScope.launch {
            setupNote = withContext(Dispatchers.IO) {
                runCatching { engine.installModel(uris) { p -> viewModelScope.launch(Dispatchers.Main) { setupNote = p } } }
                    .getOrElse { "Copy failed: ${it.message}" }
            }
            delay(50)
            llamaReady = engine.advisor.installed
        }
    }

    fun removeModel() {
        viewModelScope.launch {
            withContext(Dispatchers.IO) { engine.removeModel() }
            llamaReady = engine.advisor.installed
            setupNote = "Llama files removed"
        }
    }

    fun writeText(uri: Uri, text: String) {
        viewModelScope.launch {
            val ok = withContext(Dispatchers.IO) {
                runCatching { getApplication<Application>().contentResolver.openOutputStream(uri)!!.use { it.write(text.toByteArray()) } }.isSuccess
            }
            message = if (ok) "File saved" else "File could not be saved"
        }
    }

    fun enter() {
        engine.advisor.allowed = true
    }

    fun leave() {
        engine.advisor.allowed = false
        auto = false
        if (conveyor) toggleConveyor()
        stopBench = true
        engine.releaseLlama()
    }

    fun viewMessage(text: String) {
        viewModelScope.launch(Dispatchers.Main) { message = text }
    }

    override fun onCleared() {
        engine.advisor.stop()
        engine.releaseLlama()
    }
}

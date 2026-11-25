package org.pcbqc.edge.llm

import android.app.ActivityManager
import android.content.Context
import android.os.SystemClock
import org.json.JSONObject
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Verdict
import org.pytorch.executorch.extension.llm.LlmCallback
import org.pytorch.executorch.extension.llm.LlmGenerationConfig
import org.pytorch.executorch.extension.llm.LlmModule
import org.pytorch.executorch.extension.llm.LlmModuleConfig
import java.io.File

data class LlmStats(
    val ttftMs: Long = 0,
    val totalMs: Long = 0,
    val promptTokens: Int = 0,
    val generatedTokens: Int = 0,
    val prefillTps: Double = 0.0,
    val decodeTps: Double = 0.0,
)

data class Advice(val text: String, val fromLlama: Boolean, val note: String, val stats: LlmStats?, val raw: String = "")

class Advisor(private val ctx: Context) {
    companion object {
        const val MODEL_FILE = "llama.pte"
        const val TOKENIZER_FILE = "tokenizer.model"
        private const val NEEDED_BYTES = 1_300_000_000L
        private const val SYSTEM =
            "You explain circuit board inspection results to a factory operator in simple English. Copy the facts you are given. Use this layout: SUMMARY, LIKELY CAUSE, ACTION with 3 numbered steps."
        private const val EXAMPLE_IN =
            "Board: B-00000\nVerdict: REWORK\nMain defect: missing hole\nSeverity: major\nWhere: 2 places, top left\nAlso found: pin hole\n" +
                "Known cause: A drilled hole is absent, usually from a broken drill or a drilling program error.\n" +
                "Approved steps: Hold the board for drilling rework; Check the drill bit and program; Count the holes on the next board"
        private const val EXAMPLE_OUT =
            "SUMMARY: Board B-00000 is marked REWORK. The main defect is a major missing hole in 2 places, top left. Pin hole was also found.\n" +
                "LIKELY CAUSE: A drilled hole is absent, usually from a broken drill or a drilling program error.\n" +
                "ACTION:\n1. Hold the board for drilling rework.\n2. Check the drill bit and program.\n3. Count the holes on the next board."
    }

    private var module: LlmModule? = null

    @Volatile
    var allowed = true

    var loadMs = 0L
        private set
    var loadError: String? = null
        private set

    private fun dirs() = listOfNotNull(File(ctx.filesDir, "llm"), ctx.getExternalFilesDir("llm"))

    private fun find(vararg names: String): File? =
        dirs().flatMap { d -> names.map { File(d, it) } }.firstOrNull { it.isFile && it.length() > 1000 }

    fun modelFile(): File? = find(MODEL_FILE) ?: dirs().flatMap { it.listFiles()?.toList() ?: emptyList() }
        .firstOrNull { it.extension == "pte" && it.length() > 100_000_000 }

    fun tokenizerFile(): File? = find(TOKENIZER_FILE)

    val installed get() = modelFile() != null && tokenizerFile() != null
    val loaded get() = module != null

    fun freeMemory(): Long {
        val mi = ActivityManager.MemoryInfo()
        (ctx.getSystemService(Context.ACTIVITY_SERVICE) as ActivityManager).getMemoryInfo(mi)
        return mi.availMem
    }

    @Synchronized
    fun load(): Boolean {
        if (!allowed) return fail("the app is in the background")
        if (module != null) return true
        val m = modelFile() ?: return fail("Llama model file is not installed")
        val t = tokenizerFile() ?: return fail("Tokenizer file is not installed")
        var tries = 0
        while (freeMemory() < NEEDED_BYTES && tries < 6) {
            tries++
            System.gc()
            Thread.sleep(700)
        }
        if (freeMemory() < NEEDED_BYTES) return fail("Not enough free memory for Llama (${freeMemory() / 1_000_000} MB free)")
        val t0 = SystemClock.elapsedRealtime()
        return try {
            val mod = LlmModule(
                LlmModuleConfig.create().modulePath(m.absolutePath).tokenizerPath(t.absolutePath)
                    .temperature(0.1f).loadMode(LlmModuleConfig.LOAD_MODE_MMAP).build(),
            )
            mod.load()
            module = mod
            loadMs = SystemClock.elapsedRealtime() - t0
            loadError = null
            true
        } catch (e: Throwable) {
            fail("Llama could not load: ${e.message ?: e.javaClass.simpleName}")
        }
    }

    private fun fail(msg: String): Boolean {
        loadError = msg
        return false
    }

    @Synchronized
    fun unload() {
        try {
            module?.close()
        } catch (_: Throwable) {
        }
        module = null
    }

    fun stop() {
        try {
            module?.stop()
        } catch (_: Throwable) {
        }
    }

    private fun chat(system: String, user: String) =
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$system<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n$user<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"

    private fun shown(system: String, inText: String, outText: String, user: String, start: String) =
        "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n$system<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n$inText<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n$outText<|eot_id|>" +
            "<|start_header_id|>user<|end_header_id|>\n\n$user<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n$start"

    private fun clean(s: String) = s.replace(Regex("<\\|[^|]*\\|>"), "")

    @Synchronized
    private fun generate(prompt: String, maxNew: Int, onText: (String) -> Unit): Pair<String, LlmStats> {
        val mod = module ?: error("Llama is not loaded")
        mod.resetContext()
        val sb = StringBuilder()
        var first = 0L
        var pieces = 0
        var failure: String? = null
        var raw: JSONObject? = null
        val t0 = SystemClock.elapsedRealtime()
        val cfg = LlmGenerationConfig.create().echo(false).maxNewTokens(maxNew).temperature(0.1f).build()
        mod.generate(
            prompt, cfg,
            object : LlmCallback {
                override fun onResult(result: String) {
                    if (first == 0L) first = SystemClock.elapsedRealtime()
                    pieces++
                    sb.append(result)
                    onText(clean(sb.toString()))
                    if (pieces >= maxNew) mod.stop()
                }

                override fun onStats(stats: String) {
                    raw = runCatching { JSONObject(stats) }.getOrNull()
                }

                override fun onError(errorCode: Int, message: String) {
                    failure = "$message ($errorCode)"
                }
            },
        )
        val end = SystemClock.elapsedRealtime()
        failure?.let { error(it) }
        val r = raw
        val stats = LlmStats(
            ttftMs = if (first > 0) first - t0 else 0,
            totalMs = end - t0,
            promptTokens = r?.optInt("prompt_tokens") ?: 0,
            generatedTokens = r?.optInt("generated_tokens") ?: pieces,
            prefillTps = r?.optDouble("prefill_token_per_sec", 0.0) ?: 0.0,
            decodeTps = r?.optDouble("decode_token_per_sec", 0.0)
                ?: if (end > first && first > 0) (pieces - 1) * 1000.0 / (end - first) else 0.0,
        )
        return clean(sb.toString()).trim() to stats
    }

    fun advise(ins: Inspection, onText: (String) -> Unit): Advice {
        if (ins.verdict == Verdict.PASS || ins.verdict == Verdict.RETAKE) {
            val t = Playbook.template(ins)
            onText(t)
            return Advice(t, false, "Nothing for Llama to explain on this board", null)
        }
        if (!load()) {
            val t = Playbook.template(ins)
            onText(t)
            return Advice(t, false, "Playbook text, not Llama: $loadError", null)
        }
        return try {
            val start = "SUMMARY: Board ${ins.boardId} is marked ${ins.verdict}. The main defect is"
            val prompt = shown(SYSTEM, EXAMPLE_IN, EXAMPLE_OUT, Playbook.facts(ins), start)
            val (more, stats) = generate(prompt, 170) { onText(Guard.tidy("$start $it")) }
            val text = Guard.tidy("$start $more")
            val bad = Guard.problem(text, ins)
            if (bad == null) Advice(text, true, "Llama 3.2 1B on ExecuTorch", stats, text)
            else {
                val t = Playbook.template(ins)
                onText(t)
                Advice(t, false, "Playbook text: Llama's $bad", stats, text)
            }
        } catch (e: Throwable) {
            val t = Playbook.template(ins)
            onText(t)
            Advice(t, false, "Playbook text: Llama stopped with an error (${e.message})", null)
        }
    }

    fun ask(ins: Inspection, question: String, onText: (String) -> Unit): Advice {
        if (!load()) return Advice("Llama is not available: $loadError", false, "", null)
        return try {
            val sys = "You help an operator on a circuit board line. Answer in at most three short sentences using only the facts. If the facts do not contain the answer, say you do not know and that a supervisor should check."
            val (raw, stats) = generate(chat(sys, "${Playbook.facts(ins)}\n\nQuestion: ${question.take(300)}"), 100) { onText(Guard.tidy(it)) }
            val text = Guard.tidy(raw)
            val bad = Guard.answerProblem(text, ins)
            if (bad == null) Advice(text, true, "Llama 3.2 1B on ExecuTorch", stats)
            else Advice("Llama's answer was withheld because the $bad. Please ask a supervisor.", false, "", stats)
        } catch (e: Throwable) {
            Advice("Llama stopped with an error: ${e.message}", false, "", null)
        }
    }

    fun shift(facts: JSONObject, onText: (String) -> Unit): Advice {
        if (!load()) return Advice("Llama is not available: $loadError", false, "", null)
        return try {
            val sys = "You write a three-sentence shift note for a circuit board line in simple English. Copy the facts. Add nothing."
            val example = "Boards: 9\nVerdicts: 5 PASS, 4 REWORK\nMost common defect: pin hole, 6 times\nRepeating place: none\nProcess hint: none\nCheck first: Check dust control and plating"
            val reply = "In the last 12 hours, 9 boards were inspected: 5 PASS and 4 REWORK. The most common defect was pin hole, 6 times. First, check dust control and plating."
            val start = "In the last 12 hours,"
            val (more, stats) = generate(shown(sys, example, reply, Playbook.shiftFacts(facts), start), 110) { onText(Guard.tidy("$start $it")) }
            val text = Guard.tidy("$start $more")
            val bad = Guard.shiftProblem(text, facts)
            if (bad == null) Advice(text, true, "Llama 3.2 1B on ExecuTorch", stats)
            else Advice("${Playbook.shift(facts)}\n(Plain numbers shown: Llama's $bad.)", false, "", stats)
        } catch (e: Throwable) {
            Advice("Llama stopped with an error: ${e.message}", false, "", null)
        }
    }
}

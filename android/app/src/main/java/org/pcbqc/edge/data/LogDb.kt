package org.pcbqc.edge.data

import android.content.ContentValues
import android.content.Context
import android.database.sqlite.SQLiteDatabase
import android.database.sqlite.SQLiteOpenHelper
import org.json.JSONArray
import org.json.JSONObject
import org.pcbqc.edge.core.Drift
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.llm.Advice
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

data class LogRow(
    val id: Long,
    val time: String,
    val board: String,
    val verdict: String,
    val defects: Int,
    val visionMs: Float,
    val image: String,
    val record: String,
    val advice: String,
    val adviceSource: String,
    val operator: String,
    val reasonCode: String,
    val reason: String,
    val note: String,
    val finalVerdict: String,
) {
    val why: List<String>
        get() = runCatching { JSONObject(record).optJSONArray("why")?.let { a -> (0 until a.length()).map { a.getString(it) } } }.getOrNull().orEmpty()
}

class LogDb(ctx: Context) : SQLiteOpenHelper(ctx, "inspections.db", null, 3) {
    override fun onCreate(db: SQLiteDatabase) {
        db.execSQL(
            "CREATE TABLE inspections(id INTEGER PRIMARY KEY AUTOINCREMENT, ts INTEGER, board TEXT, verdict TEXT, defects INTEGER, vision_ms REAL, " +
                "image TEXT, record TEXT, advice TEXT, advice_source TEXT, llama INTEGER, ttft_ms INTEGER, llm_ms INTEGER, tokens_per_s REAL, llama_raw TEXT, operator TEXT DEFAULT '', " +
                "reason_code TEXT DEFAULT '', reason TEXT DEFAULT '', note TEXT DEFAULT '', final_verdict TEXT DEFAULT '')",
        )
    }

    override fun onUpgrade(db: SQLiteDatabase, oldVersion: Int, newVersion: Int) {
        if (oldVersion < 2) listOf("reason_code", "reason", "note").forEach { db.execSQL("ALTER TABLE inspections ADD COLUMN $it TEXT DEFAULT ''") }
        if (oldVersion < 3) db.execSQL("ALTER TABLE inspections ADD COLUMN final_verdict TEXT DEFAULT ''")
    }

    private fun ContentValues.reasons(ins: Inspection) {
        val r = ins.why
        put("reason_code", r.code)
        put("reason", r.short)
    }

    fun nextBoard(): String {
        val n = readableDatabase.rawQuery("SELECT COALESCE(MAX(id),0)+1 FROM inspections", null).use { it.moveToFirst(); it.getLong(0) }
        return "B-%05d".format(Locale.US, n)
    }

    fun insert(ins: Inspection, image: String): Long = writableDatabase.insert(
        "inspections", null,
        ContentValues().apply {
            put("ts", System.currentTimeMillis())
            put("board", ins.boardId)
            put("verdict", ins.verdict.name)
            put("defects", ins.findings.size)
            put("vision_ms", ins.times.totalMs)
            put("image", image)
            put("record", ins.record().toString())
            put("advice", "")
            put("advice_source", "pending")
            reasons(ins)
        },
    )

    fun setAdvice(id: Long, a: Advice) {
        writableDatabase.update(
            "inspections",
            ContentValues().apply {
                put("advice", a.text)
                put("advice_source", a.note)
                put("llama_raw", a.raw)
                put("llama", if (a.fromLlama) 1 else 0)
                a.stats?.let {
                    put("ttft_ms", it.ttftMs)
                    put("llm_ms", it.totalMs)
                    put("tokens_per_s", it.decodeTps)
                }
            },
            "id=?", arrayOf(id.toString()),
        )
    }

    fun replace(id: Long, ins: Inspection) {
        writableDatabase.update(
            "inspections",
            ContentValues().apply {
                put("verdict", ins.verdict.name)
                put("defects", ins.findings.size)
                put("record", ins.record().toString())
                reasons(ins)
            },
            "id=?", arrayOf(id.toString()),
        )
    }

    fun spots(refId: Long): Pair<Int, List<Drift.Spot>> {
        val out = ArrayList<Drift.Spot>()
        var boards = 0
        readableDatabase.rawQuery("SELECT board,record,final_verdict FROM inspections ORDER BY id DESC LIMIT 400", null).use { c ->
            while (c.moveToNext() && boards < Drift.HISTORY) {
                val r = JSONObject(c.getString(1))
                if (r.optLong("ref_id") != refId || r.optBoolean("not_board")) continue
                boards++
                if (c.getString(2) == "PASS") continue
                val d = r.optJSONArray("defects") ?: continue
                for (i in 0 until d.length()) {
                    val o = d.getJSONObject(i)
                    val p = o.optJSONArray("ref_pos") ?: continue
                    if (p.getDouble(0) !in 0.0..1.0 || p.getDouble(1) !in 0.0..1.0) continue
                    out += Drift.Spot(c.getString(0), o.getString("type"), p.getDouble(0).toFloat(), p.getDouble(1).toFloat())
                }
            }
        }
        return boards to out
    }

    fun setOperator(id: Long, decision: String, note: String, finalVerdict: String) {
        writableDatabase.update(
            "inspections",
            ContentValues().apply {
                put("operator", decision)
                put("note", note)
                put("final_verdict", finalVerdict)
            },
            "id=?", arrayOf(id.toString()),
        )
    }

    fun reasonCounts(): List<Pair<String, Int>> = readableDatabase.rawQuery(
        "SELECT reason_code,COUNT(*) FROM inspections WHERE reason_code NOT IN ('','PASS') AND (final_verdict='' OR final_verdict=verdict) GROUP BY reason_code ORDER BY COUNT(*) DESC, reason_code LIMIT 8", null,
    ).use { c -> buildList { while (c.moveToNext()) add(c.getString(0) to c.getInt(1)) } }

    fun recent(limit: Int = 200): List<LogRow> {
        val f = SimpleDateFormat("dd MMM HH:mm:ss", Locale.US)
        return readableDatabase.rawQuery(
            "SELECT id,ts,board,verdict,defects,vision_ms,image,record,advice,advice_source,operator,reason_code,reason,note,final_verdict FROM inspections ORDER BY id DESC LIMIT $limit", null,
        ).use { c ->
            buildList {
                while (c.moveToNext()) add(
                    LogRow(
                        c.getLong(0), f.format(Date(c.getLong(1))), c.getString(2), c.getString(3), c.getInt(4), c.getFloat(5),
                        c.getString(6) ?: "", c.getString(7) ?: "", c.getString(8) ?: "", c.getString(9) ?: "", c.getString(10) ?: "",
                        c.getString(11) ?: "", c.getString(12) ?: "", c.getString(13) ?: "", c.getString(14) ?: "",
                    ),
                )
            }
        }
    }

    fun shiftFacts(): JSONObject {
        val since = System.currentTimeMillis() - 12 * 3600_000L
        val out = JSONObject()
        val verdicts = JSONObject()
        var boards = 0
        val types = HashMap<String, Int>()
        readableDatabase.rawQuery("SELECT COALESCE(NULLIF(final_verdict,''),verdict),record FROM inspections WHERE ts>$since", null).use { c ->
            while (c.moveToNext()) {
                boards++
                verdicts.put(c.getString(0), verdicts.optInt(c.getString(0)) + 1)
                val rec = JSONObject(c.getString(1))
                val d = if (rec.optBoolean("not_board") || c.getString(0) == "PASS") JSONArray() else rec.optJSONArray("defects") ?: JSONArray()
                for (i in 0 until d.length()) types.merge(d.getJSONObject(i).getString("type").replace('_', ' '), 1, Int::plus)
            }
        }
        out.put("period", "last 12 hours").put("boards", boards).put("verdicts", verdicts)
        out.put("defects", JSONObject().apply { types.entries.sortedByDescending { it.value }.take(5).forEach { put(it.key, it.value) } })
        return out
    }

    fun csv(): String {
        fun q(s: String?) = "\"" + (s ?: "").replace("\"", "\"\"").replace("\n", " | ") + "\""
        val f = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
        val sb = StringBuilder("id,time,board,verdict,reason_code,reason,defects,vision_ms,advice_source,llama,ttft_ms,llm_ms,tokens_per_s,operator,final_verdict,operator_note,record,advice\n")
        readableDatabase.rawQuery(
            "SELECT id,ts,board,verdict,defects,vision_ms,advice_source,llama,ttft_ms,llm_ms,tokens_per_s,operator,record,advice,reason_code,reason,note,final_verdict FROM inspections ORDER BY id", null,
        ).use { c ->
            while (c.moveToNext()) {
                sb.append(c.getLong(0)).append(',').append(f.format(Date(c.getLong(1)))).append(',').append(q(c.getString(2))).append(',')
                    .append(c.getString(3)).append(',').append(q(c.getString(14))).append(',').append(q(c.getString(15))).append(',').append(c.getInt(4)).append(',').append("%.1f".format(Locale.US, c.getFloat(5))).append(',')
                    .append(q(c.getString(6))).append(',').append(c.getInt(7)).append(',').append(c.getLong(8)).append(',').append(c.getLong(9)).append(',')
                    .append("%.2f".format(Locale.US, c.getDouble(10))).append(',').append(q(c.getString(11))).append(',').append(c.getString(17) ?: "").append(',').append(q(c.getString(16))).append(',').append(q(c.getString(12))).append(',')
                    .append(q(c.getString(13))).append('\n')
            }
        }
        return sb.toString()
    }

    fun clear() {
        writableDatabase.delete("inspections", null, null)
    }
}

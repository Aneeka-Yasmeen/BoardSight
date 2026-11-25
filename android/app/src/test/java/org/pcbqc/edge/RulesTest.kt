package org.pcbqc.edge

import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.pcbqc.edge.core.Box
import org.pcbqc.edge.core.Drift
import org.pcbqc.edge.core.Finding
import org.pcbqc.edge.core.Inspection
import org.pcbqc.edge.core.Quality
import org.pcbqc.edge.core.Rules
import org.pcbqc.edge.core.SecondLook
import org.pcbqc.edge.core.Severity
import org.pcbqc.edge.core.StageTimes
import org.pcbqc.edge.core.Verdict
import org.pcbqc.edge.llm.Guard
import org.pcbqc.edge.llm.Playbook

class RulesTest {
    private val good = Quality(200f, 120f, 0f, emptyList())
    private val blurred = Quality(5f, 120f, 0f, listOf("image is blurred"))

    private fun f(type: String, conf: Float = 0.9f, x: Float = 100f, y: Float = 100f, s: Float = 20f, src: String = "yolo") =
        Finding(type, conf, Box(x, y, x + s, y + s), src)

    private fun decide(list: List<Finding>, q: Quality = good) = Rules.verdict(Rules.grade(list, 2000, 1500), q).first

    private fun inspection(list: List<Finding>, q: Quality = good): Inspection {
        val g = Rules.grade(list, 2000, 1500)
        val (v, r) = Rules.verdict(g, q)
        return Inspection("B-1", 2000, 1500, g, v, r, q, StageTimes(), "")
    }

    @Test fun reasonCodesMatchTheVerdict() {
        assertEquals("PASS", inspection(emptyList()).why.code)
        assertEquals("IMG-BLUR", inspection(emptyList(), blurred).why.code)
        assertEquals("REJ-SHT", inspection(listOf(f("short"))).why.code)
        assertEquals("RWK-MHL", inspection(listOf(f("missing_hole"))).why.code)
        assertEquals("REV-LOW-SHT", inspection(listOf(f("short", 0.38f))).why.code)
        assertEquals("REV-LOW-MHL", inspection(listOf(f("missing_hole", 0.38f))).why.code)
        assertEquals("REV-MIN-SPR", inspection(listOf(f("spur"))).why.code)
        assertEquals("REV-MIN-DIF", inspection(listOf(f("anomaly", src = "reference"))).why.code)
    }

    @Test fun sameDefectsGiveTheSameCodeInAnyOrder() =
        assertEquals(
            inspection(listOf(f("short", 0.9f), f("open_circuit", 0.6f, x = 900f))).why.code,
            inspection(listOf(f("short", 0.6f), f("open_circuit", 0.9f, x = 900f))).why.code,
        )

    @Test fun rejectReasonNamesOnlyCertainCriticalDefects() {
        val r = inspection(listOf(f("short"), f("open_circuit", 0.4f, x = 900f), f("spur", x = 1500f))).why
        assertEquals("REJ-SHT", r.code)
        assertTrue(r.short.contains("short x1 (90%)"))
        assertTrue(r.details.any { it.startsWith("OPN critical 40%") })
    }

    @Test fun reviewReasonShowsTheCertainMajorBehindAnUncertainCritical() {
        val r = inspection(listOf(f("short", 0.4f), f("missing_hole", x = 900f))).why
        assertEquals("REV-LOW-SHT", r.code)
        assertTrue(r.short.contains("certain major missing hole x1"))
    }

    @Test fun everyReasonIsShortAndLoggedInTheRecord() {
        for (v in listOf(emptyList(), listOf(f("short")), listOf(f("short", 0.4f)), listOf(f("spur")), listOf(f("missing_pad")))) {
            val i = inspection(v)
            assertTrue(i.why.short.length <= 120)
            val rec = i.record()
            assertEquals(i.why.code, rec.getString("reason_code"))
            assertEquals(i.why.short, rec.getString("reason"))
            assertEquals(i.why.details.size, rec.getJSONArray("why").length())
        }
    }

    private fun at(type: String, conf: Float, x1: Float, y1: Float, x2: Float, y2: Float) = Finding(type, conf, Box(x1, y1, x2, y2), "yolo")

    @Test fun halvesFromTwoTilesBecomeOneBox() {
        val r = Rules.fuse(listOf(at("spurious_copper", 0.9f, 190f, 470f, 320f, 590f), at("spurious_copper", 0.6f, 190f, 470f, 320f, 530f)))
        assertEquals(1, r.size)
        assertEquals(590f, r[0].box.y2, 0.1f)
        assertEquals(0.9f, r[0].confidence, 0.001f)
    }

    @Test fun separateDefectsStaySeparate() =
        assertEquals(2, Rules.fuse(listOf(at("spur", 0.9f, 0f, 0f, 20f, 20f), at("spur", 0.8f, 30f, 0f, 50f, 20f))).size)

    @Test fun aShortIsNeverDroppedForAMilderType() {
        val r = Rules.fuse(listOf(at("spur", 0.9f, 0f, 0f, 20f, 20f), at("short", 0.6f, 1f, 1f, 20f, 20f)))
        assertEquals(setOf("spur", "short"), r.map { it.type }.toSet())
        assertEquals(1, Rules.fuse(listOf(at("short", 0.9f, 0f, 0f, 20f, 20f), at("spur", 0.6f, 1f, 1f, 20f, 20f))).size)
    }

    @Test fun defectCutByThePictureEdgeIsNotCertain() {
        val g = Rules.grade(listOf(at("open_circuit", 0.9f, 1960f, 700f, 2000f, 740f)), 2000, 1500)
        assertTrue(g[0].cut)
        assertEquals(Verdict.REVIEW, Rules.verdict(g, good).first)
        val i = inspection(listOf(at("open_circuit", 0.9f, 1960f, 700f, 2000f, 740f)))
        assertTrue(i.why.short.contains("cut off by the picture edge"))
        assertNotNull(SecondLook.target(i))
        assertEquals(Verdict.REJECT, SecondLook.apply(i, i.findings[0], listOf(f("open_circuit", 0.8f)), good, true).first.verdict)
    }

    @Test fun aSceneThatIsNotABoardIsNeverJudged() {
        assertEquals(Verdict.RETAKE, Rules.verdict(Rules.grade(listOf(f("short")), 2000, 1500), good, true).first)
        assertEquals(Verdict.RETAKE, Rules.verdict(emptyList(), good, true).first)
        val i = inspection(listOf(f("short"))).copy(verdict = Verdict.RETAKE, notBoard = true)
        assertEquals("IMG-NOBOARD", i.why.code)
        val many = inspection((0..40).map { f("spurious_copper", x = 40f * it) }).copy(verdict = Verdict.RETAKE, notBoard = true)
        assertTrue(many.why.details[0].contains("41 marks"))
        assertTrue(i.why.details.none { it.startsWith("SHT") })
        assertTrue(i.record().getBoolean("not_board"))
        assertTrue(Playbook.template(i).contains("does not look like"))
    }

    @Test fun aDarkCameraPictureIsNeverJudged() {
        val dark = Quality(200f, 12f, 0f, listOf("image is too dark"), true)
        assertEquals(Verdict.RETAKE, decide(listOf(f("short")), dark))
        assertEquals(Verdict.RETAKE, decide(emptyList(), dark))
        assertEquals("IMG-DARK", inspection(listOf(f("short")), dark).why.code)
    }

    @Test fun cleanBoardPasses() = assertEquals(Verdict.PASS, decide(emptyList()))

    @Test fun blurredCleanBoardNeverPasses() = assertEquals(Verdict.RETAKE, decide(emptyList(), blurred))

    @Test fun blurredImageStillRejectsAShort() = assertEquals(Verdict.REJECT, decide(listOf(f("short")), blurred))

    @Test fun shortAndOpenReject() {
        assertEquals(Verdict.REJECT, decide(listOf(f("short"))))
        assertEquals(Verdict.REJECT, decide(listOf(f("open_circuit"))))
    }

    @Test fun uncertainCriticalGoesToAPerson() = assertEquals(Verdict.REVIEW, decide(listOf(f("short", 0.31f))))

    @Test fun missingHoleNeedsRework() = assertEquals(Verdict.REWORK, decide(listOf(f("missing_hole"))))

    @Test fun smallIsolatedSpurIsMinor() {
        val g = Rules.grade(listOf(f("spur")), 2000, 1500)
        assertEquals(Severity.MINOR, g[0].severity)
        assertEquals(Verdict.REVIEW, decide(listOf(f("spur"))))
    }

    @Test fun confidenceDoesNotChangeSeverity() =
        assertEquals(Rules.grade(listOf(f("spur", 0.3f)), 2000, 1500)[0].severity, Rules.grade(listOf(f("spur", 0.99f)), 2000, 1500)[0].severity)

    @Test fun repeatedDefectIsMajor() {
        val g = Rules.grade(listOf(f("spur", x = 100f), f("spur", x = 700f), f("spur", x = 1300f)), 2000, 1500)
        assertTrue(g.all { it.severity == Severity.MAJOR })
    }

    @Test fun neighbouringDefectsAreMajor() {
        val g = Rules.grade(listOf(f("spur", x = 100f), f("mouse_bite", x = 125f)), 2000, 1500)
        assertTrue(g.all { it.severity == Severity.MAJOR })
    }

    @Test fun referenceBoxAlreadyExplainedIsDropped() {
        val m = Rules.merge(listOf(f("short")), listOf(f("anomaly", x = 104f, y = 104f, s = 12f, src = "reference"), f("anomaly", x = 900f, src = "reference")))
        assertEquals(listOf("short", "anomaly"), m.map { it.type })
    }

    @Test fun overlappingBoxesAreMerged() = assertEquals(1, Rules.nms(listOf(f("short", 0.9f), f("short", 0.6f, x = 102f))).size)

    @Test fun zonesAreNamed() {
        assertEquals("top left", Rules.zone(Box(0f, 0f, 10f, 10f), 900, 900))
        assertEquals("centre", Rules.zone(Box(440f, 440f, 460f, 460f), 900, 900))
        assertEquals("bottom right", Rules.zone(Box(890f, 890f, 900f, 900f), 900, 900))
    }

    private val faithful = "SUMMARY: Board B-1 is marked REJECT because of 1 short at the top left.\nLIKELY CAUSE: Copper joins two tracks, usually from under-etching.\n" +
        "ACTION:\n1. Quarantine the board and mark the spot.\n2. Check etching time and the mask at that area.\n3. Inspect the neighbouring boards of this batch."

    @Test fun guardAcceptsAFaithfulReply() = assertNull(Guard.problem(faithful, inspection(listOf(f("short")))))

    @Test fun llamaIsToldOnlyAboutCertainPlaces() {
        val i = inspection(listOf(f("short", 0.79f), f("short", 0.3f, x = 900f)))
        assertTrue(Playbook.facts(i).contains("Where: 1 place,"))
        assertNull(Guard.problem(faithful, i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "3 shorts"), i))
    }

    @Test fun guardRejectsRealBadRepliesFromLlama() {
        val i = inspection(listOf(f("short"), f("open_circuit", x = 900f)))
        val steps = "ACTION:\n1. Quarantine the board and mark the spot.\n2. Check etching time and the mask at the area.\n3. Inspect the neighbouring boards of this batch."
        assertNotNull(Guard.problem(Guard.tidy("**Summary:** The inspection result is final and there are no defects or changes to the verdict.\n\n**Likely Cause:** The inspection result is final because the verdict is \"REJECT\" and there are no defects.\n\n**$steps"), i))
        assertNotNull(Guard.problem(Guard.tidy("**Summary:** The inspection result is final and there are no defects that need to be changed or added.\n\n**Likely Cause:** The inspection process was not thorough enough, and some defects were overlooked.\n\n$steps"), i))
        assertNotNull(Guard.problem(Guard.tidy("Here is the response:\n\n**SUMMARY:** The inspection result is final and there are no defects that need to be changed.\n\n**LIKELY CAUSE:** The board was not properly etched, resulting in a short circuit.\n\n$steps"), i))
    }

    @Test fun guardRejectsEachKindOfMistake() {
        val i = inspection(listOf(f("short")))
        assertNotNull(Guard.problem("SUMMARY: fine", i))
        assertNotNull(Guard.problem(faithful.replace("REJECT", "PASS"), i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "3 shorts"), i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "a missing hole"), i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "a fault"), i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "5 critical defects, one short"), i))
        assertNotNull(Guard.problem(faithful.replace("1 short", "a short in 4 places"), i))
        assertNotNull(Guard.problem(faithful.replace("Copper joins two tracks, usually from under-etching", "The operator was tired"), i))
        assertNotNull(Guard.problem(faithful.replace("Quarantine the board and mark the spot", "Throw the machine away"), i))
        assertNotNull(Guard.problem(faithful.replace("at the top left", "at the top left, safe to use"), i))
    }

    @Test fun answersAndShiftNotesAreChecked() {
        val i = inspection(listOf(f("short")))
        assertNull(Guard.answerProblem("Mark the short at the top left first.", i))
        assertNotNull(Guard.answerProblem("The board has no defects.", i))
        assertNotNull(Guard.answerProblem("Look at the mouse bite.", i))
        val facts = org.json.JSONObject().put("boards", 5).put("defects", org.json.JSONObject().put("short", 3))
        assertNull(Guard.shiftProblem("5 boards were checked. The most common defect was short, 3 times.", facts))
        assertNotNull(Guard.shiftProblem("7 boards were checked.", facts))
        assertNotNull(Guard.shiftProblem("5 boards had mouse bites.", facts))
        assertNotNull(Guard.shiftProblem("Number of boards: 5. Most common defect: short. Hint: look at the boards for any signs of damage or wear caused by careless operators.", facts))
        assertNotNull(Guard.shiftProblem("5 boards were inspected.", facts))
        assertNull(Guard.shiftProblem("In the last 12 hours, 5 boards were inspected. The most common defect was short, 3 times. First, check etching time and the mask at that area.", facts))
    }

    @Test fun everyTemplatePassesTheGuard() {
        for (t in listOf("short", "open_circuit", "mouse_bite", "spur", "spurious_copper", "pin_hole", "missing_hole", "missing_pad", "anomaly")) {
            val i = inspection(listOf(f(t)))
            assertNull(t, Guard.problem(Playbook.template(i), i))
        }
    }

    @Test fun recordIsValidJson() {
        val r = inspection(listOf(f("short"))).record()
        assertEquals("REJECT", r.getString("verdict"))
        assertEquals(4, r.getJSONArray("defects").getJSONObject(0).getJSONArray("bbox").length())
    }

    private fun spot(b: Int, x: Float, y: Float, t: String = "short") = Drift.Spot("B-$b", t, x, y)

    @Test fun samePlaceOnThreeBoardsIsAHotspot() {
        val r = Drift.analyse(5, listOf(spot(1, 0.50f, 0.50f), spot(2, 0.51f, 0.49f), spot(3, 0.50f, 0.51f), spot(4, 0.9f, 0.1f)))
        assertEquals(1, r.hotspots.size)
        assertEquals(3, r.hotspots[0].boards)
        assertEquals("centre", r.hotspots[0].zone)
        assertEquals(1, r.scattered)
    }

    @Test fun manyDefectsOnOneBoardAreNotAHotspot() =
        assertEquals(0, Drift.analyse(5, listOf(spot(1, 0.5f, 0.5f), spot(1, 0.51f, 0.5f), spot(1, 0.5f, 0.51f))).hotspots.size)

    @Test fun scatteredDefectsAreNotAHotspot() =
        assertEquals(0, Drift.analyse(5, listOf(spot(1, 0.1f, 0.1f), spot(2, 0.5f, 0.5f), spot(3, 0.9f, 0.9f))).hotspots.size)

    @Test fun tooFewBoardsGivesNoConclusion() = assertTrue(Drift.analyse(2, emptyList()).hint.contains("at least"))

    @Test fun secondLookIsOnlyOfferedWhenUnsure() {
        assertNotNull(SecondLook.target(inspection(listOf(f("short", 0.31f)))))
        assertNull(SecondLook.target(inspection(listOf(f("short", 0.9f)))))
        assertNull(SecondLook.target(inspection(emptyList())))
    }

    @Test fun secondLookConfirms() {
        val i = inspection(listOf(f("short", 0.31f)))
        val (n, o) = SecondLook.apply(i, i.findings[0], listOf(f("short", 0.8f)), good, true)
        assertEquals(SecondLook.Outcome.CONFIRMED, o)
        assertEquals(Verdict.REJECT, n.verdict)
        assertTrue(n.record().getString("second_look").contains("REVIEW to REJECT"))
    }

    @Test fun secondLookNeverClearsACriticalDefect() {
        val i = inspection(listOf(f("short", 0.31f)))
        val (n, o) = SecondLook.apply(i, i.findings[0], emptyList(), good, true)
        assertEquals(SecondLook.Outcome.UNRESOLVED, o)
        assertEquals(Verdict.REVIEW, n.verdict)
    }

    @Test fun secondLookClearsAMajorOnlyWithASharpEmptyCloseUp() {
        val i = inspection(listOf(f("missing_hole", 0.31f)))
        assertEquals(Verdict.PASS, SecondLook.apply(i, i.findings[0], emptyList(), good, true).first.verdict)
        assertEquals(Verdict.REVIEW, SecondLook.apply(i, i.findings[0], emptyList(), blurred, true).first.verdict)
        assertEquals(Verdict.REVIEW, SecondLook.apply(i, i.findings[0], emptyList(), good, false).first.verdict)
        assertEquals(Verdict.REVIEW, SecondLook.apply(i, i.findings[0], listOf(f("spur", 0.5f)), good, true).first.verdict)
    }

    @Test fun hitsAlreadyOnTheReferenceAreIgnoredOnlyAtTheSameSpotAndType() {
        val known = listOf(Rules.Known("short", 0.5f, 0.5f))
        val list = listOf(
            f("short").copy(refX = 0.505f, refY = 0.5f), f("short").copy(refX = 0.55f, refY = 0.5f),
            f("open_circuit").copy(refX = 0.5f, refY = 0.5f), f("short"),
        )
        val (kept, dropped) = Rules.dropKnown(list, known)
        assertEquals(1, dropped)
        assertEquals(listOf("short", "open_circuit", "short"), kept.map { it.type })
    }
}

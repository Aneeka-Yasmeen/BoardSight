import json
import os
import shutil
import subprocess

RUNNER = os.environ.get("LLAMA_RUNNER", "llama_main")
MODEL = os.environ.get("LLAMA_PTE", "models/llama.pte")
TOKENIZER = os.environ.get("LLAMA_TOK", "models/tokenizer.model")
SYSTEM = (
    "You are a quality-control assistant for a printed circuit board line. The inspection result is final: never change the verdict and never add defects. "
    "Use only the facts and approved actions you are given. Write in simple English in exactly this layout:\n"
    "SUMMARY: one or two sentences.\nLIKELY CAUSE: one sentence.\nACTION:\n1. step\n2. step\n3. step"
)
PLAYBOOK = {
    "short": ("Copper joins two tracks, usually from under-etching or a masking fault.", ["Quarantine the board and mark the spot", "Check etching time and the mask at that area", "Inspect the neighbouring boards of this batch"]),
    "open_circuit": ("A track is broken, usually from over-etching, a scratch or a masking fault.", ["Quarantine the board and mark the break", "Check the mask and etching strength", "Inspect the previous boards of this batch"]),
    "mouse_bite": ("Copper is missing along a track edge, usually from etching or mask damage.", ["Measure the remaining track width", "Send to rework if the track is too thin", "Check the mask edge and etching bath"]),
    "spur": ("A copper spike sticks out of a track, usually from incomplete etching.", ["Measure the gap to the next track", "Trim the spike if the gap is too small", "Check etching time"]),
    "spurious_copper": ("Copper was left where there should be none, usually from incomplete etching.", ["Check whether it touches a track or pad", "Remove the copper or send to rework", "Check etching time and board cleaning"]),
    "pin_hole": ("A tiny hole in the copper, usually from dust or plating trouble.", ["Check whether it lies on a pad or narrow track", "Send to rework if it does", "Check dust control and plating"]),
    "missing_hole": ("A drilled hole is absent, usually from a broken drill or a drilling program error.", ["Hold the board for drilling rework", "Check the drill bit and program", "Count the holes on the next board"]),
    "missing_pad": ("A solder pad is absent or badly formed, usually from a mask or etching fault.", ["Hold the board for rework or scrap", "Check the mask at that pad", "Inspect the same pad on the next board"]),
    "anomaly": ("This area differs from the reference board, and the type is unknown.", ["Look at the marked area under good light", "Clean the area and scan again", "Ask a supervisor if the difference remains"]),
}


def template(ev):
    if ev["verdict"] == "PASS":
        return "SUMMARY: No defects were found in a usable image.\nLIKELY CAUSE: none\nACTION:\n1. Send the board to the next stage"
    if ev["verdict"] == "RETAKE":
        return "SUMMARY: The picture is not good enough to judge this board.\nLIKELY CAUSE: camera or lighting\nACTION:\n1. Scan again in good light"
    cause, steps = PLAYBOOK.get(ev["defects"][0]["cls"], PLAYBOOK["anomaly"])
    kinds = ", ".join(sorted({d["cls"].replace("_", " ") for d in ev["defects"]}))
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(steps))
    return f"SUMMARY: {len(ev['defects'])} issue(s) found: {kinds}. Verdict {ev['verdict']}.\nLIKELY CAUSE: {cause}\nACTION:\n{numbered}"


def explain(ev, timeout=120):
    if ev["verdict"] in ("PASS", "RETAKE"):
        return template(ev), False
    if not (shutil.which(RUNNER) and os.path.exists(MODEL) and os.path.exists(TOKENIZER)):
        return template(ev), False
    facts = {
        "verdict": ev["verdict"],
        "defects": [{"type": d["cls"].replace("_", " "), "severity": d["severity"], "where": d["zone"]} for d in ev["defects"][:6]],
        "approved_actions": PLAYBOOK.get(ev["defects"][0]["cls"], PLAYBOOK["anomaly"]),
    }
    prompt = (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{SYSTEM}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n"
        f"Inspection facts:\n{json.dumps(facts)}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    )
    cmd = [RUNNER, f"--model_path={MODEL}", f"--tokenizer_path={TOKENIZER}", f"--prompt={prompt}", "--temperature=0.1", "--max_new_tokens=170"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        out = r.stdout.split("<|start_header_id|>assistant<|end_header_id|>")[-1].split("<|eot_id|>")[0].split("PyTorchObserver")[0].strip()
        if r.returncode == 0 and "SUMMARY" in out.upper() and "ACTION" in out.upper():
            return out, True
    except (OSError, subprocess.SubprocessError):
        pass
    return template(ev), False

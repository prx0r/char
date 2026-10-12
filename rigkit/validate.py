"""Validate rig.json files against rigkit/rig.schema.json and check clip retargetability (every bone a shared clip
touches exists in the rig). python3 rigkit/validate.py <slug> [...]"""
import json, os, sys, glob
import jsonschema
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
schema = json.load(open(os.path.join(ROOT, "rigkit", "rig.schema.json")))
clips = {n: json.load(open(os.path.join(ROOT, "clips", n + ".json"))) for n in json.load(open(os.path.join(ROOT, "clips", "index.json")))}
ok = True
for slug in sys.argv[1:]:
    rig = json.load(open(os.path.join(ROOT, "characters", slug, "rig", "rig.json")))
    errs = sorted(jsonschema.Draft202012Validator(schema).iter_errors(rig), key=lambda e: list(e.path))
    bones = {b["name"] for b in rig["skeleton"]["bones"]}
    missing = set()
    for c in clips.values():
        for t in c.get("tracks", {}):
            head = t.rsplit(".", 1)[0]
            if head in ("face", "fx") or head.startswith("ik"): continue
            for b in ([head.replace("*", "r"), head.replace("*", "l")] if "*" in head else [head]):
                if b not in bones: missing.add(b)
        for ik in (c.get("ik") or {}).values():
            if ik.get("target") not in rig["anchors"]: missing.add("anchor:" + str(ik.get("target")))
    print(json.dumps({"slug": slug, "schema_errors": [f"{list(e.path)}: {e.message[:120]}" for e in errs[:10]], "n_errors": len(errs),
                      "clips": len(clips), "unresolved_clip_targets": sorted(missing)}))
    ok &= not errs and not missing
sys.exit(0 if ok else 1)

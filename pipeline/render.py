"""Frame-accurate render of a timeline in headless Chromium (Playwright) -> H.264 + audio mux."""
import base64, json, os, shutil, subprocess, sys, time, threading, http.server, socketserver, functools
from multiprocessing import Process

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
CHROMIUM = os.environ.get("CHROMIUM", "/usr/bin/chromium")

class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *a): pass

def serve(port):
    h = functools.partial(_Quiet, directory=ROOT)
    socketserver.TCPServer.allow_reuse_address = True
    httpd = socketserver.ThreadingTCPServer(("127.0.0.1", port), h)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd

def _worker(port, tl_url, f0, f1, out, fps, q):
    from playwright.sync_api import sync_playwright
    ff = subprocess.Popen(["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "image2pipe", "-framerate", str(fps), "-c:v", "mjpeg", "-i", "-",
                           "-c:v", "libx264", "-preset", "medium", "-crf", "17", "-pix_fmt", "yuv420p", "-r", str(fps), out], stdin=subprocess.PIPE)
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox", "--disable-gpu", "--force-color-profile=srgb"])
        pg = b.new_page(viewport={"width": 1080, "height": 1920})
        pg.goto("http://127.0.0.1:%d/renderer/stage.html" % port)
        tl = json.load(open(tl_url))
        pg.evaluate("tl => window.STAGE_INIT(tl)", tl)
        for i in range(f0, f1):
            d = pg.evaluate("([i,q]) => window.STAGE_FRAME(i,q)", [i, q])
            ff.stdin.write(base64.b64decode(d.split(",", 1)[1]))
        b.close()
    ff.stdin.close(); ff.wait()

def render(timeline_path, audio_path, out_mp4, workers=2, port=8765, q=0.93, frames=None):
    tl = json.load(open(timeline_path))
    fps = tl["fps"]; n = int(round(tl["duration"] * fps)) if frames is None else frames
    httpd = serve(port)
    t = time.time()
    chunks = []; procs = []
    per = (n + workers - 1) // workers
    tmp = out_mp4 + ".parts"; os.makedirs(tmp, exist_ok=True)
    for w in range(workers):
        f0, f1 = w * per, min(n, (w + 1) * per)
        if f0 >= f1: continue
        part = os.path.join(tmp, "part%02d.mp4" % w); chunks.append(part)
        pr = Process(target=_worker, args=(port, timeline_path, f0, f1, part, fps, q)); pr.start(); procs.append(pr)
    for pr in procs: pr.join()
    lst = os.path.join(tmp, "list.txt")
    open(lst, "w").write("".join("file '%s'\n" % os.path.abspath(c) for c in chunks))
    cmd = ["ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "concat", "-safe", "0", "-i", lst]
    if audio_path: cmd += ["-i", audio_path, "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else: cmd += ["-c", "copy"]
    tmp_out = "/tmp/_mux_%d.mp4" % os.getpid()   # faststart rewrite fails on the JuiceFS mount; mux locally then copy
    cmd += ["-movflags", "+faststart", tmp_out]
    subprocess.run(cmd, check=True)
    shutil.copyfile(tmp_out, out_mp4); os.remove(tmp_out)
    shutil.rmtree(tmp, ignore_errors=True)
    httpd.shutdown()
    el = time.time() - t
    return {"frames": n, "render_s": round(el, 1), "fps_render": round(n / el, 2)}

def still(timeline_path, t_sec, out_png, port=8766):
    """Render single frames for inspection."""
    from playwright.sync_api import sync_playwright
    httpd = serve(port)
    tl = json.load(open(timeline_path))
    ts = t_sec if isinstance(t_sec, list) else [t_sec]
    outs = []
    with sync_playwright() as p:
        b = p.chromium.launch(executable_path=CHROMIUM, args=["--no-sandbox", "--disable-gpu"])
        pg = b.new_page(viewport={"width": 1080, "height": 1920})
        pg.on("console", lambda m: print("console:", m.text))
        pg.on("pageerror", lambda e: print("pageerror:", e))
        pg.goto("http://127.0.0.1:%d/renderer/stage.html" % port)
        pg.evaluate("tl => window.STAGE_INIT(tl)", tl)
        for k, tt in enumerate(ts):
            d = pg.evaluate("([i,q]) => window.STAGE_FRAME(i,q)", [int(round(tt * tl["fps"])), 0.92])
            o = out_png.replace(".jpg", "_%02d.jpg" % k) if len(ts) > 1 else out_png
            open(o, "wb").write(base64.b64decode(d.split(",", 1)[1])); outs.append(o)
        b.close()
    httpd.shutdown()
    return outs

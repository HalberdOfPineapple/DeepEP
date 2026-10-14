"""Minimal HTML report writer sharing the look of the analysis note. Figures are inline SVG strings."""
import html, os, time

CSS = r"""
:root { --bg:#F5F6F8; --surface:#FFFFFF; --surface-2:#ECEFF3; --ink:#17202A; --muted:#5A6572; --line:#D5DBE2; --accent:#2B59C3; --accent-soft:#E3EAFA;
  --ok:#11806E; --ok-soft:#DDF2EE; --warn:#B0620B; --warn-soft:#FBEBD8; --bad:#B42A2A; --bad-soft:#F9E1E1; --type:#7446B5; --type-soft:#EDE5F8; --code-bg:#EEF1F5;
  --font-body:"IBM Plex Sans",system-ui,sans-serif; --font-head:"IBM Plex Sans Condensed","Arial Narrow",system-ui,sans-serif; --font-mono:"IBM Plex Mono",ui-monospace,Menlo,monospace; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { color-scheme:dark; --bg:#0E1318; --surface:#151C23; --surface-2:#1C252F; --ink:#E3E8EE; --muted:#97A2AE; --line:#2B3541; --accent:#8AABFF; --accent-soft:#1D2A48; --ok:#4FC8B2; --ok-soft:#123330; --warn:#E8A650; --warn-soft:#3A2A14; --bad:#F08A8A; --bad-soft:#3D1B1B; --type:#C4A7F4; --type-soft:#2A2040; --code-bg:#111920; } }
:root[data-theme="dark"] { color-scheme:dark; --bg:#0E1318; --surface:#151C23; --surface-2:#1C252F; --ink:#E3E8EE; --muted:#97A2AE; --line:#2B3541; --accent:#8AABFF; --accent-soft:#1D2A48; --ok:#4FC8B2; --ok-soft:#123330; --warn:#E8A650; --warn-soft:#3A2A14; --bad:#F08A8A; --bad-soft:#3D1B1B; --type:#C4A7F4; --type-soft:#2A2040; --code-bg:#111920; }
*{box-sizing:border-box} body{background:var(--bg);color:var(--ink);font-family:var(--font-body);font-size:16px;line-height:1.6;margin:0;padding:0 16px 64px}
main{max-width:900px;margin:0 auto;padding-top:36px} h1,h2,h3{font-family:var(--font-head);line-height:1.15} h1{font-size:clamp(30px,4.5vw,42px);margin:0 0 8px}
h2{font-size:26px;margin:48px 0 8px;padding-top:8px;border-top:1px solid var(--line)} h3{font-size:19px;margin:28px 0 6px} p{margin:0 0 14px} .lede{font-size:18px;color:var(--muted)}
.meta{font-size:13px;color:var(--muted);font-family:var(--font-mono);margin-bottom:24px} code{font-family:var(--font-mono);font-size:.88em;background:var(--code-bg);padding:1px 5px;border-radius:4px}
pre{font-family:var(--font-mono);font-size:12.5px;background:var(--code-bg);padding:12px 14px;border-radius:8px;overflow-x:auto;line-height:1.5} a{color:var(--accent)}
table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 18px} th,td{text-align:left;vertical-align:top;padding:7px 9px;border-bottom:1px solid var(--line)}
th{font-family:var(--font-head);font-size:13px;text-transform:uppercase;letter-spacing:.05em;color:var(--muted)} td.num,th.num{text-align:right;font-family:var(--font-mono);font-size:13px}
.card{background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:16px 20px;margin:14px 0 20px} .callout{border-left:4px solid var(--accent);background:var(--accent-soft);padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0}
.finding{border-left:4px solid var(--ok);background:var(--ok-soft);padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0} .warnbox{border-left:4px solid var(--warn);background:var(--warn-soft);padding:12px 16px;border-radius:0 8px 8px 0;margin:14px 0}
figure{margin:16px 0 22px} figure svg{max-width:100%;height:auto;display:block;background:var(--surface);border:1px solid var(--line);border-radius:10px;padding:8px}
figure svg.chart{padding:10px 6px} figcaption{font-size:13px;color:var(--muted);margin-top:8px}
svg text{font-family:var(--font-body);font-size:12px;fill:var(--ink)} svg .mono{font-family:var(--font-mono);font-size:11px} svg .small{font-size:10.5px;fill:var(--muted)} svg .head{font-family:var(--font-head);font-size:13px;font-weight:600}
svg .box{fill:var(--surface-2);stroke:var(--line)} svg .box-ok{fill:var(--ok-soft);stroke:var(--ok)} svg .box-warn{fill:var(--warn-soft);stroke:var(--warn)} svg .box-bad{fill:var(--bad-soft);stroke:var(--bad)} svg .box-acc{fill:var(--accent-soft);stroke:var(--accent)} svg .box-type{fill:var(--type-soft);stroke:var(--type)}
svg .ln{stroke:var(--ink);stroke-width:1.4;fill:none} svg .ln-acc{stroke:var(--accent);stroke-width:1.6;fill:none} svg .ln-warn{stroke:var(--warn);stroke-width:1.6;fill:none} svg .ln-bad{stroke:var(--bad);stroke-width:1.6;fill:none;stroke-dasharray:5 4} svg .ln-mut{stroke:var(--muted);stroke-width:1;fill:none;stroke-dasharray:3 3}
svg .fill-acc{fill:var(--accent)} svg .fill-warn{fill:var(--warn)} svg .fill-bad{fill:var(--bad)} svg .fill-ok{fill:var(--ok)} svg .fill-mut{fill:var(--muted)}
svg.chart text{fill:var(--ink)} svg.chart .ytick text, svg.chart .xtick text{fill:var(--muted)}
.tag{display:inline-block;font-family:var(--font-mono);font-size:11px;padding:1px 7px;border-radius:999px;border:1px solid var(--line);margin-right:4px} .tag.ok{background:var(--ok-soft);color:var(--ok);border-color:transparent} .tag.bad{background:var(--bad-soft);color:var(--bad);border-color:transparent} .tag.warn{background:var(--warn-soft);color:var(--warn);border-color:transparent}
ul,ol{padding-left:22px} li{margin-bottom:6px} .kv{display:grid;grid-template-columns:max-content 1fr;gap:4px 14px;font-size:14px;margin:8px 0 14px} .kv .k{color:var(--muted);font-family:var(--font-mono);font-size:12.5px}
"""


def page(title: str, lede: str, meta: str, body: str, description: str = '') -> str:
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>{html.escape(title)}</title><meta name="description" content="{html.escape(description or lede)}">
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans+Condensed:wght@500;600;700&family=IBM+Plex+Sans:ital,wght@0,400;0,500;0,600;1,400&display=swap">
<style>{CSS}</style></head><body><main>
<h1>{html.escape(title)}</h1><p class="lede">{lede}</p><div class="meta">{meta}</div>
{body}
<p class="meta">Generated {time.strftime('%Y-%m-%d %H:%M')} by the experiment's analysis script; raw data in results/.</p>
</main></body></html>'''


def figure(svg: str, caption: str) -> str:
    return f'<figure>{svg}<figcaption>{caption}</figcaption></figure>'


def table(headers, rows, num_cols=()):
    h = ''.join(f'<th class="{"num" if i in num_cols else ""}">{html.escape(str(c))}</th>' for i, c in enumerate(headers))
    b = ''
    for r in rows:
        b += '<tr>' + ''.join(f'<td class="{"num" if i in num_cols else ""}">{c if isinstance(c, str) and c.startswith("<") else html.escape(str(c))}</td>' for i, c in enumerate(r)) + '</tr>'
    return f'<table><thead><tr>{h}</tr></thead><tbody>{b}</tbody></table>'


PIPELINE_SVG = '''<svg viewBox="0 0 880 250" role="img" aria-label="Procedure shared by the dispatch experiments: build the instrumented kernel, start one process per GPU, warm up, time 20 untraced dispatches with Kineto, then record N traced dispatches each bracketed by a clock calibration, and analyse the saved traces">
<defs><marker id="pp" markerWidth="8" markerHeight="8" refX="6" refY="4" orient="auto"><path d="M0 0 L8 4 L0 8 z" class="fill-mut"/></marker></defs>
<rect x="15" y="20" width="150" height="62" rx="8" class="box-type"/><text x="90" y="42" text-anchor="middle" class="head">1 · build kernel</text><text x="90" y="60" text-anchor="middle" class="small">apply_trace_patch.py</text><text x="90" y="74" text-anchor="middle" class="small">dispatch.cuh + probes</text>
<rect x="185" y="20" width="160" height="62" rx="8" class="box"/><text x="265" y="42" text-anchor="middle" class="head">2 · 4 processes</text><text x="265" y="60" text-anchor="middle" class="small">one per GPU 4–7</text><text x="265" y="74" text-anchor="middle" class="small">random tokens + routing</text>
<rect x="365" y="20" width="120" height="62" rx="8" class="box"/><text x="425" y="42" text-anchor="middle" class="head">3 · warm-up</text><text x="425" y="60" text-anchor="middle" class="small">3 dispatches</text><text x="425" y="74" text-anchor="middle" class="small">(JIT, first touch)</text>
<rect x="505" y="20" width="170" height="62" rx="8" class="box-acc"/><text x="590" y="42" text-anchor="middle" class="head">4 · kernel time</text><text x="590" y="60" text-anchor="middle" class="small">20 dispatches under Kineto</text><text x="590" y="74" text-anchor="middle" class="small">probes off → "kernel µs"</text>
<rect x="695" y="20" width="170" height="62" rx="8" class="box-acc"/><text x="780" y="42" text-anchor="middle" class="head">5 · traced repeats</text><text x="780" y="60" text-anchor="middle" class="small">N dispatches, probes on</text><text x="780" y="74" text-anchor="middle" class="small">one trace file each</text>
<path d="M165 51 L185 51" class="ln-mut" marker-end="url(#pp)"/><path d="M345 51 L365 51" class="ln-mut" marker-end="url(#pp)"/><path d="M485 51 L505 51" class="ln-mut" marker-end="url(#pp)"/><path d="M675 51 L695 51" class="ln-mut" marker-end="url(#pp)"/>
<path d="M780 82 L780 100" class="ln-mut" marker-end="url(#pp)"/>
<rect x="15" y="100" width="850" height="84" rx="8" class="box"/><text x="25" y="118" class="head">one traced repeat</text>
<rect x="150" y="132" width="130" height="46" rx="6" class="box-type"/><text x="215" y="152" text-anchor="middle" class="small">clock calibration</text><text x="215" y="168" text-anchor="middle" class="small">ping-pong, all pairs</text>
<rect x="300" y="132" width="110" height="46" rx="6" class="box"/><text x="355" y="152" text-anchor="middle" class="small">zero the trace</text><text x="355" y="168" text-anchor="middle" class="small">+ barrier</text>
<rect x="430" y="132" width="150" height="46" rx="6" class="box-acc"/><text x="505" y="152" text-anchor="middle" class="small">buffer.dispatch(...)</text><text x="505" y="168" text-anchor="middle" class="small">kernel writes stamps</text>
<rect x="600" y="132" width="110" height="46" rx="6" class="box"/><text x="655" y="152" text-anchor="middle" class="small">copy trace</text><text x="655" y="168" text-anchor="middle" class="small">to host</text>
<rect x="730" y="132" width="125" height="46" rx="6" class="box-type"/><text x="792" y="152" text-anchor="middle" class="small">clock calibration</text><text x="792" y="168" text-anchor="middle" class="small">again</text>
<path d="M280 155 L300 155" class="ln-mut" marker-end="url(#pp)"/><path d="M410 155 L430 155" class="ln-mut" marker-end="url(#pp)"/><path d="M580 155 L600 155" class="ln-mut" marker-end="url(#pp)"/><path d="M710 155 L730 155" class="ln-mut" marker-end="url(#pp)"/>
<path d="M440 184 L440 200" class="ln-mut" marker-end="url(#pp)"/>
<rect x="15" y="200" width="850" height="40" rx="8" class="box-ok"/><text x="25" y="225" class="head">6 · analysis script</text><text x="200" y="225" class="small">reads every rank's trace, computes the durations defined below, draws the figures, writes report.html</text>
</svg>'''


def dispatch_procedure(what_varies: str, repeats: int, kineto_iters: int = 20, variant_note: str = '') -> str:
    """Step-by-step description of common/run_dispatch.py for the dispatch-based experiments."""
    return f'''
<h3>2.1 Procedure</h3>
{figure(PIPELINE_SVG, 'Figure M. The procedure every configuration goes through (common/run_dispatch.py). Steps 4 and 5 use the same compiled kernel; the probes are enabled by a magic word in the trace header, absent in step 4.')}
<ol>
<li><b>Build the kernel under test.</b> <code>common/apply_trace_patch.py</code> takes the unmodified DeepEP dispatch kernel and inserts timestamp probes: at a handful of places a thread reads the GPU's nanosecond clock (<code>%globaltimer</code>) and writes it to a trace buffer in global memory. {variant_note}The kernel logic is otherwise unchanged. DeepEP compiles it just in time on first use.</li>
<li><b>Start four processes, one per GPU 4–7.</b> Each creates a DeepEP buffer, draws 7168-wide random BF16 tokens and random top-8 routing (DeepEP's own generator, so the routing statistics match its tests), and allocates the trace buffer. The trace buffer is handed to the kernel through an existing optional argument whose allocation is made larger than the kernel expects.</li>
<li><b>Warm up.</b> Three dispatches, so that JIT compilation and first-touch page mapping are out of the way.</li>
<li><b>Measure kernel time with the probes off.</b> {kineto_iters} dispatches under the Kineto profiler, the same method DeepEP's test suite uses: before each one the L2 cache is flushed and the four ranks synchronise with an all-reduce, so each dispatch starts cold and together. The profiler reports the duration of the dispatch kernel on each GPU; the "kernel µs" in every table is that duration. The run is then repeated with the probes on to measure their overhead.</li>
<li><b>Record {repeats} traced dispatches.</b> For each one: measure the clock offset between every pair of GPUs (a 400-round ping-pong through shared memory, so that stamps from different GPUs can be compared), zero the trace buffer, synchronise, run one dispatch with the probes on, copy the trace buffer to the host, and measure the clock offsets again. The two offset measurements bracket the dispatch; their mean is used, and their difference bounds the clock drift during the dispatch.</li>
<li><b>Analyse.</b> All ranks' traces and kernel times are saved to one <code>.pt</code> file. The analysis script turns stamp pairs into durations (definitions in 2.2 below), pools them over ranks, warps and repeats, and draws the figures.</li>
</ol>
<p>{what_varies}</p>
'''

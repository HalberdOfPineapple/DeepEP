"""Rasterise every <svg> in the given HTML files to PNG for visual inspection (scratch dir)."""
import re, sys, os, cairosvg
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
css = open(os.path.join(os.path.dirname(__file__), 'report.py')).read()
for f in sys.argv[2:]:
    s = open(f).read(); tag = os.path.basename(os.path.dirname(os.path.abspath(f))) or 'x'
    for i, m in enumerate(re.finditer(r'<svg.*?</svg>', s, flags=re.S)):
        svg = m.group(0)
        # give hand-drawn svgs the page's class styles (light theme values) so boxes/text render
        style = '<style>text{font-family:sans-serif;font-size:12px;fill:#17202A}.mono{font-family:monospace;font-size:11px}.small{font-size:10.5px;fill:#5A6572}.head{font-size:13px;font-weight:600}.box{fill:#ECEFF3;stroke:#D5DBE2}.box-ok{fill:#DDF2EE;stroke:#11806E}.box-warn{fill:#FBEBD8;stroke:#B0620B}.box-bad{fill:#F9E1E1;stroke:#B42A2A}.box-acc{fill:#E3EAFA;stroke:#2B59C3}.box-type{fill:#EDE5F8;stroke:#7446B5}.ln{stroke:#17202A;stroke-width:1.4;fill:none}.ln-acc{stroke:#2B59C3;stroke-width:1.6;fill:none}.ln-warn{stroke:#B0620B;stroke-width:1.6;fill:none}.ln-bad{stroke:#B42A2A;stroke-width:1.6;fill:none;stroke-dasharray:5 4}.ln-mut{stroke:#5A6572;stroke-width:1;fill:none;stroke-dasharray:3 3}.fill-acc{fill:#2B59C3}.fill-warn{fill:#B0620B}.fill-bad{fill:#B42A2A}.fill-ok{fill:#11806E}.fill-mut{fill:#5A6572}</style>'
        if 'class="chart"' not in svg[:200]:
            # cairosvg's CSS support is partial: inline the class styles as attributes
            import re as _re
            rules = dict(_re.findall(r'\.([\w-]+)\{([^}]*)\}', style))
            rules['text'] = 'font-family:sans-serif;font-size:12px;fill:#17202A'
            def inline(m):
                classes = m.group(1).split(); st = ';'.join(rules.get(c, '') for c in classes if c in rules)
                return f'style="{st}"'
            svg = _re.sub(r'class="([^"]+)"', inline, svg)
            # text elements without a class get the default text style
            svg = _re.sub(r'<text (?![^>]*style=)', '<text style="font-family:sans-serif;font-size:12px;fill:#17202A" ', svg)
            # text inside a styled <g> inherits nothing in cairosvg: copy the g style onto its text children
            def gfix(m):
                gst = m.group(1); body = m.group(2)
                body = _re.sub(r'<text style="([^"]*)"', lambda t: f'<text style="{t.group(1)};{gst}"', body)
                return f'<g style="{gst}">{body}</g>'
            svg = _re.sub(r'<g style="([^"]*)">(.*?)</g>', gfix, svg, flags=_re.S)
        svg = svg.replace('<svg ', '<svg style="background:white" ', 1)
        p = os.path.join(out, f'{tag}_{i}.png')
        try:
            cairosvg.svg2png(bytestring=svg.encode(), write_to=p, output_width=1400, background_color='white')
        except Exception as e:
            print('fail', p, e)
    print(f, i + 1, 'figures')

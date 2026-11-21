// LaTeX -> SVG for the Discord bot.
//
// Two engines, in this order:
//
// 1. KaTeX parses and validates the LaTeX, so a broken expression is
//    reported instead of being typeset as literal text (which is what
//    happened with matplotlib's mathtext).
// 2. MathJax does the actual typesetting, because it lays out every
//    environment correctly.
//
// KaTeX alone is not enough: it rejects environments that MathJax handles
// perfectly (\begin{eqnarray}, \begin{multline}, \begin{align*}, ...), and
// those are exactly the ones models like to write. So a KaTeX rejection is
// not treated as fatal - MathJax gets the final say, and its output is
// checked for an error marker instead.
//
// Reads one JSON object per line on stdin and writes one JSON object per
// line on stdout, so one long-lived process can serve many renders.

const katex = require('katex');
const { mathjax } = require('mathjax-full/js/mathjax.js');
const { TeX } = require('mathjax-full/js/input/tex.js');
const { SVG } = require('mathjax-full/js/output/svg.js');
const { liteAdaptor } = require('mathjax-full/js/adaptors/liteAdaptor.js');
const { RegisterHTMLHandler } = require('mathjax-full/js/handlers/html.js');
const { AllPackages } = require('mathjax-full/js/input/tex/AllPackages.js');

const adaptor = liteAdaptor();
RegisterHTMLHandler(adaptor);

const tex = new TeX({ packages: AllPackages });
const svgOut = new SVG({ fontCache: 'none' });
const html = mathjax.document('', { InputJax: tex, OutputJax: svgOut });

function render(latex) {
  // 1. KaTeX validates the expression. Its verdict is advisory: MathJax
  //    supports environments KaTeX does not.
  let katexError = null;
  try {
    katex.renderToString(latex, { throwOnError: true, displayMode: true });
  } catch (e) {
    katexError = String((e && e.message) || e);
  }

  // 2. MathJax typesets it.
  const node = html.convert(latex, {
    display: true, em: 16, ex: 8, containerWidth: 2400,
  });
  const svg = adaptor.outerHTML(node);

  // MathJax reports its own errors inside the SVG rather than throwing.
  const m = svg.match(/data-mjx-error="([^"]*)"/);
  if (m) {
    throw new Error(`MathJax: ${m[1]}` + (katexError ? ` (KaTeX: ${katexError})` : ''));
  }
  if (!/<svg[\s>]/.test(svg)) {
    throw new Error('renderer produced no svg' + (katexError ? ` (KaTeX: ${katexError})` : ''));
  }
  return svg;
}

let buffer = '';
process.stdin.setEncoding('utf8');
process.stdin.on('data', (chunk) => {
  buffer += chunk;
  let index;
  while ((index = buffer.indexOf('\n')) !== -1) {
    const line = buffer.slice(0, index).trim();
    buffer = buffer.slice(index + 1);
    if (!line) continue;
    let out;
    try {
      const latex = JSON.parse(line).latex;
      out = { svg: render(latex) };
    } catch (e) {
      out = { error: String((e && e.message) || e) };
    }
    process.stdout.write(JSON.stringify(out) + '\n');
  }
});

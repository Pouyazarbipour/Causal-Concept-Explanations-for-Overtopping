// Renders a JSON block list (from build_content.py) to .docx in Times New Roman.
// Usage: node build_docx.js content.json out.docx
const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType, Table, TableRow, TableCell,
  WidthType, BorderStyle, ShadingType, ImageRun, LevelFormat, PageBreak, Footer, PageNumber,
} = require("docx");

const [, , inFile, outFile] = process.argv;
const spec = JSON.parse(fs.readFileSync(inFile, "utf8"));
const FONT = "Times New Roman";
const PAGE_W = 11906, MARGIN = 1440, TEXT_W = PAGE_W - 2 * MARGIN;   // A4, 1-inch margins

// inline markup: *italic*, **bold**, ^{sup}, _{sub}
function runs(text, base = {}) {
  const out = [];
  const re = /(\*\*[^*]+\*\*|\*[^*]+\*|\^\{[^}]+\}|_\{[^}]+\})/g;
  let last = 0, m;
  while ((m = re.exec(text)) !== null) {
    if (m.index > last) out.push(new TextRun({ text: text.slice(last, m.index), font: FONT, ...base }));
    const tok = m[0];
    if (tok.startsWith("**")) out.push(new TextRun({ text: tok.slice(2, -2), bold: true, font: FONT, ...base }));
    else if (tok.startsWith("*")) out.push(new TextRun({ text: tok.slice(1, -1), italics: true, font: FONT, ...base }));
    else if (tok.startsWith("^")) out.push(new TextRun({ text: tok.slice(2, -1), superScript: true, font: FONT, ...base }));
    else out.push(new TextRun({ text: tok.slice(2, -1), subScript: true, font: FONT, ...base }));
    last = m.index + tok.length;
  }
  if (last < text.length) out.push(new TextRun({ text: text.slice(last), font: FONT, ...base }));
  return out;
}

const border = { style: BorderStyle.SINGLE, size: 4, color: "000000" };
const none = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };

function table(b) {
  const n = b.header.length;
  const rel = b.widths || Array(n).fill(1);
  const tot = rel.reduce((a, c) => a + c, 0);
  const widths = rel.map((w) => Math.floor((w / tot) * TEXT_W));
  widths[widths.length - 1] += TEXT_W - widths.reduce((a, c) => a + c, 0);
  const fs = b.font_size || 18;
  const mk = (cells, isHead, isLast) => new TableRow({
    tableHeader: isHead,
    children: cells.map((c, i) => new TableCell({
      width: { size: widths[i], type: WidthType.DXA },
      margins: { top: 40, bottom: 40, left: 70, right: 70 },
      borders: { top: isHead ? border : none, bottom: (isHead || isLast) ? border : none, left: none, right: none },
      shading: isHead ? { type: ShadingType.CLEAR, fill: "F2F2F2", color: "auto" } : undefined,
      children: [new Paragraph({ alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER,
        children: runs(String(c), { size: fs, bold: isHead }) })],
    })),
  });
  const rows = [mk(b.header, true, false)].concat(b.rows.map((r, k) => mk(r, false, k === b.rows.length - 1)));
  return new Table({ width: { size: TEXT_W, type: WidthType.DXA }, columnWidths: widths, rows });
}

const children = [];
for (const b of spec.blocks) {
  if (b.t === "title") children.push(new Paragraph({ heading: HeadingLevel.TITLE, alignment: AlignmentType.CENTER, children: runs(b.text) }));
  else if (b.t === "subtitle") children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { after: 120 }, children: runs(b.text, { size: 22, italics: !!b.italic }) }));
  else if (b.t === "h1") children.push(new Paragraph({ heading: HeadingLevel.HEADING_1, children: runs(b.text) }));
  else if (b.t === "h2") children.push(new Paragraph({ heading: HeadingLevel.HEADING_2, children: runs(b.text) }));
  else if (b.t === "h3") children.push(new Paragraph({ heading: HeadingLevel.HEADING_3, children: runs(b.text) }));
  else if (b.t === "p") children.push(new Paragraph({ alignment: AlignmentType.JUSTIFIED, spacing: { after: 120, line: 300 }, children: runs(b.text) }));
  else if (b.t === "note") children.push(new Paragraph({ spacing: { after: 120 }, children: runs(b.text, { size: 20, color: "7A2E0E", italics: true }) }));
  else if (b.t === "eq") children.push(new Paragraph({ alignment: AlignmentType.CENTER, spacing: { before: 80, after: 120 }, children: runs(b.text, { italics: false }) }));
  else if (b.t === "bullets") for (const it of b.items) children.push(new Paragraph({ numbering: { reference: "bullets", level: b.level || 0 }, spacing: { after: 60, line: 290 }, alignment: AlignmentType.JUSTIFIED, children: runs(it) }));
  else if (b.t === "numbered") for (const it of b.items) children.push(new Paragraph({ numbering: { reference: b.ref || "numbers", level: 0 }, spacing: { after: 60, line: 290 }, children: runs(it) }));
  else if (b.t === "caption") children.push(new Paragraph({ keepNext: !!b.keepNext, spacing: { before: 120, after: 120 }, alignment: AlignmentType.LEFT, children: runs(b.text, { size: 20 }) }));
  else if (b.t === "table") {
    children.push(new Paragraph({ keepNext: true, spacing: { before: 160, after: 80 }, children: runs(b.caption, { size: 20 }) }));
    children.push(table(b));
    if (b.note) children.push(new Paragraph({ spacing: { before: 60, after: 160 }, children: runs(b.note, { size: 17 }) }));
    else children.push(new Paragraph({ spacing: { after: 120 }, children: [] }));
  } else if (b.t === "figure") {
    const img = fs.readFileSync(path.resolve(path.dirname(inFile), b.path));
    const w = Math.round((b.width_in || 6.3) * 96), h = Math.round(w * b.aspect);
    children.push(new Paragraph({ alignment: AlignmentType.CENTER, keepNext: true, spacing: { before: 160 },
      children: [new ImageRun({ type: "png", data: img, transformation: { width: w, height: h },
        altText: { title: b.alt || "figure", description: b.alt || "figure", name: path.basename(b.path) } })] }));
    children.push(new Paragraph({ spacing: { before: 80, after: 200 }, alignment: AlignmentType.JUSTIFIED, children: runs(b.caption, { size: 20 }) }));
  } else if (b.t === "pagebreak") children.push(new Paragraph({ children: [new PageBreak()] }));
  else if (b.t === "ref") children.push(new Paragraph({ spacing: { after: 80 }, indent: { left: 360, hanging: 360 }, alignment: AlignmentType.LEFT, children: runs(b.text, { size: 21 }) }));
}

const H = (size, before, after) => ({ run: { font: FONT, size, bold: true }, paragraph: { spacing: { before, after }, keepNext: true } });
const doc = new Document({
  creator: spec.author || "", title: spec.title || "", description: spec.description || "",
  styles: {
    default: { document: { run: { font: FONT, size: 24 } } },
    paragraphStyles: [
      { id: "Title", name: "Title", basedOn: "Normal", next: "Normal", run: { font: FONT, size: 34, bold: true }, paragraph: { spacing: { before: 0, after: 200 }, keepNext: true } },
      { id: "Heading1", name: "Heading 1", basedOn: "Normal", next: "Normal", quickFormat: true, ...H(28, 300, 140), paragraph: { spacing: { before: 300, after: 140 }, keepNext: true, outlineLevel: 0 } },
      { id: "Heading2", name: "Heading 2", basedOn: "Normal", next: "Normal", quickFormat: true, ...H(25, 220, 100), paragraph: { spacing: { before: 220, after: 100 }, keepNext: true, outlineLevel: 1 } },
      { id: "Heading3", name: "Heading 3", basedOn: "Normal", next: "Normal", quickFormat: true, run: { font: FONT, size: 24, bold: true, italics: true }, paragraph: { spacing: { before: 160, after: 80 }, keepNext: true, outlineLevel: 2 } },
    ],
  },
  numbering: { config: [
    { reference: "bullets", levels: [
      { level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 270 } } } },
      { level: 1, format: LevelFormat.BULLET, text: "–", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 1000, hanging: 270 } } } }] },
    { reference: "numbers", levels: [{ level: 0, format: LevelFormat.DECIMAL, text: "%1.", alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 540, hanging: 360 } } } }] },
  ] },
  sections: [{
    properties: { page: { size: { width: PAGE_W, height: 16838 }, margin: { top: MARGIN, bottom: MARGIN, left: MARGIN, right: MARGIN } } },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [new TextRun({ children: [PageNumber.CURRENT], font: FONT, size: 20 })] })] }) },
    children,
  }],
});
Packer.toBuffer(doc).then((buf) => fs.writeFileSync(outFile, buf));

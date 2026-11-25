const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const OUT = path.join(ROOT, 'outputs');

function parseLine(line) {
  const out = [];
  let cur = '', quoted = false;
  for (let i = 0; i < line.length; i++) {
    const ch = line[i];
    if (ch === '"') {
      if (quoted && line[i + 1] === '"') { cur += '"'; i++; }
      else quoted = !quoted;
    } else if (ch === ',' && !quoted) { out.push(cur); cur = ''; }
    else cur += ch;
  }
  out.push(cur);
  return out;
}
function readCsv(name) {
  const lines = fs.readFileSync(path.join(OUT, name), 'utf8').replace(/^\uFEFF/, '').trim().split(/\r?\n/);
  const header = parseLine(lines[0]);
  return lines.slice(1).filter(Boolean).map(line => {
    const vals = parseLine(line); const row = {};
    header.forEach((h, i) => row[h] = vals[i] ?? '');
    return row;
  });
}
function csvEscape(v) {
  const s = String(v ?? '');
  return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
}
function writeCsv(name, header, rows) {
  const body = [header, ...rows].map(r => r.map(csvEscape).join(',')).join('\n') + '\n';
  fs.writeFileSync(path.join(OUT, name), body, 'utf8');
}
function num(v) { return Number(v); }
function key(parts) { return parts.join('\u0001'); }
function conditionName(c) {
  return String(c).replace(/^blur(\d+)$/, 'blur_0$1');
}

const dino = readCsv('D_DINOv3_extra_drop_detail_dev.csv');
const groupByDog = new Map();
for (const r of dino) groupByDog.set(r.dog_id, r.breed_group);

const detail = [];
for (const r of dino) {
  for (const [metric, col] of [['capture_top3', 'extra_drop_capture_top3'], ['overlap', 'extra_drop_overlap'], ['kl', 'extra_drop_kl']]) {
    detail.push([r.extractor, r.seed, r.dog_id, r.breed_group, conditionName(r.condition), metric, num(r[col])]);
  }
}
const clipDino = readCsv('CLIP_DINOv2_extra_drop_detail_dev.csv');
for (const r of clipDino) {
  const group = groupByDog.get(r.dog_id) || 'unassigned';
  for (const [metric, col] of [['capture_top3', 'extra_drop_capture_top3'], ['overlap', 'extra_drop_overlap'], ['kl', 'extra_drop_kl']]) {
    detail.push([r.extractor, r.seed, r.dog_id, group, conditionName(r.condition), metric, num(r[col])]);
  }
}
const sift = readCsv('SIFT_extra_drop_detail_dev.csv');
for (const r of sift) {
  const group = groupByDog.get(r.dog_id) || 'unassigned';
  detail.push(['sift', r.seed, r.dog_id, group, conditionName(r.condition), r.metric, num(r.extra_drop)]);
}

writeCsv('D_breed_group_extra_drop_detail.csv',
  ['extractor', 'seed', 'dog_id', 'breed_group', 'condition', 'metric', 'extra_drop'], detail);

const buckets = new Map();
for (const r of detail) {
  const k = key([r[0], r[3], r[4], r[5]]);
  if (!buckets.has(k)) buckets.set(k, {extractor:r[0], breed_group:r[3], condition:r[4], metric:r[5], values:[], dogs:new Set()});
  const b = buckets.get(k); b.values.push(r[6]); b.dogs.add(r[2]);
}
function stats(values) {
  const mean = values.reduce((a,b)=>a+b,0)/values.length;
  const sd = Math.sqrt(values.reduce((a,b)=>a+(b-mean)**2,0)/Math.max(1, values.length-1));
  return {mean, sd, min:Math.min(...values), max:Math.max(...values)};
}
const summary = [...buckets.values()].sort((a,b)=>key([a.extractor,a.metric,a.condition,a.breed_group]).localeCompare(key([b.extractor,b.metric,b.condition,b.breed_group]))).map(b => {
  const s=stats(b.values);
  return [b.extractor,b.breed_group,b.condition,b.metric,s.mean,s.sd,s.min,s.max,b.values.length,b.dogs.size];
});
writeCsv('D_breed_group_extra_drop_summary.csv',
  ['extractor','breed_group','condition','metric','mean_extra_drop','sd_extra_drop','min_extra_drop','max_extra_drop','n_rows','n_dogs'], summary);

const status = {
  status: 'complete',
  source_detail_rows: detail.length,
  extractors: [...new Set(detail.map(r=>r[0]))].sort(),
  metrics: ['capture_top3','overlap','kl'],
  groups: [...new Set(detail.map(r=>r[3]))].sort(),
  note: 'The same dog-to-breed-group map is applied to all four extractor detail tables; SIFT blur names are normalized to blur_01/03/06/12.'
};
fs.writeFileSync(path.join(OUT,'D_breed_group_extra_drop_status.json'), JSON.stringify(status,null,2)+'\n', 'utf8');

// Simple SVG line chart: overlap extra-drop by breed group, one panel per extractor.
const extractors = ['clip','dinov2','dinov3','sift'];
const colors = ['#2563eb','#dc2626','#059669','#9333ea','#d97706'];
const W=1400,H=920, panelW=650,panelH=390;
let svg=[]; svg.push(`<svg xmlns="http://www.w3.org/2000/svg" width="${W}" height="${H}" viewBox="0 0 ${W} ${H}">`);
svg.push('<rect width="100%" height="100%" fill="white"/>');
for (let pi=0; pi<extractors.length; pi++) {
  const ex=extractors[pi], ox=(pi%2)*panelW+55, oy=Math.floor(pi/2)*panelH+45;
  const rows=summary.filter(r=>r[0]===ex && r[3]==='overlap');
  const conds=[...new Set(rows.map(r=>r[2]))]; const groups=[...new Set(rows.map(r=>r[1]))];
  const vals=rows.map(r=>Math.abs(Number(r[4]))); const ymax=Math.max(0.01,...vals)*1.25;
  const x0=ox+60,y0=oy+45,pw=panelW-100,ph=panelH-100;
  svg.push(`<text x="${ox}" y="${oy+15}" font-family="Arial" font-size="20" font-weight="bold">${ex} · overlap extra-drop</text>`);
  svg.push(`<line x1="${x0}" y1="${y0+ph}" x2="${x0+pw}" y2="${y0+ph}" stroke="#222"/><line x1="${x0}" y1="${y0}" x2="${x0}" y2="${y0+ph}" stroke="#222"/>`);
  for(let i=0;i<conds.length;i++) { const x=x0+(conds.length===1?0:i*pw/(conds.length-1)); svg.push(`<text x="${x}" y="${y0+ph+22}" text-anchor="middle" font-family="Arial" font-size="12" transform="rotate(25 ${x} ${y0+ph+22})">${conds[i]}</text>`); }
  for(let gi=0;gi<groups.length;gi++) { const g=groups[gi], points=[]; for(let i=0;i<conds.length;i++){ const r=rows.find(z=>z[1]===g&&z[2]===conds[i]); if(!r) continue; const x=x0+(conds.length===1?0:i*pw/(conds.length-1)); const y=y0+ph-(Number(r[4])+ymax)/(2*ymax)*ph; points.push(`${x},${y}`); } if(points.length){ const c=colors[gi%colors.length]; svg.push(`<polyline points="${points.join(' ')}" fill="none" stroke="${c}" stroke-width="2"/><text x="${x0+pw-150}" y="${y0+16+gi*16}" font-family="Arial" font-size="12" fill="${c}">${g}</text>`); } }
  svg.push(`<text x="${x0-40}" y="${y0+8}" font-family="Arial" font-size="11">+${ymax.toFixed(3)}</text><text x="${x0-25}" y="${y0+ph/2+4}" font-family="Arial" font-size="11">0</text><text x="${x0-40}" y="${y0+ph}" font-family="Arial" font-size="11">-${ymax.toFixed(3)}</text>`);
}
svg.push('</svg>');
fs.writeFileSync(path.join(OUT,'D_breed_group_extra_drop_overlap.svg'),svg.join('\n'),'utf8');
console.log(JSON.stringify({...status, summary_rows:summary.length},null,2));

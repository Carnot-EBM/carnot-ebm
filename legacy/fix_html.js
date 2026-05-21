const fs = require('fs');
const { marked } = require('marked');

const md = fs.readFileSync('docs/technical-report.md', 'utf8');
let htmlTemplate = fs.readFileSync('docs/technical-report.html', 'utf8');

// Update <title> and <meta> in htmlTemplate
htmlTemplate = htmlTemplate.replace(
  /A Technical Report — .*?\)/g,
  'A Technical Report — 2,907 Experiments Across the Public Record, 234 Archived Milestone Records, 25,306 Python Test Items Collected (Results and Ops Retros Through Exp 2205)'
);

const htmlContent = marked.parse(md);

const headerEnd = htmlTemplate.indexOf('<article class="markdown-body">') + '<article class="markdown-body">'.length;
const footerStart = htmlTemplate.lastIndexOf('</article>');

const header = htmlTemplate.substring(0, headerEnd);
const footer = htmlTemplate.substring(footerStart);

const newHtml = header + '\n' + htmlContent + '\n' + footer;

fs.writeFileSync('docs/technical-report.html', newHtml);
console.log('Rendered successfully!');

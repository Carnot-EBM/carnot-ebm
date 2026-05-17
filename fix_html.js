const fs = require('fs');
const { marked } = require('marked');

// Read the markdown content
const md = fs.readFileSync('docs/technical-report.md', 'utf8');
const htmlContent = marked.parse(md);

// Read the corrupted HTML template
let htmlTemplate = fs.readFileSync('docs/technical-report.html', 'utf8');

const headerEnd = htmlTemplate.indexOf('<article class="markdown-body">') + '<article class="markdown-body">'.length;
const footerStart = htmlTemplate.indexOf('</article>');

// Ensure we only use the first instance of header and footer
const header = htmlTemplate.substring(0, headerEnd);
let footer = htmlTemplate.substring(footerStart);

// Clean up footer if it contains trailing corrupted data after </html>
const htmlEnd = footer.indexOf('</html>') + '</html>'.length;
if (htmlEnd > '</html>'.length) {
    footer = footer.substring(0, htmlEnd);
}

const newHtml = header + '\n' + htmlContent + '\n' + footer;

fs.writeFileSync('docs/technical-report.html', newHtml);
console.log('Fixed and rendered successfully!');

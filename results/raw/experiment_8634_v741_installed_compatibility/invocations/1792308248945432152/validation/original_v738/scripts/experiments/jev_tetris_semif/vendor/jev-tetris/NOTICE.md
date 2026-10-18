# Vendored source notice

Source: https://github.com/thelau/jev-tetris  
Commit: `9869b602965cf002afff766013f8c068846d36aa`  
License: MIT. See `LICENSE`.

These six files are unmodified upstream code:

- `src/tetris.js`
- `src/describe.js`
- `src/heuristic.js`
- `src/words.js`
- `spike/play.js`
- `LICENSE`

Local additions are not upstream files. They include the loopback client, its
`src/jev.js` compatibility re-export, local harness, package metadata, and this
notice. The re-export satisfies the unchanged harness's static import without
restoring the upstream cloud client.

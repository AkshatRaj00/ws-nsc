Write-Host "1. Generating Figures from data/results.json..." -ForegroundColor Cyan
python src/plot_results.py

Write-Host "`n2. Project Setup is 100% Complete!" -ForegroundColor Green
Write-Host "   - Model Code:  src/ws_nsc.py"
Write-Host "   - Plot Code:   src/plot_results.py"
Write-Host "   - Real Data:   data/results.json"
Write-Host "   - Graphs:      paper_figures/ (fig1 & fig2 PNG/PDF ready)"

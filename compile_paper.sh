#!/bin/bash

echo "=================================================="
echo "COMPILING RESEARCH PAPER"
echo "=================================================="

# Check if pdflatex is available
if ! command -v pdflatex &> /dev/null; then
    echo "❌ pdflatex not found. Please install LaTeX distribution."
    echo "On Ubuntu/Debian: sudo apt-get install texlive-full"
    echo "On macOS: brew install mactex"
    exit 1
fi

# Check if bibtex is available
if ! command -v bibtex &> /dev/null; then
    echo "❌ bibtex not found. Please install LaTeX distribution."
    exit 1
fi

echo "✓ LaTeX tools found"

# Clean previous builds
echo "🧹 Cleaning previous builds..."
rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fdb_latexmk *.fls *.synctex.gz

# Compile paper (multiple passes for references)
echo "📝 Compiling paper (pass 1/4)..."
pdflatex -interaction=nonstopmode paper_main.tex > compile.log 2>&1

echo "📚 Processing bibliography..."
bibtex paper_main >> compile.log 2>&1

echo "📝 Compiling paper (pass 2/4)..."
pdflatex -interaction=nonstopmode paper_main.tex >> compile.log 2>&1

echo "📝 Compiling paper (pass 3/4)..."
pdflatex -interaction=nonstopmode paper_main.tex >> compile.log 2>&1

echo "📝 Final compilation (pass 4/4)..."
pdflatex -interaction=nonstopmode paper_main.tex >> compile.log 2>&1

# Check if PDF was generated successfully
if [ -f "paper_main.pdf" ]; then
    echo "✅ Paper compiled successfully!"
    echo "📄 Output: paper_main.pdf"
    
    # Show file size
    size=$(du -h paper_main.pdf | cut -f1)
    echo "📏 File size: $size"
    
    # Count pages
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo paper_main.pdf | grep "Pages:" | awk '{print $2}')
        echo "📃 Pages: $pages"
    fi
    
    echo ""
    echo "Paper successfully generated for ISAS 2025 submission!"
    
else
    echo "❌ Compilation failed!"
    echo "Check compile.log for errors:"
    tail -20 compile.log
    exit 1
fi

echo "=================================================="
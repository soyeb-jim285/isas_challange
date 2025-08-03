#!/bin/bash

echo "=================================================="
echo "COMPILING IOP CONFERENCE SERIES PAPER"
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
rm -f paper_iop.aux paper_iop.bbl paper_iop.blg paper_iop.log paper_iop.out paper_iop.toc paper_iop.fdb_latexmk paper_iop.fls paper_iop.synctex.gz

# Compile paper (multiple passes for references)
echo "📝 Compiling IOP paper (pass 1/4)..."
pdflatex -interaction=nonstopmode paper_iop.tex > compile_iop.log 2>&1

echo "📚 Processing bibliography..."
bibtex paper_iop >> compile_iop.log 2>&1

echo "📝 Compiling IOP paper (pass 2/4)..."
pdflatex -interaction=nonstopmode paper_iop.tex >> compile_iop.log 2>&1

echo "📝 Compiling IOP paper (pass 3/4)..."
pdflatex -interaction=nonstopmode paper_iop.tex >> compile_iop.log 2>&1

echo "📝 Final compilation (pass 4/4)..."
pdflatex -interaction=nonstopmode paper_iop.tex >> compile_iop.log 2>&1

# Check if PDF was generated successfully
if [ -f "paper_iop.pdf" ]; then
    echo "✅ IOP Paper compiled successfully!"
    echo "📄 Output: paper_iop.pdf"
    
    # Show file size
    size=$(du -h paper_iop.pdf | cut -f1)
    echo "📏 File size: $size"
    
    # Count pages
    if command -v pdfinfo &> /dev/null; then
        pages=$(pdfinfo paper_iop.pdf | grep "Pages:" | awk '{print $2}')
        echo "📃 Pages: $pages"
    fi
    
    echo ""
    echo "IOP Conference Series paper successfully generated!"
    echo "Ready for submission to ISAS 2025!"
    
else
    echo "❌ Compilation failed!"
    echo "Check compile_iop.log for errors:"
    tail -20 compile_iop.log
    exit 1
fi

echo "=================================================="
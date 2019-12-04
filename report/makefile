.PHONY: report.pdf all clean

all: report.pdf

report.pdf: report.tex
	latexmk -pdf -pdflatex="pdflatex -interaction=nonstopmode" -use-make $<

clean:
	latexmk -CA

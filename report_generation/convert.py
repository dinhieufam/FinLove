"""
Markdown to PDF Converter
Converts the report.md file to a professionally formatted PDF with Times New Roman, 12pt font.
"""

import os
import sys
from pathlib import Path

# Import required libraries for markdown to PDF conversion
try:
    import markdown
    from weasyprint import HTML, CSS
    # Try to import pypandoc for better LaTeX support
    try:
        import pypandoc
        PYPANDOC_AVAILABLE = True
    except ImportError:
        PYPANDOC_AVAILABLE = False
except ImportError:
    print("Error: Required libraries not found.")
    print("Please install them using: pip install markdown weasyprint")
    print("For better LaTeX support, also install: pip install pypandoc")
    sys.exit(1)


def create_css_style() -> str:
    """
    Creates professional CSS styling for the PDF report.
    Returns CSS string with Times New Roman font, 12pt size, and proper formatting.
    """
    css = """
    @page {
        size: A4;
        margin: 2.5cm 2cm 2.5cm 2cm;
        @top-center {
            content: "";
        }
        @bottom-center {
            content: "Page " counter(page);
            font-family: "Times New Roman", serif;
            font-size: 10pt;
        }
    }
    
    body {
        font-family: "Times New Roman", "Times", serif;
        font-size: 12pt;
        line-height: 1.6;
        color: #000000;
        text-align: justify;
    }
    
    /* Headings styling */
    h1 {
        font-size: 18pt;
        font-weight: bold;
        margin-top: 24pt;
        margin-bottom: 12pt;
        page-break-after: avoid;
        color: #000000;
    }
    
    h2 {
        font-size: 16pt;
        font-weight: bold;
        margin-top: 20pt;
        margin-bottom: 10pt;
        page-break-after: avoid;
        color: #000000;
    }
    
    h3 {
        font-size: 14pt;
        font-weight: bold;
        margin-top: 16pt;
        margin-bottom: 8pt;
        page-break-after: avoid;
        color: #000000;
    }
    
    h4 {
        font-size: 12pt;
        font-weight: bold;
        margin-top: 12pt;
        margin-bottom: 6pt;
        page-break-after: avoid;
        color: #000000;
    }
    
    /* Paragraphs */
    p {
        margin-top: 6pt;
        margin-bottom: 6pt;
        text-align: justify;
    }
    
    /* Lists */
    ul, ol {
        margin-top: 6pt;
        margin-bottom: 6pt;
        padding-left: 24pt;
    }
    
    li {
        margin-top: 3pt;
        margin-bottom: 3pt;
    }
    
    /* Tables */
    table {
        width: 100%;
        border-collapse: collapse;
        margin-top: 12pt;
        margin-bottom: 12pt;
        page-break-inside: avoid;
        font-size: 11pt;
    }
    
    th, td {
        border: 1px solid #000000;
        padding: 6pt 8pt;
        text-align: left;
    }
    
    th {
        background-color: #f0f0f0;
        font-weight: bold;
    }
    
    /* Horizontal rules */
    hr {
        border: none;
        border-top: 1px solid #000000;
        margin-top: 12pt;
        margin-bottom: 12pt;
    }
    
    /* Code blocks (if any) */
    code {
        font-family: "Courier New", monospace;
        font-size: 10pt;
        background-color: #f5f5f5;
        padding: 2pt 4pt;
    }
    
    pre {
        background-color: #f5f5f5;
        padding: 8pt;
        border: 1px solid #cccccc;
        overflow-x: auto;
        page-break-inside: avoid;
        font-size: 10pt;
    }
    
    /* Math equations (LaTeX) */
    .math {
        font-family: "Times New Roman", serif;
        font-style: italic;
    }
    
    .math.display {
        display: block;
        text-align: center;
        margin: 12pt 0;
    }
    
    .math.inline {
        display: inline;
    }
    
    /* Strong and emphasis */
    strong {
        font-weight: bold;
    }
    
    em {
        font-style: italic;
    }
    
    /* Links */
    a {
        color: #000000;
        text-decoration: underline;
    }
    
    /* Page breaks */
    .page-break {
        page-break-before: always;
    }
    """
    return css


def convert_markdown_to_pdf(markdown_path: str, output_path: str) -> None:
    """
    Converts a markdown file to a professionally formatted PDF.
    Supports LaTeX equations in the markdown.
    
    Args:
        markdown_path: Path to the input markdown file
        output_path: Path where the PDF will be saved
    """
    # Read the markdown file
    print(f"Reading markdown file: {markdown_path}")
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_content = f.read()
    except FileNotFoundError:
        print(f"Error: File '{markdown_path}' not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {e}")
        sys.exit(1)
    
    # Convert markdown to HTML
    print("Converting markdown to HTML...")
    
    # Use pypandoc if available for better LaTeX support
    if PYPANDOC_AVAILABLE:
        try:
            # Convert markdown to HTML with LaTeX math rendered as MathML (weasyprint compatible)
            html_content = pypandoc.convert_text(
                markdown_content,
                'html',
                format='markdown',
                extra_args=['--mathml', '--standalone']
            )
            print("Using pypandoc for LaTeX rendering (MathML format)")
        except Exception as e:
            print(f"Warning: pypandoc conversion failed: {e}")
            print("Falling back to standard markdown library...")
            # Fall back to standard markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])
            html_content = md.convert(markdown_content)
    else:
        # Use standard markdown library with math extension if available
        try:
            # Try to use markdown with math support
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br', 'pymdownx.arithmatex'])
            html_content = md.convert(markdown_content)
        except Exception:
            # Fall back to basic markdown
            md = markdown.Markdown(extensions=['tables', 'fenced_code', 'nl2br'])
            html_content = md.convert(markdown_content)
            print("Note: LaTeX equations may not render properly.")
            print("For full LaTeX support, install: pip install pypandoc")
            print("And ensure pandoc is installed: https://pandoc.org/installing.html")
    
    # Wrap HTML content with proper structure and CSS
    css_style = create_css_style()
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            {css_style}
        </style>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Convert HTML to PDF
    print(f"Generating PDF: {output_path}")
    try:
        HTML(string=full_html).write_pdf(output_path)
        print(f"✓ Successfully created PDF: {output_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")
        print("\nTroubleshooting tips:")
        print("1. Make sure weasyprint is properly installed: pip install weasyprint")
        print("2. On Linux, you may need: sudo apt-get install python3-cffi python3-brotli libpango-1.0-0 libpangoft2-1.0-0")
        sys.exit(1)


def main():
    """
    Main function to convert report.md to PDF.
    """
    # Get the directory where this script is located
    script_dir = Path(__file__).parent
    
    # Define input and output paths
    markdown_file = script_dir / "report.md"
    output_file = script_dir / "report.pdf"
    
    # Check if markdown file exists
    if not markdown_file.exists():
        print(f"Error: Markdown file not found at {markdown_file}")
        sys.exit(1)
    
    # Convert markdown to PDF
    convert_markdown_to_pdf(str(markdown_file), str(output_file))
    
    print(f"\nConversion complete! PDF saved to: {output_file}")


if __name__ == "__main__":
    main()


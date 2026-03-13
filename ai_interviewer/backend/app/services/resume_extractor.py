import pymupdf

def extract_text_from_pdf(contents: bytes) -> str:
    """
    Extracts text from a given PDF document byte stream.

    Args:
        contents (bytes): Byte content of the uploaded PDF file.

    Returns:
        str: Extracted text from all pages.
    """
    pdf_document = pymupdf.open(stream=contents, filetype="pdf")
    text_content = ""
    for page_num in range(len(pdf_document)):
        page = pdf_document[page_num]
        text_content += page.get_text()
        
    return text_content.strip()

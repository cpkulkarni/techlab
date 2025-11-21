from xhtml2pdf import pisa
import requests

def convert_url_to_pdf(url, pdf_path):
    # Fetch the HTML content from the URL
    response = requests.get(url)
    if response.status_code != 200:
        print("Failed to fetch URL:",url)
        return False
    
    html_content = response.text
    
    # Generate PDF
    with open(pdf_path, "wb") as pdf_file:
        pisa_status = pisa.CreatePDF(html_content, dest=pdf_file)
        
    return not pisa_status.err

# URL to fetch
url_to_fetch = "https://www.geeksforgeeks.org/python-program-for-word-guessing-game/"

# PDF path to save
pdf_path = "test.pdf"

# Generate PDF
if convert_url_to_pdf(url_to_fetch, pdf_path):
    print(f"PDF generated and saved at {pdf_path}")
else:
    print("PDF generation failed")
    

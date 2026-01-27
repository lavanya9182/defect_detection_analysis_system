from fpdf import FPDF
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import tempfile

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Pharma QA Inspection Report', 0, 1, 'C')
        self.ln(10)

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def generate_qa_report(period, stats, df_defects, charts):
    """
    Generate a PDF report.
    period: str (Daily/Weekly/Monthly)
    stats: dict (summary stats)
    df_defects: dataframe of recent defects
    charts: list of matplotlib figures
    """
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 1. Title & Info
    pdf.set_font("Arial", 'B', 12)
    pdf.cell(0, 10, f"Report Period: {period}", 0, 1)
    
    from datetime import datetime
    pdf.cell(0, 10, f"Generated On: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", 0, 1)
    pdf.ln(5)
    
    # 2. Summary Statistics
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Summary Statistics", 0, 1)
    pdf.set_font("Arial", size=12)
    
    data = [
        ["Total Capsules Inspected", str(stats['total'])],
        ["Good Capsules", str(stats['good_count'])],
        ["Defective Capsules", str(stats['defect_count'])],
        ["Defect Rate", f"{stats['defect_rate']:.2f}%"],
        ["Most Frequent Defect", str(stats['most_frequent_defect'])]
    ]
    
    # Simple table
    col_width = pdf.w / 2.5
    row_height = 8
    for row in data:
        pdf.cell(col_width, row_height, row[0], border=1)
        pdf.cell(col_width, row_height, row[1], border=1)
        pdf.ln(row_height)
        
    pdf.ln(10)
    
    # 3. QA Decision
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Assessment", 0, 1)
    pdf.set_font("Arial", size=12)
    
    # Dummy logic for decision: Reject if defect rate > 5%
    decision = "ACCEPT BATCH" if stats['defect_rate'] < 5.0 else "REJECT BATCH / REQUIRES REVIEW"
    color = (0, 150, 0) if stats['defect_rate'] < 5.0 else (200, 0, 0)
    
    pdf.set_text_color(*color)
    pdf.cell(0, 10, f"Conclusion: {decision}", 0, 1)
    pdf.set_text_color(0, 0, 0) # Reset
    pdf.ln(5)

    # 4. Charts
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Visual Analytics", 0, 1)
    pdf.ln(5)
    
    # Save charts to temp files to embed
    for fig in charts:
        if fig:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                fig.savefig(tmp_file.name, format="png")
                pdf.image(tmp_file.name, w=170)
                pdf.ln(10)
                # Cleanup handled by tempfile deletion usually, 
                # but delete=False means we should ideally clean up, 
                # but OS cleans /tmp eventually.
                
    # 5. Recent Defects Table
    pdf.add_page()
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(0, 10, "Recent Defect Log (Last 10)", 0, 1)
    pdf.ln(5)
    
    pdf.set_font("Arial", size=10)
    # Headers
    headers = ["Time", "ID", "Type", "Score", "Severity"]
    col_widths = [40, 40, 40, 30, 30]
    
    for i, h in enumerate(headers):
        pdf.cell(col_widths[i], 8, h, 1, 0, 'C')
    pdf.ln()
    
    # Rows
    if not df_defects.empty:
        # Take last 10
        recent = df_defects.tail(10)
        for index, row in recent.iterrows():
            ts = row['timestamp'].strftime('%Y-%m-%d %H:%M') if hasattr(row['timestamp'], 'strftime') else str(row['timestamp'])
            
            pdf.cell(col_widths[0], 8, ts, 1)
            pdf.cell(col_widths[1], 8, str(row['image_id'])[:15], 1) # Truncate ID
            pdf.cell(col_widths[2], 8, str(row['defect_type']), 1)
            pdf.cell(col_widths[3], 8, f"{row['score']:.2f}", 1)
            pdf.cell(col_widths[4], 8, str(row['severity']), 1)
            pdf.ln()
            
    output_filename = f"QA_Report_{period}_{datetime.now().strftime('%Y%m%d')}.pdf"
    pdf.output(output_filename)
    return output_filename
